"""``get_config()`` for YAY-Robot foresight correction with the SimLingo SteerVLA actor.

The SimLingo twin of ``steervla_yay_robot_train_config.py``, and the YAY twin of
``simlingo_steervla_cast_relabel_train_config.py``: the same foresight loop -- every CoT query is
shown to a VLM, which keeps or corrects the language, and the correction conditions the policy on
the spot -- with the OpenPI Pi0-CoT actor swapped for SimLingo's HL planner -> LL waypoint policy.

The swap is genuinely small, because SimLingo's HL speaks plain text:
``SimLingoSteerVLAActor._maybe_intervene_on_cot`` applies the same ``cot_intervention_fn`` inside
``_sample_cot``, and the corrected subtask reaches the LL through the ``Command:`` line rather than
through re-tokenized CoT tokens. The stored HL samples are the same
``steervla_hl_dataset_format`` ones, which SimLingo's ``update_hl`` already reads (it takes the
rollout ``original_reasoning`` as context and supervises only the corrected subtask, since a
SteerVLA-style reasoning string is not SimLingo's answer format).

Two things differ from the OpenPI config and both are about call rate:

* SimLingo re-plans its HL every ``actions_per_cot`` LL queries, and the LL runs every
  ``actions_per_model_query`` env steps, so a CoT is queried every ~6 env steps rather than ~25.
  Left alone that is a blocking Gemini call four times more often than the OpenPI arm.
  ``query_every_n_cot_queries`` below thins it back to roughly one correction per 24 env steps.
* ``image_key`` is the native 1024x512 ``image_viz``; ``main_carla`` stores that frame in the HL
  samples for both coaches.

Needs ``uv sync --extra all-gpu --extra simlingo``, the simlingo-steervla checkout, and
``GEMINI_API_KEY``. For replay, build a pool with ``impls/vlas/extract_simlingo_hl_replay.py``.

    ./run_carla.sh --agent-config impls/configs/simlingo_steervla_yay_robot_train_config.py \\
      --train-gpu 0 --hl-gpu 1 --render-adapter 4 ...
"""

from configs.simlingo_steervla_cast_relabel_train_config import apply_simlingo_steervla
from configs.steervla_yay_robot_train_config import get_config as get_yay_robot_train_config

# HL update cadence of the b2d/f2d sweeps: 10 gradient steps every 200 env steps.
HL_UPDATE_EVERY_ENV_STEPS = 200
HL_UPDATE_NUM_STEPS = 10
# One correction per this many env steps, approximately. With the SimLingo cadence below a CoT is
# queried every ~6 env steps, so 4 CoT queries is ~24 env steps -- the OpenPI arm's rate.
TARGET_ENV_STEPS_PER_CORRECTION = 24


def get_config():
    config = get_yay_robot_train_config()
    s = apply_simlingo_steervla(config.steervla)

    # HL (torch) update, converted from env steps exactly as the SimLingo CAST config does so the
    # two arms take the same number of gradient steps per env step.
    update_interval = max(1, int(config.get("update_interval", 1)))
    updates_per_step = max(1, int(config.get("updates_per_step", 1)))
    s.hl_update_every = max(1, round(HL_UPDATE_EVERY_ENV_STEPS * updates_per_step / update_interval))
    s.hl_update_num_steps = HL_UPDATE_NUM_STEPS
    s.hl_lr = 1e-5
    s.hl_weight_decay = 0.1
    s.hl_grad_clip = 1.0
    s.hl_update_batch_size = 64
    s.hl_micro_batch_size = 8
    s.hl_freeze_regexes = None  # OpenPI-only; SimLingo trains its own trainable set.

    # Replay pools are SimLingo-format HL frames, not OpenPI ones.
    s.hl_replay_root = "/raid/users/cglossop/simlingo_hl_pools"
    s.hl_replay_pools = [dict(name="simlingo_hl_simplified", weight=0.5)]
    s.hl_online_weight = 0.5
    s.hl_online_bad_fraction = 0.90
    s.hl_online_precursor_fraction = 60.0 / 90.0
    s.use_adaptive_sampling = False

    # Thin the corrections back to the OpenPI arm's rate (see the module docstring). SimLingo ages
    # the held CoT by one per ENV step but can only re-plan on an LL query, so the CoT period is
    # the ``actions_per_model_query`` multiple at or above ``actions_per_cot`` -- 6 env steps at
    # 5/3. NOT their product: that reads as 15 and thins the corrections to half the intended rate.
    _ll_period = max(1, int(s.get("actions_per_model_query", 3)))
    _cot_age = max(1, int(s.get("actions_per_cot", 5)))
    _cot_period = -(-_cot_age // _ll_period) * _ll_period
    config.yay_robot.query_every_n_cot_queries = max(
        1, round(TARGET_ENV_STEPS_PER_CORRECTION / _cot_period)
    )
    # The lead-up window is in seconds, so it needs no adjustment -- but the stride should stay at
    # one action chunk of THIS actor, which shares action_horizon with the OpenPI arm.
    config.yay_robot.action_chunk_steps = config.action_horizon
    config.yay_robot.hl_action_dim = s.action_dim

    return config
