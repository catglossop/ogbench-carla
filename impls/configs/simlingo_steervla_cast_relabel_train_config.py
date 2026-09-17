"""``get_config()`` for CAST-relabel HL training with the SimLingo SteerVLA (``vlas/simlingo_steervla.py``).

Identical run to ``steervla_cast_relabel_train_config.py`` -- DSRL rollout with RL/BC off, CAST relabel
windows writing HL samples, ``update_hl`` on the configured cadence, periodic checkpoints -- with the
OpenPI Pi0-CoT actor swapped for the SimLingo HL planner -> LL waypoint policy. Only the ``steervla``
block changes; the swap itself is ``steervla.vla``. The LL is frozen; the HL trains the same parameter
set SimLingo training did (LoRA + vision tower).

    ./run_carla.sh --agent-config impls/configs/simlingo_steervla_cast_relabel_train_config.py \\
      --train-gpu 0 --hl-gpu 1 --render-adapter 4 ...

Needs ``uv sync --extra all-gpu --extra simlingo`` and, for replay, a pool from
``impls/vlas/extract_simlingo_hl_replay.py``.
"""

from configs.steervla_cast_relabel_train_config import get_config as get_cast_relabel_train_config

_CKPT_ROOT = "/raid/users/celine/steervla-ckpts"
# HL update cadence of the b2d/f2d sweeps (steervla_cast_relabel_hl200x10_adaptive_config.py):
# 10 gradient steps every 200 env steps.
HL_UPDATE_EVERY_ENV_STEPS = 200
HL_UPDATE_NUM_STEPS = 10
# Low-level (waypoint) query cadence in env steps; chunks are held and re-anchored in between.
ACTIONS_PER_MODEL_QUERY = 3


def apply_simlingo_steervla(s):
    """Point a ``config.steervla`` block at the SimLingo HL -> LL policy (shared with the residual config)."""
    s.vla = "simlingo_steervla"
    s.simlingo_source_root = "/home/cglossop/simlingo-steervla"
    s.hl_checkpoint = f"{_CKPT_ROOT}/2026_05_24_06_52_33_simlingo_seed1_bellman/checkpoints/epoch=019.ckpt"
    s.ll_checkpoint = f"{_CKPT_ROOT}/2026_05_23_21_39_41_simlingo_ll_vla_meta_conditioned/checkpoints/epoch=029.ckpt"
    # main_carla's generic "is a VLA configured" gate; OpenPI-only keys are unused by this actor.
    s.checkpoint = s.hl_checkpoint
    s.actor_config = ""
    # Native 1024x512 rgb_front; same mount/fov as SimLingo's rgb_simlingo camera. Also the frame
    # cast_relabel stores in HL samples (main_carla reads steervla.image_key there).
    s.image_key = "image_viz"
    # HL CoT decoding: 0 = greedy (run_carla.sh --cot-temperature overrides).
    s.cot_temperature = 0.0
    s.actions_per_model_query = ACTIONS_PER_MODEL_QUERY
    return s


def get_config():
    config = get_cast_relabel_train_config()
    s = apply_simlingo_steervla(config.steervla)

    # HL (torch) update. hl_training_gpu_rank (inherited) places the HL model on its own GPU.
    # ``hl_update_every`` counts update_with_vla calls, not env steps -- converted exactly as in
    # steervla_cast_relabel_hl200x10_adaptive_config.py so it tracks update_interval / updates_per_step.
    update_interval = max(1, int(config.get("update_interval", 1)))
    updates_per_step = max(1, int(config.get("updates_per_step", 1)))
    s.hl_update_every = max(1, round(HL_UPDATE_EVERY_ENV_STEPS * updates_per_step / update_interval))
    s.hl_update_num_steps = HL_UPDATE_NUM_STEPS
    s.hl_lr = 1e-5
    s.hl_weight_decay = 0.1
    s.hl_grad_clip = 1.0
    # HL batch recipe of the kl005 sweep (steervla_cast_relabel_hl16_mix5050_config.py): 64 rows, half
    # online / half replay, 90% corrective within the online half (2/3 of that precursor). Peak memory
    # is set by hl_micro_batch_size, not the batch size.
    s.hl_update_batch_size = 64
    s.hl_micro_batch_size = 8
    s.hl_freeze_regexes = None  # OpenPI-only; SimLingo trains its own trainable set.

    # Replay: SimLingo HL training frames (build once with extract_simlingo_hl_replay.py). Missing pools
    # only warn and fall back to online-only.
    s.hl_replay_root = "/raid/users/cglossop/simlingo_hl_pools"
    # The kl005 run split its 0.5 replay share 0.4 SimLingo / 0.1 simplified-reasoning across OpenPI pools;
    # the SimLingo HL pool (simplified reasoning, SimLingo targets) takes the whole share here.
    s.hl_replay_pools = [dict(name="simlingo_hl_simplified", weight=0.5)]
    s.hl_online_weight = 0.5
    s.hl_online_bad_fraction = 0.90
    s.hl_online_precursor_fraction = 60.0 / 90.0
    s.use_adaptive_sampling = False  # kl005 had it on; off by choice for the SimLingo runs.

    # The HL (actions_per_cot, inherited 5) is only re-planned on an LL query step once it is that old,
    # so with the LL every 3 steps the HL refreshes every 6.
    return config
