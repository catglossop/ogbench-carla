"""GRPO on the SimLingo SteerVLA high-level planner, scored by a VLM critic.

Use with ``--agent=impls/configs/steervla_grpo_config.py``. ``main_carla`` dispatches to the GRPO path
when ``online_training_mode == "grpo_hl"`` (see :func:`main_carla._run_grpo_entry`). At each scored
decision state along a single rolling rollout, the actor samples ``grpo.group_size`` candidate CoTs for
the current observation and a VLM critic scores each candidate subtask in [0, 1] from the current frame +
env-reward context (speed / route progress / cumulative + last-step reward). The scores form the group
(``A_k=(s_k-mean)/std``); the K candidate CoTs are recorded with those advantages, the top-scored
candidate is executed through the frozen SimLingo low-level policy, and
:meth:`vlas.simlingo_steervla.SimLingoSteerVLAActor.update_hl_grpo` updates only the high-level planner.

Requires ``steervla.load_trainable_params=True`` and a positive ``grpo.score_temperature``. The VLM
critic uses the inherited ``vlm_coach`` block (provider / model).
"""

import ml_collections

from configs.steervla_dsrl_config import get_config as get_dsrl_config
from configs.simlingo_steervla_cast_relabel_train_config import apply_simlingo_steervla


def get_config():
    config = get_dsrl_config()

    # GRPO runs its own loop; the DSRL critic is unused (the VLM critic is built directly from the
    # vlm_coach block in _run_grpo_entry). Keep the DSRL critic language source off.
    config.language_feedback.source = "expert"
    config.language_feedback.expert_mode = "none"
    config.online_training_mode = "grpo_hl"
    config.training_gpu_rank = 0
    config.siglip_device = "cuda:0"

    # Env-step budget (total across all groups); one rollout ends at the route's terminal/truncation.
    config.buffer_capacity = 1_000  # unused by the GRPO loop; kept so shared plumbing stays happy.

    # Rollout video (first rollout of each group is captured and logged to W&B as grpo/rollout_video).
    config.log_episode_video = True
    config.episode_video_fps = 10.0
    config.episode_video_every = 2

    # Train the SimLingo high-level CoT planner; keep its low-level waypoint policy frozen.
    apply_simlingo_steervla(config.steervla)
    config.steervla.load_trainable_params = True
    config.steervla.actions_per_cot = 5
    # Keep the HL update on its dedicated torch device. run_carla.sh --hl-gpu overrides this.
    config.steervla.hl_training_gpu_rank = 1
    # Candidate groups are pooled and microbatched by SimLingoSteerVLAActor.update_hl_grpo.
    config.steervla.hl_update_batch_size = 32
    config.steervla.hl_micro_batch_size = 8
    config.steervla.hl_lr = 1e-5
    config.steervla.hl_log_kl = True

    config.grpo = ml_collections.ConfigDict(
        dict(
            # Drive the GREEDY base (cot_temperature=0, no scoring/pooling/updates) for the first N env
            # steps to log an in-run base baseline before GRPO exploration begins. 0 = off.
            warmup_steps=50,
            # Which scored candidate to EXECUTE (pooling/advantages/updates are unaffected): "argmax"
            # (VLM-best, default), "random" (diagnostic: isolates selection bias), or "first".
            select_mode="argmax",
            # Candidate CoTs sampled + VLM-scored per decision state (all K scored in one call); >=2.
            group_size=4,
            # Candidate sampling temperature; >0 for diversity (greedy -> zero-variance scores).
            score_temperature=1.0,
            # GRPO needs per-state candidate comparisons; keep scoring dense enough for learning,
            # while reducing VLM cost from every control step.
            score_every=20,
            # Pool eight scored states (32 records at group_size=4), matching the HL update batch size.
            update_every_states=8,
            # KL(πθ‖π_ref) penalty weight vs the frozen base HL policy (0 = pure REINFORCE).
            beta_kl=0.01,
            # Passes over the pooled group per update (minibatched inside update_hl_grpo).
            num_update_steps=1,
            advantage_eps=1e-6,
            # VLM scoring attempts before giving up on a state (transient/malformed replies get retried;
            # after this many failures the state is skipped and the base policy drives that step).
            vlm_score_retries=3,
            # Export the SimLingo HL to <save_dir>/steervla_hl_ckpt/<step> every N env steps (0 = final only).
            checkpoint_every_steps=2000,
            # Debug stop task: env reward -> -ego_speed and the VLM is told to prefer stopping (verifies
            # the policy learns to select slow/stopping candidates from the surfaced reward).
            debug_task=False,
            # Debug: replace one sampled candidate with a canned stop CoT (below) so we can check the
            # policy learns to select a known-good option. Works with or without debug_task.
            inject_stop_candidate=False,
            stop_reasoning="The vehicle must come to a stop. Brake to a complete stop.",
            stop_subtask="The vehicle comes to a complete stop and remains stationary.",
        )
    )

    return config
