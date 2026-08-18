"""``get_config()`` for GRPO on the SteerVLA high-level (CoT/subtask) policy, scored by a VLM critic.

Use with ``--agent=impls/configs/steervla_grpo_config.py``. ``main_carla`` dispatches to the GRPO path
when ``online_training_mode == "grpo_hl"`` (see :func:`main_carla._run_grpo_entry`). At each scored
decision state along a single rolling rollout, the actor samples ``grpo.group_size`` candidate CoTs for
the current observation and a VLM critic scores each candidate subtask in [0, 1] from the current frame +
env-reward context (speed / route progress / cumulative + last-step reward). The scores form the group
(``A_k=(s_k-mean)/std``); the K candidate CoTs are recorded with those advantages, the top-scored
candidate is executed to advance the episode, and :meth:`vlas.steervla.SteerVLAActor.update_hl_grpo`
takes a group-relative policy-gradient step on the CoT tokens with a KL penalty to a frozen reference.
The action expert is never modified (the CoT cross-entropy is independent of it).

Requires ``steervla.load_trainable_params=True`` (the HL backbone must be a full OpenPI TrainState) and a
positive ``grpo.score_temperature`` (greedy candidates are identical -> zero-variance scores -> no signal).
The VLM critic uses the inherited ``vlm_coach`` block (provider / model).
"""

import ml_collections

from configs.steervla_dsrl_config import get_config as get_dsrl_config


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

    # Trainable HL backbone (full OpenPI TrainState) + the frozen KL reference snapshot are built in
    # SteerVLAActor.setup when load_trainable_params=True.
    config.steervla.load_trainable_params = True
    # Isolate the HL (VLM-backbone) gradient step on its own JAX GPU (index into jax.devices("gpu"),
    # -1 keeps it on training_gpu_rank. run_carla.sh --hl-gpu overrides.
    config.steervla.hl_training_gpu_rank = 1
    # Freeze the memory-heavy pretrained subtrees for the HL step (SigLIP tower + tied token embedder);
    # the CoT policy gradient only needs the LLM transformer blocks + CoT heads. [] = full fine-tune.
    config.steervla.hl_freeze_regexes = [".*img.*", ".*embedder.*"]
    # Minibatch size for the GRPO update: a pooled group of candidate CoTs can exceed one Pi0-CoT
    # forward+backward, so update_hl_grpo shuffles the pooled CoTs into minibatches of this size.
    config.steervla.hl_update_batch_size = 32
    # Flat HL learning rate for RL fine-tuning. The pretraining schedule ramps to 1e-4 (BC rate); that
    # hot LR drove the observed late-run CoT collapse, so GRPO pins a small constant rate instead.
    config.steervla.hl_lr = 1e-5

    config.grpo = ml_collections.ConfigDict(
        dict(
            # Candidate CoTs sampled + VLM-scored per decision state (all K scored in one call); >=2.
            group_size=8,
            # Candidate sampling temperature; >0 for diversity (greedy -> zero-variance scores).
            score_temperature=1.0,
            # Score every N env steps; the frozen base policy drives the rest. 1 = every decision state.
            score_every=1,
            # Scored states (K candidates each) pooled before one update_hl_grpo step.
            update_every_states=4,
            # KL(πθ‖π_ref) penalty weight vs the frozen base HL policy (0 = pure REINFORCE).
            beta_kl=0.01,
            # Passes over the pooled group per update (minibatched inside update_hl_grpo).
            num_update_steps=1,
            advantage_eps=1e-6,
            # VLM scoring attempts before giving up on a state (transient/malformed replies get retried;
            # after this many failures the state is skipped and the base policy drives that step).
            vlm_score_retries=3,
            # Export the HL backbone to <save_dir>/steervla_hl_ckpt/<step> every N env steps (0 = final only).
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
