"""``get_config()`` for OGPO + SteerVLA on CARLA Bench2Drive.

Use with ``--agent=impls/configs/ogpo_carla_config.py`` (or ``--agent-mode ogpo``).

SteerVLA is **enabled by default** — it provides high-quality critic bootstrap
targets via VLA next-state actions from the first training step
(``OGPOAgent.update_with_vla``).  Set ``steervla.enabled=False`` to fall back
to standalone OGPO with a 2-d ``[accel, steer]`` action space and
``OGPOAgent.update`` for critic / PPO.

Key differences from DSRL config:
  - No separate NoiseActor: OGPO's FlowActor operates directly in the VLA
    chunk action space (``vla_action_horizon × vla_action_dim`` = 40-d).
  - PPO group sampling (``grpo_num_samples=32``) replaces DSRL's SAC noise actor.
  - ``succ_buffer_capacity > 0`` enables OGPO+ success-buffer BC loss.
"""

from __future__ import annotations

import ml_collections

from jax_agents import ogpo as ogpo_agent


def get_config():
    config = ogpo_agent.get_config()

    # Online CARLA defaults (tuned for PPO group sampling cost)
    config.batch_size = 64
    config.buffer_capacity = 5_000
    config.warmup_steps = 500
    config.updates_per_step = 1
    config.enable_updates = True
    config.image_log_curr_interval = 10

    # Paper-canonical PPO hyperparameters (OGPO, Sec. 4)
    config.grpo_num_samples = 16
    config.clip_epsilon = 0.01
    config.ema_decay = 0.995
    config.conservative_advantage = True
    config.bc_coeff = 1.0
    config.q_variance_reduction = True
    config.n_vr_samples = 4
    config.critic_agg = "subsample"

    # χ² regularization: recommended for pixel-based tasks (OGPO paper Sec. 4, Appendix E.2).
    # Maintains a slow EMA policy (τ_slow=0.999 ≪ τ_ema=0.995) and penalizes drift from it.
    config.chi2_reg = True
    config.chi2_beta_init = 1.0
    config.slow_ema_decay = 0.999

    # VLA action space: chunk layout matching SteerVLA output
    config.vla_action_horizon = 10
    config.vla_action_dim = 4
    config.critic_action_dim = 4

    # SigLIP image embeddings
    config.observation_mode = "image"
    config.image_encoder = "siglip"
    config.siglip_model_id = "google/siglip2-so400m-patch14-384"
    config.siglip_include_prompt_subtask = True
    config.siglip_device = "cuda:0"

    config.training_gpu_rank = 0

    # No language critic feedback by default (OGPO relies on Q-values alone)
    config.critic_feedback_mode = "none"
    config.language_label_dim = 0

    # Success buffer for OGPO+ BC loss.
    # Transitions from successful episodes are sampled as ``succ_batch`` in
    # agent.update() / agent.update_with_vla().  Set to 0 to disable.
    config.succ_buffer_capacity = 10_000

    # SteerVLA rollout + critic bootstrap (enabled by default).
    # Disable with ``steervla.enabled=False`` for standalone OGPO (2-d actions).
    config.steervla = ml_collections.ConfigDict(
        dict(
            enabled=True,
            actor_config="pi05_steervla_cot_simplified_reasoning",
            checkpoint="gs://cat-logs/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning_20260523_222304/8000",
            routing_command="Follow the route and stay in lane.",
            cot_temperature=0.0,
            include_ego_history=False,
            proprio_norm=True,
            use_pi_action_chunk_for_env=True,
            action_horizon=10,
            action_dim=4,
            actions_per_model_query=1,
            actions_per_cot=1,
            output_action_format="DELTA_XY_T_DELTA_XY_SPACE",
            sample_actions_num_steps=10,
        )
    )

    return config
