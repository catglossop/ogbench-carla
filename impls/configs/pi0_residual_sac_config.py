"""Config for Pi0-based Residual SAC on CARLA Bench2Drive.

Extends the SteerVLA DSRL config with residual SAC training mode:
  - Pi0 (SteerVLA) is kept fully frozen; no flow / noise-actor updates.
  - A small residual MLP is trained via SAC Q-gradient from the DSRL critic.
  - The critic sees ``obs_encoder(obs) + language_label`` and evaluates Q on
    the composed action ``base + residual * scale`` (physical units; no clip).

Use with:
    uv run python impls/main_carla.py \\
      --agent=impls/configs/pi0_residual_sac_config.py \\
      --route=parking-cut-in-001 \\
      --online_steps=50000
Or via run_carla.sh:
    bash run_carla.sh --train-mode sac_residual --agent-config impls/configs/pi0_residual_sac_config.py
"""

from __future__ import annotations

from pathlib import Path
import runpy

import ml_collections

_BASE_PATH = Path(__file__).parent / "steervla_dsrl_config.py"
_BASE_GET_CONFIG = runpy.run_path(str(_BASE_PATH))["get_config"]


def get_config():
    config = _BASE_GET_CONFIG()

    # ── Training mode ──────────────────────────────────────────────────────
    # Pi0 frozen; only the residual MLP and DSRL critic are updated.
    config.online_training_mode = "sac_residual"

    # ── Observation for DSRL critic / obs_encoder ──────────────────────────
    # "policy_embed": pooled frozen Pi0-CoT prefix hidden (fastest, no IMPALA CNN).
    # "image": IMPALA CNN trained from raw CARLA RGB (more expressive, slower).
    config.observation_mode = "policy_embed"
    config.policy_embed_dim = 2048

    # ── Residual actor observation ─────────────────────────────────────────
    # When True, the raw CARLA ego-kinematics + routing-command one-hot (25-dim)
    # is appended to the DSRL obs_encoder output so the residual MLP conditions
    # on both the frozen Pi0 semantic features AND explicit proprioceptive state.
    # The state suffix is normalized with a running mean/std computed during
    # residual_warmup_steps; obs_dim of ResidualActor = policy_embed_dim + residual_obs_dim.
    config.residual_append_state = True
    config.residual_obs_dim = 19  # obs["state"][6:] — drops world-frame x/y/z pos and rpy orientation
    config.residual_append_base_action = True  # prepend base Pi0 action to actor obs, matching torch

    # ── Residual actor Pi-feature options ─────────────────────────────────
    # When True, the residual MLP takes frozen Pi prefix features as its
    # observation instead of DSRL's obs_encoder output.
    config.residual_use_pi_image_features = False
    config.residual_pi_feature_source = "prefix"

    config.residual_actor_hidden_dims = (256, 256)

    # ── Residual action space ──────────────────────────────────────────────
    # "accel_steer": the Pi0 waypoint chunk is PID-decoded (SimlingoStyleWaypointDecoder,
    # rollout-side) to a 2-D [accel, steer] in [-1, 1] BEFORE the residual; the
    # residual/critic/replay all operate on the bounded 2-D action and the env
    # executes it via _action_to_control. Matches torch_agents/residual_sac.py
    # (run_simlingo). "waypoint_chunk" keeps the residual on the raw 40-D chunk
    # in physical DELTA_XY meters (set residual_action_clip=None there).
    config.residual_action_space = "accel_steer"
    # Per-dim residual scale (accel, steer) — torch parity (res_scale_accel=2.0,
    # res_scale_steer=0.6); composed action is clipped to ±residual_action_clip.
    # del first: the base config types this field as float and ml_collections
    # refuses a tuple override on a typed field.
    del config.residual_action_scale
    config.residual_action_scale = (2.0, 0.6)
    config.residual_action_clip = 1.0
    # Entropy coefficient for the residual actor SAC loss.
    config.residual_alpha = 0.1
    config.residual_lr = 3e-4
    config.residual_log_std_min = -5.0
    config.residual_log_std_max = 2.0
    config.residual_layer_norm = False
    # Run pure Pi0 (zero residual) for this many steps before applying the
    # trained residual to avoid corrupting rollouts early in training.
    config.residual_warmup_steps = 500

    # ── Critic / buffer ───────────────────────────────────────────────────
    config.batch_size = 16
    config.buffer_capacity = 5_000
    config.warmup_steps = 500
    config.updates_per_step = 5
    config.discount = 0.99
    config.tau = 0.005
    # Standard DSRL critic alpha (noise actor; separate from residual_alpha).
    config.alpha = 0.1
    config.noise_scale = 1.0

    # Critic language feedback — action_delta is compact and stable.
    config.critic_feedback_mode = "action_delta"

    # ── GPU placement ─────────────────────────────────────────────────────
    # Override per-run via --agent.training_gpu_rank or run_carla.sh --train-gpu.
    config.training_gpu_rank = 0

    return config
