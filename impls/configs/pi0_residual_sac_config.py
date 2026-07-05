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
    # Frozen SigLIP embeddings: [image_embed, prompt_embed, subtask_embed] (3×1152-d).
    config.observation_mode = "image"
    config.image_encoder = "siglip"
    config.siglip_model_id = "google/siglip2-so400m-patch14-384"
    config.siglip_include_prompt_subtask = True
    # siglip_device is left unset so main_carla.py picks the training GPU automatically.

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
    config.residual_action_scale = (0.6, 0.6)
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
    # Rollout flow-latent scale for the frozen Pi0 base chunk: tanh(N(0,1)) * scale
    # when != 1.0 (master uses its untrained noise actor * noise_scale=5.0, whose
    # variance is what stochastically breaks the standstill brake-ratio stall —
    # A/B on generalization-wall-1095, 2026-06-11). Kept at 1.0 (plain unit normal).
    config.vla_noise_scale = 1.0

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

    # Critic language feedback. "action_delta" (expert − agent first step) is
    # compact but depends on the logged action; "expert_action" (expert first
    # step + validity flag) is the state-only, Bellman-consistent alternative.
    # In accel_steer mode both labels are PID-decoded [accel, steer] controls.
    config.critic_feedback_mode = "action_delta"

    # ── GPU placement ─────────────────────────────────────────────────────
    # Override per-run via --agent.training_gpu_rank or run_carla.sh --train-gpu.
    config.training_gpu_rank = 0

    return config
