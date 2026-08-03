"""``get_config()`` for TMRL + SteerVLA on CARLA Bench2Drive.

Use with ``--agent=impls/configs/steervla_tmrl_config.py``. TMRL is DSRL plus a learned
context-noise-level policy: the SAC noise actor gains a ``timestep`` head (mapped to the SteerVLA
Context-Smoothed Pre-training level ``t_context in [0, 1]``) and a ``context_noise`` head, so an
RL-finetuned high-level network selects *how much to noise the context* alongside the steering
vector (see :mod:`jax_agents.tmrl`).

This is a thin wrapper around :func:`jax_agents.tmrl.get_config` that reuses the DSRL SteerVLA
defaults. The key difference from ``steervla_dsrl_config.py`` is the CSP block: instead of drawing
``t_context`` uniformly (``sample_t_context=True``), TMRL drives it from the actor at rollout via
:meth:`vlas.steervla.SteerVLAActor.set_t_context` and from the ``t_context`` kwarg during RL updates.
The checkpoint MUST be a context-smoothing (``*_csp``) variant, otherwise TMRL falls back to
DSRL steering-only behaviour (``config.use_vla_context_smoothing`` gates this).
"""

import ml_collections

from jax_agents import tmrl as tmrl_agent


def get_config():
    config = tmrl_agent.get_config()

    config.lr = 3e-5
    config.batch_size = 24
    config.vla_update_flow_steps = 5
    config.noise_scale = 1.0
    # SAC temperatures: steering (alpha) + the two context-noise-level heads.
    config.alpha = 0.1
    config.alpha_timestep = 0.1
    config.alpha_context = 0.1
    config.warmup_steps = 200
    config.warmup_use_random_actions = False
    config.updates_per_step = 5
    config.update_interval = 10
    config.enable_updates = True
    config.enable_updates_rl = True
    config.enable_updates_bc = False
    config.enable_updates_bc_hl = False
    config.buffer_capacity = 1_000
    config.debug_task = False
    config.debug_noise = True
    config.debug_noise_samples = 8
    config.debug_noise_log_every_n_steps = 10
    config.use_best_noise = False
    config.image_log_curr_interval = 10
    config.critic_action_dim = 4
    config.vla_action_dim = 4
    config.vla_action_horizon = 10
    config.actor_action_dim = 32
    config.action_horizon = 10
    config.observation_mode = "image"
    config.image_encoder = "siglip"
    config.siglip_model_id = "google/siglip2-so400m-patch14-384"
    config.siglip_include_prompt_subtask = True
    config.siglip_device = "cuda:1"
    config.image_keys = ("base_0_rgb",)
    config.training_gpu_rank = 0
    config.online_training_mode = "rl"

    # TMRL context-noise policy knobs (see jax_agents.tmrl.get_config).
    # context_dim defaults to the encoder width (leave unset).
    config.timestep_max = 1.0
    config.context_noise_bound = 1.0
    config.use_vla_context_smoothing = True

    config.language_feedback = ml_collections.ConfigDict(
        dict(
            source="vlm",
            expert_mode="commentary_bow",
        )
    )
    config.language_feedback.source = "vlm"

    config.vlm_coach = ml_collections.ConfigDict(
        dict(
            query_every_n_episode_steps=128,
            query_on_episode_end=True,
            provider="gemini",
            gemini_model="gemini-3.5-flash",
            include_plots_in_prompt=False,
            action_chunk_steps=10,
            action_chunk_duration_sec=0.5,
            bad_event_radius_chunks=2,
            annotate_video=False,
            save_artifacts=True,
            video_fps=10.0,
            video_frame_stride=2,
        )
    )

    config.steervla = ml_collections.ConfigDict(
        dict(
            enabled=True,
            # Must be a context-smoothing (CSP) checkpoint so ``t_context`` is honored -- the TMRL
            # actor selects this level per state instead of the uniform draw DSRL uses.
            actor_config="pi05_steervla_cot_simplified_reasoning_csp",
            checkpoint="gs://cat-logs/pi05_steervla_cot_simplified_reasoning_csp/pi05_steervla_cot_simplified_reasoning_csp/pi05_steervla_cot_simplified_reasoning_csp_20260713_095815/10000",
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
            # TMRL drives t_context from the actor (set_t_context per rollout query); leave the
            # uniform-sampling CSP path OFF so the initial level is the learned one, not a draw.
            sample_t_context=False,
            t_context=0.0,
            t_context_min=0.0,
            t_context_max=1.0,
        )
    )

    return config
