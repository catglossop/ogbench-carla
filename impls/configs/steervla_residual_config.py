"""``get_config()`` for the residual-RL stack on CARLA Bench2Drive.

Use with ``--agent=impls/configs/steervla_residual_config.py``; ``main_carla.py`` dispatches
to the residual path when this config's ``agent_name == "sac_residual"``. This wraps
:func:`jax_agents.sac_residual.get_config`
(the residual SAC agent) and adds the run-level wiring: the state encoder used to
build the RL state and the frozen SteerVLA base policy.
"""

from __future__ import annotations

import ml_collections

from jax_agents import sac_residual as residual_agent


def get_config():
    config = residual_agent.get_config()

    # ----- run wiring -------------------------------------------------------- #
    # JAX RL device: ``-1`` = unset. CARLA uses ``gpu_rank`` in carla_config.yaml.
    config.training_gpu_rank = 0
    # If False, collect transitions without RL gradient updates (rollout-only).
    config.enable_updates = True
    # No-RL baseline: roll out the frozen base policy only (no residual agent,
    # encoder, buffer, or updates). state_encoder is ignored when this is True.
    config.base_only = False
    config.buffer_capacity = 100_000

    # ----- logging ----------------------------------------------------------- #
    # Log a W&B rollout video per episode (frames from obs["image_viz"]).
    config.log_episode_video = True
    config.episode_video_fps = 10.0
    # Capture every Nth env step (plus the terminal frame) to keep videos light.
    config.episode_video_every = 2

    # RL state encoder (see impls/encoders/). The residual agent is encoder-
    # agnostic: whatever this produces (a single fixed-size vector) is concatenated
    # with the base action chunk and fed to the residual MLP.
    #   "siglip_pool" : frozen HF SigLIP2 image embedding concatenated with a SigLIP2
    #                   text embedding of the routing-command prompt (no Gemma LLM).
    #                   Image + command share SigLIP's aligned space (config.siglip below).
    # Requires local SteerVLA (remote HTTP mode does not expose the base prompt).
    config.state_encoder = "siglip_pool"

    # ----- siglip_pool encoder ----------------------------------------------- #
    # Frozen HF SigLIP2 dual encoder. The state is [image_embed, prompt_embed]
    # (both L2-normalized SigLIP embeddings in one aligned space). Set
    # include_prompt=False for the image-only lower-bound ablation.
    config.siglip = ml_collections.ConfigDict(
        dict(
            model_id="google/siglip2-so400m-patch14-384",
            device="cuda",
            include_prompt=True,
        )
    )

    # ----- frozen SteerVLA base policy --------------------------------------- #
    config.steervla = ml_collections.ConfigDict(
        dict(
            enabled=True,
            actor_config="pi05_steervla_cot_simplified_reasoning",
            checkpoint="gs://cat-logs/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning_20260523_222304/8000/",
            routing_command="Follow the route and stay in lane.",
            cot_temperature=0.0,
            include_ego_history=False,
            proprio_norm=True,
            # CARLA step + replay buffer use the OpenPI chunk layout
            # (action_horizon x action_dim), executed via cumsum + PID
            # (steervla_simlingo_control). Set False for legacy [accel, steer].
            use_pi_action_chunk_for_env=True,
            action_horizon=10,
            action_dim=4,
            # Query Pi0-CoT every env step (1) vs. reuse the chunk open-loop.
            actions_per_model_query=1,
            # Reuse sampled CoT reasoning/subtask across this many env steps.
            actions_per_cot=1,
            output_action_format="DELTA_XY_T_DELTA_XY_SPACE",
            sample_actions_num_steps=10,
            # PID brake threshold (m/s) in the waypoint decoder: desired speeds below this brake.
            # The default (0.1) can trap the car at a standstill when the base predicts a near-stop
            # speed (cold-start brake trap); lower it (e.g. 0.01) so the car starts moving.
            brake_speed=0.1,
            # Remote HTTP actor (leave unset for local checkpoint load).
            # actor_url="http://127.0.0.1:8000",
        )
    )

    return config
