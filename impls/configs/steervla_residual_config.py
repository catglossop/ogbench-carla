"""``get_config()`` for the residual-RL stack on CARLA Bench2Drive.

Use with ``--agent=impls/configs/steervla_residual_config.py`` (the default for
``impls/main_carla_residual.py``). This wraps :func:`jax_agents.sac_residual.get_config`
(the residual SAC agent) and adds the run-level wiring: the proprio slice used as
the RL state and the frozen SteerVLA base policy. (RLT, when added, plugs in as a
state encoder feeding this same agent — it is not an agent itself, so it reuses
this config/entrypoint via a ``config.encoder`` knob rather than new files.)

``config.training_gpu_rank`` pins JAX (RL agent + SteerVLA checkpoint restore) to
one GPU; CARLA's render GPU is ``gpu_rank`` in ``impls/configs/carla_config.yaml``.
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
    config.buffer_capacity = 100_000

    # RL state = proprio slice of obs["state"] (carla_utils 25-dim layout):
    # kinematics + last control = state[6:19] (13-dim). [0:6]=global pose,
    # [19:25]=command (already in the VLA prompt).
    config.ego_state_slice = (6, 19)

    # ----- frozen SteerVLA base policy --------------------------------------- #
    config.steervla = ml_collections.ConfigDict(
        dict(
            enabled=True,
            actor_config="pi05_steervla_cot_simplified_reasoning",
            checkpoint=(
                "gs://cat-logs/pi05_steervla_cot_simplified_reasoning_no_attention/"
                "pi05_steervla_cot_simplified_reasoning_no_attention/"
                "pi05_steervla_cot_simplified_reasoning_no_attention_20260526_175924/2000"
            ),
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
            # Remote HTTP actor (leave unset for local checkpoint load). The
            # token-based encoders (pi_prefix / rl_token) require local mode.
            # actor_url="http://127.0.0.1:8000",
        )
    )

    return config
