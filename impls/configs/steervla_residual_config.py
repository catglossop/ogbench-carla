"""``get_config()`` for the residual-RL stack on CARLA Bench2Drive.

Use with ``--agent=impls/configs/steervla_residual_config.py`` (the default for
``impls/main_carla_residual.py``). This wraps :func:`jax_agents.sac_residual.get_config`
(the residual SAC agent) and adds the run-level wiring: the state encoder used to
build the RL state and the frozen SteerVLA base policy. (RLT, when added, plugs in
as another ``state_encoder`` feeding this same agent — it is not an agent itself,
so it reuses this config/entrypoint via the ``config.state_encoder`` knob rather
than new run files.)

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
    # agnostic: whatever this produces (a single fixed-size vector) is
    # concatenated with the base action chunk and fed to the residual MLP. Options:
    #   "pi_prefix"   : frozen, mean-pooled SteerVLA prefix feature (full PaliGemma
    #                   forward over image + prompt, then mean-pool; deterministic,
    #                   stop-gradient). Speed + routing ride in via the prompt, so
    #                   no separate proprio vector is needed.
    #   "siglip_pool" : frozen, mean-pooled SigLIP image feature (vision tower only,
    #                   no Gemma LLM) — a perception-only lower-bound ablation.
    #   "rl_token"    : frozen RLT autoencoder (trained offline) over the un-pooled
    #                   prefix tokens — same VLM-backbone input as pi_prefix, but a
    #                   learned compression to z_rl instead of mean-pool. Requires a
    #                   checkpoint (config.rl_token.checkpoint_path) and PyTorch.
    # All require local SteerVLA (remote HTTP mode does not expose Pi features).
    config.state_encoder = "pi_prefix"

    # ----- rl_token encoder (used only when state_encoder == "rl_token") ------ #
    # The autoencoder is trained separately by train_rl_token_ae.py on prefix
    # embeddings dumped (dump_rl_token_embeddings.py) from the *same* frozen
    # SteerVLA checkpoint below; its config (d_model, layers, max_seq_len) is read
    # from the checkpoint, so only the path is needed here. device="cpu" keeps the
    # small Torch AE off the JAX GPU; set "cuda" to run it on-GPU.
    config.rl_token = ml_collections.ConfigDict(
        dict(
            checkpoint_path="",
            device="cpu",
        )
    )

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
