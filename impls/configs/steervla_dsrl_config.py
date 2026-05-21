"""``get_config()`` for DSRL + (optional) SteerVLA on CARLA Bench2Drive.

Use with ``--agent=impls/configs/steervla_dsrl_config.py``. This is a thin
convenience wrapper around :func:`jax_agents.dsrl.get_config` with sensible
defaults for online single-route runs and a ``steervla`` block describing how
to plug in a frozen SteerVLA flow.

Set ``config.observation_mode`` to ``"state"`` (default) or ``"image"`` so DSRL
reads either the vector ``obs['state']`` or RGB ``obs['image']`` from the CARLA env.

Set ``config.training_gpu_rank`` to pin JAX RL and SteerVLA OpenPI checkpoint restore
to one GPU (``-1`` = default device / GPU 0 for restores). CARLA's sim GPU is ``gpu_rank``
in ``impls/configs/carla_config.yaml``.

Plugging in SteerVLA Pi0-CoT: set ``config.steervla.enabled`` and checkpoint paths; see
:class:`vlas.steervla.SteerVLAActor`. ``main_carla`` builds ``vla_sample_fn`` and uses
:meth:`jax_agents.dsrl.DSRLAgent.sample_actions_with_vla`; the DSRL network still uses an
internal :class:`jax_agents.dsrl.FlowActor` for init and training losses (SteerVLA is not
substituted as the Flax ``flow`` module).
Action-expert-only fine-tuning must be done in OpenPI (``TrainConfig.freeze_filter``),
not inside DSRL's Flax losses yet.
"""

from __future__ import annotations

import ml_collections

from jax_agents import dsrl as dsrl_agent


def get_config():
    config = dsrl_agent.get_config()

    config.lr = 3e-4
    config.batch_size = 64
    config.flow_steps = 10
    config.noise_scale = 1.0
    config.alpha = 0.1
    # Collect transitions with the rollout policy (SteerVLA / DSRL) but skip RL updates.
    config.warmup_steps = 0
    # If True, use env.action_space.sample() during warmup instead of the policy.
    config.warmup_use_random_actions = False
<<<<<<< Updated upstream
    config.updates_per_step = 10
    config.buffer_capacity = 1_000
=======
    config.updates_per_step = 1
    config.buffer_capacity = 10_000
>>>>>>> Stashed changes
    config.image_log_curr_interval = 10
    config.critic_action_dim = 4
    config.vla_action_dim = 4
    config.vla_action_horizon = 10
    config.actor_action_dim = 32
    config.action_horizon = 10
    # config.steervla = None
    # DSRL trains on ``observation_mode`` only; env step always returns both keys.
    config.observation_mode = "image"
    config.image_keys = ("base_0_rgb",)
    # JAX RL device: ``-1`` = unset. CARLA uses ``gpu_rank`` in carla_config.yaml.
    config.training_gpu_rank = 0
    # Critic feedback mode — three options:
    #   "commentary_bow"      (default): expert commentary BOW from the SimLingo-style labeler.
    #   "action_delta"                 : (critic_action_dim)-dim = expert first step minus agent first step.
    #   "delta_commentary_bow"        : corrective language BOW from expert-vs-agent action delta
    #                                   (e.g. "Adjust right. Decelerate more heavily.").
    # To switch to action_delta, uncomment:
    # config.critic_feedback_mode = "action_delta"
    # Online training regime:
    #   "rl"          : standard DSRL online RL
    #   "dagger"      : on-policy data aggregation with expert actions as supervision for the DSRL BC flow
    #   "dagger_direct" : expert imitation directly on the Pi action head (action_out_proj + time_mlp)
    #   "sac_direct"  : SAC on the Pi action head — DSRL critic is updated each step, then
    #                   action_out_proj + time_mlp are trained to maximize Q(s, a) where a is
    #                   a single-step Euler approximation: a = clip(noise + v_θ(s, noise, 0), -1, 1)
    config.online_training_mode = "rl"
    # language_label_dim is auto-set for commentary_bow / delta_commentary_bow in main_carla.py,
    # and critic_action_dim is used directly when mode is action_delta.

    # Residual actor defaults (sac_residual / dagger_residual).
    # scale=0.3 ≈ 2m correction capacity in DELTA_XY normalized space (7× phys).
    config.residual_action_scale = 0.3
    # alpha=0.3 provides reasonable entropy pressure for the 40-dim action space
    # (action_horizon=10 * action_dim=4); higher than the default 0.1 used for
    # smaller action spaces.
    config.residual_alpha = 0.3
    # Steps to execute pure Pi0 (zero residual) before enabling the MLP so
    # random-init weights do not degrade driving quality from step 1.
    config.residual_warmup_steps = 100
    # Default residual input: frozen Pi prefix feature from the same conditioned
    # transformer path used by direct DAgger, not a separately trained convnet.
    config.residual_use_pi_image_features = True
    # "prefix" matches the current default. "suffix" uses frozen action-suffix
    # hidden states conditioned on the base action chunk.
    config.residual_pi_feature_source = "prefix"
    # Optional critic ablation: feed frozen Pi prefix features into the critic
    # and noise_critic instead of DSRL's learned obs_encoder.
    config.critic_use_pi_prefix_features = False

    # Log VLA action distribution vs expert every N steps (0 = disabled).
    config.steervla_debug_action_dist_interval = 300
    config.steervla_debug_action_dist_num_samples = 32

    config.steervla = ml_collections.ConfigDict(
        dict(
            enabled=True,
            # Local OpenPI inference (ignored when actor_url is set):
<<<<<<< Updated upstream
            actor_config="pi05_steervla_cot_simplified_reasoning",
            # checkpoint="gs://cat-logs/pi05_steervla_cot_ki/pi05_steervla_cot_ki/90000",
            checkpoint="gs://cat-logs/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning_20260521_021239/6000",
            # checkpoint="gs://cat-logs/pi05_steervla_cot_ki_simplified_reasoning/pi05_steervla_cot_ki_simplified_reasoning/pi05_steervla_cot_ki_simplified_reasoning_20260512_144250/50000",
=======
            actor_config="pi05_steervla_cot_ki_inference",
            checkpoint="gs://cat-logs/pi05_steervla_cot_ki/pi05_steervla_cot_ki/90000",
            # checkpoint="gs://cat-logs/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning_20260521_021239/50000",
>>>>>>> Stashed changes
            routing_command="Follow the route and stay in lane.",
            cot_temperature=0.0,
            include_ego_history=False,
            proprio_norm=True,
            # Replay buffer + CARLA ``step`` use OpenPI chunk layout (``action_horizon`` × ``action_dim``),
            # executed like ``simlingo/team_code/agent_steervla.py`` (cumsums + PID). Set
            # ``use_pi_action_chunk_for_env: false`` for legacy ``[accel, steer]`` controls only.
            use_pi_action_chunk_for_env=True,
            action_horizon=10,
            action_dim=4,
            # Query Pi0-CoT once, then execute this many rows from the returned action chunk
            # before querying again. Set to 1 to query every env step.
            actions_per_model_query=1,
            # Reuse sampled CoT reasoning/subtask for this many env actions before sampling
            # CoT again. With actions_per_model_query=5, this reuses CoT across two action chunks.
            actions_per_cot=1,
            output_action_format="DELTA_XY_T_DELTA_XY_SPACE",
            sample_actions_num_steps=10,
            # Head-only OpenPI backward still retains full-model activations; keep this tiny.
            direct_dagger_microbatch_size=64,
            # Rebuild rollout inference wrappers only every N direct updates.
            direct_dagger_inference_refresh_interval=16,
            # Remote HTTP actor (leave unset or falsy for local checkpoint load):
            # actor_url="http://35.186.30.251:8000",
        )
    )

    return config
