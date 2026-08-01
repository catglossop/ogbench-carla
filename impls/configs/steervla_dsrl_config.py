"""``get_config()`` for DSRL + (optional) SteerVLA on CARLA Bench2Drive.

Use with ``--agent=impls/configs/steervla_dsrl_config.py``. This is a thin
convenience wrapper around :func:`jax_agents.dsrl.get_config` with sensible
defaults for online single-route runs and a ``steervla`` block describing how
to plug in a frozen SteerVLA flow.

Set ``config.observation_mode`` to ``"state"`` or ``"image"``.
For ``observation_mode="image"``, set ``config.image_encoder`` to ``"impala"`` (learned)
or ``"siglip"`` (frozen HuggingFace SigLIP embeddings for actor/critic inputs).
With ``siglip_include_prompt_subtask=True``, observations are
``[image_embed, prompt_embed, subtask_embed]`` (each 1152-d for SigLIP2 SO400M).

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

import ml_collections

from jax_agents import dsrl as dsrl_agent


def get_config():
    config = dsrl_agent.get_config()

    config.lr = 3e-5
    config.batch_size = 128
    # Denoise steps for frozen VLA forwards during RL updates (rollout uses steervla.sample_actions_num_steps).
    config.vla_update_flow_steps = 5
    config.noise_scale = 1.0
    config.alpha = 0.1
    # Collect transitions with the rollout policy (SteerVLA / DSRL) but skip RL updates.
    config.warmup_steps = 200
    # If True, use env.action_space.sample() during warmup instead of the policy.
    config.warmup_use_random_actions = False
    config.updates_per_step = 5
    # Run RL gradient updates every N env steps (still ``updates_per_step`` per update).
    config.update_interval = 10
    # Master switch: set to false for rollout-only runs (no gradient updates of any kind).
    config.enable_updates = True
    # Per-kind switches, each ANDed with ``enable_updates``:
    #   rl    -> DSRL critic/actor (RL) updates
    #   bc    -> full BC / DAgger imitation path (``update_dagger``)
    #   bc_hl -> high-level VLM backbone update (``update_hl`` on cast_relabel data)
    config.enable_updates_rl = True
    config.enable_updates_bc = False
    config.enable_updates_bc_hl = False
    config.buffer_capacity = 1_000
    # When true, RL updates use reward = -ego_speed (m/s) instead of env reward.
    config.debug_task = True
    # Rollout-only: best-of-N random VLA noises minimizing first-step speed delta_xy.
    config.debug_noise = False
    config.debug_noise_samples = 8
    config.debug_noise_log_every_n_steps = 10
    # When debug_noise=True: if True, execute the slowest candidate; if False, log only and use actor noise.
    config.use_best_noise = False
    config.image_log_curr_interval = 10
    config.critic_action_dim = 4
    config.vla_action_dim = 4
    config.vla_action_horizon = 10
    config.actor_action_dim = 32
    config.action_horizon = 10
    # config.steervla = None
    # DSRL trains on ``observation_mode`` only; env step always returns both keys.
    config.observation_mode = "image"
    # ``impala``: learned Flax encoder inside DSRL. ``siglip``: frozen SigLIP embeddings.
    config.image_encoder = "siglip"
    config.siglip_model_id = "google/siglip2-so400m-patch14-384"
    config.siglip_include_prompt_subtask = True
    config.siglip_device = "cuda:0"
    config.image_keys = ("base_0_rgb",)
    # JAX RL device: ``-1`` = unset. CARLA uses ``gpu_rank`` in carla_config.yaml.
    config.training_gpu_rank = 0
    # Critic feedback mode — four options:
    #   "commentary_bow"      (default): expert commentary BOW from the SimLingo-style labeler.
    #   "expert_action"                : (critic_action_dim + 1)-dim = expert first step + validity
    #                                    flag. State-only (independent of the agent's action), so it
    #                                    is consistent across the TD bootstrap and the actor's
    #                                    Q-query. PID-decoded [accel, steer] in accel_steer mode.
    #   "action_delta"                 : (critic_action_dim)-dim = expert first step minus agent
    #                                    first step. Depends on the logged action; the next-state
    #                                    label is backfilled one step later in main_carla.py.
    #   "delta_commentary_bow"        : corrective language BOW from expert-vs-agent action delta
    #                                   (e.g. "Adjust right. Decelerate more heavily.").
    # To switch, uncomment:
    # config.critic_feedback_mode = "expert_action"
    # Online training regime:
    #   "rl"     : standard DSRL online RL
    #   "dagger" : on-policy data aggregation with expert actions as supervision
    config.online_training_mode = "rl"
    # language_label_dim is auto-set from language_feedback / critic_feedback_mode in main_carla.py.

    config.language_feedback = ml_collections.ConfigDict(
        dict(
            # Where DSRL critic language labels come from:
            #   "expert" — SimLingo-style expert commentary / action-delta coaches
            #   "vlm"    — Gemini/Perceptron VLM chunk feedback (see ``vlm_coach``)
            source="expert",
            # Used when source="expert". One of commentary_bow | expert_action |
            # action_delta | delta_commentary_bow | none
            expert_mode="commentary_bow",
        )
    )
    # Enable VLM chunk coaching for the DSRL critic:
    config.language_feedback.source = "vlm"

    config.vlm_coach = ml_collections.ConfigDict(
        dict(
            # Query the VLM every N episode env steps (0 = only at episode end).
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
            # Local OpenPI inference (ignored when actor_url is set):
            # Must be the _csp variant to match the CSP checkpoint below: it is the only config
            # whose model carries context_smoothing (and hence the ctx_time_mlp_* params that the
            # checkpoint contains and that t_context needs).
            actor_config="pi05_steervla_cot_simplified_reasoning_no_ego_history",
            # actor_config="pi05_steervla_cot_simplified_reasoning",
            # checkpoint="gs://cat-logs/pi05_steervla_cot_ki/pi05_steervla_cot_ki/90000",
            # Prefer the gs:// form: the /home/carla/.cache/openpi/... local mirror of this same
            # checkpoint is machine-specific. OpenPI caches GCS downloads under ~/.cache/openpi.
            # checkpoint="gs://cat-logs/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning_20260523_222304/8000",
            # checkpoint="gs://cat-logs/pi05_steervla_cot_ki_simplified_reasoning/pi05_steervla_cot_ki_simplified_reasoning/pi05_steervla_cot_ki_simplified_reasoning_20260512_144250/50000",
            # checkpoint="gs://cat-logs/pi05_steervla_cot_simplified_reasoning_no_attention/pi05_steervla_cot_simplified_reasoning_no_attention/pi05_steervla_cot_simplified_reasoning_no_attention_20260526_175924/4000",
            # checkpoint="gs://cat-logs/pi05_steervla_cot_simplified_reasoning_csp/pi05_steervla_cot_simplified_reasoning_csp/pi05_steervla_cot_simplified_reasoning_csp_20260713_095815/10000",
            # Active checkpoint must match actor_config above (..._no_ego_history).
            checkpoint="gs://cat-logs/pi05_steervla_cot_simplified_reasoning_no_ego_history/pi05_steervla_simplified_reasoning_no_ego_history_v1/pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000",
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
            sample_actions_num_steps=10, # default is 10
            # When set, skip sample_cot and teacher-force CoT like inspect_outputs.ipynb.
            # fixed_subtask_text="The vehicle follows the route with steady lane keeping, normally maintaining its current speed.",
            # fixed_reasoning_text="Follow the route.",
            # Context-Smoothed Pre-training (CSP) noise level for the action expert. Only valid
            # for a checkpoint trained with ``Pi0CoTConfig.context_smoothing`` enabled; omit both
            # keys (the default) for the clean, precise-imitation regime.
            #   sample_t_context=True  -> fresh t_context ~ U[t_context_min, t_context_max] per
            #                             model query (independent per best-of-N candidate).
            #   t_context=<float>      -> pin a fixed level (0 = clean, 1 = uninformative context).
            # sample_t_context=True,
            # t_context_min=0.0,
            # t_context_max=1.0,
            # Remote HTTP actor (leave unset or falsy for local checkpoint load):
            # actor_url="http://35.186.30.251:8000",
        )
    )

    return config
