"""``get_config()`` for the Best-of-N + SteerVLA agent on CARLA Bench2Drive.

Use with ``--agent=impls/configs/steervla_best_of_n_config.py``. This mirrors
``steervla_dsrl_config.py`` but targets :class:`jax_agents.best_of_n.BestOfNAgent`.

Best-of-N has **no noise actor**: at each env step the attached
:class:`vlas.steervla.SteerVLAActor` samples ``best_of_n`` chains-of-thought at
``vla_cot_temperature`` (one batched forward), the critic scores each candidate
chunk, and the argmax-Q candidate is executed. Each candidate's subtask string is
embedded with SigLIP and used as the critic language label, so selection is
``argmax_i Q([obs_e; subtask_i], action_i)`` and the executed candidate's subtask
embedding is persisted into the replay transition (``critic_feedback_mode="subtask_siglip"``).

Notes vs. the DSRL config:
  * **No** ``language_feedback`` / ``vlm_coach`` block — those would override the
    critic mode (``resolve_critic_feedback_mode`` only honors the legacy
    ``critic_feedback_mode`` field when ``language_feedback`` is absent). Best-of-N
    relies on ``critic_feedback_mode="subtask_siglip"``.
  * ``observation_mode="image"`` + ``image_encoder="siglip"`` so ``main_carla`` builds
    one SigLIP encoder and shares it for both the obs embedding and the subtask labels.
  * ``main_carla`` injects ``steervla_actor`` (needs ``sample_candidates``) and the shared
    ``siglip_encoder`` into ``BestOfNAgent.create`` automatically.

Set ``config.training_gpu_rank`` to pin JAX RL + SteerVLA OpenPI restore to one GPU
(``-1`` = default). CARLA's sim GPU is ``gpu_rank`` in ``impls/configs/carla_config.yaml``.
"""

import ml_collections

from jax_agents import best_of_n as best_of_n_agent


def get_config():
    config = best_of_n_agent.get_config()

    config.lr = 3e-4
    config.batch_size = 256
    # Number of CoT candidates sampled per env step; the critic picks the argmax-Q chunk.
    # Each step runs one batched CoT forward plus ``action_decode_batch_size`` micro-batches
    # for action decoding — keep this modest for interactive CARLA rollouts.
    config.best_of_n = 10
    config.bon_viz_interval = 100
    # Temperature for the best-of-N CoT sampling (>0 so candidates differ).
    config.vla_cot_temperature = 1.0
    # Denoise steps for the frozen VLA bootstrap forward during RL up\dates.
    config.vla_update_flow_steps = 5
    config.noise_scale = 1.0
    config.alpha = 0.1
    # Collect transitions with the rollout policy but skip RL updates for this many env steps.
    config.warmup_steps = 500
    config.warmup_use_random_actions = False
    config.updates_per_step = 5
    # Run RL gradient updates every N env steps (still ``updates_per_step`` per update).
    config.update_interval = 10
    # Master switch: set to false for rollout-only runs (no gradient updates of any kind).
    config.enable_updates = False
    # Per-kind switches, each ANDed with ``enable_updates``:
    #   rl    -> critic/actor (RL) updates
    #   bc    -> full BC / DAgger imitation path (``update_dagger``)
    #   bc_hl -> high-level VLM backbone update (no-op for best-of-N; kept for parity)
    config.enable_updates_rl = False
    config.enable_updates_bc = False
    config.enable_updates_bc_hl = False
    config.buffer_capacity = 5_000
    # When true, RL updates use reward = -ego_speed (m/s) instead of env reward.
    config.debug_task = False
    # Best-of-N selects over CoT candidates (not random noise), so the noise-sweep debug is off.
    config.debug_noise = False
    config.debug_noise_samples = 8
    config.debug_noise_log_every_n_steps = 10
    config.use_best_noise = False
    config.image_log_curr_interval = 10
    config.critic_action_dim = 4
    config.vla_action_dim = 4
    config.vla_action_horizon = 10
    config.actor_action_dim = 32
    config.action_horizon = 10

    # DSRL/best-of-N train on ``observation_mode`` only; env step always returns both keys.
    config.observation_mode = "image"
    # ``siglip``: frozen SigLIP embeddings, shared with the subtask critic labels below.
    config.image_encoder = "siglip"
    config.siglip_model_id = "google/siglip2-so400m-patch14-384"
    # Subtask is the critic label, so the observation can stay image-only (no prompt/subtask concat).
    config.siglip_include_prompt_subtask = True
    # Run SigLIP on GPU. Within the run's CUDA_VISIBLE_DEVICES this maps to the same
    # physical GPU as JAX (the sim renderer is on a different GPU via carla_config.gpu_rank).
    # Keep this on GPU on many-core machines: SigLIP-on-CPU oversubscribes torch's intra-op
    # thread pool (defaults to core count) and thrashes the box.
    # Relative to CUDA_VISIBLE_DEVICES (set to the physical training GPU, e.g. 6).
    config.siglip_device = "cuda:0"
    config.image_keys = ("base_0_rgb",)
    # JAX RL device: ``-1`` = unset. CARLA uses ``gpu_rank`` in carla_config.yaml.
    config.training_gpu_rank = 0

    # Critic feedback: SigLIP text embedding of the executed candidate's subtask.
    # Keep this as the only critic-mode signal — do NOT add a ``language_feedback`` block,
    # which would route ``resolve_critic_feedback_mode`` away from ``subtask_siglip``.
    config.critic_feedback_mode = "subtask_siglip"
    # Online training regime: "rl" (standard online RL) or "dagger".
    config.online_training_mode = "rl"
    # language_label_dim is auto-set from siglip_embed_dim in main_carla.py.
    
    # Critic weights
    config.critic_pretrained_weights = "/raid/users/cglossop/critic_carla/run_20260701_233000/4000"

    config.steervla = ml_collections.ConfigDict(
        dict(
            enabled=True,
            # Local OpenPI inference (ignored when actor_url is set):
            actor_config="pi05_steervla_cot_simplified_reasoning",
            checkpoint="gs://cat-logs/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning_20260523_222304/8000",
            routing_command="Follow the route and stay in lane.",
            # Per-step CoT sampling temperature for the rollout actor. Best-of-N samples
            # ``best_of_n`` CoTs via ``sample_candidates`` at ``vla_cot_temperature`` (set above),
            # so this default only affects any non-candidate single-sample paths.
            cot_temperature=1.0,
            include_ego_history=False,
            # Raw m/s and degrees. Every ``pi05_steervla_cot_simplified_reasoning*``
            # TrainConfig sets proprio_norm=False, so the old ``True`` (speed/20,
            # course/180) fed the model a state layout it never trained on.
            proprio_norm=False,
            # Replay buffer + CARLA ``step`` use OpenPI chunk layout (``action_horizon`` × ``action_dim``),
            # executed like ``simlingo/team_code/agent_steervla.py`` (cumsums + PID).
            use_pi_action_chunk_for_env=True,
            action_horizon=10,
            action_dim=4,
            # Query Pi0-CoT once, then execute this many rows before re-querying (1 = every step).
            actions_per_model_query=1,
            # Reuse sampled CoT reasoning/subtask for this many env actions before re-sampling CoT.
            actions_per_cot=1,
            output_action_format="DELTA_XY_T_DELTA_XY_SPACE",
            sample_actions_num_steps=10,
            # Decode action chunks in small micro-batches (CoT stays batched). Large
            # batched _sample_actions forwards can trigger native aborts on some drivers.
            action_decode_batch_size=2,
            # Context-Smoothed Pre-training (CSP) noise level for the action expert. Only valid
            # for a checkpoint trained with ``Pi0CoTConfig.context_smoothing`` enabled; omit both
            # keys (the default) for the clean, precise-imitation regime.
            #   sample_t_context=True  -> fresh t_context ~ U[t_context_min, t_context_max] per
            #                             model query (independent per best-of-N candidate).
            #   t_context=<float>      -> pin a fixed level (0 = clean, 1 = uninformative context).
            # sample_t_context=True,
            # t_context_min=0.0,
            # t_context_max=1.0,
            # Remote HTTP actor is NOT supported for best-of-N (needs local sample_candidates):
            # actor_url="http://35.186.30.251:8000",
        )
    )

    return config
