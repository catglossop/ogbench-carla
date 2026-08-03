"""``get_config()`` for a ROLLOUT-ONLY SteerVLA eval — ``*_no_ego_history`` checkpoint (job 1).

One of three sibling configs used to compare SteerVLA checkpoints on the same route by
rolling each one out with **no gradient updates of any kind**:

  * ``steervla_rollout_no_ego_history_config.py``      (this file)
  * ``steervla_rollout_ego_history_config.py``
  * ``steervla_rollout_simplified_reasoning_config.py``

Structure follows ``steervla_cast_relabel_config.py`` (inherit ``steervla_dsrl_config`` and
override), but everything that learns or calls a VLM is switched off:

  * ``enable_updates`` and all three per-kind switches are False — the DSRL agent is still
    built (it supplies the steering noise that seeds the flow) but never takes a gradient step.
  * No ``cast_relabel`` block, so ``OnlineCastRelabelSession`` never starts.
  * ``language_feedback.source="expert"`` + ``expert_mode="none"``. This matters: NOT setting it
    would leave the inherited ``source="vlm"``, and ``coaches.critic_feedback.resolve_critic_feedback_mode``
    returns ``vlm_chunk_bow`` from ``source`` *before* it ever looks at ``critic_feedback_mode`` —
    so ``run_carla.sh --critic-mode none`` alone would NOT stop the Gemini critic coach.
  * ``steervla.load_trainable_params=False`` — inference-only param restore, no OpenPI optimizer
    state. Required here: this job shares ONE GPU with its own CARLA renderer.

Checkpoint-matching knobs (these must mirror the OpenPI ``TrainConfig`` the checkpoint was
pretrained with, or the model sees differently-scaled inputs than it was trained on):

  * ``include_ego_history=False`` — matches ``pi05_steervla_cot_simplified_reasoning_no_ego_history``.
  * ``proprio_norm=False`` — matches it too. NOTE this differs from the inherited ``True``:
    ``openpi.policies.steervla_policy.normalize_ego_state`` divides speed by 20 and course by 180
    when True, so ``True`` against a ``proprio_norm=False`` checkpoint feeds speed 0.5 where the
    model was trained on 10.0.
  * ``sample_t_context=False`` and no ``t_context`` key — the clean (non-CSP) regime. The inherited
    ``sample_t_context=True`` is only valid for a ``*_csp`` context-smoothing checkpoint; this one
    is not, and has no ``ctx_time_mlp_*`` params.

GPU layout — this job needs BOTH cards. Measured on 2x RTX 4090 (24.5 GB each), one run costs:

  * ~22 GB on the JAX card (the Pi0-CoT restore peaks here; params are 12.5 GiB of fp32), and
  * ~5-7 GB on the CARLA card for the renderer.

That ~28 GB total does NOT fit on one 24.5 GB card, so two of these runs cannot go side by side
on a 2-GPU box — run them sequentially.

Launch it as::

    OPENPI_DATA_HOME=/home/carla/.cache/openpi \\
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    ./carla_job.sh start --job 0 --train-gpu 0 --render-adapter 1 ...

Why each piece:

  * ``CUDA_VISIBLE_DEVICES=1`` — ``main_carla._configure_jax_training_device`` calls
    ``jax.devices("gpu")``, which inits the JAX backend across EVERY visible card and reserves a
    pool on each. Restricting the view keeps JAX off the renderer's card. ``training_gpu_rank`` and
    ``siglip_device`` are therefore **0**/``cuda:0`` — they index the *visible* devices.
  * ``--render-adapter 1`` puts CARLA on physical GPU **0**. CARLA's adapter order is swapped
    relative to ``nvidia-smi`` (confirmed empirically; the "maps to physical GPU N" comment at
    ``carla_utils.py`` ~line 1537 is wrong, ``carla_config.yaml``'s note is right).
  * ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` — JAX otherwise grabs ~92% of its card up front.
    Capping instead with ``XLA_PYTHON_CLIENT_MEM_FRACTION`` does NOT work here: the restore OOMs
    on a single 4.83 GB allocation at a 0.5 cap (12.2 GB), and also at ~17.5 GB.

``OPENPI_DATA_HOME`` is required because ``vlas/steervla.py`` hardcodes the OpenPI cache to
``/raid/users/cglossop/openpi``, which does not exist on this box (it is applied with
``setdefault``, so the env var wins).
"""

from configs.steervla_dsrl_config import get_config as get_dsrl_config


def get_config():
    config = get_dsrl_config()

    # ---- Rollout only: no learning of any kind. ----
    config.enable_updates = False
    config.enable_updates_rl = False
    config.enable_updates_bc = False
    config.enable_updates_bc_hl = False
    config.online_training_mode = "rl"
    # Run the real policy from step 0 (no random-action warmup period).
    config.warmup_steps = 0
    config.warmup_use_random_actions = False
    # Log/store the true env reward, not the -ego_speed debug reward.
    config.debug_task = False
    # Skip the extra best-of-N noise candidate sampling; it costs a VLA forward per candidate.
    config.debug_noise = False
    config.use_best_noise = False

    # ---- No coaches: no expert labeler, no Gemini critic coach. ----
    config.language_feedback.source = "expert"
    config.language_feedback.expert_mode = "none"
    config.critic_feedback_mode = "none"

    # ---- Single GPU for this job (JAX + SigLIP together). ----
    config.training_gpu_rank = 0
    config.siglip_device = "cuda:0"

    # ---- SteerVLA: frozen inference actor for THIS checkpoint. ----
    config.steervla.load_trainable_params = False
    config.steervla.hl_training_gpu_rank = -1
    config.steervla.actor_config = "pi05_steervla_cot_simplified_reasoning_no_ego_history"
    config.steervla.checkpoint = (
        "gs://cat-logs/pi05_steervla_cot_simplified_reasoning_no_ego_history/"
        "pi05_steervla_simplified_reasoning_no_ego_history_v1/"
        "pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000"
    )
    # Match the pretraining TrainConfig (see module docstring).
    config.steervla.include_ego_history = False
    config.steervla.proprio_norm = False
    # Non-CSP checkpoint -> clean regime (leave ``t_context`` unset/None).
    config.steervla.sample_t_context = False
    # Greedy CoT decoding so the three checkpoints are compared without sampling noise.
    config.steervla.cot_temperature = 0.0

    return config
