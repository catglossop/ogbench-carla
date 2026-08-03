"""``get_config()`` for a ROLLOUT-ONLY SteerVLA eval — ``*_ego_history`` checkpoint (job 2).

Sibling of ``steervla_rollout_no_ego_history_config.py``; see that file's docstring for why
every learning path and every VLM coach is switched off. The only differences here are the
checkpoint being evaluated, its matching ego-history setting, and the GPU it is pinned to.

  * ``include_ego_history=True`` — matches ``pi05_steervla_cot_simplified_reasoning_ego_history``,
    which was pretrained with ego history on. ``normalize_ego_state`` then returns the last 4
    (speed, course) pairs instead of just the current one, so this MUST agree with the checkpoint.
  * ``proprio_norm=False`` — matches the pretraining TrainConfig (differs from the inherited True).
  * ``sample_t_context=False`` — non-CSP checkpoint, clean regime.

GPU layout and launch env are identical to ``steervla_rollout_no_ego_history_config.py`` — read
that docstring for why ``CUDA_VISIBLE_DEVICES``, ``--render-adapter``,
``XLA_PYTHON_CLIENT_PREALLOCATE=false`` and ``OPENPI_DATA_HOME`` are all required. Short version::

    OPENPI_DATA_HOME=/home/carla/.cache/openpi \\
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    ./carla_job.sh start --job 1 --train-gpu 0 --render-adapter 1 ...

This run needs ~22 GB on the JAX card plus ~5-7 GB on the CARLA card, so it must NOT be run
concurrently with a sibling rollout on a 2-GPU box.
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
    config.warmup_steps = 0
    config.warmup_use_random_actions = False
    config.debug_task = False
    config.debug_noise = False
    config.use_best_noise = False

    # ---- No coaches: no expert labeler, no Gemini critic coach. ----
    config.language_feedback.source = "expert"
    config.language_feedback.expert_mode = "none"
    config.critic_feedback_mode = "none"

    # ---- Single GPU for this job (JAX + SigLIP together). ----
    # Indices into the CUDA_VISIBLE_DEVICES-filtered view; run with CUDA_VISIBLE_DEVICES=1.
    config.training_gpu_rank = 0
    config.siglip_device = "cuda:0"

    # ---- SteerVLA: frozen inference actor for THIS checkpoint. ----
    config.steervla.load_trainable_params = False
    config.steervla.hl_training_gpu_rank = -1
    config.steervla.actor_config = "pi05_steervla_cot_simplified_reasoning_ego_history"
    config.steervla.checkpoint = (
        "gs://cat-logs/pi05_steervla_cot_simplified_reasoning_ego_history/"
        "pi05_steervla_simplified_reasoning_ego_history_v1/"
        "pi05_steervla_simplified_reasoning_ego_history_v1_20260717_183503/10000"
    )
    # Match the pretraining TrainConfig: this checkpoint DOES use ego history.
    config.steervla.include_ego_history = True
    config.steervla.proprio_norm = False
    # Non-CSP checkpoint -> clean regime (leave ``t_context`` unset/None).
    config.steervla.sample_t_context = False
    config.steervla.cot_temperature = 0.0

    return config
