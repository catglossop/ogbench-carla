"""``get_config()`` for a ROLLOUT-ONLY SteerVLA eval — base ``simplified_reasoning`` ckpt (job 3).

Sibling of ``steervla_rollout_no_ego_history_config.py``; see that file's docstring for why
every learning path and every VLM coach is switched off. This one evaluates the original
``pi05_steervla_cot_simplified_reasoning`` checkpoint.

  * ``include_ego_history=False`` — matches ``pi05_steervla_cot_simplified_reasoning``.
  * ``proprio_norm=False`` — matches the pretraining TrainConfig (differs from the inherited True).
  * ``sample_t_context=False`` — non-CSP checkpoint, clean regime.

This job runs AFTER the other two finish — a single rollout needs ~22 GB (JAX) + ~5-7 GB (CARLA),
which spans both cards of a 2x24 GB box, so these three configs cannot overlap. Launch env is the
same as ``steervla_rollout_no_ego_history_config.py`` (read that docstring for the reasoning)::

    OPENPI_DATA_HOME=/home/carla/.cache/openpi \\
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    ./carla_job.sh start --job 2 --train-gpu 0 --render-adapter 1 ...
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
    # Indices into the CUDA_VISIBLE_DEVICES-filtered view; pick the card at launch.
    config.training_gpu_rank = 0
    config.siglip_device = "cuda:0"

    # ---- SteerVLA: frozen inference actor for THIS checkpoint. ----
    config.steervla.load_trainable_params = False
    config.steervla.hl_training_gpu_rank = -1
    config.steervla.actor_config = "pi05_steervla_cot_simplified_reasoning"
    config.steervla.checkpoint = (
        "gs://cat-logs/pi05_steervla_cot_simplified_reasoning/"
        "pi05_steervla_cot_simplified_reasoning/"
        "pi05_steervla_cot_simplified_reasoning_20260523_222304/8000"
    )
    # Match the pretraining TrainConfig (see module docstring).
    config.steervla.include_ego_history = False
    config.steervla.proprio_norm = False
    # Non-CSP checkpoint -> clean regime (leave ``t_context`` unset/None).
    config.steervla.sample_t_context = False
    config.steervla.cot_temperature = 0.0

    return config
