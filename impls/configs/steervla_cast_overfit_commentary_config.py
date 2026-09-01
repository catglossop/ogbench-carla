"""CAST-relabel **overfitting-observation** runs on the 0823 commentary checkpoint.

Same pipeline as ``steervla_cast_relabel_config.py`` (window rollout -> VLM review ->
per-chunk GOOD/BAD credit -> corrected subtask + reasoning -> HL dataset -> online
``SteerVLAActor.update_hl``). Two things differ:

1. The policy is the **commentary** SteerVLA checkpoint
   ``.../pi05_steervla_cot_simplfied_reasoning_commentary_0823_20260823_154520/14000``,
   whose ``actor_config`` is ``pi05_steervla_cot_simplified_reasoning_commentary``. That
   config is byte-identical to ``pi05_steervla_cot_simplified_reasoning_no_ego_history``
   except for ``hl_cot_reasoning_key`` ('commentary' vs 'gemini_refined_label'), which is a
   *training*-time dataset key — the inference contract (state layout, ``include_ego_history
   =False``, action horizon 10, action dim 4) is unchanged.

2. It is aimed at watching the backbone **overfit the small online cast_relabel pool**. The
   training recipe is deliberately left exactly as the base config has it; what is added is
   measurement. ``SteerVLAActor.update_hl`` now logs, on every applied update:

     vla_hl/policy_updates                  cumulative optimizer steps applied to the backbone
     vla_hl/updates_applied                 update_hl bodies that reached the gradient loop
     vla_hl/update_calls                    attempts, incl. the ones hl_update_every threw away
     vla_hl/reuse_mean                      uses per distinct sample ever used (all pools)
     vla_hl/reuse_max                       most-reused single sample
     vla_hl/reuse_online_mean               same, restricted to the online cast_relabel pool
     vla_hl/reuse_online_per_pool_sample    online uses / current online pool size (counts
                                            never-drawn samples as zero — the honest "how many
                                            times has a given sample been trained on")
     vla_hl/reuse_online_pool_coverage      fraction of the online pool ever drawn
     vla_hl/online_pool_size                online cast_relabel samples on disk

   A sample is credited once per gradient step it participates in, so a batch fed to
   ``hl_update_num_steps=2`` counts twice, and a row ``_pad_hl_batch`` duplicated counts once
   per copy — both really are extra gradient exposure.

Single-card layout, same as ``steervla_cast_relabel_job1_slt001.py``: renderer, JAX
inference, DSRL, SigLIP and the HL gradient step all live on one GPU, so **launch under a
``CUDA_VISIBLE_DEVICES=<physical gpu>`` mask** — ``training_gpu_rank`` is an index into
``jax.devices("gpu")``, not a physical id. Inside a single-GPU mask both are 0.

Launch (route comes from ``carla_job.sh --route``; ports/display from ``--job``)::

    CUDA_VISIBLE_DEVICES=5 ./carla_job.sh start --job 410 --train-gpu 0 --render-adapter 5 \\
        --route signalized-junction-left-turn-001 -- \\
        --hl-gpu 0 --agent-config impls/configs/steervla_cast_overfit_commentary_config.py \\
        --train-mode rl --critic-mode none --online-steps 8000 \\
        -- --save_dir=/home/cglossop/exps
"""

from configs.steervla_cast_relabel_config import get_config as get_cast_relabel_config

# Index within CUDA_VISIBLE_DEVICES, not a physical GPU id. With a single-GPU mask this is 0
# for JAX and for torch. run_carla.sh also overwrites training_gpu_rank / hl_training_gpu_rank
# from --train-gpu / --hl-gpu, so keep those flags at 0 too.
JOB_GPU = 0
# Same rank as JOB_GPU -> SteerVLAActor.setup takes the non-split_hl branch and the HL train
# state stays on the inference GPU (no per-update cross-device weight copy).
HL_GPU = JOB_GPU

CHECKPOINT = (
    "gs://cat-logs/pi05_steervla_cot_simplified_reasoning_commentary/"
    "pi05_steervla_cot_simplfied_reasoning_commentary_0823/"
    "pi05_steervla_cot_simplfied_reasoning_commentary_0823_20260823_154520/14000"
)


def get_config():
    config = get_cast_relabel_config()

    # (1) The policy under study.
    config.steervla.checkpoint = CHECKPOINT
    config.steervla.actor_config = "pi05_steervla_cot_simplified_reasoning_commentary"

    # (2) Single-card pinning. siglip_device is NOT touched by run_carla.sh — without this every
    # concurrent job piles SigLIP onto the first visible card.
    config.training_gpu_rank = JOB_GPU
    config.siglip_device = f"cuda:{JOB_GPU}"
    config.steervla.hl_training_gpu_rank = HL_GPU
    # Dropped from the base config's 64: the HL forward+backward shares VRAM with the inference
    # model on a single card (same reasoning as steervla_cast_relabel_job1_slt001.py).
    config.steervla.hl_update_batch_size = 16

    # Base config already has enable_updates / enable_updates_bc_hl / cast_relabel.enabled /
    # store_hl_dataset / load_trainable_params on, and enable_updates_rl off. Left as-is: the
    # point of this run is to *observe* the recipe, not to change it.
    return config
