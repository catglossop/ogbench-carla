"""Job 1 — CAST relabel **training** run on ``signalized-junction-left-turn-001``.

Same pipeline as ``steervla_cast_relabel_config.py`` (window rollout -> VLM review ->
per-chunk GOOD/BAD credit -> corrected subtask + reasoning -> HL dataset -> online
``SteerVLAActor.update_hl``), with updates left ON. The only deviation from the base
config is *which* GPUs it uses: this job is confined to a **single card**.

Everything on one GPU:
  * CARLA renderer (``--render-adapter``; the wrapper builds the UE4 subprocess its own
    scrubbed env, so the parent's ``CUDA_VISIBLE_DEVICES`` does not affect it),
  * JAX inference + DSRL RL updates (``training_gpu_rank``),
  * SigLIP (``siglip_device``),
  * the HL (VLM-backbone) gradient step. ``hl_training_gpu_rank == training_gpu_rank``
    makes ``SteerVLAActor.setup`` take the non-``split_hl`` branch, so the train state
    lives on the inference GPU and no cross-device weight copy happens per HL update.

**Launch this job with ``CUDA_VISIBLE_DEVICES=<physical gpu>``.** ``training_gpu_rank`` is
an *index into* ``jax.devices("gpu")``, NOT a physical GPU id, and those two diverge on this
box: GPUs 0 and 6 are in nvidia-smi compute mode ``Prohibited``, so JAX enumerates only
``[1,2,3,4,5,7]`` and rank 1 silently lands on physical cuda:2. Masking with
``CUDA_VISIBLE_DEVICES`` collapses the ambiguity — JAX and torch then both see exactly one
device at index 0 — and also stops JAX from grabbing a ~0.5 GB stub on every other card.

Because the HL train state (params + Adam mu/nu + backward activations) shares VRAM with
the inference model, ``hl_update_batch_size`` is dropped from the base config's 64 (which
assumes a dedicated card) to 16. To go back to a two-card split, drop the
``CUDA_VISIBLE_DEVICES`` mask, set ``HL_GPU`` to a free JAX rank other than ``JOB_GPU``,
and restore the batch to 64.

Launch on physical GPU 1 (ports/display come from ``carla_job.sh --job 1``)::

    CUDA_VISIBLE_DEVICES=1 ./carla_job.sh start --job 1 --train-gpu 0 --render-adapter 1 \\
        --route signalized-junction-left-turn-001 -- \\
        --hl-gpu 0 \\
        --agent-config impls/configs/steervla_cast_relabel_job1_slt001.py \\
        --train-mode rl --critic-mode none --online-steps 8000 --save-buffer true \\
        -- --save_dir=/home/cglossop/exps

Note ``--render-adapter`` stays *physical* (1) while ``--train-gpu``/``--hl-gpu`` are 0,
the only index inside the mask.
"""

from configs.steervla_cast_relabel_config import get_config as get_cast_relabel_config

# Index within CUDA_VISIBLE_DEVICES, not a physical GPU id — see the module docstring.
# With a single-GPU mask this is always 0 for both JAX and torch.
JOB_GPU = 0
# Same rank as JOB_GPU -> the HL update is NOT isolated; it runs on the inference GPU.
HL_GPU = JOB_GPU


def get_config():
    config = get_cast_relabel_config()

    # Index into jax.devices("gpu"). run_carla.sh also overwrites this from --train-gpu, so
    # keep the two in sync when launching.
    config.training_gpu_rank = JOB_GPU
    # Torch device for the frozen SigLIP encoder. NOT touched by run_carla.sh — it must be
    # set here or every concurrent job piles SigLIP onto the first visible card.
    config.siglip_device = f"cuda:{JOB_GPU}"

    # Equal to training_gpu_rank -> steervla.py :: SteerVLAActor.setup computes
    # hl_device == infer_device, so split_hl is False and the whole train state stays on the
    # single job GPU.
    config.steervla.hl_training_gpu_rank = HL_GPU
    # Reduced from the base config's 64: the HL forward+backward now shares VRAM with the
    # inference model, so keep the batch modest to avoid OOM.
    config.steervla.hl_update_batch_size = 16

    # Base config already has: enable_updates=True, enable_updates_rl=True,
    # enable_updates_bc_hl=True, cast_relabel.enabled=True, store_hl_dataset=True,
    # load_trainable_params=True. Left as-is — this is the training job.
    return config
