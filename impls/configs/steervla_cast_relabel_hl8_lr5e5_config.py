"""CAST-relabel / HL-DAgger training run — HL update every 8 calls, flat LR 5e-5.

One config, used by BOTH concurrent route jobs (GPU 5 and GPU 6). It differs from
``steervla_cast_relabel_train_config.py`` in exactly two ways:

  * ``steervla.hl_lr = 5e-5`` — flat LR for the high-level (VLM-backbone) optimizer,
    replacing the base config's ``1e-5``. Setting ``hl_lr`` swaps the schedule for a
    constant LR with **no warmup ramp** (``vlas/steervla.py`` ~L1692), and the actor prints
    ``[steervla] HL optimizer LR overridden to flat 5e-05`` at startup — check for that line.
  * single-GPU pinning, so each of the two jobs is confined to its own card.

``hl_update_every = 8`` is inherited unchanged from
``steervla_cast_relabel_train_config.py`` — it means "run ``update_hl`` once every 8
``update_with_vla`` calls", i.e. every 8 *DSRL update* calls, not every 8 env steps. The
env-step cadence is the separate ``config.update_interval`` knob, left at the base value.

The HL gradient step really does run: ``jax_agents/dsrl.py:1327`` calls
``steervla_actor.update_hl(global_step=...)``. (The parent config's docstring still claims
the step "is wired in a later step" — that is stale.)

GPU pinning. ``training_gpu_rank`` indexes ``jax.devices("gpu")``, NOT physical GPU ids, so
**launch each job under ``CUDA_VISIBLE_DEVICES=<physical gpu>``**: JAX and torch then both
see exactly one device at index 0, the same config works for either card, and JAX does not
grab a ~0.5 GB stub on every other GPU. ``--render-adapter`` stays *physical* — the wrapper
builds CARLA's subprocess with a scrubbed env, so the mask does not apply to it.

(Historical note: ``steervla_cast_relabel_job1_slt001.py`` says GPUs 0 and 6 are compute
mode ``Prohibited`` on this box. That is no longer true — all 8 are ``Default`` as of
2026-09-06 — but the mask is still the right way to launch, for the reasons above.)

Launch (ports/display derived from ``carla_job.sh --job k``)::

    CUDA_VISIBLE_DEVICES=5 ./carla_job.sh start --job <k> --train-gpu 0 --render-adapter 5 \\
        --route generalization-wall-1095 -- \\
        --hl-gpu 0 \\
        --agent-config impls/configs/steervla_cast_relabel_hl8_lr5e5_config.py \\
        --train-mode rl --critic-mode none --online-steps 8000 --save-buffer true

``--train-mode rl`` is not optional: ``run_carla.sh`` defaults it to ``dagger``, which
changes the update dispatch. ``GEMINI_API_KEY`` must be exported — the CAST relabel observer
calls Gemini to review each window.
"""

from configs.steervla_cast_relabel_train_config import get_config as get_cast_relabel_train_config

# Index *within* CUDA_VISIBLE_DEVICES, not a physical GPU id — see the module docstring.
# With a single-GPU mask this is 0 for both JAX and torch, on either card.
JOB_GPU = 0
# Equal to JOB_GPU -> SteerVLAActor.setup takes the non-``split_hl`` branch, so the HL train
# state lives on the inference GPU and no cross-device weight copy happens per HL update.
HL_GPU = JOB_GPU

HL_LR = 5e-5

# The matched-crop, UNNORMALIZED ll_heavy run @ step 6000 -- the checkpoint the 220-route
# leaderboard was run on (DS 73.85 / RC 91.38), replacing the base config's older
# ``no_ego_history`` @ 6000. Params-only mirror; ``load_trainable_params=True`` builds a FRESH
# train state from ``<ckpt>/params`` via CheckpointWeightLoader and never reads ``train_state``,
# so the 37 GB optimizer directory is not needed.
#   GCS: gs://cat-logs/pi05_steervla_cot_simplified_reasoning_ll_heavy/ll_heavy_unnormed_matchcrop
#        /ll_heavy_unnormed_matchcrop_20260904_152800/6000
CHECKPOINT = "/raid/users/cglossop/steervla_pi_ckpts/ll_heavy_unnormed_matchcrop/6000"
ACTOR_CONFIG = "pi05_steervla_cot_simplified_reasoning_ll_heavy"


def get_config():
    config = get_cast_relabel_train_config()

    # --- the two requested settings -------------------------------------------------
    # Flat LR for the HL optimizer (base config ships 1e-5).
    config.steervla.hl_lr = HL_LR
    # Inherited from the parent, restated so the intent is visible at the point of use and a
    # change to the parent cannot silently alter this run's cadence.
    config.steervla.hl_update_every = 8
    # 64 across the whole hl-dagger/cast-relabel family as of 2026-09-06.
    config.steervla.hl_update_batch_size = 64

    # --- checkpoint under training ----------------------------------------------------
    config.steervla.checkpoint = CHECKPOINT
    config.steervla.actor_config = ACTOR_CONFIG
    # Properties of THIS checkpoint's openpi TrainConfig, not free choices. ll_heavy sets
    # data.include_ego_history=False (prompt-state width 2) and data.proprio_norm=False (raw m/s
    # and degrees). Restated here even though the whole family now defaults to False
    # (2026-09-06) -- it is a property of this checkpoint, not a global preference, so a future
    # change to the base must not silently alter it. ll_heavy is also
    # ``skip_norm_stats=True`` and ships no ``assets/*/norm_stats.json``, so
    # ``resolve_openpi_norm_enabled`` correctly leaves OpenPI Normalize/Unnormalize off.
    config.steervla.include_ego_history = False
    config.steervla.proprio_norm = False

    # --- single-card pinning --------------------------------------------------------
    config.training_gpu_rank = JOB_GPU
    # Torch device for the frozen SigLIP encoder. run_carla.sh does NOT override this, so it
    # must be set here or both jobs pile SigLIP onto the first visible card.
    config.siglip_device = f"cuda:{JOB_GPU}"
    config.steervla.hl_training_gpu_rank = HL_GPU

    return config
