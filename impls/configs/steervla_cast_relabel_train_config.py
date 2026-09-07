"""``get_config()`` for the CAST-relabel **training** experiment.

Same observer pipeline as ``steervla_cast_relabel_config.py`` (window rollout -> VLM
review -> per-chunk GOOD/BAD credit -> corrected subtask + fresh reasoning trace), but
this variant is the entrypoint for the high-level (VLM-backbone) update loop:

  1. SteerVLA is loaded as a **trainable** model — the full OpenPI ``TrainState``
     (optimizer + opt_state + freeze/trainable filters), like ``scripts/train.py`` — via
     ``steervla.load_trainable_params=True`` (see ``vlas/steervla.py``).
  2. Every BAD/relabeled action chunk is persisted as a SteerVLA high-level training sample
     in the ``steervla_hl_dataset_format`` schema (image + ego state + prompt + corrected
     subtask + new reasoning trace + masked action chunk), via ``cast_relabel.store_hl_dataset``.

The actual VLM-backbone gradient step is wired in a later step; for now this config just
loads the trainable model and writes the high-level dataset so the storage path can be
exercised end-to-end.

Everything else (routes, DSRL/SteerVLA knobs, GPU pinning, disabled DSRL critic coach) is
inherited from ``steervla_cast_relabel_config.py``.
"""

from configs.steervla_cast_relabel_config import get_config as get_cast_relabel_config


def get_config():
    config = get_cast_relabel_config()

    # (1) Load SteerVLA as a trainable model (full train state), not inference-only.
    config.steervla.load_trainable_params = True
    # High-level (VLM-backbone) update cadence, run from DSRL ``update_with_vla``. Throttle so a
    # full VLM forward/backward doesn't run on every DSRL update.
    config.steervla.hl_update_every = 8
    # Matches ``steervla_cast_relabel_config``'s 64. The former value of 2 made each HL step
    # almost pure noise; 64 is the batch the base config was tuned around.
    config.steervla.hl_update_batch_size = 64
    config.steervla.hl_update_num_steps = 1

    # Policy checkpoints for later inference. This is the config that actually *trains* the
    # backbone, so it is the one where checkpointing matters: params-only exports to
    # ``<hl_checkpoint_dir>/<step>/params``, redeployable with load_trainable_params=False.
    # Overridable per run via run_carla.sh's --hl-ckpt-dir / --hl-ckpt-every / --hl-ckpt-keep-last.
    config.steervla.hl_checkpoint_every_steps = 2000
    # Empty -> <save_dir>/checkpoints, alongside videos/ and trajectories/. Set an absolute path
    # to collect every run's checkpoints under one root instead (mind the ~10 GB each).
    config.steervla.hl_checkpoint_dir = ""
    config.steervla.hl_checkpoint_keep_last = 3

    # (2) Persist BAD/relabeled chunks as high-level (VLM-backbone) samples.
    config.cast_relabel.store_hl_dataset = True
    # Keep the training dataset separate from a plain observer run's artifacts.
    config.cast_relabel.hl_dataset_subdir = "cast_relabel_hl_dataset"
    # Match config.steervla.action_dim so the (unsupervised) stored chunk has the right shape.
    config.cast_relabel.hl_action_dim = config.steervla.action_dim

    return config
