"""HL-DAgger (CAST-relabel) training arm for the route study.

The *second* half of each queued pair in ``run_hl_dagger_queue.sh``. Starts from the same
``no_ego_history`` v1 @ step 6000 weights as
``steervla_hl_dagger_base_config.py`` and then actually trains the high-level
(VLM-backbone) head on CAST-relabeled data collected on that route:

  rollout window -> Gemini reviews the window video -> GOOD/BAD events -> causal credit
  onto individual action chunks -> corrected subtask + fresh reasoning trace -> stored as
  a ``steervla_hl_dataset_format`` sample -> ``SteerVLAActor.update_hl``.

Only the CoT/VLM backbone is supervised: stored chunks carry an all-``False``
``action_loss_mask``, so the action expert is untouched. That is the "HL" in HL-DAgger,
and it is why ``enable_updates_bc`` (the low-level DAgger path) stays off.

Everything substantive is already correct in ``steervla_cast_relabel_config.py`` —
``load_trainable_params=True``, ``enable_updates_bc_hl=True``,
``cast_relabel.store_hl_dataset=True``, the no-ego-history checkpoint and actor config,
and ``hl_replay_root=/raid/users/cglossop/steervla_hl_pools``. This module exists to (a)
name the experiment, (b) assert those invariants so a future edit to the shared
cast-relabel config cannot silently turn this arm into an observer-only run, and (c) park
the published policy checkpoints under their own root.

``run_carla.sh`` overrides ``training_gpu_rank``/``siglip_device`` from ``--train-gpu``,
``hl_training_gpu_rank`` from ``--hl-gpu``, and ``hl_checkpoint_dir`` from
``--hl-ckpt-dir``.

Launch (ports/display derived from ``carla_job.sh --job k``)::

    ./carla_job.sh start --job 210 --train-gpu 3 --render-adapter 3 -- \\
        --route hazard-at-side-lane-005 \\
        --agent-config impls/configs/steervla_hl_dagger_train_config.py \\
        --train-mode rl --online-steps 20000 --hl-gpu 3 \\
        --run-group bench2drive_hl_dagger_train

Requires ``GEMINI_API_KEY`` to be exported — the CAST observer is a Gemini client and the
run collects no training data without it.
"""

from configs.steervla_cast_relabel_config import get_config as get_cast_relabel_config

HL_CHECKPOINT_ROOT = "/raid/users/cglossop/hl_dagger/checkpoints"


def get_config():
    config = get_cast_relabel_config()

    # ── Assert the training invariants rather than trusting the shared base ──────
    # Each of these is what separates "HL-DAgger training run" from "observer run".
    config.enable_updates = True
    config.enable_updates_bc_hl = True  # the VLM-backbone update
    config.enable_updates_rl = False  # no DSRL critic/actor
    config.enable_updates_bc = False  # no low-level DAgger
    config.steervla.load_trainable_params = True  # full TrainState, not bare params
    config.cast_relabel.enabled = True
    config.cast_relabel.store_hl_dataset = True

    # Published params-only checkpoints land under one root instead of each run's
    # save_dir, so the 10 routes' policies are directly comparable side by side.
    # run_carla.sh --hl-ckpt-dir overrides this per job.
    config.steervla.hl_checkpoint_dir = HL_CHECKPOINT_ROOT

    return config
