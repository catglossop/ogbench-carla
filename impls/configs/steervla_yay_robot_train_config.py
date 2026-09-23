"""``get_config()`` for the YAY-Robot **train-and-redeploy** experiment.

Same foresight corrector as ``steervla_yay_robot_config.py`` -- every CoT query is shown to a
VLM, which either keeps the language or corrects it, and the correction conditions the flow
expert on the spot -- but this is the variant that closes the loop:

  1. Interventions (and the 2 s of frames leading up to each) accumulate as
     ``steervla_hl_dataset_format`` samples in ``<save_dir>/yay_robot_hl_dataset``.
  2. ``SteerVLAActor.update_hl``, driven from ``DSRLAgent.update_with_vla(run_hl=True)``, trains
     the CoT/VLM backbone on them online. Only the backbone: every sample carries an all-``False``
     ``action_loss_mask``, so the action expert is untouched.
  3. Every **2000 env steps** the fine-tuned backbone is exported params-only to
     ``<hl_checkpoint_dir>/<step>/params``, redeployable as a frozen policy with
     ``steervla.checkpoint="<hl_checkpoint_dir>/<step>", load_trainable_params=False``.

Step 3 is the "redeploy every 2k steps" cadence. It is the same
``steervla.hl_checkpoint_every_steps`` mechanism ``steervla_cast_relabel_train_config.py`` uses,
so the exports are directly comparable between the hindsight and foresight arms. Mind the disk:
each export is ~10 GB, hence ``hl_checkpoint_keep_last``.

The measure of success is the intervention rate falling over the run (``yay_robot/intervention_rate``
in W&B): the backbone is learning to say what the VLM would have corrected it to, unprompted. That
is also why the post-stop eval episodes run with the correction hook **detached** -- the deployable
artifact is the backbone alone, with no VLM in the loop.

This module exists to (a) name the experiment and (b) assert the training invariants, so a later
edit to the shared ``steervla_yay_robot_config.py`` cannot silently turn this arm into a
collect-only run.

Requires ``GEMINI_API_KEY`` -- the corrector is a Gemini client, and without it the run collects
no interventions at all.

Launch (ports/display derived from ``carla_job.sh --job k``)::

    ./carla_job.sh start --job 220 --train-gpu 3 --render-adapter 3 -- \\
        --route hazard-at-side-lane-005 \\
        --agent-config impls/configs/steervla_yay_robot_train_config.py \\
        --train-mode rl --online-steps 20000 --hl-gpu 3 \\
        --run-group bench2drive_yay_robot_train
"""

from configs.steervla_yay_robot_config import get_config as get_yay_robot_config

# Where the redeployable params-only exports land. Empty -> ``<save_dir>/checkpoints``, next to
# videos/ and trajectories/. ``run_carla.sh --hl-ckpt-dir`` overrides this per job.
HL_CHECKPOINT_ROOT = ""


def get_config():
    config = get_yay_robot_config()

    # ── Assert the training invariants rather than trusting the shared base ──────────────
    # Each of these is what separates a training run from a collect-only one.
    config.enable_updates = True
    config.enable_updates_bc_hl = True  # the VLM-backbone update
    config.enable_updates_rl = False  # no DSRL critic/actor
    config.enable_updates_bc = False  # no low-level DAgger
    config.steervla.load_trainable_params = True  # full TrainState, not bare params
    config.yay_robot.enabled = True
    config.yay_robot.store_hl_dataset = True
    # cast_relabel and yay_robot cannot both own the HL dataset dir; main_carla raises if they do.
    config.cast_relabel.enabled = False

    # High-level update cadence, run from DSRL ``update_with_vla``. Matches the CAST training arm
    # so the two are comparable: throttled so a full VLM forward/backward is not run on every DSRL
    # update, at the batch size the base config was tuned around.
    config.steervla.hl_update_every = 8
    config.steervla.hl_update_batch_size = 64
    config.steervla.hl_update_num_steps = 1

    # ── Redeploy cadence: export the fine-tuned backbone every 2k env steps ──────────────
    # Params-only, no optimizer state. Redeploy one frozen with
    #   steervla.checkpoint="<hl_checkpoint_dir>/<step>", load_trainable_params=False
    # (same actor_config). Overridable per run via run_carla.sh --hl-ckpt-every /
    # --hl-ckpt-dir / --hl-ckpt-keep-last.
    config.steervla.hl_checkpoint_every_steps = 2000
    config.steervla.hl_checkpoint_dir = HL_CHECKPOINT_ROOT
    # ~10 GB per export, so a 20k-step run at 2k spacing would leave ~100 GB. Keep the newest 3.
    config.steervla.hl_checkpoint_keep_last = 3

    return config
