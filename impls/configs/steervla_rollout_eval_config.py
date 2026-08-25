"""``get_config()`` for a **pure inference / rollout-only** evaluation of a saved SteerVLA
checkpoint — no gradients, no VLM, no data written.

This is the deployment half of the CAST loop. A ``steervla_cast_relabel_train_config.py`` run
exports its HL-fine-tuned backbone every ``steervla.hl_checkpoint_every_steps`` env steps to
``<hl_checkpoint_dir>/<step>/params`` (params only, no optimizer state). This config redeploys
one of those frozen and rolls it out on arbitrary routes so the fine-tune can be scored against
routes it was never trained on.

Differences from ``steervla_cast_collect_config.py``, which it inherits:

- ``cast_relabel.enabled = False``. Collection runs the VLM review pass on every window; an
  eval run must not. Beyond the Gemini spend, the review pass is the dominant per-window
  latency and it writes an HL corpus we would then have to remember not to train on.
- ``steervla.checkpoint`` points at the exported HL checkpoint rather than the GCS pretrain.
  ``load_trainable_params`` stays ``False`` (inherited), which is what an exported
  params-only checkpoint supports — see ``SteerVLAActor.save_checkpoint``.
- ``steervla.hl_checkpoint_every_steps = 0``. Nothing is being trained, so there is nothing to
  export; leaving it on would only produce copies of the input weights.

``actor_config`` is inherited unchanged and **must** match the architecture the checkpoint was
saved under (``pi05_steervla_cot_simplified_reasoning_no_ego_history``). Restoring a checkpoint
under a mismatched config fails in OpenPI's param restore rather than degrading quietly.

Usage::

    STEERVLA_EVAL_CHECKPOINT=/path/to/checkpoints/5000 \\
    ./carla_job.sh start --job 80 --train-gpu 0 --render-adapter 0 \\
        --route enter-actor-flow-004 -- \\
        --agent-config impls/configs/steervla_rollout_eval_config.py \\
        --online-steps 5000 --enable-updates false --run-group CastEval5k \\
        -- --save_dir=/raid/users/cglossop/exps

``GEMINI_API_KEY`` is *not* needed (no VLM calls). Set ``STEERVLA_EVAL_CHECKPOINT`` to evaluate
a different step without editing this file. The default is a stable copy of the step-6000 export
from the ``generalization-wall-1095`` CAST training run.
"""

import os

from configs.steervla_cast_collect_config import get_config as get_collect_config

# HL export from the wall-1095 CAST training run (wandb snz0n8eq), copied OUT of that run's
# checkpoints/ directory on purpose.
#
# ``steervla.hl_checkpoint_keep_last`` (3 by default, 10 for that run) prunes the oldest export
# every time a new one lands, so a checkpoint inside a *live* training run is a moving target:
# the step-5000 export evaluated here was deleted mid-eval when the source run reached step
# 15000, which killed a queued rollout with ``FileNotFoundError`` at actor startup. Always
# evaluate a copy, never the original — and copy it before the run advances 10 more intervals.
DEFAULT_EVAL_CHECKPOINT = '/raid/users/cglossop/cast_eval_ckpts/wall1095_155243_step6000'


def get_config():
    config = get_collect_config()

    # ── No VLM. This is the whole difference from a collection run. ──────────────────────
    config.cast_relabel.enabled = False
    config.cast_relabel.store_hl_dataset = False
    config.cast_relabel.debug = False
    config.cast_relabel.save_artifacts = False

    # ── Redeploy the exported checkpoint, frozen. ────────────────────────────────────────
    config.steervla.checkpoint = os.environ.get('STEERVLA_EVAL_CHECKPOINT', DEFAULT_EVAL_CHECKPOINT)
    config.steervla.load_trainable_params = False
    config.steervla.hl_checkpoint_every_steps = 0
    config.steervla.hl_checkpoint_dir = ''

    return config
