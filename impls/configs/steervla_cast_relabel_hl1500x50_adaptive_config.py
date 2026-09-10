"""CAST-relabel / HL-DAgger: the adaptive-sampling recipe, but batched into rare large bursts.

Identical to ``steervla_cast_relabel_hl16_adaptive_config.py`` in every other respect -- same
matched-crop 6000 checkpoint, ``proprio_norm=False``, ``hl_update_batch_size=64``, ``hl_lr=1e-5``,
``cot_temperature=0.5``, the same 50/40/10 pool mixture and the same adaptive severity weighting --
so a comparison against those runs isolates the update *cadence*.

What changes: instead of one gradient step every 16 ``update_with_vla`` calls, fifty steps every
1500. Same total gradient budget, delivered in a few large bursts rather than continuously.

  hl_update_every      16 -> 1500
  hl_update_num_steps   1 -> 50

Cadence arithmetic, because the numbers only work out over a full-length run. With the inherited
``update_interval=10`` and ``updates_per_step=5`` the loop makes ~0.5 ``update_with_vla`` calls per
env step, so a 10 000-step budget yields ~5000 calls -> ~3 ``update_hl`` calls -> ~150 applied
updates. That is exactly the ``--max_hl_updates=150`` cap these runs use, reached near the END of
the budget rather than partway through. Two consequences worth knowing before reading the results:

* A run that stops early -- on ``--stop_on_driving_score`` or a short episode budget -- may take
  **zero or one** HL update. Under the hl16 recipe the same run would have taken dozens. A shorter
  ``--online-steps`` is not a proportionally shorter version of this config; it can be a no-op.
* ``--updates_after_driving_score`` is quantised to 50 here. The countdown is checked between
  ``update_hl`` calls, and each call applies all 50 steps, so a request for "20 more updates"
  actually delivers 50.
"""

from configs.steervla_cast_relabel_hl16_adaptive_config import get_config as get_hl16_adaptive_config

HL_UPDATE_EVERY = 1500
HL_UPDATE_NUM_STEPS = 50


def get_config():
    config = get_hl16_adaptive_config()
    config.steervla.hl_update_every = HL_UPDATE_EVERY
    config.steervla.hl_update_num_steps = HL_UPDATE_NUM_STEPS
    return config
