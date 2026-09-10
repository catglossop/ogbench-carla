"""CAST-relabel / HL-DAgger: the adaptive-sampling recipe, batched into rare large bursts.

Identical to ``steervla_cast_relabel_hl16_adaptive_config.py`` in every other respect -- same
matched-crop 6000 checkpoint, ``proprio_norm=False``, ``hl_update_batch_size=64``, ``hl_lr=1e-5``,
``cot_temperature=0.5``, the same 50/40/10 pool mixture and the same adaptive severity weighting --
so a comparison against those runs isolates the update *cadence*.

What changes: **fifty gradient steps every 1500 ENVIRONMENT STEPS**, instead of one step every 16
``update_with_vla`` calls.

``hl_update_every`` does NOT count environment steps -- it counts ``update_with_vla`` calls
(``steervla.py``: ``self._hl_update_calls % self.hl_update_every``). The loop makes
``updates_per_step`` calls once every ``update_interval`` env steps, so with the inherited
``update_interval=10`` / ``updates_per_step=5`` there are 0.5 calls per env step and 1500 env steps
is **750** calls, not 1500. Setting the knob to a literal 1500 would silently give a 3000-env-step
cadence -- twice what was asked for. So the value is derived from the run's own cadence below
rather than hard-coded, and stays correct if update_interval / updates_per_step are ever changed.

Cadence arithmetic over a full-length run: a 10 000-step budget is ~6 ``update_hl`` calls x 50
steps = ~300 applied updates, so ``--max_hl_updates=150`` binds first, at ~3 calls / ~4500 env
steps. Two consequences worth knowing before reading results:

* A run that stops early -- on ``--stop_on_driving_score``, or with a short ``--online-steps`` --
  may take **zero or one** HL update, where the hl16 recipe would have taken dozens. A shorter
  budget is not a proportionally shorter version of this config; it can be a no-op.
* ``--updates_after_driving_score`` is quantised to 50. The countdown is only checked between
  ``update_hl`` calls and each call applies all 50 steps, so "20 more updates" delivers 50.
"""

from configs.steervla_cast_relabel_hl16_adaptive_config import get_config as get_hl16_adaptive_config

# The cadence this config is actually specified in.
HL_UPDATE_EVERY_ENV_STEPS = 1500
HL_UPDATE_NUM_STEPS = 50


def get_config():
    config = get_hl16_adaptive_config()
    # calls-per-env-step = updates_per_step / update_interval; invert it to convert the env-step
    # cadence above into the call count the throttle actually compares against.
    update_interval = max(1, int(config.get("update_interval", 1)))
    updates_per_step = max(1, int(config.get("updates_per_step", 1)))
    calls_per_env_step = updates_per_step / update_interval
    config.steervla.hl_update_every = max(1, round(HL_UPDATE_EVERY_ENV_STEPS * calls_per_env_step))
    config.steervla.hl_update_num_steps = HL_UPDATE_NUM_STEPS
    return config
