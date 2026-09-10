"""CAST-relabel / HL-DAgger: adaptive sampling, 15 gradient steps every 100 environment steps.

Identical to ``steervla_cast_relabel_hl16_adaptive_config.py`` in every other respect -- same
matched-crop 6000 checkpoint, ``proprio_norm=False``, ``hl_update_batch_size=64``, ``hl_lr=1e-5``,
``cot_temperature=0.5``, the same 50/40/10 pool mixture and the same adaptive severity weighting.

NOT rate-matched to its siblings, unlike hl750x25 vs hl1500x50:

    hl1500x50    50 steps / 1500 env steps  =  1 step per 30.0 env steps
    hl750x25     25 steps /  750 env steps  =  1 step per 30.0 env steps
    hl100x15     15 steps /  100 env steps  =  1 step per  6.7 env steps   <- this one

So a comparison against those two confounds cadence with roughly 4.5x more total training. That
may well be the point -- but it is not the controlled granularity comparison the other pair is,
and reading it as one would be wrong.

Consequence worth knowing before launching. At 15 applied steps per burst, ``--max_hl_updates=150``
is reached in TEN bursts, i.e. after about **1000 env steps** -- a tenth of the usual 10 000-step
budget. Under the eval-mode recipe the run will stop, export, and go to frozen eval almost
immediately. If the intent is a full-length run at this density, raise ``--max_hl_updates``
accordingly (e.g. 1500 to use the whole budget).

``hl_update_every`` counts ``update_with_vla`` calls, not env steps (``steervla.py``:
``self._hl_update_calls % self.hl_update_every``), so the env-step cadence is converted below
rather than hard-coded, and stays correct if update_interval / updates_per_step change.
"""

from configs.steervla_cast_relabel_hl16_adaptive_config import get_config as get_hl16_adaptive_config

# The cadence this config is actually specified in.
HL_UPDATE_EVERY_ENV_STEPS = 100
HL_UPDATE_NUM_STEPS = 15


def get_config():
    config = get_hl16_adaptive_config()
    update_interval = max(1, int(config.get("update_interval", 1)))
    updates_per_step = max(1, int(config.get("updates_per_step", 1)))
    calls_per_env_step = updates_per_step / update_interval
    config.steervla.hl_update_every = max(1, round(HL_UPDATE_EVERY_ENV_STEPS * calls_per_env_step))
    config.steervla.hl_update_num_steps = HL_UPDATE_NUM_STEPS
    return config
