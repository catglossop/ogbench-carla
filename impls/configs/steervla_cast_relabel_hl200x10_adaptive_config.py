"""CAST-relabel / HL-DAgger: adaptive sampling, 10 gradient steps every 200 environment steps.

Identical to ``steervla_cast_relabel_hl16_adaptive_config.py`` in every other respect -- same
matched-crop 6000 checkpoint, ``proprio_norm=False``, ``hl_update_batch_size=64``, ``hl_lr=1e-5``,
``cot_temperature=0.5``, the same 50/40/10 pool mixture and the same adaptive severity weighting.

Where it sits among the cadence variants:

    hl1500x50    50 steps / 1500 env steps  =  1 step per 30.0 env steps   cap at ~4500 env steps
    hl750x25     25 steps /  750 env steps  =  1 step per 30.0 env steps   cap at ~4500 env steps
    hl200x10     10 steps /  200 env steps  =  1 step per 20.0 env steps   cap at ~3000 env steps
    hl100x15     15 steps /  100 env steps  =  1 step per  6.7 env steps   cap at ~1000 env steps

So this is a finer-grained cadence than the hl750/hl1500 pair and only mildly denser (1.5x rather
than hl100x15's 4.5x), reaching ``--max_hl_updates=150`` at roughly 3000 env steps -- well inside a
10 000-step budget, so the score-stop has a real chance to fire first.

``hl_update_every`` counts ``update_with_vla`` calls, not env steps (``steervla.py``:
``self._hl_update_calls % self.hl_update_every``), so the env-step cadence is converted below
rather than hard-coded, and stays correct if update_interval / updates_per_step change.
"""

from configs.steervla_cast_relabel_hl16_adaptive_config import get_config as get_hl16_adaptive_config

# The cadence this config is actually specified in.
HL_UPDATE_EVERY_ENV_STEPS = 200
HL_UPDATE_NUM_STEPS = 10


def get_config():
    config = get_hl16_adaptive_config()
    update_interval = max(1, int(config.get("update_interval", 1)))
    updates_per_step = max(1, int(config.get("updates_per_step", 1)))
    calls_per_env_step = updates_per_step / update_interval
    config.steervla.hl_update_every = max(1, round(HL_UPDATE_EVERY_ENV_STEPS * calls_per_env_step))
    config.steervla.hl_update_num_steps = HL_UPDATE_NUM_STEPS
    return config
