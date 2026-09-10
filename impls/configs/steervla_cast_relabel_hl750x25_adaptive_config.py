"""CAST-relabel / HL-DAgger: adaptive sampling, 25 gradient steps every 750 environment steps.

Identical to ``steervla_cast_relabel_hl16_adaptive_config.py`` in every other respect -- same
matched-crop 6000 checkpoint, ``proprio_norm=False``, ``hl_update_batch_size=64``, ``hl_lr=1e-5``,
``cot_temperature=0.5``, the same 50/40/10 pool mixture and the same adaptive severity weighting.

Sibling of ``steervla_cast_relabel_hl1500x50_adaptive_config.py`` and deliberately at the SAME
average rate: 25 steps / 750 env steps and 50 steps / 1500 env steps are both one gradient step
per 30 env steps. So a comparison between the two isolates burst GRANULARITY at constant total
budget -- more frequent, smaller updates against rarer, larger ones -- rather than confounding it
with how much training happened.

``hl_update_every`` counts ``update_with_vla`` calls, not env steps (``steervla.py``:
``self._hl_update_calls % self.hl_update_every``). The loop makes ``updates_per_step`` calls once
every ``update_interval`` env steps, so the env-step cadence is converted below rather than
hard-coded, and stays correct if either of those is changed.
"""

from configs.steervla_cast_relabel_hl16_adaptive_config import get_config as get_hl16_adaptive_config

# The cadence this config is actually specified in.
HL_UPDATE_EVERY_ENV_STEPS = 750
HL_UPDATE_NUM_STEPS = 25


def get_config():
    config = get_hl16_adaptive_config()
    update_interval = max(1, int(config.get("update_interval", 1)))
    updates_per_step = max(1, int(config.get("updates_per_step", 1)))
    calls_per_env_step = updates_per_step / update_interval
    config.steervla.hl_update_every = max(1, round(HL_UPDATE_EVERY_ENV_STEPS * calls_per_env_step))
    config.steervla.hl_update_num_steps = HL_UPDATE_NUM_STEPS
    return config
