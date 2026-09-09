"""Residual-SAC evaluation settings for the selected base-policy checkpoint.

The 10k-step evaluation halves the residual schedule from the normal 20k-step
sweep: 1000 base-only warmup steps followed by a 1500-step residual ramp.
"""

from pathlib import Path
import runpy


_BASE_GET_CONFIG = runpy.run_path(
    str(Path(__file__).with_name("steervla_residual_config.py"))
)["get_config"]


CHECKPOINT = (
    "gs://cat-logs/pi05_steervla_cot_simplified_reasoning_ll_heavy/"
    "ll_heavy_unnormed_matchcrop/"
    "ll_heavy_unnormed_matchcrop_20260904_152800/6000"
)


def get_config():
    config = _BASE_GET_CONFIG()
    config.residual_warmup_steps = 1000
    config.residual_ramp_steps = 1500
    config.residual_accel_scale = 0.2
    config.residual_steer_scale = 0.2
    config.residual_bc_beta = 0.1
    config.residual_bc_normalize = False
    config.expo = False
    config.best_of_n = 1
    config.otf_td_backup = False

    config.steervla.checkpoint = CHECKPOINT
    config.steervla.actor_config = "pi05_steervla_cot_simplified_reasoning_ll_heavy"
    config.steervla.proprio_norm = False
    config.steervla.actions_per_model_query = 3
    config.steervla.actions_per_cot = 5
    return config
