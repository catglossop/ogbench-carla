"""Residual-SAC settings for evaluating a frozen CAST-relabel policy checkpoint.

The paired CAST-to-residual runner supplies ``steervla.checkpoint`` for each
route/seed. This config fixes the compatible actor architecture and residual
hyperparameters while keeping the CAST checkpoint itself frozen.
"""

from pathlib import Path
import runpy


_BASE_GET_CONFIG = runpy.run_path(
    str(Path(__file__).with_name("steervla_residual_config.py"))
)["get_config"]


def get_config():
    config = _BASE_GET_CONFIG()
    # Same residual schedule and hyperparameters as the 10k residual evaluation.
    config.residual_warmup_steps = 1000
    config.residual_ramp_steps = 1500
    config.residual_accel_scale = 0.2
    config.residual_steer_scale = 0.2
    config.residual_bc_beta = 0.1
    config.residual_bc_normalize = False
    config.expo = False
    config.best_of_n = 1
    config.otf_td_backup = False

    # Must match the policy architecture used by the HL16 adaptive CAST trainer.
    config.steervla.actor_config = "pi05_steervla_cot_simplified_reasoning_ll_heavy"
    config.steervla.proprio_norm = False
    config.steervla.actions_per_model_query = 3
    config.steervla.actions_per_cot = 5
    return config
