"""Residual SAC on the frozen mixed policy (InternVL2 HL + pi05 LL), with Surya's SimLingo residual settings.

``steervla_mixed_residual_config.py`` swaps the base to the mixed policy but keeps the generic residual
defaults; this applies the hyperparameters of the SimLingo residual-RL eval sweep
(``simlingo_steervla_residual_config.py``, ``run_residual_eval.sh``) and the mixed policy's cadence /
paths from ``_mixed_policy.py``. Only the residual SAC agent trains.
"""
from configs._mixed_policy import apply_mixed_policy
from configs.steervla_mixed_residual_config import get_config as get_mixed_residual_config


def get_config():
    config = get_mixed_residual_config()
    # == simlingo_steervla_residual_config.py
    config.state_encoder = "siglip_pool"
    config.residual_warmup_steps = 1000
    config.residual_ramp_steps = 1500
    config.residual_accel_scale = 0.1
    config.residual_steer_scale = 0.1
    config.residual_bc_beta = 1.0
    config.residual_bc_normalize = False
    config.expo = False
    config.best_of_n = 1
    config.otf_td_backup = False
    apply_mixed_policy(config.steervla)
    return config
