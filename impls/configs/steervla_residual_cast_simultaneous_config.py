"""Simultaneous CAST relabel + standalone SAC-residual training.

CAST updates the trainable SteerVLA high-level policy while ``sac_residual``
independently learns a residual controller from environment reward.
"""

from configs.steervla_cast_relabel_hl200x10_adaptive_config import get_config as get_cast_config
from configs.steervla_residual_eval_config import get_config as get_residual_config


def get_config():
    config = get_residual_config()
    cast = get_cast_config()

    config.cast_relabel = cast.cast_relabel
    for key in (
        "load_trainable_params",
        "hl_training_gpu_rank",
        "hl_update_batch_size",
        "hl_update_num_steps",
        "hl_checkpoint_every_steps",
        "hl_checkpoint_dir",
        "hl_checkpoint_keep_last",
    ):
        config.steervla[key] = cast.steervla[key]
    # Standalone SAC calls update_hl once per env step; the DSRL-derived CAST
    # config's value is converted for two update calls per step.
    config.steervla.hl_update_every = 200
    return config
