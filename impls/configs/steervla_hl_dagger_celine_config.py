"""Machine-local HL CAST/DAgger config for Celine's RAID storage."""

from configs.steervla_cast_relabel_config import get_config as get_cast_relabel_config


def get_config():
    config = get_cast_relabel_config()
    config.steervla.hl_replay_root = "/raid/users/celine/steervla_hl_pools"
    config.steervla.checkpoint = (
        "/raid/users/celine/openpi/cat-logs/"
        "pi05_steervla_cot_simplified_reasoning_no_ego_history/"
        "pi05_steervla_simplified_reasoning_no_ego_history_v1/"
        "pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000"
    )
    config.steervla.hl_checkpoint_dir = "/raid/users/celine/hl_dagger/checkpoints"
    return config
