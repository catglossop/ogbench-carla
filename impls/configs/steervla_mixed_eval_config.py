"""Frozen InternVL2 hierarchical planner + pi05 6000 backbone/action expert."""
from pathlib import Path
from configs.steervla_rollout_job2_eaf004_no_ego_history import get_config as base_config


def get_config():
    config = base_config()
    root = Path(__file__).resolve().parents[2]
    config.steervla.hl_provider = 'internvl2'
    config.steervla.hl_checkpoint = str(
        root / 'simlingo_checkpoints/2026_05_24_06_52_33_simlingo_seed1_bellman'
        '/checkpoints/epoch=019.ckpt/converted/pytorch_model.bin')
    config.steervla.simlingo_source_root = '/home/celinet/simlingo-steervla'
    config.steervla.hl_python = '/home/celinet/miniconda3/envs/simlingo/bin/python'
    config.steervla.actions_per_model_query = 3
    config.steervla.debug_noise = False
    return config
