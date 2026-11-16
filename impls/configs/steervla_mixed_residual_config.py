"""Residual SAC over frozen InternVL2 HL + Pi05 action expert."""
from __future__ import annotations

import sys

from configs.steervla_residual_config import get_config as get_residual_config


_PI05_CKPT = (
    "/raid/users/surya/openpi/cat-logs/pi05_steervla_cot_simplified_reasoning_no_ego_history/"
    "pi05_steervla_simplified_reasoning_no_ego_history_v1/"
    "pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000"
)
_SIMLINGO_HL_CKPT = (
    "/raid/users/celine/steervla-ckpts/"
    "2026_05_24_06_52_33_simlingo_seed1_bellman/checkpoints/epoch=019.ckpt"
)


def get_config():
    config = get_residual_config()
    actor = config.steervla
    actor.checkpoint = _PI05_CKPT
    actor.hl_provider = "internvl2"
    actor.hl_checkpoint = _SIMLINGO_HL_CKPT
    actor.simlingo_source_root = "/raid/users/surya/simlingo-steervla"
    actor.hl_python = sys.executable
    return config
