"""Rollout-only hierarchical InternVL2 SteerVLA for Qwen Best-of-N evaluation."""

import os

from configs.simlingo_steervla_cast_relabel_train_config import apply_simlingo_steervla
from configs.steervla_dsrl_config import get_config as get_dsrl_config


def get_config():
    config = get_dsrl_config()
    config.enable_updates = False
    config.enable_updates_rl = False
    config.enable_updates_bc = False
    config.enable_updates_bc_hl = False
    config.warmup_steps = 0
    config.language_feedback.source = "expert"
    config.language_feedback.expert_mode = "none"

    actor = apply_simlingo_steervla(config.steervla)
    actor.simlingo_source_root = os.environ.get(
        "SIMLINGO_SOURCE_ROOT", "/raid/users/celine/simlingo-comp-inference")
    actor.hl_checkpoint = os.environ.get("SIMLINGO_HL_CHECKPOINT", actor.hl_checkpoint)
    actor.ll_checkpoint = os.environ.get("SIMLINGO_LL_CHECKPOINT", actor.ll_checkpoint)
    actor.checkpoint = actor.hl_checkpoint
    actor.cot_temperature = 1.0
    actor.actions_per_model_query = 3
    actor.actions_per_cot = 5
    actor.proprio_norm = False
    return config
