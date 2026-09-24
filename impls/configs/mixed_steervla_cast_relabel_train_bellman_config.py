"""Bellman (/raid) paths for the mixed CAST-relabel sweep.

Identical recipe to ``mixed_steervla_cast_relabel_train_config.py`` -- the HL optimizer, the replay
mixture, the update cadence and every other knob come from it unchanged -- with only the four
machine-specific paths repointed at bellman's ``/raid``. Results from a run here and a run on the
other machine merge into one table, because nothing that affects training differs.

    ./run_carla.sh --agent-config impls/configs/mixed_steervla_cast_relabel_train_bellman_config.py \\
      --train-gpu <policy gpu> --hl-gpu <hl gpu> --render-adapter <policy gpu> ...

Each path is still overridable by the environment variable in brackets, so a third machine needs a
launcher, not another config.
"""

import os

from configs.mixed_steervla_cast_relabel_train_config import get_config as get_dgx6_config

# The original SimLingo HL (seed1 bellman, epoch 19) -- the same weights, under bellman's own copy.
# [MIXED_HL_CHECKPOINT]
HL_CHECKPOINT = os.environ.get(
    "MIXED_HL_CHECKPOINT",
    "/raid/users/celine/steervla-ckpts/2026_05_24_06_52_33_simlingo_seed1_bellman/checkpoints/epoch=019.ckpt",
)
# [SIMLINGO_SOURCE_ROOT] Source only; simlingo_training is imported from it. The f2d SimLingo sweep
# ran against this path on bellman -- dev's committed default points at another user's checkout, so
# check it before launching rather than trusting either value.
SIMLINGO_SOURCE_ROOT = os.environ.get("SIMLINGO_SOURCE_ROOT", "/home/cglossop/simlingo-steervla")
# [MIXED_HL_PYTHON] The InternVL2 worker needs torch plus the simlingo extra (peft, hydra,
# lightning, timm). If bellman's .venv was synced without that extra, point this at a wrapper that
# puts /raid/users/cglossop/ogbench-simlingo-deps on PYTHONPATH -- see hl_python.sh.template.
HL_PYTHON = os.environ.get("MIXED_HL_PYTHON", "/home/cglossop/ogbench-carla/.venv/bin/python")
# [HL_REPLAY_ROOT] / [HL_REPLAY_POOL] Bellman's pool references frames in the SimLingo database by
# absolute path, so it carries no "_img" copy of them.
HL_REPLAY_ROOT = os.environ.get("HL_REPLAY_ROOT", "/raid/users/cglossop/simlingo_hl_pools")
HL_REPLAY_POOL = os.environ.get("HL_REPLAY_POOL", "simlingo_hl_simplified")


def get_config():
    config = get_dgx6_config()
    s = config.steervla
    s.hl_checkpoint = HL_CHECKPOINT
    s.simlingo_source_root = SIMLINGO_SOURCE_ROOT
    s.hl_python = HL_PYTHON
    s.hl_replay_root = HL_REPLAY_ROOT
    # Weight unchanged (0.5); only the pool's directory name differs between the machines.
    s.hl_replay_pools = [dict(name=HL_REPLAY_POOL, weight=0.5)]
    return config
