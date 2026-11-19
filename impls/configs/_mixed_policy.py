"""The "mixed" SteerVLA policy of the HARP generalization / pooled-HL evals, as one place to point a config at.

InternVL2 HL (SimLingo HL checkpoint, run in a torch worker by ``vlas/mixed_steervla.py``) plans text;
pi05 ``no_ego_history_v1 @ 6000`` reads the camera and emits the action chunk. Cadence and HL sampling
are those of ``.run_carla/qwen_zs_bon_run.sh ACTOR=mixed`` (``steervla_mixed_eval_config`` + overrides):
HL every 5 LL queries, LL every 3 env steps, HL sampled at ``cot_temperature`` 1.0, ``proprio_norm`` off.
"""

MIXED_LL_CHECKPOINT = (
    "/raid/users/cglossop/openpi/cat-logs/pi05_steervla_cot_simplified_reasoning_no_ego_history/"
    "pi05_steervla_simplified_reasoning_no_ego_history_v1/"
    "pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000"
)
MIXED_LL_ACTOR_CONFIG = "pi05_steervla_cot_simplified_reasoning_no_ego_history"
# Base (un-finetuned) SimLingo HL, the same file the SimLingo residual / YAY sweeps start from.
MIXED_HL_CHECKPOINT = (
    "/raid/users/celine/steervla-ckpts/2026_05_24_06_52_33_simlingo_seed1_bellman/checkpoints/epoch=019.ckpt"
)
SIMLINGO_SOURCE_ROOT = "/home/cglossop/simlingo-steervla"
# .venv python + the SimLingo deps on PYTHONPATH (MixedSteerVLAActor strips the parent's PYTHONPATH).
MIXED_HL_PYTHON = "/raid/users/cglossop/sweep_results/generalization_b2d_sources/hl_python.sh"


def apply_mixed_policy(s):
    """Point a pi05 ``config.steervla`` block at the mixed policy (in place)."""
    s.vla = "steervla"
    s.checkpoint = MIXED_LL_CHECKPOINT
    s.actor_config = MIXED_LL_ACTOR_CONFIG
    s.hl_provider = "internvl2"
    s.hl_checkpoint = MIXED_HL_CHECKPOINT
    s.simlingo_source_root = SIMLINGO_SOURCE_ROOT
    s.hl_python = MIXED_HL_PYTHON
    s.actions_per_cot = 5
    s.actions_per_model_query = 3
    s.proprio_norm = False
    s.cot_temperature = 1.0
    return s
