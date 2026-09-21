#!/usr/bin/env bash
# Second-machine half of the f2d mixed-policy zero-shot Qwen BoN sweep
# (RUN_GROUP f2dsteervla_kl005_seed0_qwenzs_bon_mixed). Same protocol as launch_f2d_mixed.sh on
# bellman, walked in REVERSE: routes last-to-first, CARLA seeds 2 -> 1 -> 0. SKIP_WANDB_CLAIMED=1
# makes every worker skip a cell another host has running/finished/crashed on W&B, so the two
# machines meet in the middle without running a cell twice. See HANDOVER_qwen_zs_bon_second_machine.md.
#
#   SWEEP_GPUS="0 1" QWEN_GPUS="0 1" QWEN_PORTS="18850 18851" \
#     setsid nohup ./.run_carla/handover/launch_f2d_mixed_reverse.sh > sweep_b.log 2>&1 &
#   ./.run_carla/handover/launch_f2d_mixed_reverse.sh --status | --results | --stop | --dry-run
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
command -v uv >/dev/null || { echo "[launch-B] ABORT: uv not on PATH" >&2; exit 1; }

export BENCH=f2d
export SOURCE_SWEEP=f2dsteervla_simlingo_fixedcarla_kl005_seed0
export RUN_GROUP=f2dsteervla_kl005_seed0_qwenzs_bon_mixed
export ROUTES_FILE="$ROOT_DIR/.run_carla/handover/routes_f2d_reversed.txt"
export CARLA_SEEDS="2 1 0" N_EVAL=1 EVAL_SEED_OFFSET=0
export ACTOR=mixed N_CANDIDATES=4
export SKIP_WANDB_CLAIMED=1
export WANDB_KEY_FILE="${WANDB_KEY_FILE:-$HOME/.wandb_school_key}"
export WANDB_ENTITY=catherineglossop
export MIXED_HL_PYTHON="${MIXED_HL_PYTHON:-/raid/users/cglossop/sweep_results/qwenzs_mixed_b2d_sources/hl_python.sh}"
MODE="${1:---arm}"

if [ "$MODE" = --arm ]; then
  : "${SWEEP_GPUS:?set SWEEP_GPUS (one worker per listed GPU) -- never defaulted}"
  : "${QWEN_GPUS:?set QWEN_GPUS (critic GPU per worker, lined up with SWEEP_GPUS)}"
  : "${QWEN_PORTS:?set QWEN_PORTS (critic port per worker, lined up with SWEEP_GPUS)}"
  export SWEEP_GPUS QWEN_GPUS QWEN_PORTS
  # Log only as catherine_glossop, never the ~/.netrc (catglossop) fallthrough.
  who=$(WANDB_API_KEY="$(cat "$WANDB_KEY_FILE")" "$ROOT_DIR/.venv/bin/python" -c \
        'import wandb; print(wandb.Api().viewer.username)' 2>/dev/null | tail -1)
  [ "$who" = catherine_glossop ] || { echo "[launch-B] ABORT: W&B key resolves to '${who:-?}', not catherine_glossop" >&2; exit 1; }
  [ -d "/raid/users/cglossop/sweeps/${SOURCE_SWEEP}" ] \
    || { echo "[launch-B] ABORT: source checkpoints not mirrored (see f2d_assets.txt)" >&2; exit 1; }
fi
./.run_carla/qwen_zs_bon_sweep.sh "$MODE"
