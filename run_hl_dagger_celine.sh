#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARLA_ROOT="${CARLA_ROOT:-/raid/users/celine/carla}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/raid/users/celine/openpi}"
export HF_HOME="${HF_HOME:-/raid/users/celine/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/raid/users/celine/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/raid/users/celine/cache/xdg}"
export OGBENCH_SAVE_DIR="${OGBENCH_SAVE_DIR:-/raid/users/celine/hl_dagger/runs}"

if [[ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]]; then
  echo "[run_hl_dagger_celine] CARLA is missing at $CARLA_ROOT" >&2
  exit 1
fi
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "[run_hl_dagger_celine] GEMINI_API_KEY must be exported for CAST relabeling." >&2
  exit 1
fi

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$OGBENCH_SAVE_DIR"

exec "$ROOT_DIR/run_carla.sh" \
  --agent-config "$ROOT_DIR/impls/configs/steervla_hl_dagger_celine_config.py" \
  --train-mode rl \
  --train-gpu 1 \
  --hl-gpu 1 \
  --render-adapter 7 \
  --hl-ckpt-dir /raid/users/celine/hl_dagger/checkpoints \
  --run-group HL-Dagger \
  "$@"
