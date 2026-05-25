#!/usr/bin/env bash
# run_simlingo.sh — launcher for SimLingo residual SAC on OGBench-CARLA.
#
# This script drives impls/main_carla_simlingo.py under the `simlingo`
# conda env.  The CARLA server (carla_env_server.py) is launched
# automatically as a subprocess by main_carla_simlingo.py.
#
# Quick examples:
#
#   # Base-policy eval only (no SAC)
#   bash run_simlingo.sh --eval-only --route bench2drive_00
#
#   # SAC training
#   bash run_simlingo.sh --route parking-cut-in-001 --steps 50000
#
#   # Second instance on different ports (train GPU 1, CARLA adapter 1)
#   bash run_simlingo.sh --instance 1 \
#       --route signalized-junction-left-turn-001
#
#   # Offline W&B (no internet)
#   WANDB_MODE=offline bash run_simlingo.sh --route parking-cut-in-001

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
SIMLINGO_CKPT="/home/celinet/simlingo_checkpoints/simlingo/checkpoints/epoch=013.ckpt"
ROUTE="bench2drive_00"
STEPS="10000"
WARMUP="500"
LEARNING_STARTS="500"
CHUNK_SIZE="10"
RES_SCALE="0.1"
BATCH_SIZE="256"
BUFFER_CAP="10000"
UPDATES_PER_STEP="4"
ACTOR_LR="1e-4"
CRITIC_LR="1e-4"
SEED="0"
RUN_GROUP="Debug"
WANDB_MODE="${WANDB_MODE:-online}"
LOG_INTERVAL="1"
VIDEO_LOG_INTERVAL="1"
SAVE_INTERVAL="2000"
DEVICE="cuda"
CARLA_CFG="impls/configs/carla_config.yaml"
SIMLINGO_PYTHON="/home/celinet/miniconda3/envs/simlingo/bin/python"
EVAL_ONLY="false"
DEBUG_NEG_SPEED="false"
SAVE_VIDEO="true"
DRY_RUN="false"
TRAIN_GPU=""          # empty = preserve inherited CUDA_VISIBLE_DEVICES
CARLA_HOST=""         # empty = use yaml value
CARLA_PORT=""         # empty = use yaml value
CARLA_STREAMING_PORT="" # empty = use yaml value or 0 when generating a temp config
CARLA_TM_PORT=""      # empty = derive from --port (+6000) or use yaml value
X_DISPLAY_NUM=""      # empty = derive from --port when generating a temp config
GPU_RANK=""           # CARLA -graphicsadapter; empty = use yaml value
RENDER_ADAPTER=""     # alias for GPU_RANK
INSTANCE=""           # compact parallel-run index
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash run_simlingo.sh [options] [-- extra args passed to main_carla_simlingo.py]

Mode:
  --eval-only               Run base policy only, no SAC training
  --debug-neg-speed         Replace reward with -speed (m/s) — SAC should brake

Routing / environment:
  --route NAME              Bench2Drive route (scenario name, file basename, or route id)
                            Default: bench2drive_00
  --carla-config PATH       Path to carla_config.yaml
                            Default: impls/configs/carla_config.yaml

Multi-instance (ports):
  --instance N              Convenience preset for one concurrent run:
                              train GPU=N, CARLA adapter=N, port=2201+10*N,
                              traffic-manager port=8201+10*N
  --gpu N                   Alias for --train-gpu N --sim-gpu N
  --train-gpu N             CUDA_VISIBLE_DEVICES for SimLingo/PyTorch.
                            If omitted, preserves the current CUDA_VISIBLE_DEVICES.
  --sim-gpu N               CARLA rendering adapter (-graphicsadapter).
  --render-adapter N        Alias for --sim-gpu N.
  --gpu-rank N              Backward-compatible alias for --sim-gpu N.
  --carla-host HOST         CARLA host override. Default: yaml value.
  --port N                  CARLA server port. Generates a temp config overriding the yaml.
  --carla-port N            Alias for --port.
                            Derive traffic-manager port automatically as N+6000 unless --tm-port set.
                            Also scopes simulation_results.json and save_dir to this port.
  --tm-port N               Traffic manager port (only used when --port is also set).
  --carla-streaming-port N  CARLA streaming port. Default: yaml value or 0 for temp configs.
  --x-display-num N         Xvfb display number. Default: derived from --port.

Model:
  --checkpoint PATH         SimLingo checkpoint directory
                            Default: /home/celinet/simlingo_checkpoints/simlingo/checkpoints/epoch=013.ckpt
  --device DEVICE           Torch device. Default: cuda
  --chunk-size N            Waypoints per VLM call (1–10). Default: 10

SAC hyperparameters:
  --steps N                 Total env steps. Default: 10000
  --warmup N                Warmup steps (random/zero residual). Default: 500
  --learning-starts N       Buffer threshold before updates. Default: 500
  --res-scale F             Residual action scale. Default: 0.1
  --batch-size N            SAC mini-batch size. Default: 256
  --buffer-cap N            Replay buffer capacity. Default: 10000
  --updates-per-step N      SAC updates per env step. Default: 4
  --actor-lr F              Actor learning rate. Default: 1e-4
  --critic-lr F             Critic learning rate. Default: 1e-4
  --gamma F                 Discount factor. Default: 0.97
  --tau F                   Target network tau. Default: 0.01

Logging:
  --run-group NAME          W&B run group. Default: Debug
  --wandb-mode MODE         online|offline|disabled. Default: $WANDB_MODE or online
  --log-interval N          Log training metrics every N steps. Default: 1
  --video-log-interval N    Upload video every N episodes (0=never). Default: 5
  --save-interval N         Save SAC checkpoint every N steps. Default: 2000
  --no-video                Disable local mp4 saving
  --dry-run                 Print resolved config/command without launching

  -h, --help                Show this help

Examples:
  # Eval only — check base policy on route 28035
  bash run_simlingo.sh --eval-only

  # SAC training with wandb
  bash run_simlingo.sh --route parking-cut-in-001 --steps 50000 --run-group SAC-v1

  # Two parallel instances on the same machine (different ports, different GPUs)
  bash run_simlingo.sh --instance 0 --route bench2drive_00 &
  bash run_simlingo.sh --instance 1 --route parking-cut-in-001 &

  # Explicit form, equivalent to a custom run_carla.sh-style launch
  bash run_simlingo.sh --train-gpu 1 --sim-gpu 1 --port 2211 --route bench2drive_00

  # Offline W&B
  WANDB_MODE=offline bash run_simlingo.sh --route parking-cut-in-001 --steps 20000
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-only)           EVAL_ONLY="true"; shift ;;
    --debug-neg-speed)     DEBUG_NEG_SPEED="true"; shift ;;
    --route)               ROUTE="$2"; shift 2 ;;
    --carla-config)        CARLA_CFG="$2"; shift 2 ;;
    --checkpoint)          SIMLINGO_CKPT="$2"; shift 2 ;;
    --device)              DEVICE="$2"; shift 2 ;;
    --chunk-size)          CHUNK_SIZE="$2"; shift 2 ;;
    --steps)               STEPS="$2"; shift 2 ;;
    --warmup)              WARMUP="$2"; shift 2 ;;
    --learning-starts)     LEARNING_STARTS="$2"; shift 2 ;;
    --res-scale)           RES_SCALE="$2"; shift 2 ;;
    --batch-size)          BATCH_SIZE="$2"; shift 2 ;;
    --buffer-cap)          BUFFER_CAP="$2"; shift 2 ;;
    --updates-per-step)    UPDATES_PER_STEP="$2"; shift 2 ;;
    --actor-lr)            ACTOR_LR="$2"; shift 2 ;;
    --critic-lr)           CRITIC_LR="$2"; shift 2 ;;
    --gamma)               GAMMA="${2}"; shift 2 ;;
    --tau)                 TAU="${2}"; shift 2 ;;
    --run-group)           RUN_GROUP="$2"; shift 2 ;;
    --wandb-mode)          WANDB_MODE="$2"; shift 2 ;;
    --log-interval)        LOG_INTERVAL="$2"; shift 2 ;;
    --video-log-interval)  VIDEO_LOG_INTERVAL="$2"; shift 2 ;;
    --save-interval)       SAVE_INTERVAL="$2"; shift 2 ;;
    --seed)                SEED="$2"; shift 2 ;;
    --no-video)            SAVE_VIDEO="false"; shift ;;
    --dry-run)             DRY_RUN="true"; shift ;;
    --instance)            INSTANCE="$2"; shift 2 ;;
    --gpu)                 TRAIN_GPU="$2"; GPU_RANK="$2"; shift 2 ;;
    --train-gpu|--cuda-gpu) TRAIN_GPU="$2"; shift 2 ;;
    --sim-gpu|--render-adapter) GPU_RANK="$2"; shift 2 ;;
    --carla-host)          CARLA_HOST="$2"; shift 2 ;;
    --port)                CARLA_PORT="$2"; shift 2 ;;
    --carla-port)          CARLA_PORT="$2"; shift 2 ;;
    --tm-port)             CARLA_TM_PORT="$2"; shift 2 ;;
    --carla-streaming-port) CARLA_STREAMING_PORT="$2"; shift 2 ;;
    --x-display-num)       X_DISPLAY_NUM="$2"; shift 2 ;;
    --gpu-rank)            GPU_RANK="$2"; shift 2 ;;
    -h|--help)             usage; exit 0 ;;
    --)                    shift; EXTRA_ARGS+=("$@"); break ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

# ── Environment check ─────────────────────────────────────────────────────────
# Install any packages missing from the simlingo env before launching.
REQUIRED_PKGS=(ml_collections)
for pkg in "${REQUIRED_PKGS[@]}"; do
  import_name="${pkg//-/_}"
  if ! "$SIMLINGO_PYTHON" -c "import ${import_name}" 2>/dev/null; then
    echo "[run_simlingo.sh] Installing missing package into simlingo env: $pkg"
    "$SIMLINGO_PYTHON" -m pip install "$pkg" -q
  fi
done

# ── Runtime overrides: generate a scoped temp config ───────────────────────────
if [[ -n "$INSTANCE" ]]; then
  if [[ -z "$TRAIN_GPU" ]]; then
    TRAIN_GPU="$INSTANCE"
  fi
  if [[ -z "$GPU_RANK" ]]; then
    GPU_RANK="$INSTANCE"
  fi
  if [[ -z "$CARLA_PORT" ]]; then
    CARLA_PORT=$(( 2201 + 10 * INSTANCE ))
  fi
  if [[ -z "$CARLA_TM_PORT" ]]; then
    CARLA_TM_PORT=$(( 8201 + 10 * INSTANCE ))
  fi
fi

if [[ -n "$RENDER_ADAPTER" ]]; then
  GPU_RANK="$RENDER_ADAPTER"
fi

TMP_CFG=""
NEEDS_TMP_CFG="false"
for _v in "$CARLA_HOST" "$CARLA_PORT" "$CARLA_STREAMING_PORT" "$CARLA_TM_PORT" "$X_DISPLAY_NUM" "$GPU_RANK"; do
  if [[ -n "$_v" ]]; then
    NEEDS_TMP_CFG="true"
  fi
done

if [[ "$NEEDS_TMP_CFG" == "true" ]]; then
  if [[ -z "$CARLA_TM_PORT" ]]; then
    if [[ -n "$CARLA_PORT" ]]; then
      CARLA_TM_PORT=$(( CARLA_PORT + 6000 ))
    fi
  fi
  if [[ -z "$CARLA_STREAMING_PORT" && -n "$CARLA_PORT" ]]; then
    CARLA_STREAMING_PORT="0"
  fi
  if [[ -z "$X_DISPLAY_NUM" && -n "$CARLA_PORT" ]]; then
    X_DISPLAY_NUM=$(( 10 + (CARLA_PORT % 90) ))
  fi

  TMP_ROOT="$ROOT_DIR/.run_simlingo"
  mkdir -p "$TMP_ROOT"
  TMP_DIR="$(mktemp -d "$TMP_ROOT/run.XXXXXX")"
  TMP_CFG="$TMP_DIR/carla_config.yaml"
  trap 'rm -rf "$TMP_DIR"' EXIT

  "$SIMLINGO_PYTHON" - "$CARLA_CFG" "$TMP_CFG" \
    "$CARLA_HOST" "$CARLA_PORT" "$CARLA_STREAMING_PORT" "$CARLA_TM_PORT" "$X_DISPLAY_NUM" "$GPU_RANK" <<'PY'
import sys
from pathlib import Path

import yaml

base_path, out_path, host, port, streaming_port, tm_port, x_display_num, gpu_rank = sys.argv[1:9]
cfg = yaml.safe_load(Path(base_path).read_text())

if host:
    cfg["host"] = host
if port:
    cfg["port"] = int(port)
    cfg["checkpoint"] = f"./simulation_results_port{port}.json"
    cfg["debug_checkpoint"] = f"./live_results_port{port}.txt"
if streaming_port:
    cfg["streaming_port"] = int(streaming_port)
if tm_port:
    cfg["traffic_manager_port"] = int(tm_port)
if x_display_num:
    cfg["x_display_num"] = int(x_display_num)
if gpu_rank:
    cfg["gpu_rank"] = int(gpu_rank)

Path(out_path).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

  CARLA_CFG="$TMP_CFG"
fi

# If no explicit CARLA adapter was given, preserve the yaml value.  This avoids
# main_carla_simlingo.py's default --gpu_rank=0 overriding carla_config_gpu1.yaml.
if [[ -z "$GPU_RANK" ]]; then
  GPU_RANK="$("$SIMLINGO_PYTHON" - "$CARLA_CFG" <<'PY'
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(int(cfg.get("gpu_rank", 0)))
PY
)"
fi

# ── Build the argument list ───────────────────────────────────────────────────
ARGS=(
  --simlingo_checkpoint="$SIMLINGO_CKPT"
  --route="$ROUTE"
  --carla_config="$CARLA_CFG"
  --device="$DEVICE"
  --chunk_size="$CHUNK_SIZE"
  --res_scale="$RES_SCALE"
  --seed="$SEED"
  --run_group="$RUN_GROUP"
  --wandb_mode="$WANDB_MODE"
  --log_interval="$LOG_INTERVAL"
  --video_log_interval="$VIDEO_LOG_INTERVAL"
  --save_interval="$SAVE_INTERVAL"
  --save_video="$SAVE_VIDEO"
)

ARGS+=(--gpu_rank="$GPU_RANK")

# Scope save_dir per port so parallel runs don't share checkpoints/logs.
if [[ -n "$CARLA_PORT" ]]; then
  ARGS+=(--save_dir="./logs/simlingo_residual_port${CARLA_PORT}")
fi

if [[ "$EVAL_ONLY" == "true" ]]; then
  ARGS+=(--eval_only)
else
  ARGS+=(
    --total_steps="$STEPS"
    --warmup_steps="$WARMUP"
    --learning_starts="$LEARNING_STARTS"
    --batch_size="$BATCH_SIZE"
    --buffer_capacity="$BUFFER_CAP"
    --updates_per_step="$UPDATES_PER_STEP"
    --actor_lr="$ACTOR_LR"
    --critic_lr="$CRITIC_LR"
  )
  [[ -n "${GAMMA:-}" ]] && ARGS+=(--gamma="$GAMMA")
  [[ -n "${TAU:-}" ]]   && ARGS+=(--tau="$TAU")
fi

if [[ "$DEBUG_NEG_SPEED" == "true" ]]; then
  ARGS+=(--debug_neg_speed_reward)
fi

ARGS+=("${EXTRA_ARGS[@]}")

echo "[run_simlingo.sh] route=$ROUTE  eval_only=$EVAL_ONLY  debug_neg_speed=$DEBUG_NEG_SPEED"
echo "[run_simlingo.sh] steps=$STEPS  warmup=$WARMUP  chunk_size=$CHUNK_SIZE  res_scale=$RES_SCALE"
echo "[run_simlingo.sh] wandb_mode=$WANDB_MODE  run_group=$RUN_GROUP"
echo "[run_simlingo.sh] checkpoint=$SIMLINGO_CKPT"
echo "[run_simlingo.sh] carla_config=$CARLA_CFG"
if [[ -n "$TRAIN_GPU" ]]; then
  echo "[run_simlingo.sh] train_gpu=$TRAIN_GPU  carla_gpu=$GPU_RANK"
else
  echo "[run_simlingo.sh] train_gpu=${CUDA_VISIBLE_DEVICES:-<inherited/all>}  carla_gpu=$GPU_RANK"
fi
if [[ -n "$CARLA_PORT" ]]; then
  echo "[run_simlingo.sh] carla_host=${CARLA_HOST:-<yaml>}  carla_port=$CARLA_PORT  tm_port=$CARLA_TM_PORT  streaming_port=$CARLA_STREAMING_PORT  x_display=:${X_DISPLAY_NUM}"
fi
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
  printf '[run_simlingo.sh] command: '
  if [[ -n "$TRAIN_GPU" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$TRAIN_GPU"
  fi
  printf 'WANDB_MODE=%q %q impls/main_carla_simlingo.py' "$WANDB_MODE" "$SIMLINGO_PYTHON"
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

if [[ -n "$TRAIN_GPU" ]]; then
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" WANDB_MODE="$WANDB_MODE" "$SIMLINGO_PYTHON" impls/main_carla_simlingo.py "${ARGS[@]}"
else
  WANDB_MODE="$WANDB_MODE" "$SIMLINGO_PYTHON" impls/main_carla_simlingo.py "${ARGS[@]}"
fi
