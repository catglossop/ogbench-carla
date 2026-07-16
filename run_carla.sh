#!/usr/bin/env bash
set -euo pipefail

# Single launch wrapper for the residual-RL CARLA stack.
# Writes temp agent + CARLA configs under .run_carla/ (so your base configs are
# never edited in place), pins the two GPUs (JAX learner vs. CARLA renderer),
# and invokes impls/main_carla.py (residual/EXPO path, selected by the agent config).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ROUTE="parking-cut-in-001"
ONLINE_STEPS="50000"
SEED="0"
RUN_GROUP="Debug"
SAVE_BUFFER="true"
WANDB_MODE="${WANDB_MODE:-online}"

TRAIN_GPU_RANK="2"
SIM_GPU_RANK="3"
RENDER_ADAPTER=""
CARLA_HOST="localhost"
CARLA_PORT="2020"
CARLA_STREAMING_PORT="0"
TM_PORT="8020"
X_DISPLAY_NUM=""

ENABLE_UPDATES="true"
BASE_ONLY=""
STATE_ENCODER=""
RLT_CHECKPOINT=""

BASE_AGENT_CFG="impls/configs/steervla_residual_config.py"
BASE_CARLA_CFG="impls/configs/carla_config.yaml"

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash run_carla.sh [options] [-- extra args passed to impls/main_carla.py]

Options:
  --route NAME              Bench2Drive route name/id. Default: parking-cut-in-001
  --online-steps N          Number of env steps. Default: 50000
  --seed N                  Random seed. Default: 0
  --run-group NAME          W&B / experiment group. Default: Debug
  --save-buffer BOOL        true|false. Default: true
  --wandb-mode MODE         online|offline|disabled. Default: WANDB_MODE or online

  --train-gpu N             JAX / learner GPU rank. Default: 2
  --sim-gpu N               Alias for --render-adapter. Default: 3
  --render-adapter N        CARLA -graphicsadapter value. Default: 3

  --carla-host HOST         CARLA host. Default: localhost
  --carla-port PORT         CARLA RPC port. Default: 2020
  --carla-streaming-port P  CARLA streaming port. Default: 0 (auto)
  --tm-port PORT            Traffic manager port. Default: 8020
  --x-display-num N         Xvfb display number. Default: derived from carla port

  --enable-updates BOOL     true|false. false = rollout/buffer only (no RL updates). Default: true
  --base-only BOOL          true|false. true = no-RL baseline: roll out the frozen base policy only. Default: false
  --state-encoder NAME      RL state encoder: pi_prefix|siglip_pool|rl_token. Default: config value (pi_prefix)
  --rlt-checkpoint PATH     RLT autoencoder checkpoint (only for --state-encoder rl_token).

  --agent-config PATH       Base agent config. Default: impls/configs/steervla_residual_config.py
  --carla-config PATH       Base CARLA yaml. Default: impls/configs/carla_config.yaml
  -h, --help                Show this help

Examples:
  bash run_carla.sh --train-gpu 0 --sim-gpu 4 --route parking-cut-in-001
  bash run_carla.sh --online-steps 5000 --wandb-mode offline
  bash run_carla.sh --enable-updates false --save-buffer true   # rollout-only data collection
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --route) ROUTE="$2"; shift 2 ;;
    --online-steps) ONLINE_STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run-group) RUN_GROUP="$2"; shift 2 ;;
    --save-buffer) SAVE_BUFFER="$2"; shift 2 ;;
    --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
    --train-gpu) TRAIN_GPU_RANK="$2"; shift 2 ;;
    --sim-gpu) SIM_GPU_RANK="$2"; shift 2 ;;
    --render-adapter) RENDER_ADAPTER="$2"; shift 2 ;;
    --carla-host) CARLA_HOST="$2"; shift 2 ;;
    --carla-port) CARLA_PORT="$2"; shift 2 ;;
    --carla-streaming-port) CARLA_STREAMING_PORT="$2"; shift 2 ;;
    --tm-port) TM_PORT="$2"; shift 2 ;;
    --x-display-num) X_DISPLAY_NUM="$2"; shift 2 ;;
    --enable-updates) ENABLE_UPDATES="$2"; shift 2 ;;
    --base-only) BASE_ONLY="$2"; shift 2 ;;
    --state-encoder) STATE_ENCODER="$2"; shift 2 ;;
    --rlt-checkpoint) RLT_CHECKPOINT="$2"; shift 2 ;;
    --agent-config) BASE_AGENT_CFG="$2"; shift 2 ;;
    --carla-config) BASE_CARLA_CFG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$ENABLE_UPDATES" in
  true|false) ;;
  *)
    echo "Invalid --enable-updates: $ENABLE_UPDATES (expected true|false)" >&2
    exit 2
    ;;
esac

case "$BASE_ONLY" in
  ""|true|false) ;;
  *)
    echo "Invalid --base-only: $BASE_ONLY (expected true|false)" >&2
    exit 2
    ;;
esac

TMP_ROOT="$ROOT_DIR/.run_carla"
mkdir -p "$TMP_ROOT"
TMP_DIR="$(mktemp -d "$TMP_ROOT/run.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

AGENT_CFG_TMP="$TMP_DIR/agent_config.py"
CARLA_CFG_TMP="$TMP_DIR/carla_config.yaml"

BASE_AGENT_CFG_ABS="$(cd "$(dirname "$BASE_AGENT_CFG")" && pwd)/$(basename "$BASE_AGENT_CFG")"
BASE_CARLA_CFG_ABS="$(cd "$(dirname "$BASE_CARLA_CFG")" && pwd)/$(basename "$BASE_CARLA_CFG")"

if [[ -z "$X_DISPLAY_NUM" ]]; then
  X_DISPLAY_NUM="$((10 + (CARLA_PORT % 90)))"
fi

if [[ -n "$RENDER_ADAPTER" ]]; then
  SIM_GPU_RANK="$RENDER_ADAPTER"
fi

STATE_ENCODER_LINE=""
if [[ -n "$STATE_ENCODER" ]]; then
  STATE_ENCODER_LINE="config.state_encoder = \"${STATE_ENCODER}\""
fi
RLT_CHECKPOINT_LINE=""
if [[ -n "$RLT_CHECKPOINT" ]]; then
  RLT_CHECKPOINT_LINE="config.rl_token.checkpoint_path = r\"${RLT_CHECKPOINT}\""
fi
BASE_ONLY_LINE=""
if [[ -n "$BASE_ONLY" ]]; then
  BASE_ONLY_LINE="config.base_only = ${BASE_ONLY^}"
fi

cat > "$AGENT_CFG_TMP" <<EOF
from pathlib import Path
import runpy

_BASE_PATH = Path(r"${BASE_AGENT_CFG_ABS}")
_BASE_GET_CONFIG = runpy.run_path(str(_BASE_PATH))["get_config"]


def get_config():
    config = _BASE_GET_CONFIG()
    config.training_gpu_rank = ${TRAIN_GPU_RANK}
    config.enable_updates = ${ENABLE_UPDATES^}
    ${STATE_ENCODER_LINE}
    ${RLT_CHECKPOINT_LINE}
    ${BASE_ONLY_LINE}
    return config
EOF

uv run python - <<EOF
from pathlib import Path
import yaml

base_path = Path(r"${BASE_CARLA_CFG_ABS}")
cfg = yaml.safe_load(base_path.read_text())
cfg["host"] = "${CARLA_HOST}"
cfg["port"] = int("${CARLA_PORT}")
cfg["streaming_port"] = int("${CARLA_STREAMING_PORT}")
cfg["traffic_manager_port"] = int("${TM_PORT}")
cfg["gpu_rank"] = int("${SIM_GPU_RANK}")
cfg["x_display_num"] = int("${X_DISPLAY_NUM}")
cfg["use_cuda_visible_devices"] = False
Path(r"${CARLA_CFG_TMP}").write_text(yaml.safe_dump(cfg, sort_keys=False))
EOF

echo "[run_carla.sh] route=${ROUTE}"
echo "[run_carla.sh] enable_updates=${ENABLE_UPDATES} base_only=${BASE_ONLY:-<config default>} state_encoder=${STATE_ENCODER:-<config default>}${RLT_CHECKPOINT:+ rlt_checkpoint=${RLT_CHECKPOINT}}"
echo "[run_carla.sh] train_gpu_rank=${TRAIN_GPU_RANK} render_adapter=${SIM_GPU_RANK}"
echo "[run_carla.sh] carla_host=${CARLA_HOST} carla_port=${CARLA_PORT} streaming_port=${CARLA_STREAMING_PORT} tm_port=${TM_PORT} x_display=:${X_DISPLAY_NUM}"
echo "[run_carla.sh] save_buffer=${SAVE_BUFFER} online_steps=${ONLINE_STEPS}"
echo "[run_carla.sh] temp agent config: ${AGENT_CFG_TMP}"
echo "[run_carla.sh] temp carla config: ${CARLA_CFG_TMP}"

# main_carla.py dispatches to the residual/EXPO path when the agent config's
# agent_name == "sac_residual" (steervla_residual_config.py). log_interval/save_interval are
# the residual cadence (main_carla's DSRL defaults are 1 / 100000); placed before EXTRA_ARGS so
# a caller can still override them via `-- --log_interval=...`.
WANDB_MODE="${WANDB_MODE}" uv run python impls/main_carla.py \
  --agent="${AGENT_CFG_TMP}" \
  --carla_config="${CARLA_CFG_TMP}" \
  --route="${ROUTE}" \
  --online_steps="${ONLINE_STEPS}" \
  --save_buffer="${SAVE_BUFFER}" \
  --seed="${SEED}" \
  --run_group="${RUN_GROUP}" \
  --log_interval=10 \
  --save_interval=5000 \
  "${EXTRA_ARGS[@]}"
