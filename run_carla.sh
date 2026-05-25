#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ROUTE=""
ONLINE_STEPS="50000"
SEED="0"
RUN_GROUP="Debug"
SAVE_BUFFER="true"
EXPERT_DEBUG="false"
EXPERT_RECOVER_DEBUG="false"
WANDB_MODE="${WANDB_MODE:-online}"

TRAIN_GPU_RANK="0"
SIM_GPU_RANK="0"
RENDER_ADAPTER=""
CARLA_HOST="localhost"
CARLA_PORT="2020"
CARLA_STREAMING_PORT=""
TM_PORT=""
X_DISPLAY_NUM=""

CRITIC_MODE="none"
TRAIN_MODE="rl"
AGENT_MODE="ogpo"

BASE_AGENT_CFG=""
BASE_CARLA_CFG="impls/configs/carla_config.yaml"

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash run_carla.sh [options] [-- extra args passed to impls/main_carla.py]

Options:
  --route NAME              Bench2Drive route name/id. Default: parking-cut-in-001
  --online-steps N          Number of env steps. Default: 5000
  --seed N                  Random seed. Default: 0
  --run-group NAME          W&B / experiment group. Default: Debug
  --save-buffer BOOL        true|false. Default: true
  --expert-debug BOOL       true|false. Default: false
  --expert-recover-debug BOOL  true|false. Default: false
  --wandb-mode MODE         online|offline|disabled. Default: WANDB_MODE or offline

  --train-gpu N             JAX / learner GPU rank. Default: 2
  --sim-gpu N               Alias for --render-adapter. Default: 3
  --render-adapter N        CARLA -graphicsadapter value. Default: 3

  --carla-host HOST         CARLA host. Default: localhost
  --carla-port PORT         CARLA RPC port. Default: 2020
  --carla-streaming-port P  CARLA streaming port. Default: CARLA_PORT+1
  --tm-port PORT            Traffic manager port. Default: CARLA_PORT+6000
  --x-display-num N         Xvfb display number. Default: derived from CARLA_PORT

  --critic-mode MODE        one of:
                              none         -> no extra critic info
                              delta        -> numeric action delta
                              delta-lang   -> language on expert-agent delta
                              expert-lang  -> language on expert action
                            Default: delta

  --train-mode MODE         rl|dagger. Default: rl

  --agent-mode MODE         dsrl|ogpo. Default: dsrl
                              dsrl  -> steervla_dsrl_config.py (SteerVLA + DSRL)
                              ogpo  -> ogpo_carla_config.py    (standalone OGPO flow)
                            Overridden by --agent-config if both are set.

  --agent-config PATH       Base agent config (overrides --agent-mode).
                            Default: derived from --agent-mode
  --carla-config PATH       Base CARLA yaml. Default: impls/configs/carla_config.yaml
  -h, --help                Show this help

Examples:
  bash run_carla.sh --critic-mode delta-lang --train-gpu 0 --sim-gpu 4
  bash run_carla.sh --route parking-cut-in-001 --carla-port 2002
  bash run_carla.sh --critic-mode none --expert-debug true --save-buffer false
  bash run_carla.sh --train-mode dagger --critic-mode delta-lang
  bash run_carla.sh --agent-mode ogpo --carla-port 2020 --train-gpu 0 --sim-gpu 4
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --route) ROUTE="$2"; shift 2 ;;
    --online-steps) ONLINE_STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run-group) RUN_GROUP="$2"; shift 2 ;;
    --save-buffer) SAVE_BUFFER="$2"; shift 2 ;;
    --expert-debug) EXPERT_DEBUG="$2"; shift 2 ;;
    --expert-recover-debug) EXPERT_RECOVER_DEBUG="$2"; shift 2 ;;
    --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
    --train-gpu) TRAIN_GPU_RANK="$2"; shift 2 ;;
    --sim-gpu) SIM_GPU_RANK="$2"; shift 2 ;;
    --render-adapter) RENDER_ADAPTER="$2"; shift 2 ;;
    --carla-host) CARLA_HOST="$2"; shift 2 ;;
    --carla-port) CARLA_PORT="$2"; shift 2 ;;
    --carla-streaming-port) CARLA_STREAMING_PORT="$2"; shift 2 ;;
    --tm-port) TM_PORT="$2"; shift 2 ;;
    --x-display-num) X_DISPLAY_NUM="$2"; shift 2 ;;
    --critic-mode) CRITIC_MODE="$2"; shift 2 ;;
    --train-mode) TRAIN_MODE="$2"; shift 2 ;;
    --agent-mode) AGENT_MODE="$2"; shift 2 ;;
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

case "$CRITIC_MODE" in
  none) CRITIC_FEEDBACK_MODE="none" ;;
  delta) CRITIC_FEEDBACK_MODE="action_delta" ;;
  delta-lang) CRITIC_FEEDBACK_MODE="delta_commentary_bow" ;;
  expert-lang) CRITIC_FEEDBACK_MODE="commentary_bow" ;;
  *)
    echo "Invalid --critic-mode: $CRITIC_MODE" >&2
    echo "Expected one of: none, delta, delta-lang, expert-lang" >&2
    exit 2
    ;;
esac

case "$TRAIN_MODE" in
  rl|dagger) ;;
  *)
    echo "Invalid --train-mode: $TRAIN_MODE" >&2
    echo "Expected one of: rl, dagger" >&2
    exit 2
    ;;
esac

# Resolve base agent config from --agent-mode if --agent-config was not given.
if [[ -z "$BASE_AGENT_CFG" ]]; then
  case "$AGENT_MODE" in
    dsrl) BASE_AGENT_CFG="impls/configs/steervla_dsrl_config.py" ;;
    ogpo) BASE_AGENT_CFG="impls/configs/ogpo_carla_config.py" ;;
    *)
      echo "Invalid --agent-mode: $AGENT_MODE" >&2
      echo "Expected one of: dsrl, ogpo" >&2
      exit 2
      ;;
  esac
fi

TMP_ROOT="$ROOT_DIR/.run_carla"
mkdir -p "$TMP_ROOT"
TMP_DIR="$(mktemp -d "$TMP_ROOT/run.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

AGENT_CFG_TMP="$TMP_DIR/agent_config.py"
CARLA_CFG_TMP="$TMP_DIR/carla_config.yaml"

BASE_AGENT_CFG_ABS="$(cd "$(dirname "$BASE_AGENT_CFG")" && pwd)/$(basename "$BASE_AGENT_CFG")"
BASE_CARLA_CFG_ABS="$(cd "$(dirname "$BASE_CARLA_CFG")" && pwd)/$(basename "$BASE_CARLA_CFG")"

CARLA_STREAMING_PORT="${CARLA_STREAMING_PORT:-$((CARLA_PORT + 1))}"
TM_PORT="${TM_PORT:-$((CARLA_PORT + 6000))}"
X_DISPLAY_NUM="${X_DISPLAY_NUM:-$((10 + (CARLA_PORT % 90)))}"

if [[ -n "$RENDER_ADAPTER" ]]; then
  SIM_GPU_RANK="$RENDER_ADAPTER"
fi

if [[ "$AGENT_MODE" == "ogpo" ]]; then
cat > "$AGENT_CFG_TMP" <<EOF
from pathlib import Path
import runpy

_BASE_PATH = Path(r"${BASE_AGENT_CFG_ABS}")
_BASE_GET_CONFIG = runpy.run_path(str(_BASE_PATH))["get_config"]


def get_config():
    config = _BASE_GET_CONFIG()
    config.training_gpu_rank = ${TRAIN_GPU_RANK}
    config.siglip_device = "cuda:${TRAIN_GPU_RANK}"
    config.critic_feedback_mode = "${CRITIC_FEEDBACK_MODE}"
    # grpo_num_samples, bc_coeff, etc. come from ogpo_carla_config.py base.
    # online_training_mode is DSRL-specific; OGPO has no dagger mode.
    if config.critic_feedback_mode == "none":
        config.language_label_dim = 0
    return config
EOF
else
cat > "$AGENT_CFG_TMP" <<EOF
from pathlib import Path
import runpy

_BASE_PATH = Path(r"${BASE_AGENT_CFG_ABS}")
_BASE_GET_CONFIG = runpy.run_path(str(_BASE_PATH))["get_config"]


def get_config():
    config = _BASE_GET_CONFIG()
    config.training_gpu_rank = ${TRAIN_GPU_RANK}
    config.siglip_device = "cuda:${TRAIN_GPU_RANK}"
    config.critic_feedback_mode = "${CRITIC_FEEDBACK_MODE}"
    config.online_training_mode = "${TRAIN_MODE}"
    if config.critic_feedback_mode == "none":
        config.language_label_dim = 0
    return config
EOF
fi

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
echo "[run_carla.sh] agent_mode=${AGENT_MODE} train_mode=${TRAIN_MODE}"
echo "[run_carla.sh] critic_mode=${CRITIC_FEEDBACK_MODE}"
echo "[run_carla.sh] train_gpu_rank=${TRAIN_GPU_RANK} render_adapter=${SIM_GPU_RANK}"
echo "[run_carla.sh] carla_host=${CARLA_HOST} carla_port=${CARLA_PORT} streaming_port=${CARLA_STREAMING_PORT} tm_port=${TM_PORT} x_display=:${X_DISPLAY_NUM}"
echo "[run_carla.sh] expert_debug=${EXPERT_DEBUG} expert_recover_debug=${EXPERT_RECOVER_DEBUG} save_buffer=${SAVE_BUFFER} online_steps=${ONLINE_STEPS}"
echo "[run_carla.sh] temp agent config: ${AGENT_CFG_TMP}"
echo "[run_carla.sh] temp carla config: ${CARLA_CFG_TMP}"

WANDB_MODE="${WANDB_MODE}" OPENPI_DATA_HOME="/raid/users/celine/.cache/openpi" uv run python impls/main_carla.py \
  --agent="${AGENT_CFG_TMP}" \
  --carla_config="${CARLA_CFG_TMP}" \
  --route="${ROUTE}" \
  --online_steps="${ONLINE_STEPS}" \
  --save_buffer="${SAVE_BUFFER}" \
  --seed="${SEED}" \
  --run_group="${RUN_GROUP}" \
  --expert_debug="${EXPERT_DEBUG}" \
  --expert_recover_debug="${EXPERT_RECOVER_DEBUG}" \
  "${EXTRA_ARGS[@]}"
