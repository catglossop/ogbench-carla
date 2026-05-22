#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ROUTE="parking-cut-in-001"
ONLINE_STEPS="50000"
SEED="0"
RUN_GROUP="Debug"
SAVE_BUFFER="true"
BUFFER_CAPACITY=""
EXPERT_DEBUG="false"
EXPERT_RECOVER_DEBUG="false"
WANDB_MODE="${WANDB_MODE:-online}"

TRAIN_GPU_RANK="2"
SIM_GPU_RANK="3"
RENDER_ADAPTER=""
CARLA_HOST="localhost"
CARLA_PORT="2020"
CARLA_STREAMING_PORT="0"
TM_PORT="8020"
X_DISPLAY_NUM=""

CRITIC_MODE="none"
TRAIN_MODE="rl"
DAGGER_RESIDUAL_TRAIN_OBS_ENCODER="false"
CRITIC_USE_PI_PREFIX_FEATURES=""
REWARD_MODE="event"

BASE_AGENT_CFG="impls/configs/steervla_dsrl_config.py"
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
  --buffer-capacity N       Replay buffer size. Default: value in agent config (1000)
  --expert-debug BOOL       true|false. Default: false
  --expert-recover-debug BOOL  true|false. Default: false
  --wandb-mode MODE         online|offline|disabled. Default: WANDB_MODE or online

  --train-gpu N             JAX / learner GPU rank. Default: 2
  --sim-gpu N               Alias for --render-adapter. Default: 3
  --render-adapter N        CARLA -graphicsadapter value. Default: 3

  --carla-host HOST         CARLA host. Default: localhost
  --carla-port PORT         CARLA RPC port. Default: 2020
  --carla-streaming-port P  CARLA streaming port. Default: auto (0)
  --tm-port PORT            Traffic manager port. Default: 8020
  --x-display-num N         Xvfb display number. Default: derived from carla port

  --critic-mode MODE        one of:
                              none          -> no extra critic info
                              delta         -> numeric action delta
                              delta-lang    -> language on expert-agent delta
                              expert-lang   -> language on expert action
                              expert-action -> expert action itself (no delta)
                            Default: delta

  --train-mode MODE         rl|dagger|dagger_direct|sac_direct|sac_residual|dagger_residual. Default: rl
  --train-obs-encoder BOOL  true|false. Sets agent.dagger_residual_train_obs_encoder. Default: false
  --critic-pi-prefix BOOL   true|false. Overrides agent.critic_use_pi_prefix_features.
                            Default: config default

  --reward-mode MODE        one of:
                              event         -> existing shaped/event reward
                              soft-penalty  -> ca5be36 soft-penalty reward (progress * factors);
                                              speed-limit penalty is automatically enabled
                            Default: event

  --agent-config PATH       Base agent config. Default: impls/configs/steervla_dsrl_config.py
  --carla-config PATH       Base CARLA yaml. Default: impls/configs/carla_config.yaml
  -h, --help                Show this help

Examples:
  bash run_carla.sh --critic-mode delta-lang --train-gpu 0 --sim-gpu 4
  bash run_carla.sh --route parking-cut-in-001 --carla-port 2002 --carla-streaming-port 2003 --tm-port 8002 --x-display-num 12
  bash run_carla.sh --critic-mode none --expert-debug true --save-buffer false
  bash run_carla.sh --train-mode dagger --critic-mode delta-lang
  bash run_carla.sh --train-mode dagger_direct --critic-mode delta-lang
  bash run_carla.sh --train-mode sac_direct --critic-mode delta
  bash run_carla.sh --train-mode sac_residual --critic-mode delta
  bash run_carla.sh --train-mode dagger_residual --critic-mode delta
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --route) ROUTE="$2"; shift 2 ;;
    --online-steps) ONLINE_STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run-group) RUN_GROUP="$2"; shift 2 ;;
    --save-buffer) SAVE_BUFFER="$2"; shift 2 ;;
    --buffer-capacity) BUFFER_CAPACITY="$2"; shift 2 ;;
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
    --train-obs-encoder) DAGGER_RESIDUAL_TRAIN_OBS_ENCODER="$2"; shift 2 ;;
    --critic-pi-prefix) CRITIC_USE_PI_PREFIX_FEATURES="$2"; shift 2 ;;
    --reward-mode) REWARD_MODE="$2"; shift 2 ;;
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
  expert-action) CRITIC_FEEDBACK_MODE="expert_action" ;;
  *)
    echo "Invalid --critic-mode: $CRITIC_MODE" >&2
    echo "Expected one of: none, delta, delta-lang, expert-lang, expert-action" >&2
    exit 2
    ;;
esac

case "$TRAIN_MODE" in
  rl|dagger|dagger_direct|sac_direct|sac_residual|dagger_residual) ;;
  *)
    echo "Invalid --train-mode: $TRAIN_MODE" >&2
    echo "Expected one of: rl, dagger, dagger_direct, sac_direct, sac_residual, dagger_residual" >&2
    exit 2
    ;;
esac

case "$DAGGER_RESIDUAL_TRAIN_OBS_ENCODER" in
  true|false) ;;
  *)
    echo "Invalid --train-obs-encoder: $DAGGER_RESIDUAL_TRAIN_OBS_ENCODER" >&2
    echo "Expected one of: true, false" >&2
    exit 2
    ;;
esac

if [[ -n "$CRITIC_USE_PI_PREFIX_FEATURES" ]]; then
  case "$CRITIC_USE_PI_PREFIX_FEATURES" in
    true|false) ;;
    *)
      echo "Invalid --critic-pi-prefix: $CRITIC_USE_PI_PREFIX_FEATURES" >&2
      echo "Expected one of: true, false" >&2
      exit 2
      ;;
  esac
fi

case "$REWARD_MODE" in
  event|soft-penalty) ;;
  *)
    echo "Invalid --reward-mode: $REWARD_MODE" >&2
    echo "Expected one of: event, soft-penalty" >&2
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

WANDB_RUN_NAME="${TRAIN_MODE}_${CRITIC_MODE}_$(date +%Y%m%d_%H%M%S)"

cat > "$AGENT_CFG_TMP" <<EOF
from pathlib import Path
import runpy

_BASE_PATH = Path(r"${BASE_AGENT_CFG_ABS}")
_BASE_GET_CONFIG = runpy.run_path(str(_BASE_PATH))["get_config"]


def get_config():
    config = _BASE_GET_CONFIG()
    config.training_gpu_rank = ${TRAIN_GPU_RANK}
    config.critic_feedback_mode = "${CRITIC_FEEDBACK_MODE}"
    config.online_training_mode = "${TRAIN_MODE}"
    config.dagger_residual_train_obs_encoder = ${DAGGER_RESIDUAL_TRAIN_OBS_ENCODER^}
$(if [[ -n "$CRITIC_USE_PI_PREFIX_FEATURES" ]]; then echo "    config.critic_use_pi_prefix_features = ${CRITIC_USE_PI_PREFIX_FEATURES^}"; fi)
$(if [[ -n "$BUFFER_CAPACITY" ]]; then echo "    config.buffer_capacity = ${BUFFER_CAPACITY}"; fi)
    if config.critic_feedback_mode == "none":
        config.language_label_dim = 0
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
cfg["use_soft_penalty_reward"] = "${REWARD_MODE}" == "soft-penalty"
Path(r"${CARLA_CFG_TMP}").write_text(yaml.safe_dump(cfg, sort_keys=False))
EOF

echo "[run_carla.sh] route=${ROUTE}"
echo "[run_carla.sh] train_mode=${TRAIN_MODE}"
echo "[run_carla.sh] critic_mode=${CRITIC_MODE}"
echo "[run_carla.sh] critic_pi_prefix=${CRITIC_USE_PI_PREFIX_FEATURES:-<config default>}"
echo "[run_carla.sh] reward_mode=${REWARD_MODE}"
echo "[run_carla.sh] train_obs_encoder=${DAGGER_RESIDUAL_TRAIN_OBS_ENCODER}"
echo "[run_carla.sh] wandb_run_name=${WANDB_RUN_NAME}"
echo "[run_carla.sh] train_gpu_rank=${TRAIN_GPU_RANK} render_adapter=${SIM_GPU_RANK}"
echo "[run_carla.sh] carla_host=${CARLA_HOST} carla_port=${CARLA_PORT} streaming_port=${CARLA_STREAMING_PORT} tm_port=${TM_PORT} x_display=:${X_DISPLAY_NUM}"
echo "[run_carla.sh] expert_debug=${EXPERT_DEBUG} expert_recover_debug=${EXPERT_RECOVER_DEBUG} save_buffer=${SAVE_BUFFER} buffer_capacity=${BUFFER_CAPACITY:-<config default>} online_steps=${ONLINE_STEPS}"
echo "[run_carla.sh] temp agent config: ${AGENT_CFG_TMP}"
echo "[run_carla.sh] temp carla config: ${CARLA_CFG_TMP}"

XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_autotune_level=2" \
WANDB_MODE="${WANDB_MODE}" WANDB_RUN_NAME="${WANDB_RUN_NAME}" uv run python impls/main_carla.py \
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
