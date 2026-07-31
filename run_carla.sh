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
EXPERT_DEBUG="false"
EXPERT_RECOVER_DEBUG="false"
WANDB_MODE="${WANDB_MODE:-online}"

TRAIN_GPU_RANK="2"
# Empty -> leave the base config's steervla.hl_training_gpu_rank untouched. Set to a JAX GPU rank
# to isolate the HL (VLM-backbone) update on its own GPU (see steervla_cast_relabel_config.py).
HL_TRAIN_GPU_RANK=""
SIM_GPU_RANK="3"
RENDER_ADAPTER=""
CARLA_HOST="localhost"
CARLA_PORT="2020"
CARLA_STREAMING_PORT="0"
TM_PORT="8020"
X_DISPLAY_NUM=""

CRITIC_MODE="expert-lang"
TRAIN_MODE="dagger"

ENABLE_UPDATES="true"
BASE_ONLY=""
STATE_ENCODER=""
RLT_CHECKPOINT=""

# Crash supervisor: relaunch main_carla (resuming from checkpoint) after a CARLA native crash
# (SIGSEGV/SIGABRT, exit code >=128). 0 disables the retry loop.
MAX_RETRIES="${MAX_RETRIES:-50}"

# Default stays the DSRL config (as on master / cast_relabel / routing-commands). The residual
# stack from surya/rl-token is opt-in via `--agent-config impls/configs/steervla_residual_config.py`;
# main_carla.py dispatches on the config's agent_name, so the flag is the only switch needed.
BASE_AGENT_CFG="impls/configs/steervla_dsrl_config.py"
BASE_CARLA_CFG="impls/configs/carla_config.yaml"

# Set FAIL2DRIVE_CARLA_ROOT in the env (or pass --fail2drive-carla-root) only if
# Fail2Drive routes need a different CARLA install than the default one.
FAIL2DRIVE_CARLA_ROOT="${FAIL2DRIVE_CARLA_ROOT:-}"

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
  --expert-debug BOOL       true|false. Drive with the expert autopilot. Default: false
  --expert-recover-debug BOOL  true|false. Expert-recovery debug mode. Default: false
  --wandb-mode MODE         online|offline|disabled. Default: WANDB_MODE or online

  --train-gpu N             JAX / learner GPU rank (inference + RL). Default: 2
  --hl-gpu N                JAX GPU rank for the isolated HL (VLM-backbone) update.
                            Unset -> use the base config's steervla.hl_training_gpu_rank.
  --sim-gpu N               Alias for --render-adapter. Default: 3
  --render-adapter N        CARLA -graphicsadapter value. Default: 3

  --carla-host HOST         CARLA host. Default: localhost
  --carla-port PORT         CARLA RPC port. Default: 2020
  --carla-streaming-port P  CARLA streaming port. Default: 0 (auto)
  --tm-port PORT            Traffic manager port. Default: 8020
  --x-display-num N         Xvfb display number. Default: derived from carla port

  --critic-mode MODE        none|delta|delta-lang|expert-lang (DSRL configs only). Default: expert-lang
  --train-mode MODE         rl|dagger (DSRL configs only). Default: dagger

  --enable-updates BOOL     true|false. false = rollout/buffer only (no RL updates). Default: true
  --base-only BOOL          true|false. true = no-RL baseline: roll out the frozen base policy only. Default: false
  --state-encoder NAME      RL state encoder: pi_prefix|pi_prefix_groups|siglip_pool|rl_token. Default: config value (pi_prefix)
  --rlt-checkpoint PATH     RLT autoencoder checkpoint (only for --state-encoder rl_token).

  --agent-config PATH       Base agent config. Default: impls/configs/steervla_dsrl_config.py
                            Residual/EXPO stack: impls/configs/steervla_residual_config.py
  --carla-config PATH       Base CARLA yaml. Default: impls/configs/carla_config.yaml
  --fail2drive-carla-root P CARLA install to use for Fail2Drive routes.
                            Default: \$FAIL2DRIVE_CARLA_ROOT (unset -> leave CARLA_ROOT alone)
  --max-retries N           Auto-restart+resume this many times after a CARLA crash. Default: 50 (0 disables)
  -h, --help                Show this help

Examples:
  bash run_carla.sh --train-gpu 0 --sim-gpu 4 --route parking-cut-in-001
  bash run_carla.sh --online-steps 5000 --wandb-mode offline
  bash run_carla.sh --enable-updates false --save-buffer true   # rollout-only data collection
  bash run_carla.sh --agent-config impls/configs/steervla_residual_config.py --state-encoder pi_prefix
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
    --hl-gpu) HL_TRAIN_GPU_RANK="$2"; shift 2 ;;
    --sim-gpu) SIM_GPU_RANK="$2"; shift 2 ;;
    --render-adapter) RENDER_ADAPTER="$2"; shift 2 ;;
    --carla-host) CARLA_HOST="$2"; shift 2 ;;
    --carla-port) CARLA_PORT="$2"; shift 2 ;;
    --carla-streaming-port) CARLA_STREAMING_PORT="$2"; shift 2 ;;
    --tm-port) TM_PORT="$2"; shift 2 ;;
    --x-display-num) X_DISPLAY_NUM="$2"; shift 2 ;;
    --critic-mode) CRITIC_MODE="$2"; shift 2 ;;
    --train-mode) TRAIN_MODE="$2"; shift 2 ;;
    --enable-updates) ENABLE_UPDATES="$2"; shift 2 ;;
    --base-only) BASE_ONLY="$2"; shift 2 ;;
    --state-encoder) STATE_ENCODER="$2"; shift 2 ;;
    --rlt-checkpoint) RLT_CHECKPOINT="$2"; shift 2 ;;
    --agent-config) BASE_AGENT_CFG="$2"; shift 2 ;;
    --carla-config) BASE_CARLA_CFG="$2"; shift 2 ;;
    --fail2drive-carla-root) FAIL2DRIVE_CARLA_ROOT="$2"; shift 2 ;;
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
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
    _HL_GPU = "${HL_TRAIN_GPU_RANK}"
    if _HL_GPU != "" and "steervla" in config:
        # Isolate the HL (VLM-backbone) update on its own GPU. See steervla_cast_relabel_config.py.
        config.steervla.hl_training_gpu_rank = int(_HL_GPU)
    config.enable_updates = ${ENABLE_UPDATES^}
    # DSRL-only knobs: the residual config (surya/rl-token) has no critic_feedback_mode /
    # online_training_mode, so only set them when the base config actually defines them.
    if "critic_feedback_mode" in config:
        config.critic_feedback_mode = "${CRITIC_FEEDBACK_MODE}"
        if config.critic_feedback_mode == "none":
            config.language_label_dim = 0
    if "online_training_mode" in config:
        config.online_training_mode = "${TRAIN_MODE}"
    # Residual-only knobs: emitted as empty strings unless the matching flag was passed.
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

# Stable run name so save_dir + the W&B run id survive restarts (the supervisor resumes, not forks).
ROUTE_TAG="$(printf '%s' "$ROUTE" | tr -c 'A-Za-z0-9._-' '-' | sed 's/-\{2,\}/-/g; s/^-//; s/-$//')"
if [[ "${BASE_ONLY,,}" == "true" ]]; then MODE_TAG="base"; else MODE_TAG="${STATE_ENCODER:-pi_prefix}"; fi
EXP_NAME="${ROUTE_TAG}-${MODE_TAG}-sd$(printf '%03d' "$SEED")_$(date +%Y%m%d_%H%M%S)"

# Resolve route source (bench2drive | fail2drive) via the registry, then only
# override CARLA_ROOT if FAIL2DRIVE_CARLA_ROOT is explicitly set (e.g. you want
# to point Fail2Drive routes at a separate 0.9.15 install). By default the
# vanilla 0.9.16 install ships the Fail2Drive asset packs we copied in, so we
# leave CARLA_ROOT alone and both benchmarks talk to the same server.
ROUTE_SOURCE="$(uv run python -c "from ogbench.carla.route_registry import find_route; print(find_route('${ROUTE}').source)" 2>/dev/null || true)"
if [[ "$ROUTE_SOURCE" == "fail2drive" && -n "$FAIL2DRIVE_CARLA_ROOT" ]]; then
  if [[ ! -d "$FAIL2DRIVE_CARLA_ROOT" ]]; then
    echo "[run_carla.sh] WARNING: FAIL2DRIVE_CARLA_ROOT=${FAIL2DRIVE_CARLA_ROOT} doesn't exist; not switching CARLA_ROOT." >&2
  else
    export CARLA_ROOT="$FAIL2DRIVE_CARLA_ROOT"
    export CARLA_PYTHON_API_ROOT="$FAIL2DRIVE_CARLA_ROOT/PythonAPI/carla"
    echo "[run_carla.sh] Fail2Drive route detected: CARLA_ROOT=${CARLA_ROOT}"
    echo "[run_carla.sh]   (relaunch the CARLA server from ${CARLA_ROOT}/CarlaUE4.sh to match)"
  fi
fi

echo "[run_carla.sh] route=${ROUTE} (source=${ROUTE_SOURCE:-?})"
echo "[run_carla.sh] agent_config=${BASE_AGENT_CFG}"
echo "[run_carla.sh] train_mode=${TRAIN_MODE}"
echo "[run_carla.sh] critic_mode=${CRITIC_FEEDBACK_MODE}"
echo "[run_carla.sh] enable_updates=${ENABLE_UPDATES} base_only=${BASE_ONLY:-<config default>} state_encoder=${STATE_ENCODER:-<config default>}${RLT_CHECKPOINT:+ rlt_checkpoint=${RLT_CHECKPOINT}}"
echo "[run_carla.sh] train_gpu_rank=${TRAIN_GPU_RANK} hl_gpu_rank=${HL_TRAIN_GPU_RANK:-<config>} render_adapter=${SIM_GPU_RANK}"
echo "[run_carla.sh] carla_host=${CARLA_HOST} carla_port=${CARLA_PORT} streaming_port=${CARLA_STREAMING_PORT} tm_port=${TM_PORT} x_display=:${X_DISPLAY_NUM}"
echo "[run_carla.sh] expert_debug=${EXPERT_DEBUG} expert_recover_debug=${EXPERT_RECOVER_DEBUG}"
echo "[run_carla.sh] save_buffer=${SAVE_BUFFER} online_steps=${ONLINE_STEPS} exp_name=${EXP_NAME} max_retries=${MAX_RETRIES}"
echo "[run_carla.sh] temp agent config: ${AGENT_CFG_TMP}"
echo "[run_carla.sh] temp carla config: ${CARLA_CFG_TMP}"

# main_carla.py dispatches to the residual/EXPO path when the agent config's
# agent_name == "sac_residual" (steervla_residual_config.py). log_interval/save_interval are
# the residual cadence (main_carla's DSRL defaults are 1 / 100000); placed before EXTRA_ARGS so
# a caller can still override them via `-- --log_interval=...`.
#
# Supervisor loop: CARLA's leaderboard runs in-process and periodically segfaults on long runs
# (killing our python outright, so no in-process try/except can catch it). On a crash-signal exit
# (>=128) we kill any orphaned CARLA/Xvfb on this run's ports and relaunch with --resume=true, which
# restores the agent + replay buffer + step counter from the last checkpoint. A clean exit (0),
# SIGINT (130), or a non-crash error (<128, e.g. a config bug) stops the loop.
attempt=0
RESUME_FLAG="false"
while :; do
  set +e
  WANDB_MODE="${WANDB_MODE}" uv run python impls/main_carla.py \
    --agent="${AGENT_CFG_TMP}" \
    --carla_config="${CARLA_CFG_TMP}" \
    --route="${ROUTE}" \
    --online_steps="${ONLINE_STEPS}" \
    --save_buffer="${SAVE_BUFFER}" \
    --seed="${SEED}" \
    --run_group="${RUN_GROUP}" \
    --exp_name="${EXP_NAME}" \
    --resume="${RESUME_FLAG}" \
    --expert_debug="${EXPERT_DEBUG}" \
    --expert_recover_debug="${EXPERT_RECOVER_DEBUG}" \
    --log_interval=10 \
    --save_interval=5000 \
    "${EXTRA_ARGS[@]}"
  CODE=$?
  set -e

  if [[ $CODE -eq 0 ]]; then
    echo "[run_carla.sh] run completed (exit 0)."
    break
  fi
  if [[ $CODE -eq 130 ]]; then
    echo "[run_carla.sh] interrupted (SIGINT); not restarting."
    exit 130
  fi
  if [[ $CODE -lt 128 ]]; then
    echo "[run_carla.sh] exited with code ${CODE} (not a crash signal); not restarting."
    exit "$CODE"
  fi
  attempt=$((attempt + 1))
  if [[ "$MAX_RETRIES" -le 0 || $attempt -gt "$MAX_RETRIES" ]]; then
    echo "[run_carla.sh] crash (exit ${CODE}); retry budget exhausted (${attempt}/${MAX_RETRIES}). Giving up."
    exit "$CODE"
  fi
  echo "[run_carla.sh] main_carla crashed (exit ${CODE}, likely CARLA native SIGSEGV/SIGABRT); cleaning up + resuming (attempt ${attempt}/${MAX_RETRIES})."
  pkill -9 -f "CarlaUE4.*-carla-rpc-port=${CARLA_PORT}" 2>/dev/null || true
  pkill -9 -f "Xvfb :${X_DISPLAY_NUM} " 2>/dev/null || true
  sleep 8
  RESUME_FLAG="true"
done
