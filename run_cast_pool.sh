#!/usr/bin/env bash
# Launch a pooled CAST-relabel run: N rollout workers (one route each) + 1 central trainer.
#
# Workers roll out and relabel into a shared sample pool but take no gradient steps. The trainer
# owns the only trainable model, runs a large pooled update every `round_new_samples` fresh samples,
# and publishes a params-only checkpoint. Workers hot-reload the newest published version
# mid-episode, so every route converges back onto one policy after each round.
#
# See impls/cast_pool.py for the filesystem protocol and impls/configs/steervla_cast_pooled_config.py
# for the knobs.
#
# Usage:
#   ./run_cast_pool.sh --routes r1,r2,r3 --worker-gpus 2,3,4 --trainer-gpu 7 [options] [-- extra run_carla.sh args]
#
#   --routes         comma-separated route names (one worker each)          [required]
#   --worker-gpus    comma-separated GPU per worker; one value = all share  [required]
#   --trainer-gpu    GPU for the central trainer                            [required]
#   --job-base       first carla_job.sh index; workers use base..base+N-1   [default 300]
#   --run-name       pool run name (dirs are created under --pool-dir)      [default cast_pool_<ts>]
#   --pool-dir       root for pool/ and checkpoints/                        [default /raid/users/$USER/cast_pool]
#   --agent-config   base agent config                    [default impls/configs/steervla_cast_pooled_config.py]
#   --online-steps   per-worker env steps                                   [default 100000]
#   --max-rounds     stop the trainer after N rounds (0 = forever)          [default 0]
#   --dry-run        print what would launch, then exit
#
# Stop everything with:  ./run_cast_pool.sh --stop <run-name>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ROUTES=""
WORKER_GPUS=""
TRAINER_GPU=""
JOB_BASE=300
RUN_NAME=""
POOL_DIR="/raid/users/${USER}/cast_pool"
AGENT_CFG="impls/configs/steervla_cast_pooled_config.py"
ONLINE_STEPS=100000
MAX_ROUNDS=0
DRY_RUN=false
STOP_RUN=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --routes) ROUTES="$2"; shift 2 ;;
    --worker-gpus) WORKER_GPUS="$2"; shift 2 ;;
    --trainer-gpu) TRAINER_GPU="$2"; shift 2 ;;
    --job-base) JOB_BASE="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --pool-dir) POOL_DIR="$2"; shift 2 ;;
    --agent-config) AGENT_CFG="$2"; shift 2 ;;
    --online-steps) ONLINE_STEPS="$2"; shift 2 ;;
    --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --stop) STOP_RUN="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "[run_cast_pool] unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ── stop mode ────────────────────────────────────────────────────────────────────────
if [[ -n "$STOP_RUN" ]]; then
  MANIFEST="${POOL_DIR}/${STOP_RUN}/manifest.env"
  if [[ ! -f "$MANIFEST" ]]; then
    echo "[run_cast_pool] no manifest at $MANIFEST" >&2; exit 1
  fi
  # shellcheck disable=SC1090
  source "$MANIFEST"
  echo "[run_cast_pool] stopping trainer pid=${TRAINER_PID:-none}"
  [[ -n "${TRAINER_PID:-}" ]] && kill "$TRAINER_PID" 2>/dev/null || true
  for job in ${WORKER_JOBS:-}; do
    echo "[run_cast_pool] stopping worker job $job"
    ./carla_job.sh stop "$job" || true
  done
  echo "[run_cast_pool] done."
  exit 0
fi

# ── validate ─────────────────────────────────────────────────────────────────────────
[[ -n "$ROUTES" ]]      || { echo "[run_cast_pool] --routes is required" >&2; exit 1; }
[[ -n "$WORKER_GPUS" ]] || { echo "[run_cast_pool] --worker-gpus is required" >&2; exit 1; }
[[ -n "$TRAINER_GPU" ]] || { echo "[run_cast_pool] --trainer-gpu is required" >&2; exit 1; }

IFS=',' read -r -a ROUTE_ARR <<< "$ROUTES"
IFS=',' read -r -a GPU_ARR <<< "$WORKER_GPUS"
N_ROUTES=${#ROUTE_ARR[@]}

if [[ ${#GPU_ARR[@]} -eq 1 ]]; then
  for ((i = 1; i < N_ROUTES; i++)); do GPU_ARR+=("${GPU_ARR[0]}"); done
elif [[ ${#GPU_ARR[@]} -ne $N_ROUTES ]]; then
  echo "[run_cast_pool] --worker-gpus must have 1 value or one per route (${N_ROUTES})" >&2; exit 1
fi

for g in "${GPU_ARR[@]}"; do
  if [[ "$g" == "$TRAINER_GPU" ]]; then
    echo "[run_cast_pool] WARNING: worker GPU $g is also the trainer GPU. The trainer's pooled" >&2
    echo "                update is the largest allocation in the run; co-tenanting risks OOM." >&2
  fi
done

RUN_NAME="${RUN_NAME:-cast_pool_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${POOL_DIR}/${RUN_NAME}"
POOL_ROOT="${RUN_ROOT}/pool"
CKPT_ROOT="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"

echo "[run_cast_pool] run_name=${RUN_NAME}"
echo "[run_cast_pool] routes=${ROUTES} (${N_ROUTES} workers)"
echo "[run_cast_pool] worker gpus=${GPU_ARR[*]}  trainer gpu=${TRAINER_GPU}"
echo "[run_cast_pool] pool=${POOL_ROOT}"
echo "[run_cast_pool] checkpoints=${CKPT_ROOT}"
echo "[run_cast_pool] job indices=${JOB_BASE}..$((JOB_BASE + N_ROUTES - 1))"

if [[ "$DRY_RUN" == true ]]; then
  for ((i = 0; i < N_ROUTES; i++)); do
    echo "  worker $((JOB_BASE + i)): route=${ROUTE_ARR[$i]} gpu=${GPU_ARR[$i]}"
  done
  echo "  trainer: gpu=${TRAINER_GPU} rounds=${MAX_ROUNDS}"
  exit 0
fi

mkdir -p "$POOL_ROOT" "$CKPT_ROOT" "$LOG_DIR"

# ── per-run agent config ─────────────────────────────────────────────────────────────
# Same layering idea as run_carla.sh: runpy-load the base config, then stamp in the pool paths.
# Written once and shared by every worker and the trainer, so they cannot disagree about where the
# pool and the published checkpoints live.
POOLED_CFG="${RUN_ROOT}/agent_config.py"
cat > "$POOLED_CFG" <<EOF
from pathlib import Path
import runpy

_BASE_PATH = Path(r"${REPO_ROOT}/${AGENT_CFG}")
_BASE_GET_CONFIG = runpy.run_path(str(_BASE_PATH))["get_config"]


def get_config():
    config = _BASE_GET_CONFIG()
    config.cast_pool.pool_root = r"${POOL_ROOT}"
    config.cast_pool.checkpoint_dir = r"${CKPT_ROOT}"
    config.cast_pool.max_rounds = ${MAX_ROUNDS}
    # Workers write into the shared pool; the session namespaces by run tag underneath it.
    config.cast_relabel.hl_dataset_root = r"${POOL_ROOT}"
    return config
EOF
echo "[run_cast_pool] wrote pooled agent config: ${POOLED_CFG}"

WORKER_JOBS=()
for ((i = 0; i < N_ROUTES; i++)); do
  ROUTE="${ROUTE_ARR[$i]}"
  GPU="${GPU_ARR[$i]}"
  JOB=$((JOB_BASE + i))
  echo "[run_cast_pool] starting worker job ${JOB}: route=${ROUTE} gpu=${GPU}"
  ./carla_job.sh start --job "$JOB" --train-gpu "$GPU" --render-adapter "$GPU" --route "$ROUTE" -- \
    --agent-config "$POOLED_CFG" \
    --train-mode rl --critic-mode none --enable-updates false \
    --online-steps "$ONLINE_STEPS" --save-buffer false --max-retries 50 \
    "${EXTRA_ARGS[@]}"
  WORKER_JOBS+=("$JOB")
  # Stagger: each worker boots its own Xvfb + UE4 server and restores a multi-GB checkpoint.
  # Starting them in lockstep makes those bursts collide on disk and on the CARLA RPC handshake.
  sleep 20
done

# ── trainer ──────────────────────────────────────────────────────────────────────────
TRAINER_LOG="${LOG_DIR}/trainer.log"
echo "[run_cast_pool] starting trainer on gpu ${TRAINER_GPU} -> ${TRAINER_LOG}"
nohup uv run python impls/train_hl_pooled.py \
  --agent="$POOLED_CFG" \
  --pool_root="$POOL_ROOT" \
  --checkpoint_dir="$CKPT_ROOT" \
  --training_gpu="$TRAINER_GPU" \
  --max_rounds="$MAX_ROUNDS" \
  --run_name="${RUN_NAME}_trainer" \
  > "$TRAINER_LOG" 2>&1 &
TRAINER_PID=$!

cat > "${RUN_ROOT}/manifest.env" <<EOF
RUN_NAME=${RUN_NAME}
POOL_ROOT=${POOL_ROOT}
CKPT_ROOT=${CKPT_ROOT}
TRAINER_PID=${TRAINER_PID}
TRAINER_GPU=${TRAINER_GPU}
WORKER_JOBS="${WORKER_JOBS[*]}"
ROUTES=${ROUTES}
EOF

echo "[run_cast_pool] trainer pid=${TRAINER_PID}"
echo "[run_cast_pool] manifest: ${RUN_ROOT}/manifest.env"
echo "[run_cast_pool] tail trainer:  tail -f ${TRAINER_LOG}"
echo "[run_cast_pool] tail a worker: ./carla_job.sh logs ${WORKER_JOBS[0]}"
echo "[run_cast_pool] stop all:      ./run_cast_pool.sh --stop ${RUN_NAME}"
