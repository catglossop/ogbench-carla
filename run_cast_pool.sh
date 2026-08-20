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
#   --render-gpus    CARLA -graphicsadapter per worker      [default: same as --worker-gpus]
#   --job-base       first carla_job.sh index; workers use base..base+N-1   [default 300]
#   --run-name       pool run name (dirs are created under --pool-dir)      [default cast_pool_<ts>]
#   --pool-dir       root for pool/ and checkpoints/        [default /raid/users/$USER/cast_pool]
#   --agent-config   base agent config                    [default impls/configs/steervla_cast_pooled_config.py]
#   --online-steps   per-worker env steps                                   [default 100000]
#   --max-rounds     stop the trainer after N rounds (0 = forever)          [default 0]
#   --dry-run        print what would launch, then exit
#
#   Memory (this is what lets several workers share one GPU):
#   --worker-mem-fraction F   XLA_PYTHON_CLIENT_MEM_FRACTION per worker     [default 0.30]
#   --trainer-mem-fraction F  XLA_PYTHON_CLIENT_MEM_FRACTION for trainer    [default 0.90]
#     JAX preallocates ~75% of a GPU by default, so an unfractioned worker grabs ~110 GB of an
#     H200 and exactly ONE fits per card -- regardless of the ~10-20 GB it actually needs. Setting
#     a fraction is the only reason a 5-6 route pool fits on 2-3 free GPUs. Budget per GPU:
#     N*F*VRAM + N*~7GB (each worker also runs a CarlaUE4 renderer) must stay under the card.
#
#   Round tuning (trainer):
#   --round-new-samples N  fresh samples that trigger a round  [default 15*<n_routes>, i.e. ~one
#                          window from every worker; the config default of 90 assumes 5-6 workers]
#   --round-batch-size N   pooled update batch size                    [default: config value]
#   --round-num-steps N    gradient steps per round                    [default: config value]
#
# Validate trainer VRAM before committing multi-hour CARLA workers (no CARLA, ~one round then exit):
#   XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 uv run python impls/train_hl_pooled.py \
#     --agent=impls/configs/steervla_cast_pooled_config.py \
#     --pool_root=/tmp/smoke_pool --checkpoint_dir=/tmp/smoke_ckpt \
#     --training_gpu=6 --smoke_test=true --steervla_min_online=0
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
RENDER_GPUS=""
WORKER_MEM_FRACTION="0.30"
TRAINER_MEM_FRACTION="0.90"
ROUND_NEW_SAMPLES=""
ROUND_BATCH_SIZE=""
ROUND_NUM_STEPS=""
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
    --render-gpus) RENDER_GPUS="$2"; shift 2 ;;
    --worker-mem-fraction) WORKER_MEM_FRACTION="$2"; shift 2 ;;
    --trainer-mem-fraction) TRAINER_MEM_FRACTION="$2"; shift 2 ;;
    --round-new-samples) ROUND_NEW_SAMPLES="$2"; shift 2 ;;
    --round-batch-size) ROUND_BATCH_SIZE="$2"; shift 2 ;;
    --round-num-steps) ROUND_NUM_STEPS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --stop) STOP_RUN="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
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

# Render adapters default to the worker's own GPU (run_carla.sh's default too). They are directly
# comparable to nvidia-smi indices: carla_utils.py forces the NVIDIA Vulkan ICD so -graphicsadapter=N
# maps to physical GPU N.
if [[ -z "$RENDER_GPUS" ]]; then
  RENDER_ARR=("${GPU_ARR[@]}")
else
  IFS=',' read -r -a RENDER_ARR <<< "$RENDER_GPUS"
  if [[ ${#RENDER_ARR[@]} -eq 1 ]]; then
    for ((i = 1; i < N_ROUTES; i++)); do RENDER_ARR+=("${RENDER_ARR[0]}"); done
  elif [[ ${#RENDER_ARR[@]} -ne $N_ROUTES ]]; then
    echo "[run_cast_pool] --render-gpus must have 1 value or one per route (${N_ROUTES})" >&2; exit 1
  fi
fi

# One window per worker per round: the config's 90 assumes 5-6 workers, so a smaller pool would
# wait ~2x longer for its first round with no signal that anything is wrong.
if [[ -z "$ROUND_NEW_SAMPLES" ]]; then
  ROUND_NEW_SAMPLES=$(( 15 * N_ROUTES ))
fi

for g in "${GPU_ARR[@]}"; do
  if [[ "$g" == "$TRAINER_GPU" ]]; then
    echo "[run_cast_pool] WARNING: worker GPU $g is also the trainer GPU. The trainer's pooled" >&2
    echo "                update is the largest allocation in the run; co-tenanting risks OOM." >&2
  fi
done

# Budget check. Each co-tenant on a GPU reserves worker_mem_fraction of it up front (JAX
# preallocates) AND runs a CarlaUE4 renderer alongside (~7 GB observed). Catch an over-subscribed
# card here rather than as an OOM twenty minutes into CARLA boot.
declare -A _GPU_WORKERS=()
for g in "${GPU_ARR[@]}"; do _GPU_WORKERS["$g"]=$(( ${_GPU_WORKERS["$g"]:-0} + 1 )); done
for g in "${!_GPU_WORKERS[@]}"; do
  n="${_GPU_WORKERS[$g]}"
  over="$(awk -v n="$n" -v f="$WORKER_MEM_FRACTION" 'BEGIN { print (n * f > 0.80) ? 1 : 0 }')"
  if [[ "$over" == "1" ]]; then
    echo "[run_cast_pool] WARNING: GPU ${g} hosts ${n} workers at --worker-mem-fraction ${WORKER_MEM_FRACTION}" >&2
    echo "                (= $(awk -v n="$n" -v f="$WORKER_MEM_FRACTION" 'BEGIN{printf "%.2f", n*f}') of the card, before ~${n}x7 GB of CarlaUE4 renderers)." >&2
    echo "                Lower --worker-mem-fraction or spread the routes across more GPUs." >&2
  fi
done

RUN_NAME="${RUN_NAME:-cast_pool_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${POOL_DIR}/${RUN_NAME}"
POOL_ROOT="${RUN_ROOT}/pool"
CKPT_ROOT="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"

echo "[run_cast_pool] run_name=${RUN_NAME}"
echo "[run_cast_pool] routes=${ROUTES} (${N_ROUTES} workers)"
echo "[run_cast_pool] worker gpus=${GPU_ARR[*]}  render adapters=${RENDER_ARR[*]}  trainer gpu=${TRAINER_GPU}"
echo "[run_cast_pool] mem fraction: worker=${WORKER_MEM_FRACTION} trainer=${TRAINER_MEM_FRACTION}"
echo "[run_cast_pool] round: new_samples=${ROUND_NEW_SAMPLES} batch=${ROUND_BATCH_SIZE:-<config>} steps=${ROUND_NUM_STEPS:-<config>}"
echo "[run_cast_pool] pool=${POOL_ROOT}"
echo "[run_cast_pool] checkpoints=${CKPT_ROOT}"
echo "[run_cast_pool] job indices=${JOB_BASE}..$((JOB_BASE + N_ROUTES - 1))"

if [[ "$DRY_RUN" == true ]]; then
  for ((i = 0; i < N_ROUTES; i++)); do
    echo "  worker $((JOB_BASE + i)): route=${ROUTE_ARR[$i]} train_gpu=${GPU_ARR[$i]} render=${RENDER_ARR[$i]} mem_frac=${WORKER_MEM_FRACTION}"
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
    config.cast_pool.round_new_samples = ${ROUND_NEW_SAMPLES}
    _ROUND_BATCH = "${ROUND_BATCH_SIZE}"
    if _ROUND_BATCH:
        config.cast_pool.round_batch_size = int(_ROUND_BATCH)
        config.steervla.hl_update_batch_size = int(_ROUND_BATCH)
    _ROUND_STEPS = "${ROUND_NUM_STEPS}"
    if _ROUND_STEPS:
        config.cast_pool.round_num_steps = int(_ROUND_STEPS)
        config.steervla.hl_update_num_steps = int(_ROUND_STEPS)
    # Workers write into the shared pool; the session namespaces by run tag underneath it.
    config.cast_relabel.hl_dataset_root = r"${POOL_ROOT}"
    return config
EOF
echo "[run_cast_pool] wrote pooled agent config: ${POOLED_CFG}"

WORKER_JOBS=()
for ((i = 0; i < N_ROUTES; i++)); do
  ROUTE="${ROUTE_ARR[$i]}"
  GPU="${GPU_ARR[$i]}"
  RENDER="${RENDER_ARR[$i]}"
  JOB=$((JOB_BASE + i))
  echo "[run_cast_pool] starting worker job ${JOB}: route=${ROUTE} train_gpu=${GPU} render=${RENDER} mem_frac=${WORKER_MEM_FRACTION}"
  # XLA_PYTHON_CLIENT_MEM_FRACTION is inherited straight through setsid -> run_carla.sh -> uv run
  # -> python, and is what allows more than one worker per GPU (see the header).
  XLA_PYTHON_CLIENT_MEM_FRACTION="$WORKER_MEM_FRACTION" \
  ./carla_job.sh start --job "$JOB" --train-gpu "$GPU" --render-adapter "$RENDER" --route "$ROUTE" -- \
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
# --mem_fraction is passed as a FLAG, not just an env var: train_hl_pooled.py applies it before JAX
# initializes. Relying on the exported variable alone once left the trainer on JAX's 75% default and
# OOM'd round 1, so the flag is the authoritative channel and the export is only a belt-and-braces.
export XLA_PYTHON_CLIENT_MEM_FRACTION="$TRAINER_MEM_FRACTION"
nohup uv run python impls/train_hl_pooled.py \
  --agent="$POOLED_CFG" \
  --pool_root="$POOL_ROOT" \
  --checkpoint_dir="$CKPT_ROOT" \
  --training_gpu="$TRAINER_GPU" \
  --mem_fraction="$TRAINER_MEM_FRACTION" \
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
WORKER_GPUS_USED="${GPU_ARR[*]}"
RENDER_ADAPTERS_USED="${RENDER_ARR[*]}"
WORKER_MEM_FRACTION=${WORKER_MEM_FRACTION}
TRAINER_MEM_FRACTION=${TRAINER_MEM_FRACTION}
ROUND_NEW_SAMPLES=${ROUND_NEW_SAMPLES}
ROUTES=${ROUTES}
EOF

echo "[run_cast_pool] trainer pid=${TRAINER_PID}"
echo "[run_cast_pool] manifest: ${RUN_ROOT}/manifest.env"
echo "[run_cast_pool] tail trainer:  tail -f ${TRAINER_LOG}"
echo "[run_cast_pool] tail a worker: ./carla_job.sh logs ${WORKER_JOBS[0]}"
echo "[run_cast_pool] stop all:      ./run_cast_pool.sh --stop ${RUN_NAME}"
