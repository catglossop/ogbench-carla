#!/usr/bin/env bash
# Restartable F2D YAY-Robot evaluation queue.
# Each route trains the SimLingo HL planner for 10k steps, then --eval-mode freezes it
# and rolls it out for three model seeds in the same CARLA process.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

POSITIONAL_ARGS=()
HL_PHYSICAL_GPU="${EVAL_HL_GPU:-7}"
while (($#)); do
  case "$1" in
    --hl-gpu)
      [[ $# -ge 2 ]] || { echo '--hl-gpu requires a GPU id.' >&2; exit 2; }
      HL_PHYSICAL_GPU="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: bash run_yay_robot_eval.sh [policy_gpu] [run_root] [--hl-gpu GPU]"
      exit 0
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

PHYSICAL_GPU="${POSITIONAL_ARGS[0]:-${EVAL_GPU:-0}}"
SINGLE_GPU="${EVAL_SINGLE_GPU:-0}"
RUN_ROOT="${POSITIONAL_ARGS[1]:-${EVAL_ROOT:-/raid/users/${USER}/carla_exps/evals/yay_robot}}"
EVAL_LABEL="${EVAL_LABEL:-simlingo-steervla-yay-robot-f2d-remaining-v1-20261116}"
RUN_GROUP="${EVAL_RUN_GROUP:-SimLingoSteerVLAYayRobotF2DEvalModeV2}"
# Keep the caller's local CARLA installation (with install_f2d_content.sh assets) by default.
FAIL2DRIVE_CARLA_ROOT="${EVAL_FAIL2DRIVE_CARLA_ROOT:-}"
CARLA_0915_ROOT="${CARLA_0915_ROOT:-${EVAL_CARLA_0915_ROOT:-/raid/users/${USER}/f2d_carla}}"
CARLA_0915_PYTHON="${CARLA_0915_PYTHON:-${EVAL_CARLA_0915_PYTHON:-${ROOT_DIR}/.venv-carla-0915/bin/python}}"
# A retry gets a fresh W&B name; completed routes still skip via their .done markers.
# Supply EVAL_RUN_NONCE=<number> only to deliberately reuse names.
if [[ -n "${EVAL_RUN_NONCE:-}" ]]; then
  RUN_NONCE="$EVAL_RUN_NONCE"
else
  printf -v RUN_NONCE '%05d%05d' "$RANDOM" "$RANDOM"
fi
[[ "$RUN_NONCE" =~ ^[0-9]+$ ]] || { echo 'EVAL_RUN_NONCE must be numeric.' >&2; exit 2; }
# Default to one training seed; the loop completes every route before advancing to a later seed.
SEEDS="${SEEDS:-0}"
MAX_RETRIES="${MAX_RETRIES:-50}"
WANDB_MODE="${WANDB_MODE:-online}"
SAVE_BUFFER="${SAVE_BUFFER:-false}"
SAVE_VIDEO_LOCAL="${SAVE_VIDEO_LOCAL:-true}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-true}"
DRY_RUN="${DRY_RUN:-0}"
AGENT_CONFIG="${EVAL_AGENT_CONFIG:-impls/configs/simlingo_steervla_yay_robot_train_config.py}"
EXP_PREFIX="${EVAL_EXP_PREFIX:-simlingo-steervla-yay-robot-f2d-remaining}"

# Optional: hold each cell until the GPU has this much free memory (MiB). 0 = start immediately.
WAIT_FREE_MIB="${EVAL_WAIT_FREE_MIB:-0}"
# Lanes sharing a GPU serialize their starts through a gate file: a cell starts only when the GPU has
# WAIT_FREE_MIB free AND no other lane started on it in the last EVAL_GPU_GATE_HOLD seconds (its CARLA
# and models are still loading, so free memory over-reports).
GATE_DIR="${EVAL_GPU_GATE_DIR:-/tmp/${USER}_carla_gpu_gates}"
GATE_HOLD="${EVAL_GPU_GATE_HOLD:-900}"
wait_for_gpu_memory() {
  [[ "$WAIT_FREE_MIB" -gt 0 ]] || return 0
  mkdir -p "$GATE_DIR"
  local gate="$GATE_DIR/gpu${PHYSICAL_GPU}" free last now waited=0 gate_fd ok
  while true; do
    exec {gate_fd}>"$gate.lock"
    flock "$gate_fd"
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$PHYSICAL_GPU" 2>/dev/null | tr -d ' ')
    last=$(cat "$gate" 2>/dev/null || echo 0); now=$(date +%s); ok=0
    if [[ "$free" =~ ^[0-9]+$ && "$free" -ge "$WAIT_FREE_MIB" && $((now - last)) -ge "$GATE_HOLD" ]]; then
      echo "$now" > "$gate"; ok=1
    fi
    flock -u "$gate_fd"; eval "exec ${gate_fd}>&-"
    [[ "$ok" == 1 ]] && { echo "[$(date --iso-8601=seconds)] GPU $PHYSICAL_GPU gate open (${free} MiB free)"; return 0; }
    (( waited % 1800 == 0 )) && echo "[$(date --iso-8601=seconds)] waiting for GPU $PHYSICAL_GPU: ${free:-?} MiB free (need $WAIT_FREE_MIB), last start $((now - last))s ago"
    sleep 60; waited=$((waited + 60))
  done
}
# Optional: EVAL_F2D_0915=1 runs every Fail2Drive (generalization-*) route on CARLA 0.9.15, not only animals.
F2D_0915="${EVAL_F2D_0915:-0}"
uses_0915() {
  [[ "$1" == "generalization-animals-1076" ]] && return 0
  [[ "$F2D_0915" == "1" && "$1" == generalization-* ]]
}
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "[yay_robot_eval] GEMINI_API_KEY must be exported; refusing to launch an uncorrected baseline." >&2
  exit 1
fi

case "$SINGLE_GPU" in
  0|false|1|true) ;;
  *) echo "[yay_robot_eval] EVAL_SINGLE_GPU must be 0, 1, false, or true." >&2; exit 2 ;;
esac
CUDA_DEVICES="$PHYSICAL_GPU,$HL_PHYSICAL_GPU"
HL_GPU_RANK=1
MEMORY_ENV=()
if [[ "$SINGLE_GPU" == "1" || "$SINGLE_GPU" == "true" ]]; then
  HL_PHYSICAL_GPU="$PHYSICAL_GPU"
  CUDA_DEVICES="$PHYSICAL_GPU"
  HL_GPU_RANK=0
  MEMORY_ENV=(
    "XLA_PYTHON_CLIENT_MEM_FRACTION=${EVAL_JAX_MEM_FRACTION:-0.30}"
    "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  )
elif [[ "$PHYSICAL_GPU" == "$HL_PHYSICAL_GPU" ]]; then
  echo "[yay_robot_eval] EVAL_HL_GPU must differ from the policy/CARLA GPU unless EVAL_SINGLE_GPU=1." >&2
  exit 2
fi

# Dedicated to YAY Robot, avoiding both residual-evaluation port ranges.
CARLA_PORT="${EVAL_CARLA_PORT:-$((18000 + PHYSICAL_GPU * 20))}"
CARLA_STREAMING_PORT="${EVAL_CARLA_STREAMING_PORT:-$((CARLA_PORT + 1))}"
TM_PORT="${EVAL_TM_PORT:-$((CARLA_PORT + 6000))}"
X_DISPLAY_NUM="${EVAL_X_DISPLAY_NUM:-$((700 + PHYSICAL_GPU))}"

ROUTES=(
  generalization-hard-brake-1036
  generalization-pedestrian-other-blocker-1072
  generalization-right-construction-1094
  generalization-animals-1076
)
if [[ -n "${EVAL_ROUTES:-}" ]]; then
  read -r -a ROUTES <<< "$EVAL_ROUTES"
fi
(( ${#ROUTES[@]} > 0 )) || { echo 'EVAL_ROUTES selected no routes.' >&2; exit 2; }

needs_0915=0
for r in "${ROUTES[@]}"; do uses_0915 "$r" && needs_0915=1; done
if [[ "$needs_0915" == 1 ]]; then
  [[ -x "$CARLA_0915_ROOT/CarlaUE4.sh" ]] || {
    echo "[yay_robot_eval] CARLA 0.9.15 simulator not found at $CARLA_0915_ROOT/CarlaUE4.sh" >&2; exit 2;
  }
  [[ -x "$CARLA_0915_PYTHON" ]] || {
    echo "[yay_robot_eval] CARLA 0.9.15 Python environment not found at $CARLA_0915_PYTHON" >&2; exit 2;
  }
  [[ -f "$CARLA_0915_ROOT/CarlaUE4/Content/Carla/Blueprints/Walkers/BP_Elephant.uasset" ]] || {
    echo "[yay_robot_eval] Elephant blueprint missing from CARLA 0.9.15 root: $CARLA_0915_ROOT" >&2; exit 2;
  }
fi

read -r -a SEED_LIST <<< "$SEEDS"
if (( ${#SEED_LIST[@]} == 0 )); then
  echo 'SEEDS must contain at least one seed.' >&2
  exit 2
fi
for seed in "${SEED_LIST[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
done

mkdir -p "$RUN_ROOT/logs/$EVAL_LABEL" "$RUN_ROOT/status/$EVAL_LABEL" "$RUN_ROOT/runs/$EVAL_LABEL"
cat > "$RUN_ROOT/eval_spec_${EVAL_LABEL}.txt" <<SPEC
created=$(date --iso-8601=seconds)
eval_label=$EVAL_LABEL
run_group=$RUN_GROUP
run_nonce=$RUN_NONCE
physical_gpu=$PHYSICAL_GPU
hl_physical_gpu=$HL_PHYSICAL_GPU
single_gpu=$SINGLE_GPU
cuda_visible_devices=$CUDA_DEVICES
xla_mem_fraction=${EVAL_JAX_MEM_FRACTION:-<default>}
carla_port=$CARLA_PORT
carla_streaming_port=$CARLA_STREAMING_PORT
tm_port=$TM_PORT
x_display_num=$X_DISPLAY_NUM
seeds=${SEED_LIST[*]}
routes=${ROUTES[*]}
online_steps=10000
max_episode_steps=4000
agent_config=$AGENT_CONFIG
f2d_all_0915=$F2D_0915
wait_free_mib=$WAIT_FREE_MIB
xla_preallocate=${XLA_PYTHON_CLIENT_PREALLOCATE:-<default>}
hl_checkpoint=/raid/users/celine/steervla-ckpts/2026_05_24_06_52_33_simlingo_seed1_bellman/checkpoints/epoch=019.ckpt
ll_checkpoint=/raid/users/celine/steervla-ckpts/2026_05_23_21_39_41_simlingo_ll_vla_meta_conditioned/checkpoints/epoch=029.ckpt
fail2drive_carla_root=${FAIL2DRIVE_CARLA_ROOT:-<current CARLA_ROOT>}
carla_0915_root_for_animals=$CARLA_0915_ROOT
carla_0915_python_for_animals=$CARLA_0915_PYTHON
yay_correction_period_env_steps=24
yay_correction_every_cot_queries=4
hl_update_schedule=10_gradient_steps_per_200_env_steps
hl_replay_mix=0.5_offline_simlingo_hl_simplified__0.5_online_yay

SPEC

WATCHDOG_GRACE="${EVAL_WATCHDOG_GRACE:-600}"
STALL_SECS="${EVAL_STALL_SECS:-900}"
STALL_STRIKES="${EVAL_STALL_STRIKES:-2}"
CARLA_GONE_STRIKES="${EVAL_CARLA_GONE_STRIKES:-6}"
run_with_watchdog() {
  local log_file="$1" failure_file="$2"; shift 2
  : > "$log_file"
  setsid "$@" > >(tee "$log_file") 2>&1 &
  local run_pid=$!
  (
    sleep "$WATCHDOG_GRACE"
    local stalls=0 gone=0 age
    while kill -0 "$run_pid" 2>/dev/null; do
      if ! pgrep -u "$USER" -f "carla-rpc-port=${CARLA_PORT}" >/dev/null 2>&1; then gone=$((gone + 1)); else gone=0; fi
      age=$(( $(date +%s) - $(stat -c %Y "$log_file" 2>/dev/null || date +%s) ))
      if [[ "$age" -ge "$STALL_SECS" ]]; then stalls=$((stalls + 1)); else stalls=0; fi
      if [[ "$gone" -ge "$CARLA_GONE_STRIKES" || "$stalls" -ge "$STALL_STRIKES" ]]; then
        local reason="carla_missing"
        [[ "$stalls" -ge "$STALL_STRIKES" ]] && reason="no_log_progress_${age}s"
        printf 'stage=run\nreason=%s\ntimestamp=%s\n' "$reason" "$(date --iso-8601=seconds)" > "$failure_file"
        echo "[watchdog] aborting pid $run_pid: $reason" >> "$log_file"
        kill -TERM -"$run_pid" 2>/dev/null || true
        sleep 15
        kill -KILL -"$run_pid" 2>/dev/null || true
        pgrep -u "$USER" -f "carla-rpc-port=${CARLA_PORT}" | xargs -r kill -KILL 2>/dev/null || true
        rm -f "/tmp/.X${X_DISPLAY_NUM}-lock" 2>/dev/null || true
        break
      fi
      sleep 60
    done
  ) &
  local watchdog_pid=$!
  wait "$run_pid"; local run_code=$?
  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true
  return "$run_code"
}

total=$(( ${#ROUTES[@]} * ${#SEED_LIST[@]} ))
index=0
for seed in "${SEED_LIST[@]}"; do
  for route in "${ROUTES[@]}"; do
    index=$((index + 1))
    tag="${route}__seed-${seed}"
    done_file="$RUN_ROOT/status/$EVAL_LABEL/${tag}.done"
    failure_file="$RUN_ROOT/status/$EVAL_LABEL/${tag}.failed"
    log_file="$RUN_ROOT/logs/$EVAL_LABEL/${tag}.log"
    exp_name="${EXP_PREFIX}_${RUN_NONCE}_sd-${seed}_${route}"
    carla_version_env=(CARLA_0915_ROOT= CARLA_0915_PYTHON=)
    fail2drive_root_env=(FAIL2DRIVE_CARLA_ROOT="$FAIL2DRIVE_CARLA_ROOT")
    if uses_0915 "$route"; then
      [[ -z "${EVAL_EXP_PREFIX:-}" ]] && exp_name="simlingo-steervla-yay-robot-f2d-animals-0915_${RUN_NONCE}_sd-${seed}_${route}"
      carla_version_env=(CARLA_0915_ROOT="$CARLA_0915_ROOT" CARLA_0915_PYTHON="$CARLA_0915_PYTHON")
      # run_carla.sh applies this override after the 0.9.15 switch; leave it unset for this route.
      fail2drive_root_env=(FAIL2DRIVE_CARLA_ROOT=)
    fi
    if [[ "$SKIP_COMPLETED" == "1" && -f "$done_file" ]]; then
      echo "[$index/$total] SKIP completed: $tag"
      continue
    fi
    echo "[$index/$total] START: $tag (wandb_name=$exp_name)"
    if [[ "$DRY_RUN" == "1" ]]; then
      continue
    fi

    run_dir="$RUN_ROOT/runs/$EVAL_LABEL/$tag"
    mkdir -p "$run_dir"
    set +e
    eval_seeds="$((seed + 1001)),$((seed + 1002)),$((seed + 1003))"
    wait_for_gpu_memory
    run_with_watchdog "$log_file" "$failure_file" env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "${MEMORY_ENV[@]}" "${fail2drive_root_env[@]}" OGBENCH_SAVE_DIR="$run_dir" "${carla_version_env[@]}" \
      ./run_carla.sh \
        --agent-config "$AGENT_CONFIG" \
        --route "$route" \
        --carla-seed "$seed" \
        --train-seed "$seed" \
        --eval-seeds "$eval_seeds" \
        --seed "$seed" \
        --exp-name "$exp_name" \
        --online-steps 10000 \
        --max-episode-steps 4000 \
        --train-gpu 0 \
        --hl-gpu "$HL_GPU_RANK" \
        --train-mode rl \
        --critic-mode none \
        --render-adapter "$PHYSICAL_GPU" \
        --carla-port "$CARLA_PORT" \
        --carla-streaming-port "$CARLA_STREAMING_PORT" \
        --post-stop-eval-episodes 3 \
        --tm-port "$TM_PORT" \
        --x-display-num "$X_DISPLAY_NUM" \
        --run-group "$RUN_GROUP" \
        --eval-mode \
        --save-buffer "$SAVE_BUFFER" \
        --save-video-local "$SAVE_VIDEO_LOCAL" \
        --wandb-mode "$WANDB_MODE" \
        --max-retries "$MAX_RETRIES"
    run_code=$?; tee_code=0
    set -e
    if [[ "$run_code" -eq 0 && "$tee_code" -eq 0 ]]; then
      touch "$done_file"
      echo "[$index/$total] DONE: $tag"
      continue
    fi
    if [[ "$run_code" -eq 130 ]]; then
      echo "[$index/$total] interrupted; stopping evaluation queue." >&2
      exit 130
    fi
    printf 'timestamp=%s\nrun_exit_code=%s\ntee_exit_code=%s\n' "$(date --iso-8601=seconds)" "$run_code" "$tee_code" > "$failure_file"
    echo "[$index/$total] FAILED: $tag (run=$run_code tee=$tee_code; marker=$failure_file)" >&2
    if [[ "$CONTINUE_ON_FAILURE" == "1" || "$CONTINUE_ON_FAILURE" == "true" ]]; then
      echo "[$index/$total] continuing after recorded failure." >&2
      continue
    fi
    exit "$run_code"
  done
done

echo "Evaluation queue complete: $RUN_ROOT"
