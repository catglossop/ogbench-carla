#!/usr/bin/env bash
# Restartable B2D + F2D residual-RL evaluation queue.
# Each route trains for 10k steps, then --eval-mode freezes the final policy
# and rolls it out for three model seeds in the same CARLA process.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PHYSICAL_GPU="${1:-${EVAL_GPU:-0}}"
RUN_ROOT="${2:-${EVAL_ROOT:-/raid/users/${USER}/carla_exps/evals/residual_rl}}"
EVAL_LABEL="${EVAL_LABEL:-residual-rl-b2d-f2d-evalmode-v3-20260914}"
RUN_GROUP="${EVAL_RUN_GROUP:-ResidualRLB2DF2DEvalModeV3}"
# A fresh queue invocation must not collide in W&B with an earlier failed launch.
# Supply EVAL_RUN_NONCE=<number> to deliberately retain names across invocations.
if [[ -n "${EVAL_RUN_NONCE:-}" ]]; then
  RUN_NONCE="$EVAL_RUN_NONCE"
else
  NONCE_FILE="$RUN_ROOT/status/$EVAL_LABEL/run_nonce"
  if [[ -f "$NONCE_FILE" ]]; then
    RUN_NONCE="$(<"$NONCE_FILE")"
  else
    printf -v RUN_NONCE '%05d%05d' "$RANDOM" "$RANDOM"
  fi
fi
[[ "$RUN_NONCE" =~ ^[0-9]+$ ]] || { echo 'EVAL_RUN_NONCE must be numeric.' >&2; exit 2; }
SEEDS="${SEEDS:-0 1 2}"
MAX_RETRIES="${MAX_RETRIES:-50}"
WANDB_MODE="${WANDB_MODE:-online}"
SAVE_BUFFER="${SAVE_BUFFER:-true}"
SAVE_VIDEO_LOCAL="${SAVE_VIDEO_LOCAL:-true}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-true}"
DRY_RUN="${DRY_RUN:-0}"

# Isolated from the normal sweep range (2000 + gpu * 20) and failed-rerun
# range (12000 + gpu * 20). Override these variables for another concurrent queue.
CARLA_PORT="${EVAL_CARLA_PORT:-$((14000 + PHYSICAL_GPU * 20))}"
CARLA_STREAMING_PORT="${EVAL_CARLA_STREAMING_PORT:-$((CARLA_PORT + 1))}"
TM_PORT="${EVAL_TM_PORT:-$((CARLA_PORT + 6000))}"
X_DISPLAY_NUM="${EVAL_X_DISPLAY_NUM:-$((600 + PHYSICAL_GPU))}"

ROUTES=(
  signalized-junction-left-turn-001
  enter-actor-flow-004
  non-signalized-junction-right-turn-001
  non-signalized-junction-left-turn-enter-flow-002
  signalized-junction-left-turn-enter-flow-003
  non-signalized-junction-left-turn-002
  signalized-junction-right-turn-004
  pedestrian-crossing-004
  # Sheet shorthand: "stat-in-001"; canonical registry name:
  vanilla-signalized-turn-encounter-red-light-002
  crossing-bicycle-flow-004
  vanilla-signalized-turn-encounter-green-light-004
  # Sheet label omits the registry's "r" in "merger".
  merger-into-slow-traffic-v2-001
  vehicle-turning-route-pedestrian-005
  vehicle-opens-door-two-ways-005
  sequential-lane-change-005
  interurban-actor-flow-004
  generalization-construction-permutations-1019
  generalization-custom-obstacles-1020
  generalization-pedestrians-on-road-1085
  generalization-construction-pedestrian-1011
  generalization-pedestrian-crowd-1069
  generalization-custom-obstacles-1024
  generalization-fully-blocked-1032
  generalization-hard-brake-1036
  generalization-bad-parking-1009
  generalization-pedestrian-other-blocker-1072
  generalization-right-construction-1093
  generalization-right-of-way-1056
  generalization-wall-1095
  generalization-image-on-object-1041
  generalization-obscured-stop-1048
  generalization-bad-parking-1004
  generalization-animals-1083
  generalization-wall-1097
)

read -r -a SEED_LIST <<< "$SEEDS"
if (( ${#SEED_LIST[@]} == 0 )); then
  echo 'SEEDS must contain at least one seed.' >&2
  exit 2
fi
for seed in "${SEED_LIST[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
done

mkdir -p "$RUN_ROOT/logs/$EVAL_LABEL" "$RUN_ROOT/status/$EVAL_LABEL" "$RUN_ROOT/runs/$EVAL_LABEL"
[[ -f "${NONCE_FILE:-}" ]] || printf '%s\n' "$RUN_NONCE" > "$RUN_ROOT/status/$EVAL_LABEL/run_nonce"
cat > "$RUN_ROOT/eval_spec_${EVAL_LABEL}.txt" <<SPEC
created=$(date --iso-8601=seconds)
eval_label=$EVAL_LABEL
run_group=$RUN_GROUP
run_nonce=$RUN_NONCE
physical_gpu=$PHYSICAL_GPU
carla_port=$CARLA_PORT
carla_streaming_port=$CARLA_STREAMING_PORT
tm_port=$TM_PORT
x_display_num=$X_DISPLAY_NUM
seeds=${SEED_LIST[*]}
routes=${ROUTES[*]}
online_steps=10000
max_episode_steps=4000
agent_config=impls/configs/steervla_residual_eval_config.py
checkpoint=gs://cat-logs/pi05_steervla_cot_simplified_reasoning_ll_heavy/ll_heavy_unnormed_matchcrop/ll_heavy_unnormed_matchcrop_20260904_152800/6000
state_encoder=siglip_pool
residual_accel_scale=0.1
residual_steer_scale=0.1
residual_bc_beta=1.0
residual_warmup_steps=1000
residual_ramp_steps=1500
actions_per_model_query=3
actions_per_cot=5
proprio_norm=false
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
    exp_name="residual-rl-eval_${EVAL_LABEL}_sd-${seed}_${route}_${RUN_NONCE}"
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
    run_with_watchdog "$log_file" "$failure_file" env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" OGBENCH_SAVE_DIR="$run_dir" \
      ./run_carla.sh \
        --agent-config impls/configs/steervla_residual_eval_config.py \
        --route "$route" \
        --carla-seed "$seed" \
        --train-seed "$seed" \
        --eval-seeds "$eval_seeds" \
        --seed "$seed" \
        --exp-name "$exp_name" \
        --online-steps 10000 \
        --max-episode-steps 4000 \
        --train-gpu 0 \
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
