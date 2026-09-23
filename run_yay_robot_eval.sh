#!/usr/bin/env bash
# Restartable B2D + F2D YAY-Robot evaluation queue.
# Each route trains the SimLingo HL planner for 10k steps, then --eval-mode freezes it
# and rolls it out for three model seeds in the same CARLA process.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PHYSICAL_GPU="${1:-${EVAL_GPU:-0}}"
HL_PHYSICAL_GPU="${EVAL_HL_GPU:-7}"
RUN_ROOT="${2:-${EVAL_ROOT:-/raid/users/${USER}/carla_exps/evals/yay_robot}}"
EVAL_LABEL="${EVAL_LABEL:-simlingo-steervla-yay-robot-b2d-f2d-remaining-v2-20260921}"
RUN_GROUP="${EVAL_RUN_GROUP:-SimLingoSteerVLAYayRobotB2DF2DRemainingV2}"
# A retry gets a fresh W&B name; completed routes still skip via their .done markers.
# Supply EVAL_RUN_NONCE=<number> only to deliberately reuse names.
if [[ -n "${EVAL_RUN_NONCE:-}" ]]; then
  RUN_NONCE="$EVAL_RUN_NONCE"
else
  printf -v RUN_NONCE '%05d%05d' "$RANDOM" "$RANDOM"
fi
[[ "$RUN_NONCE" =~ ^[0-9]+$ ]] || { echo 'EVAL_RUN_NONCE must be numeric.' >&2; exit 2; }
SEEDS="${SEEDS:-0 1 2}"
MAX_RETRIES="${MAX_RETRIES:-50}"
WANDB_MODE="${WANDB_MODE:-online}"
SAVE_BUFFER="${SAVE_BUFFER:-false}"
SAVE_VIDEO_LOCAL="${SAVE_VIDEO_LOCAL:-true}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-true}"
DRY_RUN="${DRY_RUN:-0}"
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "[yay_robot_eval] GEMINI_API_KEY must be exported; refusing to launch an uncorrected baseline." >&2
  exit 1
fi

if [[ "$PHYSICAL_GPU" == "$HL_PHYSICAL_GPU" ]]; then
  echo "[yay_robot_eval] EVAL_HL_GPU must differ from the policy/CARLA GPU." >&2
  exit 2
fi

# Validate the actual loaded HL model and one replay batch before allocating CARLA.
if [[ "${EVAL_PREFLIGHT:-1}" == "1" ]]; then
  echo "[yay_robot_eval] running SimLingo HL preflight (policy CUDA:0, HL CUDA:1)..."
  env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU,$HL_PHYSICAL_GPU" PYTHONPATH="$ROOT_DIR/impls${PYTHONPATH:+:$PYTHONPATH}" \
    .venv/bin/python impls/preflight_simlingo_yay_robot.py \
      --agent-config impls/configs/simlingo_steervla_yay_robot_train_config.py --train-gpu 0 --hl-gpu 1
fi

# Dedicated to YAY Robot, avoiding both residual-evaluation port ranges.
CARLA_PORT="${EVAL_CARLA_PORT:-$((18000 + PHYSICAL_GPU * 20))}"
CARLA_STREAMING_PORT="${EVAL_CARLA_STREAMING_PORT:-$((CARLA_PORT + 1))}"
TM_PORT="${EVAL_TM_PORT:-$((CARLA_PORT + 6000))}"
X_DISPLAY_NUM="${EVAL_X_DISPLAY_NUM:-$((700 + PHYSICAL_GPU))}"

ROUTES=(
  signalized-junction-left-turn-001
  t_-junction-002
  vehicle-turning-route-pedestrian-003
  highway-exit-002
  accident-two-ways-002
  interurban-actor-flow-004
  generalization-construction-permutations-1019
  generalization-custom-obstacles-1020
  generalization-pedestrians-on-road-1085
  generalization-construction-pedestrian-1011
  generalization-pedestrian-crowd-1069
  generalization-custom-obstacles-1024
  generalization-fully-blocked-1032
  generalization-hard-brake-1036
  generalization-pedestrian-other-blocker-1072
  generalization-right-construction-1094
  generalization-right-of-way-1056
  generalization-image-on-object-1041
  generalization-obscured-stop-1046
  generalization-bad-parking-1004
  generalization-animals-1076
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
cat > "$RUN_ROOT/eval_spec_${EVAL_LABEL}.txt" <<SPEC
created=$(date --iso-8601=seconds)
eval_label=$EVAL_LABEL
run_group=$RUN_GROUP
run_nonce=$RUN_NONCE
physical_gpu=$PHYSICAL_GPU
hl_physical_gpu=$HL_PHYSICAL_GPU
carla_port=$CARLA_PORT
carla_streaming_port=$CARLA_STREAMING_PORT
tm_port=$TM_PORT
x_display_num=$X_DISPLAY_NUM
seeds=${SEED_LIST[*]}
routes=${ROUTES[*]}
online_steps=10000
max_episode_steps=4000
agent_config=impls/configs/simlingo_steervla_yay_robot_train_config.py
hl_checkpoint=/raid/users/celine/steervla-ckpts/2026_05_24_06_52_33_simlingo_seed1_bellman/checkpoints/epoch=019.ckpt
ll_checkpoint=/raid/users/celine/steervla-ckpts/2026_05_23_21_39_41_simlingo_ll_vla_meta_conditioned/checkpoints/epoch=029.ckpt
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
    exp_name="simlingo-steervla-yay-robot-remaining_${RUN_NONCE}_sd-${seed}_${route}"
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
    run_with_watchdog "$log_file" "$failure_file" env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU,$HL_PHYSICAL_GPU" OGBENCH_SAVE_DIR="$run_dir" \
      ./run_carla.sh \
        --agent-config impls/configs/simlingo_steervla_yay_robot_train_config.py \
        --route "$route" \
        --carla-seed "$seed" \
        --train-seed "$seed" \
        --eval-seeds "$eval_seeds" \
        --seed "$seed" \
        --exp-name "$exp_name" \
        --online-steps 10000 \
        --max-episode-steps 4000 \
        --train-gpu 0 \
        --hl-gpu 1 \
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
