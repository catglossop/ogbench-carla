#!/usr/bin/env bash
# Restartable SimLingo SteerVLA GRPO queue: train each route, then frozen-evaluate it three times.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PHYSICAL_GPU="${1:-${EVAL_GPU:-0}}"
HL_PHYSICAL_GPU="${GRPO_HL_GPU:-1}"
RUN_ROOT="${2:-${EVAL_ROOT:-/raid/users/${USER}/carla_exps/evals/grpo}}"
GRPO_SCORER="${GRPO_SCORER:-simulator}"
GRPO_REWARD_LABEL="${GRPO_REWARD_LABEL:-$GRPO_SCORER}"
EVAL_LABEL="${EVAL_LABEL:-simlingo-steervla-grpo-${GRPO_REWARD_LABEL}-b2d-f2d-evalmode-v1-20261116}"
RUN_GROUP="${EVAL_RUN_GROUP:-SimLingoSteerVLAGRPO${GRPO_REWARD_LABEL^}B2DF2DEvalModeV1}"
# Keep the caller's local CARLA installation (with install_f2d_content.sh assets) by default.
FAIL2DRIVE_CARLA_ROOT="${EVAL_FAIL2DRIVE_CARLA_ROOT:-}"
SEEDS="${SEEDS:-0}"
MAX_RETRIES="${MAX_RETRIES:-50}"
WANDB_MODE="${WANDB_MODE:-online}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-true}"
DRY_RUN="${DRY_RUN:-0}"

[[ "$GRPO_SCORER" == simulator || "$GRPO_SCORER" == gemini ]] || {
  echo 'GRPO_SCORER must be simulator or gemini.' >&2; exit 2;
}
[[ "$GRPO_REWARD_LABEL" =~ ^[a-zA-Z0-9_-]+$ ]] || {
  echo 'GRPO_REWARD_LABEL may contain only letters, numbers, underscores, and hyphens.' >&2; exit 2;
}
if [[ "$GRPO_SCORER" == gemini && -z "${GEMINI_API_KEY:-}" ]]; then
  echo 'GEMINI_API_KEY must be exported when GRPO_SCORER=gemini.' >&2
  exit 1
fi
[[ "$PHYSICAL_GPU" != "$HL_PHYSICAL_GPU" ]] || { echo 'GRPO_HL_GPU must differ from the policy/CARLA GPU.' >&2; exit 2; }

if [[ -n "${EVAL_RUN_NONCE:-}" ]]; then RUN_NONCE="$EVAL_RUN_NONCE"; else printf -v RUN_NONCE '%05d%05d' "$RANDOM" "$RANDOM"; fi
[[ "$RUN_NONCE" =~ ^[0-9]+$ ]] || { echo 'EVAL_RUN_NONCE must be numeric.' >&2; exit 2; }
read -r -a SEED_LIST <<< "$SEEDS"
for seed in "${SEED_LIST[@]}"; do [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }; done

# Dedicated range: residual uses 16000+, YAY uses 18000+.
CARLA_PORT="${EVAL_CARLA_PORT:-$((20000 + PHYSICAL_GPU * 20))}"
CARLA_STREAMING_PORT="${EVAL_CARLA_STREAMING_PORT:-$((CARLA_PORT + 1))}"
TM_PORT="${EVAL_TM_PORT:-$((CARLA_PORT + 6000))}"
X_DISPLAY_NUM="${EVAL_X_DISPLAY_NUM:-$((900 + PHYSICAL_GPU))}"

ROUTES=(
  signalized-junction-left-turn-001 enter-actor-flow-004 non-signalized-junction-right-turn-003
  non-signalized-junction-left-turn-enter-flow-002 signalized-junction-left-turn-enter-flow-001
  non-signalized-junction-left-turn-001 signalized-junction-right-turn-004 pedestrian-crossing-004
  t_-junction-002 crossing-bicycle-flow-004 hazard-at-side-lane-002 merger-into-slow-traffic-v2-005
  vehicle-turning-route-pedestrian-003 highway-exit-002 accident-two-ways-002 interurban-actor-flow-004
  generalization-construction-permutations-1019 generalization-custom-obstacles-1020
  generalization-pedestrians-on-road-1085 generalization-construction-pedestrian-1011
  generalization-pedestrian-crowd-1069 generalization-custom-obstacles-1024 generalization-fully-blocked-1032
  generalization-hard-brake-1036 generalization-pedestrian-other-blocker-1072
  generalization-right-construction-1094 generalization-right-of-way-1056 generalization-image-on-object-1041
  generalization-obscured-stop-1046 generalization-bad-parking-1004 generalization-animals-1076
  generalization-wall-1097
)

mkdir -p "$RUN_ROOT/logs/$EVAL_LABEL" "$RUN_ROOT/status/$EVAL_LABEL" "$RUN_ROOT/runs/$EVAL_LABEL"
cat > "$RUN_ROOT/eval_spec_${EVAL_LABEL}.txt" <<SPEC
created=$(date --iso-8601=seconds)
run_group=$RUN_GROUP
scorer=$GRPO_SCORER
reward_label=$GRPO_REWARD_LABEL
run_nonce=$RUN_NONCE
policy_gpu=$PHYSICAL_GPU
hl_gpu=$HL_PHYSICAL_GPU
carla_port=$CARLA_PORT
tm_port=$TM_PORT
x_display_num=$X_DISPLAY_NUM
seeds=${SEED_LIST[*]}
routes=${ROUTES[*]}
agent_config=impls/configs/steervla_grpo_config.py
base_policy=SimLingo_SteerVLA_HL_plus_frozen_SimLingo_LL
hl_checkpoint=/raid/users/celine/steervla-ckpts/2026_05_24_06_52_33_simlingo_seed1_bellman/checkpoints/epoch=019.ckpt
ll_checkpoint=/raid/users/celine/steervla-ckpts/2026_05_23_21_39_41_simlingo_ll_vla_meta_conditioned/checkpoints/epoch=029.ckpt
online_steps=10000
eval_episodes=3
fail2drive_carla_root=${FAIL2DRIVE_CARLA_ROOT:-<current CARLA_ROOT>}
SPEC

WATCHDOG_GRACE="${EVAL_WATCHDOG_GRACE:-600}"
STALL_SECS="${EVAL_STALL_SECS:-900}"
STALL_STRIKES="${EVAL_STALL_STRIKES:-2}"
run_with_watchdog() {
  local log_file="$1"; shift
  : > "$log_file"
  setsid "$@" > >(tee "$log_file") 2>&1 & local run_pid=$!
  (
    sleep "$WATCHDOG_GRACE"; local stalls=0 age
    while kill -0 "$run_pid" 2>/dev/null; do
      age=$(( $(date +%s) - $(stat -c %Y "$log_file" 2>/dev/null || date +%s) ))
      if [[ "$age" -ge "$STALL_SECS" ]]; then stalls=$((stalls + 1)); else stalls=0; fi
      if [[ "$stalls" -ge "$STALL_STRIKES" ]]; then
        echo "[watchdog] aborting pid $run_pid: no log progress for ${age}s" >> "$log_file"
        kill -TERM -"$run_pid" 2>/dev/null || true; sleep 15; kill -KILL -"$run_pid" 2>/dev/null || true
        break
      fi
      sleep 60
    done
  ) & local watchdog_pid=$!
  wait "$run_pid"; local status=$?
  kill "$watchdog_pid" 2>/dev/null || true; wait "$watchdog_pid" 2>/dev/null || true
  return "$status"
}

total=$(( ${#ROUTES[@]} * ${#SEED_LIST[@]} )); index=0
for seed in "${SEED_LIST[@]}"; do
  for route in "${ROUTES[@]}"; do
    index=$((index + 1)); tag="${route}__seed-${seed}"
    done_file="$RUN_ROOT/status/$EVAL_LABEL/${tag}.done"
    log_file="$RUN_ROOT/logs/$EVAL_LABEL/${tag}.log"
    exp_name="simlingo-steervla-grpo-${GRPO_REWARD_LABEL}-evalmode-v1_${RUN_NONCE}_sd-${seed}_${route}"
    if [[ "$SKIP_COMPLETED" == 1 && -f "$done_file" ]]; then echo "[$index/$total] SKIP completed: $tag"; continue; fi
    echo "[$index/$total] START: $tag (wandb_name=$exp_name)"
    [[ "$DRY_RUN" == 1 ]] && continue
    run_dir="$RUN_ROOT/runs/$EVAL_LABEL/$tag"; mkdir -p "$run_dir"
    eval_seeds="$((seed + 1001)),$((seed + 1002)),$((seed + 1003))"
    set +e
    run_with_watchdog "$log_file" env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU,$HL_PHYSICAL_GPU" \
      XLA_PYTHON_CLIENT_MEM_FRACTION="${GRPO_XLA_MEM_FRACTION:-0.30}" \
      FAIL2DRIVE_CARLA_ROOT="$FAIL2DRIVE_CARLA_ROOT" OGBENCH_SAVE_DIR="$run_dir" \
      ./run_carla.sh --agent-config impls/configs/steervla_grpo_config.py --route "$route" \
      --carla-seed "$seed" --train-seed "$seed" --eval-seeds "$eval_seeds" --seed "$seed" --exp-name "$exp_name" \
      --online-steps 10000 --max-episode-steps 4000 --train-gpu 0 --hl-gpu 1 --train-mode grpo_hl --critic-mode none \
      --render-adapter "$PHYSICAL_GPU" --carla-port "$CARLA_PORT" --carla-streaming-port "$CARLA_STREAMING_PORT" \
      --tm-port "$TM_PORT" --x-display-num "$X_DISPLAY_NUM" --run-group "$RUN_GROUP" \
      --grpo-scorer "$GRPO_SCORER" --eval-mode \
      --post-stop-eval-episodes 3 --save-buffer false --save-video-local true --wandb-mode "$WANDB_MODE" --max-retries "$MAX_RETRIES"
    status=$?; set -e
    if [[ "$status" -eq 0 ]]; then touch "$done_file"; echo "[$index/$total] DONE: $tag"; continue; fi
    printf 'timestamp=%s\nrun_exit_code=%s\n' "$(date --iso-8601=seconds)" "$status" > "$RUN_ROOT/status/$EVAL_LABEL/${tag}.failed"
    echo "[$index/$total] FAILED: $tag" >&2
    [[ "$CONTINUE_ON_FAILURE" == 1 || "$CONTINUE_ON_FAILURE" == true ]] || exit "$status"
  done
done
