#!/usr/bin/env bash
# Short GRPO timing run: measure batched candidate generation and Gemini scoring without policy updates.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU="${1:-${GRPO_TIMING_GPU:-0}}"
ROUTE="${2:-${GRPO_TIMING_ROUTE:-signalized-junction-left-turn-001}}"
HL_GPU="${GRPO_HL_GPU:-1}"
STEPS="${GRPO_TIMING_STEPS:-1000}"
RUN_ROOT="${GRPO_TIMING_ROOT:-/raid/users/${USER}/carla_exps/evals/grpo_timing}"
FAIL2DRIVE_CARLA_ROOT="${EVAL_FAIL2DRIVE_CARLA_ROOT:-}"
WANDB_MODE="disabled"
RUN_GROUP="${GRPO_TIMING_RUN_GROUP:-SimLingoSteerVLAGRPOGeminiTiming}"

[[ -n "${GEMINI_API_KEY:-}" ]] || { echo 'GEMINI_API_KEY must be exported for this timing run.' >&2; exit 1; }
[[ "$GPU" =~ ^[0-9]+$ && "$HL_GPU" =~ ^[0-9]+$ ]] || { echo 'GPU ids must be integers.' >&2; exit 2; }
[[ "$GPU" != "$HL_GPU" ]] || { echo 'GRPO_HL_GPU must differ from the policy/CARLA GPU.' >&2; exit 2; }
[[ "$STEPS" =~ ^[0-9]+$ ]] && (( STEPS >= 1 && STEPS <= 1000 )) || {
  echo 'GRPO_TIMING_STEPS must be between 1 and 1000.' >&2; exit 2;
}

# Dedicated timing-run ports. Override these if another process already owns the range.
CARLA_PORT="${GRPO_TIMING_CARLA_PORT:-$((28000 + GPU * 20))}"
STREAMING_PORT="${GRPO_TIMING_STREAMING_PORT:-$((CARLA_PORT + 1))}"
TM_PORT="${GRPO_TIMING_TM_PORT:-$((CARLA_PORT + 6000))}"
DISPLAY_NUM="${GRPO_TIMING_DISPLAY_NUM:-$((1200 + GPU))}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${RANDOM}${RANDOM}"
RUN_NAME="simlingo-steervla-grpo-gemini-timing_${RUN_ID}_${ROUTE}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
LOG_FILE="$RUN_DIR/run.log"
mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/run_spec.txt" <<SPEC
route=$ROUTE
steps=$STEPS
scorer=gemini
updates=false
warmup_steps=0
score_every=6
group_size=4
policy_gpu=$GPU
hl_gpu=$HL_GPU
carla_port=$CARLA_PORT
streaming_port=$STREAMING_PORT
tm_port=$TM_PORT
display_num=$DISPLAY_NUM
run_group=$RUN_GROUP
run_name=$RUN_NAME
wandb_mode=$WANDB_MODE
SPEC

echo "[grpo_vlm_timing] route=$ROUTE steps=$STEPS scorer=gemini updates=false"
echo "[grpo_vlm_timing] GPU=$GPU HL_GPU=$HL_GPU ports=$CARLA_PORT/$STREAMING_PORT/$TM_PORT display=:$DISPLAY_NUM"
echo "[grpo_vlm_timing] W&B disabled; timings are printed and saved locally."
echo "[grpo_vlm_timing] terminal output is also saved to $LOG_FILE"

set -o pipefail
env CUDA_VISIBLE_DEVICES="$GPU,$HL_GPU" \
  FAIL2DRIVE_CARLA_ROOT="$FAIL2DRIVE_CARLA_ROOT" \
  OGBENCH_SAVE_DIR="$RUN_DIR/run" \
  ./run_carla.sh \
    --agent-config impls/configs/steervla_grpo_config.py \
    --route "$ROUTE" \
    --carla-seed 0 --train-seed 0 --seed 0 \
    --exp-name "$RUN_NAME" --run-group "$RUN_GROUP" \
    --online-steps "$STEPS" --max-episode-steps 4000 \
    --train-gpu 0 --hl-gpu 1 --render-adapter "$GPU" \
    --carla-port "$CARLA_PORT" --carla-streaming-port "$STREAMING_PORT" \
    --tm-port "$TM_PORT" --x-display-num "$DISPLAY_NUM" \
    --train-mode grpo_hl --critic-mode none \
    --grpo-scorer gemini --grpo-warmup 0 --enable-updates false \
    --save-buffer false --save-video-local false --wandb-mode "$WANDB_MODE" \
    --max-retries 0 2>&1 | tee "$LOG_FILE"
