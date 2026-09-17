#!/usr/bin/env bash
# frozen_eval_run.sh ROUTE GPU SLOT CHECKPOINT CARLA_SEED TRAIN_SEED ORIG_RUN_DIR
#
# Reload a finished run's final checkpoint and roll it out frozen for extra eval seeds.
# No gradient updates (--enable-updates false AND --frozen-eval, belt and braces), no CAST
# review, no checkpointing. The summary is written into ORIG_RUN_DIR next to the original
# run_summary.json, which is never touched.
set -uo pipefail
cd /home/cglossop/ogbench-carla

ROUTE="${1:?route}"; GPU="${2:?gpu}"; SLOT="${3:?slot}"
CKPT="${4:?checkpoint}"; CARLA_SEED="${5:?carla seed}"; TRAIN_SEED="${6:?train seed}"
ORIG_DIR="${7:?original run dir}"

EVAL_SEEDS="${EVAL_SEEDS:-1004,1005,1006}"
AGENT_CFG="${AGENT_CFG:-impls/configs/steervla_cast_relabel_hl200x10_adaptive_config.py}"
RUN_GROUP="${RUN_GROUP:-hl200x10_b2dsubset_seed0}"
ONLINE_STEPS="${ONLINE_STEPS:-6000}"
N_EVAL=$(awk -F, '{print NF}' <<< "$EVAL_SEEDS")

CARLA_PORT=$((16400 + SLOT * 20)); TM_PORT=$((16500 + SLOT * 20)); DISPLAY_NUM=$((940 + SLOT))

export CARLA_ROOT="${CARLA_ROOT:-/home/cglossop/carla}"
export OGBENCH_SAVE_DIR="${OGBENCH_SAVE_DIR:-/raid/users/cglossop/carla_exps}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_API_KEY="$(cat /home/cglossop/.wandb_school_key)"
export WANDB_ENTITY=catherineglossop
: "${GEMINI_API_KEY:?GEMINI_API_KEY must be exported}"

for p in "$CARLA_PORT" $((CARLA_PORT + 1)) "$TM_PORT"; do
  if ss -ltn 2>/dev/null | grep -q ":${p} "; then
    echo "[frozen_run] ABORT: port $p already listening" >&2; exit 1; fi
done
LOCK="/tmp/.X${DISPLAY_NUM}-lock"
if [ -e "$LOCK" ]; then
  if pgrep -u "$USER" -f "Xvfb :${DISPLAY_NUM}\b" >/dev/null 2>&1 \
     || pgrep -u "$USER" -f "carla-rpc-port=${CARLA_PORT}\b" >/dev/null 2>&1; then
    echo "[frozen_run] ABORT: display :$DISPLAY_NUM genuinely in use" >&2; exit 1
  fi
  if [ -O "$LOCK" ]; then
    echo "[frozen_run] reclaiming stale lock $LOCK" >&2; rm -f "$LOCK"
  else
    echo "[frozen_run] ABORT: $LOCK held by another user" >&2; exit 1
  fi
fi

[ -d "$CKPT" ] || { echo "[frozen_run] ABORT: checkpoint not a directory: $CKPT" >&2; exit 1; }
[ -d "$ORIG_DIR" ] || { echo "[frozen_run] ABORT: run dir missing: $ORIG_DIR" >&2; exit 1; }

echo "[frozen_run] route=$ROUTE gpu=$GPU slot=$SLOT rpc=$CARLA_PORT display=:$DISPLAY_NUM"
echo "[frozen_run] ckpt=$CKPT"
echo "[frozen_run] carla_seed=$CARLA_SEED train_seed=$TRAIN_SEED eval_seeds=$EVAL_SEEDS (${N_EVAL} episodes)"
echo "[frozen_run] summary -> ${ORIG_DIR}/run_summary_frozen_eval.json"

CUDA_VISIBLE_DEVICES="$GPU" exec bash run_carla.sh \
  --route "$ROUTE" --train-gpu 0 --hl-gpu 0 --render-adapter "$GPU" \
  --carla-port "$CARLA_PORT" --carla-streaming-port $((CARLA_PORT + 1)) \
  --tm-port "$TM_PORT" --x-display-num "$DISPLAY_NUM" \
  --agent-config "$AGENT_CFG" \
  --train-mode rl --critic-mode none \
  --online-steps "$ONLINE_STEPS" --save-buffer false --save-video-local true \
  --eval-mode --frozen-eval true --enable-updates false \
  --steervla-checkpoint "$CKPT" \
  --frozen-eval-out "$ORIG_DIR" \
  --post-stop-eval-episodes "$N_EVAL" \
  --seed "$TRAIN_SEED" --carla-seed "$CARLA_SEED" --train-seed "$TRAIN_SEED" \
  --eval-seeds "$EVAL_SEEDS" \
  --run-group "${RUN_GROUP}_frozeneval" --wandb-mode online
