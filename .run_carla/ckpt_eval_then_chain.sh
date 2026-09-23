#!/usr/bin/env bash
# ckpt_eval_then_chain.sh — once this GPU's current sweep worker exits, run a frozen 3-seed eval of one
# saved checkpoint, record it as the route's result, then hand the GPU to gpu_chain.sh.
#
#   GPU=3 CARLA_PORT=18000 TM_PORT=18100 DISPLAY_NUM=997 \
#   ROUTE=highway-exit-002 RUN_DIR=/raid/.../<run> CKPT_STEP=6000 \
#   AGENT_CFG=impls/configs/steervla_cast_relabel_hl200x10_adaptive_config.py \
#   STOP_NOTE="..." QUEUES="<sweep> <sweep>" ./.run_carla/ckpt_eval_then_chain.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:?}"; CARLA_PORT="${CARLA_PORT:?}"; TM_PORT="${TM_PORT:?}"; DISPLAY_NUM="${DISPLAY_NUM:?}"
ROUTE="${ROUTE:?}"; RUN_DIR="${RUN_DIR:?}"; CKPT_STEP="${CKPT_STEP:?}"; AGENT_CFG="${AGENT_CFG:?}"
STOP_NOTE="${STOP_NOTE:-stopped manually; checkpoint ${CKPT_STEP} evaluated frozen}"
QUEUES="${QUEUES:-}"
RUN_GROUP="$(basename "$(dirname "$RUN_DIR")")"
LOG=".run_carla/jobs/ckpt_eval_gpu${GPU}.log"
log() { echo "[ckpt-eval gpu${GPU} $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
: "${GEMINI_API_KEY:?GEMINI_API_KEY must be exported}"
export UV_PROJECT_ENVIRONMENT=/home/cglossop/ogbench-carla/.venv
export PYTHONPATH="$PWD:/raid/users/cglossop/ogbench-simlingo-deps"
unset EVAL_EVERY CARLA_0915_ROOT CARLA_0915_PYTHON
CK="$RUN_DIR/checkpoints/$CKPT_STEP"
[ -d "$CK" ] || { log "ABORT: no checkpoint at $CK"; exit 1; }

worker_alive() {
  local p
  for p in $(ps -eo pid=,args= | awk '$2=="bash" && $3 ~ /(sweep|llheavy|f2d)_extra_worker\.sh$/ {print $1}'); do
    [ "$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^GPU=//p')" = "$GPU" ] && return 0
  done
  return 1
}
log "armed: $ROUTE checkpoint $CKPT_STEP on gpu $GPU once its current worker exits; then queues: ${QUEUES:-none}"
while worker_alive; do sleep 120; done
while :; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
  [ "${u:-999999}" -lt 20000 ] && break; sleep 60
done
pgrep -u "$USER" -f "[X]vfb :${DISPLAY_NUM}( |\$)" >/dev/null || rm -f "/tmp/.X${DISPLAY_NUM}-lock"
mkdir -p "$RUN_DIR/ckpt_evals/$CKPT_STEP"
EVAL_LOG=".run_carla/jobs/$RUN_GROUP/${ROUTE}_ckpt${CKPT_STEP}_eval.log"
log "frozen eval of $CK (log $EVAL_LOG)"
CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_API_KEY="$(cat /home/cglossop/.wandb_school_key)" WANDB_ENTITY=catherineglossop \
CARLA_ROOT=/home/cglossop/carla OGBENCH_SAVE_DIR=/raid/users/cglossop/carla_exps \
  timeout -k 120 8h bash run_carla.sh \
    --route "$ROUTE" --train-gpu 0 --hl-gpu 0 --render-adapter "$GPU" \
    --carla-port "$CARLA_PORT" --carla-streaming-port $((CARLA_PORT + 1)) --tm-port "$TM_PORT" \
    --x-display-num "$DISPLAY_NUM" --agent-config "$AGENT_CFG" --train-mode rl --critic-mode none \
    --online-steps 40000 --save-buffer false --save-video-local true \
    --eval-mode --frozen-eval true --enable-updates false \
    --steervla-checkpoint "$CK" --frozen-eval-out "$RUN_DIR/ckpt_evals/$CKPT_STEP" \
    --post-stop-eval-episodes 3 --seed 0 --carla-seed 0 --train-seed 0 --eval-seeds 1001,1002,1003 \
    --fixed-carla-seed true --cot-temperature 0.1 \
    --run-group "${RUN_GROUP}_frozeneval" --wandb-mode online --max-retries 2 \
    >> "$EVAL_LOG" 2>&1
log "frozen eval exited ($?)"
# main_carla runs in its own process group: never leave it (or its simulator) on the GPU.
pkill -u "$USER" -9 -f "[m]ain_carla\.py .*--route=${ROUTE} .*--frozen_eval=true" 2>/dev/null
pkill -u "$USER" -9 -f "[c]arla-rpc-port=${CARLA_PORT}( |\$)" 2>/dev/null
pkill -u "$USER" -f "[X]vfb :${DISPLAY_NUM}( |\$)" 2>/dev/null; rm -f "/tmp/.X${DISPLAY_NUM}-lock"

python3 - "$RUN_DIR" "$CK" "$CKPT_STEP" "$STOP_NOTE" <<'PY' >> "$LOG" 2>&1
import json, sys
from pathlib import Path
run, ck, step, note = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
fe = run / "ckpt_evals" / step / "run_summary_frozen_eval.json"
if not fe.exists():
    print("  no frozen-eval summary written; nothing recorded"); sys.exit()
d = json.loads(fe.read_text()); ev = d.get("eval", [])
if len(ev) < 3:
    print(f"  only {len(ev)} eval episodes; not writing run_summary.json"); sys.exit()
if (run / "run_summary.json").exists():
    print("  run_summary.json already exists; leaving it"); sys.exit()
summary = {
    "route": d.get("route"),
    "seeds": d.get("seeds", {"carla_seed": 0, "train_seed": 0}),
    "training": {"final_driving_score": None, "stop_reason": note, "hl_updates_applied": None,
                 "env_steps": int(step), "final_checkpoint": ck},
    "eval": ev,
    "eval_mean_driving_score": sum(e["driving_score"] for e in ev) / len(ev),
    "source": f"ckpt_evals/{step}/run_summary_frozen_eval.json (frozen eval of checkpoint {step})",
}
(run / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("  wrote run_summary.json:", [e["driving_score"] for e in ev])
PY

if [ -n "$QUEUES" ]; then
  log "handing gpu $GPU to gpu_chain.sh: $QUEUES"
  exec env GPU="$GPU" CARLA_PORT="$CARLA_PORT" TM_PORT="$TM_PORT" DISPLAY_NUM="$DISPLAY_NUM" \
    QUEUES="$QUEUES" ./.run_carla/gpu_chain.sh
fi
log "done"
