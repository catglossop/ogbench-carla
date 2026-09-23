#!/usr/bin/env bash
# ckpt_ladder_run.sh -- train once, checkpointing often, then evaluate EVERY checkpoint.
#
# Same update strategy as coldcot_seqlane005 (cot_temperature=0.1, no extra updates after the
# first 100 DS), but checkpoints every 500 env steps instead of 2000 and keeps all of them.
# Afterwards each checkpoint is rolled out frozen on the same 3 eval seeds.
#
# The point: a run that reports 100 DS at its stopping checkpoint and then evaluates at 30 tells
# you nothing about WHERE it went wrong. A score-vs-checkpoint ladder does -- it separates "never
# learned it" from "learned it and then lost it", and shows whether the stop fired on a real
# improvement or on a lucky episode at a checkpoint that was already past its best.
#
# Waits for a free GPU rather than starting immediately, so it can be queued behind the b2d sweep.
set -uo pipefail
cd /home/cglossop/ogbench-carla

ROUTE="${ROUTE:-sequential-lane-change-005}"
GPU="${GPU:-5}"
SLOT="${SLOT:-3}"                 # port 16460, display :943 -- disjoint from slots 0/1/2
EXP="${EXP:-ckptladder_seqlane005}"
SEED="${SEED:-0}"
CKPT_EVERY="${CKPT_EVERY:-500}"
EVAL_SEEDS="${EVAL_SEEDS:-1001,1002,1003}"
AGENT_CFG="${AGENT_CFG:-impls/configs/steervla_cast_relabel_hl200x10_adaptive_config.py}"
SAVE_DIR="/raid/users/cglossop/experiments/${EXP}"
LOG_DIR=".run_carla/jobs/${EXP}"
mkdir -p "$LOG_DIR"
say() { echo "[ladder $(date +%H:%M:%S)] $*" | tee -a "${LOG_DIR}/ladder.log"; }

# ---- 1. wait for the GPU to be free of MY jobs (never touch anyone else's) ----
say "waiting for gpu ${GPU} to free up"
while :; do
  busy=$(ps -eo user,args --no-headers | awk -v u="$USER" '$1==u' \
         | grep "[r]un_carla.sh" | grep -c -- "--render-adapter ${GPU}")
  [ "$busy" -eq 0 ] && break
  sleep 120
done
# Someone else may hold it even if I don't; check real memory before committing.
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
  [ "${used:-999999}" -lt 20000 ] && break
  say "gpu ${GPU} still holds ${used} MiB (someone else); waiting"
  sleep 120
done
say "gpu ${GPU} free; starting training"

# ---- 2. train, checkpointing every CKPT_EVERY steps, keeping all of them ----
TRAIN_LOG="${LOG_DIR}/train.log"
SEED="$SEED" AGENT_CFG="$AGENT_CFG" RUN_GROUP="$EXP" OGBENCH_SAVE_DIR="$SAVE_DIR" \
UPDATES_AFTER_SCORE=0 MAX_HL_UPDATES=150 STOP_ON_SCORE=100 ONLINE_STEPS=10000 \
  setsid ./.run_carla/eval_subset_run.sh "$ROUTE" "$GPU" "$SLOT" \
    --cot-temperature 0.1 \
    --hl-ckpt-every "$CKPT_EVERY" \
    --hl-ckpt-keep-last 0 \
  > "$TRAIN_LOG" 2>&1
say "training exited ($?)"

RUN_DIR=$(dirname "$(find "$SAVE_DIR" -name run_summary.json 2>/dev/null | head -1)")
if [ ! -d "${RUN_DIR:-}" ]; then
  say "no run_summary.json under $SAVE_DIR -- training did not finish; nothing to evaluate"; exit 1
fi
CKPT_DIR="${RUN_DIR}/checkpoints"
say "run dir: $RUN_DIR"

# ---- 3. frozen-eval every checkpoint, oldest first ----
mapfile -t STEPS < <(ls -1 "$CKPT_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n)
say "checkpoints to evaluate: ${#STEPS[@]}  (${STEPS[*]})"
for st in "${STEPS[@]}"; do
  OUT="${RUN_DIR}/ckpt_evals/${st}"
  if [ -f "${OUT}/run_summary_frozen_eval.json" ]; then say "ckpt ${st}: already evaluated"; continue; fi
  mkdir -p "$OUT"
  say "ckpt ${st}: evaluating on seeds ${EVAL_SEEDS}"
  EVAL_SEEDS="$EVAL_SEEDS" AGENT_CFG="$AGENT_CFG" RUN_GROUP="${EXP}_ckpteval" \
  OGBENCH_SAVE_DIR="$SAVE_DIR" ONLINE_STEPS=6000 \
    setsid ./.run_carla/frozen_eval_run.sh "$ROUTE" "$GPU" "$SLOT" \
      "${CKPT_DIR}/${st}" "$SEED" "$SEED" "$OUT" \
    > "${LOG_DIR}/eval_ckpt_${st}.log" 2>&1
  say "ckpt ${st}: exit $?"
  ./.run_carla/ckpt_ladder_results.sh >/dev/null 2>&1 || true
done
./.run_carla/ckpt_ladder_results.sh || true
say "ladder complete"
