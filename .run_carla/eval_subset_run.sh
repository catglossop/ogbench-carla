#!/usr/bin/env bash
# eval_subset_run.sh ROUTE GPU SLOT [extra run_carla.sh args...]
#
# One reportable HL-DAgger / CAST-relabel run on the eval subset, with --eval-mode on.
#
# THE SEED CONTRACT. A run's seed label fixes BOTH seeds, so every "seed 0" run on the eval
# subset -- whatever route, whatever day -- uses the same (carla_seed, train_seed) pair and is
# directly comparable with every other seed-0 run. Deriving them here rather than passing them by
# hand is the point: a pair typed per-invocation drifts, and two runs that disagree about their
# seeds cannot be compared even though both say "seed 0".
#
#   seed N  ->  carla_seed = N,  train_seed = N,  eval_seeds = N+1001, N+1002, N+1003
#
# carla_seed drives the simulator (traffic manager, scenario actors, env.reset). train_seed drives
# the model (JAX PRNG, numpy/random, and the actor's CoT / action / noise sampling). The eval
# episodes replay carla_seed and vary only the model seed, so their spread measures the policy
# rather than the scenario. All of it is recorded in run_summary.json by --eval-mode.
#
#   ./.run_carla/eval_subset_run.sh signalized-junction-left-turn-001 5 0
#   SEED=1 ./.run_carla/eval_subset_run.sh generalization-wall-1095 6 1
set -uo pipefail
cd /home/cglossop/ogbench-carla

ROUTE="${1:?usage: eval_subset_run.sh ROUTE GPU SLOT [extra args]}"
GPU="${2:?need a GPU index}"
SLOT="${3:?need a slot index (0,1,... -> distinct ports/displays)}"
shift 3

SEED="${SEED:-0}"
CARLA_SEED="$SEED"
TRAIN_SEED="$SEED"
EVAL_SEEDS="$((SEED + 1001)),$((SEED + 1002)),$((SEED + 1003))"

AGENT_CFG="${AGENT_CFG:-impls/configs/steervla_cast_relabel_hl16_adaptive_config.py}"
ONLINE_STEPS="${ONLINE_STEPS:-10000}"
MAX_HL_UPDATES="${MAX_HL_UPDATES:-150}"
STOP_ON_SCORE="${STOP_ON_SCORE:-100}"
STOP_SCORE_STREAK="${STOP_SCORE_STREAK:-}"
FIXED_CARLA_SEED="${FIXED_CARLA_SEED:-}"
UPDATES_AFTER_SCORE="${UPDATES_AFTER_SCORE:-20}"
RUN_GROUP="${RUN_GROUP:-hl16_adaptive_evalmode}"

CARLA_PORT=$((16400 + SLOT * 20))
TM_PORT=$((16500 + SLOT * 20))
DISPLAY_NUM=$((940 + SLOT))

export CARLA_ROOT="${CARLA_ROOT:-/home/cglossop/carla}"
export OGBENCH_SAVE_DIR="${OGBENCH_SAVE_DIR:-/home/cglossop/carla_exps}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# W&B identity: the school account. Do NOT fall through to ~/.netrc, which is catglossop.
# Key is consumed inline and never echoed; the entity must be explicit, since the key alone
# silently lands the run under the netrc account. "catherineglossop" has no underscore.
export WANDB_API_KEY="$(cat /home/cglossop/.wandb_school_key)"
export WANDB_ENTITY=catherineglossop
: "${GEMINI_API_KEY:?GEMINI_API_KEY must be exported -- CAST relabel is a Gemini client}"

for p in "$CARLA_PORT" $((CARLA_PORT + 1)) "$TM_PORT"; do
  if ss -ltn 2>/dev/null | grep -q ":${p} "; then
    echo "[eval_run] ABORT: port $p already listening" >&2; exit 1; fi
done
# A lock file is only a real conflict if something is actually holding the display. After a
# killed run the lock outlives the Xvfb, and a bare -e test then aborts every subsequent route
# in the sweep in about a second each -- which is exactly how 7 routes were lost on 2026-09-10.
# Treat "lock exists but no live Xvfb/CARLA on our display" as stale and reclaim it.
LOCK="/tmp/.X${DISPLAY_NUM}-lock"
if [ -e "$LOCK" ]; then
  if pgrep -u "$USER" -f "Xvfb :${DISPLAY_NUM}\b" >/dev/null 2>&1 \
     || pgrep -u "$USER" -f "carla-rpc-port=${CARLA_PORT}\b" >/dev/null 2>&1; then
    echo "[eval_run] ABORT: display :$DISPLAY_NUM genuinely in use" >&2; exit 1
  fi
  if [ -O "$LOCK" ]; then
    echo "[eval_run] reclaiming stale lock $LOCK (no live Xvfb/CARLA)" >&2
    rm -f "$LOCK" || { echo "[eval_run] ABORT: could not remove stale $LOCK" >&2; exit 1; }
  else
    echo "[eval_run] ABORT: $LOCK held by another user" >&2; exit 1
  fi
fi

echo "[eval_run] route=$ROUTE gpu=$GPU slot=$SLOT"
echo "[eval_run] seed=$SEED -> carla_seed=$CARLA_SEED train_seed=$TRAIN_SEED eval_seeds=$EVAL_SEEDS"
echo "[eval_run] rpc=$CARLA_PORT tm=$TM_PORT display=:$DISPLAY_NUM group=$RUN_GROUP"

# CUDA_VISIBLE_DEVICES masks to one physical card; --train-gpu/--hl-gpu index INTO that mask
# (always 0) while --render-adapter stays physical (CARLA gets a scrubbed env).
CUDA_VISIBLE_DEVICES="$GPU" exec bash run_carla.sh \
  --route "$ROUTE" --train-gpu 0 --hl-gpu 0 --render-adapter "$GPU" \
  --carla-port "$CARLA_PORT" --carla-streaming-port $((CARLA_PORT + 1)) \
  --tm-port "$TM_PORT" --x-display-num "$DISPLAY_NUM" \
  --agent-config "$AGENT_CFG" \
  --train-mode rl --critic-mode none \
  --online-steps "$ONLINE_STEPS" --save-buffer false --save-video-local true \
  --eval-mode \
  --seed "$SEED" --carla-seed "$CARLA_SEED" --train-seed "$TRAIN_SEED" --eval-seeds "$EVAL_SEEDS" \
  --run-group "$RUN_GROUP" --wandb-mode online \
  "$@" \
  ${STOP_SCORE_STREAK:+--stop-score-streak "$STOP_SCORE_STREAK"} \
  ${FIXED_CARLA_SEED:+--fixed-carla-seed "$FIXED_CARLA_SEED"} \
  -- --max_hl_updates="$MAX_HL_UPDATES" \
     --stop_on_driving_score="$STOP_ON_SCORE" \
     --updates_after_driving_score="$UPDATES_AFTER_SCORE"
