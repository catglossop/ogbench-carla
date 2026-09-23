#!/usr/bin/env bash
# gpu_chain.sh — give one GPU a list of sweep queues to serve in order, once its current job is done.
#
#   GPU=3 CARLA_PORT=18000 TM_PORT=18100 DISPLAY_NUM=997 \
#     QUEUES="f2dsteervla_simlingo_fixedcarla_kl005_seed0 f2dllheavy_fixedcarla_kl005_newroutes_seed0" \
#     ./.run_carla/gpu_chain.sh
#
# WAIT_FOR: none (default) | b2d_driver (the b2dllheavy sweep driver) | f2d_worker (an f2d_extra_worker
# on this GPU). Each queue is served by sweep_extra_worker.sh until it is empty; the recipe is chosen by
# sweep name: f2dsteervla_* is the SimLingo SteerVLA f2d sweep (stop score 95), *llheavy* the pi0.5
# ll-heavy sweeps (stop score 100). Everything else matches across them.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:?}"; CARLA_PORT="${CARLA_PORT:?}"; TM_PORT="${TM_PORT:?}"; DISPLAY_NUM="${DISPLAY_NUM:?}"
QUEUES="${QUEUES:?space-separated sweep names}"
WAIT_FOR="${WAIT_FOR:-none}"
LOG=".run_carla/jobs/gpu_chain_gpu${GPU}.log"
log() { echo "[chain gpu${GPU} $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
: "${GEMINI_API_KEY:?GEMINI_API_KEY must be exported}"
export UV_PROJECT_ENVIRONMENT=/home/cglossop/ogbench-carla/.venv
export PYTHONPATH="$PWD:/raid/users/cglossop/ogbench-simlingo-deps"
export F2D_CARLA_0915_ROOT=/home/cglossop/f2d_carla
export F2D_CARLA_0915_PYTHON=/home/cglossop/ogbench-carla/.venv-f2d-eval/bin/python
export RESULT_TRAIN_STEP=10000
unset EVAL_EVERY CARLA_0915_ROOT CARLA_0915_PYTHON

env_of() { tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | sed -n "s/^$2=//p"; }
blocker_alive() {
  local p
  case "$WAIT_FOR" in
    b2d_driver)
      for p in $(pgrep -u "$USER" -f "[b]2d_subset_sweep.sh --arm"); do
        [ "$(env_of "$p" SWEEP_NAME)" = b2dllheavy_fixedcarla_kl005_newroutes_seed0 ] && return 0
      done ;;
    f2d_worker)
      for p in $(pgrep -u "$USER" -f "[f]2d_extra_worker.sh"); do
        [ "$(env_of "$p" GPU)" = "$GPU" ] && return 0
      done ;;
  esac
  return 1
}
routes_file() {
  case "$1" in
    f2dsteervla_*) echo f2d_steervla_subset.txt ;;
    f2dllheavy_*) echo f2d_llheavy_newroutes.txt ;;
    b2dllheavy_*) echo b2d_llheavy_newroutes.txt ;;
  esac
}

log "armed (wait_for=$WAIT_FOR, queues: $QUEUES; rpc $CARLA_PORT, tm $TM_PORT, display :$DISPLAY_NUM)"
while blocker_alive; do sleep 120; done
log "gpu $GPU is ours"

for s in $QUEUES; do
  case "$s" in
    f2dsteervla_*) cfg=impls/configs/simlingo_steervla_cast_relabel_train_config.py; stop=95 ;;
    *llheavy*)     cfg=impls/configs/steervla_cast_relabel_hl200x10_adaptive_config.py; stop=100 ;;
    *) log "unknown sweep $s; skipping"; continue ;;
  esac
  log "serving $s"
  SWEEP_NAME="$s" AGENT_CFG="$cfg" STOP_ON_SCORE="$stop" STOP_SCORE_STREAK=3 UPDATES_AFTER_SCORE=0 \
  MAX_HL_UPDATES=500 ONLINE_STEPS=10000 FIXED_CARLA_SEED=true \
  EXTRA_ARGS="--cot-temperature 0.1 --hl-kl-coef 0.05 --hl-ckpt-keep-last 5" \
  GPU="$GPU" CARLA_PORT="$CARLA_PORT" TM_PORT="$TM_PORT" DISPLAY_NUM="$DISPLAY_NUM" \
    ./.run_carla/sweep_extra_worker.sh
  log "$s worker exited ($?)"
  RESULTS_DIR="/raid/users/cglossop/sweep_results/$s" OGBENCH_SAVE_DIR="/raid/users/cglossop/sweeps/$s" \
  SWEEP="$s" ROUTES_FILE="$(routes_file "$s")" ./.run_carla/sweep_results.sh >/dev/null 2>&1 || true
done
log "chain finished"
