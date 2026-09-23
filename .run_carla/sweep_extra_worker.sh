#!/usr/bin/env bash
# sweep_extra_worker.sh — f2d_extra_worker.sh, fixed so an aborted route cannot leave its
# main_carla. run_carla.sh starts main_carla in its own process group, so killing the launcher's group
# (all the original does) left a hung main_carla holding the GPU -- 2026-09-17 on GPU 5, ~4 h lost.
# Sweep settings come from the environment (SWEEP_NAME, AGENT_CFG, STOP_ON_SCORE, ...).
# f2d_extra_worker.sh — attach ONE extra worker to the f2d SimLingo sweep that is already running
# (or restart capacity for one whose driver has exited), popping from its live queue.
#
# The sweep driver (b2d_subset_sweep.sh) owns the queue. Re-arming that script to add a GPU would
# REBUILD queue.txt from the routes file: it would re-add the route a live worker has in flight (no
# run_summary.json yet, so "not done") and overwrite sweep.pid, breaking the original sweep's
# --stop. This attaches instead -- it pops from the SAME flock'd queue, which is safe for two
# poppers, and touches nothing the driver owns.
#
#   GPU=5 ./.run_carla/f2d_extra_worker.sh
#   GPU=6 CARLA_PORT=17400 TM_PORT=17500 DISPLAY_NUM=994 ./.run_carla/f2d_extra_worker.sh
#
# It waits for the GPU to fall below GPU_FREE_MIB before each route, so it can be started while
# something else is still finishing on that card.
#
# TWO THINGS THIS GETS RIGHT, both learned the hard way on 2026-09-17:
#
# 1. THE INTERPRETER. run_carla.sh runs `uv run python`, which resolves to the *project* venv of
#    the checkout it runs in. The sweep driver was armed from a shell whose venv was the main
#    checkout's, so its routes ran on /home/cglossop/ogbench-carla/.venv. A worker started from a
#    clean environment (tmux) instead made uv CREATE a bare .venv in the worktree and every route
#    died on `import flax` in about a second. UV_PROJECT_ENVIRONMENT pins it explicitly.
#
# 2. FAIL FAST, DO NOT DRAIN. The first version treated "route exited" as "route done" and looped,
#    so that misconfiguration consumed all 14 queued routes in two minutes and the driver then saw
#    an empty queue and declared the sweep complete. A route that fails faster than
#    MIN_HEALTHY_SECS is now put BACK on the queue and the worker stops, loudly.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU="${GPU:?set GPU explicitly (no default: GPUs are chosen per launch)}"
CARLA_PORT="${CARLA_PORT:-17600}"
TM_PORT="${TM_PORT:-17700}"
DISPLAY_NUM="${DISPLAY_NUM:-996}"
GPU_FREE_MIB="${GPU_FREE_MIB:-20000}"
# A real route takes hours. Anything that exits non-zero this fast is a configuration fault that
# will repeat on every route, so it must stop the worker rather than eat the queue.
MIN_HEALTHY_SECS="${MIN_HEALTHY_SECS:-300}"

# The interpreter the sweep's own routes ran on. See note 1 above.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/home/cglossop/ogbench-carla/.venv}"

# Recipe of the live sweep, copied from its armed header so this worker's routes are directly
# comparable with the ones the driver ran (see jobs/<SWEEP>/sweep.log).
SEED="${SEED:-0}"
SWEEP="${SWEEP_NAME:-f2dsteervla_simlingo_fixedcarla_kl005_seed0}"
AGENT_CFG="${AGENT_CFG:-impls/configs/simlingo_steervla_cast_relabel_train_config.py}"
export OGBENCH_SAVE_DIR="${OGBENCH_SAVE_DIR:-/raid/users/cglossop/sweeps/${SWEEP}}"
RESULTS_DIR="${RESULTS_DIR:-/raid/users/cglossop/sweep_results/${SWEEP}}"
MAX_HL_UPDATES="${MAX_HL_UPDATES:-500}"
STOP_ON_SCORE="${STOP_ON_SCORE:-95}"
STOP_SCORE_STREAK="${STOP_SCORE_STREAK:-3}"
UPDATES_AFTER_SCORE="${UPDATES_AFTER_SCORE:-0}"
FIXED_CARLA_SEED="${FIXED_CARLA_SEED:-true}"
ONLINE_STEPS="${ONLINE_STEPS:-10000}"
EXTRA_ARGS="${EXTRA_ARGS:---cot-temperature 0.1 --hl-kl-coef 0.05 --hl-ckpt-keep-last 5}"

# Watchdogs, same thresholds as the driver's worker: CARLA dying leaves no process, CARLA
# deadlocking leaves one spinning while the log goes silent. Both have happened.
WATCHDOG_GRACE="${WATCHDOG_GRACE:-600}"
WATCHDOG_STRIKES="${WATCHDOG_STRIKES:-6}"
STALL_SECS="${STALL_SECS:-900}"
STALL_STRIKES="${STALL_STRIKES:-2}"

LOG_DIR=".run_carla/jobs/${SWEEP}"
QUEUE="${LOG_DIR}/queue.txt"
LOCK="${LOG_DIR}/queue.lock"
DIED="${LOG_DIR}/carla_died.txt"
SELF_LOG="${LOG_DIR}/extra_worker_gpu${GPU}.log"
[ -f "$QUEUE" ] || { echo "[extra] no live queue at $QUEUE — is the sweep armed?" >&2; exit 1; }
[ -e "$LOCK" ] || : > "$LOCK"

log() { echo "[extra gpu${GPU} $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$SELF_LOG"; }

: "${GEMINI_API_KEY:?GEMINI_API_KEY must be exported — CAST relabel is a Gemini client}"
[ -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ] || {
  log "UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT} has no bin/python; refusing to start"; exit 1; }
"${UV_PROJECT_ENVIRONMENT}/bin/python" -c "import flax, jax, hydra" 2>/dev/null || {
  log "${UV_PROJECT_ENVIRONMENT} cannot import flax/jax; refusing to start (this is the failure that ate the queue)"; exit 1; }

# Same pop as the driver's next_route(), against the same lock file.
next_route() { flock 9; local r; r=$(head -n1 "$QUEUE"); [ -n "$r" ] && sed -i '1d' "$QUEUE"; echo "$r"; } 9>>"$LOCK"
# Put a route BACK at the front, so a stopped worker loses nothing.
requeue_front() { flock 9; local tmp; tmp=$(mktemp); { echo "$1"; cat "$QUEUE"; } > "$tmp"; mv "$tmp" "$QUEUE"; } 9>>"$LOCK"

log "attached to $SWEEP (rpc $CARLA_PORT, tm $TM_PORT, display :$DISPLAY_NUM, python $UV_PROJECT_ENVIRONMENT); $(wc -l < "$QUEUE") routes queued"

while :; do
  route=$(next_route)
  [ -z "$route" ] && { log "queue empty; worker exiting"; break; }

  waited=0
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
    [ "${used:-999999}" -lt "$GPU_FREE_MIB" ] && break
    [ "$waited" -eq 0 ] && log "waiting for gpu $GPU (${used} MiB in use) before $route"
    waited=$((waited + 1))
    if [ "$waited" -ge 720 ]; then   # 12 h: an overnight wait that never clears is a failure
      log "gpu $GPU still busy after 12h; requeueing $route and stopping"
      requeue_front "$route"; exit 0
    fi
    sleep 60
  done

  RLOG="${LOG_DIR}/${route}.log"
  log "START $route (log $RLOG)"
  started=$(date +%s)
  SEED="$SEED" AGENT_CFG="$AGENT_CFG" RUN_GROUP="$SWEEP" OGBENCH_SAVE_DIR="$OGBENCH_SAVE_DIR" \
  UPDATES_AFTER_SCORE="$UPDATES_AFTER_SCORE" MAX_HL_UPDATES="$MAX_HL_UPDATES" \
  STOP_ON_SCORE="$STOP_ON_SCORE" STOP_SCORE_STREAK="$STOP_SCORE_STREAK" \
  FIXED_CARLA_SEED="$FIXED_CARLA_SEED" ONLINE_STEPS="$ONLINE_STEPS" \
  CARLA_PORT_BASE="$CARLA_PORT" TM_PORT_BASE="$TM_PORT" DISPLAY_BASE="$DISPLAY_NUM" \
    setsid ./.run_carla/eval_subset_run.sh "$route" "$GPU" 0 $EXTRA_ARGS > "$RLOG" 2>&1 &
  rc=$!

  ( sleep "$WATCHDOG_GRACE"; strikes=0; stalls=0
    while kill -0 "$rc" 2>/dev/null; do
      if ! pgrep -u "$USER" -f "carla-rpc-port=${CARLA_PORT}" >/dev/null 2>&1; then
        strikes=$((strikes+1))
        [ "$strikes" -ge "$WATCHDOG_STRIKES" ] && {
          log "!! CARLA gone for $route — aborting"
          echo "$route" >> "$DIED"; kill -TERM -"$rc" 2>/dev/null; sleep 15; kill -9 -"$rc" 2>/dev/null; pkill -u "$USER" -9 -f "[m]ain_carla\.py .*--route=${route} .*--run_group=${SWEEP}" 2>/dev/null; break; }
      else strikes=0; fi
      if [ -f "$RLOG" ]; then
        age=$(( $(date +%s) - $(stat -c %Y "$RLOG" 2>/dev/null || date +%s) ))
        if [ "$age" -ge "$STALL_SECS" ]; then
          stalls=$((stalls+1))
          [ "$stalls" -ge "$STALL_STRIKES" ] && {
            log "!! $route stalled ${age}s with no log output — aborting"
            echo "$route" >> "$DIED"; kill -TERM -"$rc" 2>/dev/null; sleep 15; kill -9 -"$rc" 2>/dev/null; pkill -u "$USER" -9 -f "[m]ain_carla\.py .*--route=${route} .*--run_group=${SWEEP}" 2>/dev/null
            pkill -u "$USER" -9 -f "carla-rpc-port=${CARLA_PORT}" 2>/dev/null
            rm -f "/tmp/.X${DISPLAY_NUM}-lock" 2>/dev/null
            break; }
        else stalls=0; fi
      fi
      sleep 60
    done ) &
  wd=$!

  wait "$rc"; code=$?
  kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
  # Clean up HERE, after every route. The watchdog's own kill -9 / pkill never gets to run: its
  # TERM makes `wait` above return at once, and the line above kills the watchdog mid-`sleep 15`.
  # run_carla.sh runs main_carla in its own process group, so a hung one survived the launcher's
  # death and held the GPU for hours (GPU 5 on 2026-09-17, GPU 3 and 6 on 2026-09-18). A clean exit
  # leaves nothing to match, so this is a no-op then.
  pkill -u "$USER" -9 -f "[m]ain_carla\.py .*--route=${route} .*--run_group=${SWEEP}" 2>/dev/null
  pkill -u "$USER" -9 -f "[c]arla-rpc-port=${CARLA_PORT}( |$)" 2>/dev/null
  pkill -u "$USER" -f "[X]vfb :${DISPLAY_NUM}( |$)" 2>/dev/null
  rm -f "/tmp/.X${DISPLAY_NUM}-lock" 2>/dev/null
  elapsed=$(( $(date +%s) - started ))
  log "DONE  $route (exit $code after ${elapsed}s)"

  if [ "$code" -ne 0 ] && [ "$elapsed" -lt "$MIN_HEALTHY_SECS" ]; then
    log "!! $route failed after only ${elapsed}s (< ${MIN_HEALTHY_SECS}s): treating as a configuration fault."
    log "!! requeueing $route and STOPPING so the queue is not drained. See $RLOG."
    requeue_front "$route"
    exit 1
  fi

  RESULTS_DIR="$RESULTS_DIR" OGBENCH_SAVE_DIR="$OGBENCH_SAVE_DIR" SWEEP="$SWEEP" \
    ./.run_carla/sweep_results.sh >/dev/null 2>&1 || true
done
