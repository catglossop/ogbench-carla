#!/usr/bin/env bash
# b2d_subset_sweep.sh — full eval-mode sweep of the b2d subset at one (carla,train) seed pair.
#
# Each route is launched through eval_subset_run.sh, so the seed contract lives in exactly one
# place: seed N -> carla_seed=N, train_seed=N, eval_seeds=N+1001..N+1003. Running a second seed
# later is `SEED=1 ./.run_carla/b2d_subset_sweep.sh` -- same eval-seed OFFSETS, different pair,
# which is what makes the arms comparable.
#
#   ./.run_carla/b2d_subset_sweep.sh                 # dry run: print the plan
#   ./.run_carla/b2d_subset_sweep.sh --arm           # queue it (detach with nohup)
#   SEED=1 ./.run_carla/b2d_subset_sweep.sh --arm    # a second seed pair
#   ./.run_carla/b2d_subset_sweep.sh --status
#   ./.run_carla/b2d_subset_sweep.sh --stop
#
# CHECKPOINTS ARE ON. --eval-mode keeps the 2000-env-step cadence plus the end-of-training export,
# so budget ~10-20 GB per route. That is why save_dir defaults to /raid (3.2T free) rather than
# /home (298G) -- 23 routes of checkpoints will not fit on the root filesystem.
set -uo pipefail
cd /home/cglossop/ogbench-carla

SEED="${SEED:-0}"
GPUS=(${SWEEP_GPUS:-5 6})
ROUTES_FILE="${ROUTES_FILE:-b2d_subset.txt}"
AGENT_CFG="${AGENT_CFG:-impls/configs/steervla_cast_relabel_hl200x10_adaptive_config.py}"
SWEEP="${SWEEP_NAME:-hl200x10_b2dsubset_seed${SEED}}"

# Everything on /raid: 3.2T free against 298G on /, and 23 routes of checkpoints (~10-20 GB
# each) will not fit on the root filesystem. Run artifacts and checkpoints under sweeps/<name>;
# the review-friendly summary under sweep_results/<name>, kept separate so one folder holds only
# the things you actually read.
export OGBENCH_SAVE_DIR="${OGBENCH_SAVE_DIR:-/raid/users/cglossop/sweeps/${SWEEP}}"
RESULTS_DIR="${RESULTS_DIR:-/raid/users/cglossop/sweep_results/${SWEEP}}"
LOG_DIR=".run_carla/jobs/${SWEEP}"

# Per-run knobs forwarded to eval_subset_run.sh. Empty/default values reproduce the
# original sweep exactly; set them to run a sweep with a different recipe.
UPDATES_AFTER_SCORE="${UPDATES_AFTER_SCORE:-20}"
MAX_HL_UPDATES="${MAX_HL_UPDATES:-150}"
STOP_ON_SCORE="${STOP_ON_SCORE:-100}"
STOP_SCORE_STREAK="${STOP_SCORE_STREAK:-}"
FIXED_CARLA_SEED="${FIXED_CARLA_SEED:-}"
ONLINE_STEPS="${ONLINE_STEPS:-10000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
WATCHDOG_GRACE="${WATCHDOG_GRACE:-600}"
WATCHDOG_STRIKES="${WATCHDOG_STRIKES:-6}"
# A live run writes [RC-PID] lines every tick, so silence means hung, not slow.
STALL_SECS="${STALL_SECS:-900}"
STALL_STRIKES="${STALL_STRIKES:-2}"
MODE="dry"
for a in "$@"; do case "$a" in
  --arm) MODE=arm ;; --dry-run) MODE=dry ;; --status) MODE=status ;; --stop) MODE=stop ;;
  *) echo "[sweep] unknown arg: $a" >&2; exit 1 ;;
esac; done

QUEUE="${LOG_DIR}/queue.txt"
LOCK="${LOG_DIR}/queue.lock"
DIED="${LOG_DIR}/carla_died.txt"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

log() { echo "[sweep $(date +%H:%M:%S)] $*"; }

if [ "$MODE" = status ]; then
  echo "sweep      : $SWEEP  (seed $SEED)"
  echo "save_dir   : $OGBENCH_SAVE_DIR"
  echo "results    : $RESULTS_DIR"
  echo "queued     : $(wc -l < "$QUEUE" 2>/dev/null || echo '-')"
  echo "finished   : $(find "$OGBENCH_SAVE_DIR" -name run_summary.json 2>/dev/null | wc -l)"
  [ -s "$DIED" ] && { echo "watchdog aborts:"; cat "$DIED"; }
  exit 0
fi

if [ "$MODE" = stop ]; then
  # Scoped to this sweep's own workers, by the script name AND the sweep tag -- never a bare
  # pkill on run_carla.sh, which would match every other tenant's runs on this box.
  for p in $(pgrep -u "$USER" -f "b2d_subset_sweep.sh --arm" 2>/dev/null); do
    log "killing sweep worker pgid $(ps -o pgid= -p "$p" | tr -d ' ')"
    kill -TERM -"$(ps -o pgid= -p "$p" | tr -d ' ')" 2>/dev/null
  done
  sleep 10
  for i in "${!GPUS[@]}"; do
    s_i=$(( i + ${SLOT_OFFSET:-0} ))
    port=$((16400 + s_i * 20))
    for q in $(pgrep -u "$USER" -f "carla-rpc-port=${port}" 2>/dev/null); do kill -9 "$q" 2>/dev/null; done
    rm -f "/tmp/.X$((940 + s_i))-lock"
  done
  log "stopped. Other users' jobs untouched."
  exit 0
fi

# Resume by default: a route with a run_summary.json already trained AND evaluated, so re-arming
# after a crash picks up only what is missing instead of redoing finished work. RESUME=0 forces all.
grep -v '^[[:space:]]*$' "$ROUTES_FILE" > "${QUEUE}.all"
if [ "${RESUME:-1}" = 1 ]; then
  find "$OGBENCH_SAVE_DIR" -name run_summary.json 2>/dev/null \
    | xargs -r grep -ho '"route": *"[^"]*"' 2>/dev/null \
    | sed 's/.*"route": *"\([^"]*\)".*/\1/' | sort -u > "${QUEUE}.done"
  grep -vxF -f "${QUEUE}.done" "${QUEUE}.all" > "$QUEUE" 2>/dev/null || cp "${QUEUE}.all" "$QUEUE"
  log "resume: $(wc -l < "${QUEUE}.done") done, $(wc -l < "$QUEUE") to run"
else
  cp "${QUEUE}.all" "$QUEUE"
fi
N=$(wc -l < "$QUEUE")
echo
echo "  sweep      : $SWEEP"
echo "  seed       : $SEED  ->  carla_seed=$SEED train_seed=$SEED eval_seeds=$((SEED+1001)),$((SEED+1002)),$((SEED+1003))"
echo "  routes     : $N   (from $ROUTES_FILE)"
echo "  config     : $AGENT_CFG"
echo "  gpus       : ${GPUS[*]}   (one route per gpu at a time)"
echo "  checkpoints: ON (--eval-mode; every 2000 env steps + end-of-training export)"
  echo "  recipe     : max_hl_updates=$MAX_HL_UPDATES stop_on_score=$STOP_ON_SCORE streak=${STOP_SCORE_STREAK:-1} updates_after=$UPDATES_AFTER_SCORE"
  echo "               fixed_carla_seed=${FIXED_CARLA_SEED:-false} online_steps=$ONLINE_STEPS extra=\"${EXTRA_ARGS}\""
echo "  save_dir   : $OGBENCH_SAVE_DIR   ($(df -h "$(dirname "$OGBENCH_SAVE_DIR")" 2>/dev/null | tail -1 | awk '{print $4}') free)"
echo "  results    : $RESULTS_DIR"
echo "  logs       : $LOG_DIR"
echo
if [ "$MODE" = dry ]; then
  echo "  DRY RUN — nothing launched. Re-run with --arm."
  exit 0
fi

: > "$LOCK"
: "${GEMINI_API_KEY:?GEMINI_API_KEY must be exported — CAST relabel is a Gemini client}"
log "armed: $N routes, seed $SEED, gpus ${GPUS[*]}"

next_route() { flock 9; local r; r=$(head -n1 "$QUEUE"); [ -n "$r" ] && sed -i '1d' "$QUEUE"; echo "$r"; } 9>>"$LOCK"

worker() {
  # SLOT_OFFSET keeps this sweep off the ports/displays of a sweep already running.
  local slot=$(( $1 + ${SLOT_OFFSET:-0} )) gpu=$2
  local port=$((16400 + slot * 20))
  while :; do
    local route; route=$(next_route)
    [ -z "$route" ] && { log "w$slot/gpu$gpu: queue empty"; break; }
    # Do not start a route until this worker's GPU is actually free. It may still be held by a
    # probe run of mine that has not finished, or simply need a moment to release VRAM after the
    # previous route -- starting into either gives an immediate OOM and burns a queue entry.
    local _waited=0
    while :; do
      local _used
      _used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
      [ "${_used:-999999}" -lt "${GPU_FREE_MIB:-20000}" ] && break
      [ "$_waited" -eq 0 ] && log "w$slot/gpu$gpu: waiting for gpu (${_used} MiB in use) before $route"
      _waited=$((_waited + 1))
      if [ "$_waited" -ge 300 ]; then
        log "w$slot/gpu$gpu: gpu still busy after 5h; requeueing $route and stopping this worker"
        flock "$LOCK" bash -c 'echo "$0" >> "$1"' "$route" "$QUEUE"
        return 0
      fi
      sleep 60
    done
    log "w$slot/gpu$gpu: START $route"
    SEED="$SEED" AGENT_CFG="$AGENT_CFG" RUN_GROUP="$SWEEP" OGBENCH_SAVE_DIR="$OGBENCH_SAVE_DIR" \
    UPDATES_AFTER_SCORE="$UPDATES_AFTER_SCORE" MAX_HL_UPDATES="$MAX_HL_UPDATES" \
    STOP_ON_SCORE="$STOP_ON_SCORE" STOP_SCORE_STREAK="$STOP_SCORE_STREAK" \
    FIXED_CARLA_SEED="$FIXED_CARLA_SEED" ONLINE_STEPS="$ONLINE_STEPS" \
      setsid ./.run_carla/eval_subset_run.sh "$route" "$gpu" "$slot" $EXTRA_ARGS \
      > "${LOG_DIR}/${route}.log" 2>&1 &
    local rc=$!
    # Two independent failure modes, two checks. CARLA *dying* leaves no process (strikes on
    # pgrep). CARLA *deadlocking* leaves the process spinning at 100% CPU forever while the run
    # writes nothing -- the original pgrep-only watchdog reset its strike counter every cycle and
    # sat on a hung route for 11 hours on 2026-09-10. Log mtime is what actually detects that.
    local rlog="${LOG_DIR}/${route}.log"
    ( sleep "$WATCHDOG_GRACE"; strikes=0; stalls=0
      while kill -0 "$rc" 2>/dev/null; do
        if ! pgrep -u "$USER" -f "carla-rpc-port=${port}" >/dev/null 2>&1; then
          strikes=$((strikes+1))
          [ "$strikes" -ge "$WATCHDOG_STRIKES" ] && {
            log "w$slot/gpu$gpu: !! CARLA gone for $route — aborting"
            echo "$route" >> "$DIED"; kill -TERM -"$rc" 2>/dev/null; sleep 15; kill -9 -"$rc" 2>/dev/null; break; }
        else strikes=0; fi
        if [ -f "$rlog" ]; then
          local age=$(( $(date +%s) - $(stat -c %Y "$rlog" 2>/dev/null || date +%s) ))
          if [ "$age" -ge "$STALL_SECS" ]; then
            stalls=$((stalls+1))
            [ "$stalls" -ge "$STALL_STRIKES" ] && {
              log "w$slot/gpu$gpu: !! $route stalled ${age}s with no log output — aborting"
              echo "$route" >> "$DIED"; kill -TERM -"$rc" 2>/dev/null; sleep 15; kill -9 -"$rc" 2>/dev/null
              # Reclaim the display/CARLA this route was holding so the next route can start.
              pkill -u "$USER" -9 -f "carla-rpc-port=${port}" 2>/dev/null
              rm -f "/tmp/.X$((940 + slot))-lock" 2>/dev/null
              break; }
          else stalls=0; fi
        fi
        sleep 60
      done ) &
    local wd=$!
    wait "$rc"; local code=$?
    kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
    log "w$slot/gpu$gpu: DONE  $route (exit $code)"
    RESULTS_DIR="$RESULTS_DIR" OGBENCH_SAVE_DIR="$OGBENCH_SAVE_DIR" SWEEP="$SWEEP" \
      ./.run_carla/sweep_results.sh >/dev/null 2>&1 || true
  done
}

for i in "${!GPUS[@]}"; do worker "$i" "${GPUS[$i]}" & sleep 8; done
wait
log "sweep complete"
RESULTS_DIR="$RESULTS_DIR" OGBENCH_SAVE_DIR="$OGBENCH_SAVE_DIR" SWEEP="$SWEEP" ./.run_carla/sweep_results.sh || true
