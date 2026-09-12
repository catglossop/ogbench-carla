#!/usr/bin/env bash
# frozen_eval_sweep.sh -- add eval seeds to runs that already finished training.
#
# For every completed run (one with a run_summary.json), reload that run's FINAL training
# checkpoint and roll it out frozen for N more eval episodes: no gradient updates, no CAST
# review, no checkpointing. Same carla_seed and train_seed as the original; only the model
# sampling seed changes between episodes, which is what makes the spread a property of the
# policy rather than of the scenario.
#
# The point is confidence: 3 eval seeds on a route that scored 100/32/100 does not pin down the
# mean. These are seeds 1004-1006 by default, disjoint from the original 1001-1003, so the two
# sets pool into 6 independent evals of the same weights.
#
# Results land in the ORIGINAL run directory as run_summary_frozen_eval.json (via
# --frozen-eval-out), leaving run_summary.json untouched. The collector merges both.
#
#   ./.run_carla/frozen_eval_sweep.sh --arm
#   EVAL_SEEDS=1007,1008,1009 ./.run_carla/frozen_eval_sweep.sh --arm    # a third batch
set -uo pipefail
cd /home/cglossop/ogbench-carla

SWEEP="${SWEEP_NAME:-hl200x10_b2dsubset_seed0}"
OGBENCH_SAVE_DIR="${OGBENCH_SAVE_DIR:-/raid/users/cglossop/sweeps/${SWEEP}}"
RESULTS_DIR="${RESULTS_DIR:-/raid/users/cglossop/sweep_results/${SWEEP}}"
EVAL_SEEDS="${EVAL_SEEDS:-1004,1005,1006}"
AGENT_CFG="${AGENT_CFG:-impls/configs/steervla_cast_relabel_hl200x10_adaptive_config.py}"
# Slots 2,3 -> ports 16440/16460, displays :942/:943. Slots 0,1 belong to the training sweep;
# overlapping either would have the two sweeps killing each other's CARLA.
GPUS=(${FROZEN_GPUS:-1 4})
SLOTS=(2 3)
ONLINE_STEPS="${ONLINE_STEPS:-6000}"
LOG_DIR=".run_carla/jobs/frozen_eval_${SWEEP}"
QUEUE="${LOG_DIR}/queue.txt"; LOCK="${LOG_DIR}/queue.lock"
STALL_SECS="${STALL_SECS:-900}"; STALL_STRIKES="${STALL_STRIKES:-2}"
MODE="dry"
for a in "$@"; do case "$a" in
  --arm) MODE=arm ;; --dry-run) MODE=dry ;; --status) MODE=status ;; --stop) MODE=stop ;;
esac; done
mkdir -p "$LOG_DIR"
log() { echo "[frozen $(date +%H:%M:%S)] $*"; }

# One queue line per finished run: route<TAB>checkpoint<TAB>carla_seed<TAB>train_seed<TAB>run_dir
build_queue() {
  python3 - "$OGBENCH_SAVE_DIR" <<'PY'
import json, os, sys
for root, _d, files in sorted(os.walk(sys.argv[1])):
    if "run_summary.json" not in files:
        continue
    if "run_summary_frozen_eval.json" in files:      # already has its extra seeds
        continue
    try:
        d = json.load(open(os.path.join(root, "run_summary.json")))
    except Exception:
        continue
    ck = (d.get("training") or {}).get("final_checkpoint") or ""
    if not ck or not os.path.isdir(ck):               # nothing to reload -> cannot eval
        print(f"SKIP\t{d.get('route')}\tmissing checkpoint {ck}", file=sys.stderr)
        continue
    s = d.get("seeds") or {}
    print("\t".join([str(d.get("route")), ck, str(s.get("carla_seed", 0)), str(s.get("train_seed", 0)), root]))
PY
}

if [ "$MODE" = stop ]; then
  for pg in $(ps -eo user,pgid,args --no-headers | awk -v u="$USER" '$1==u' | grep "[f]rozen_eval_run" | awk '{print $2}' | sort -u); do
    kill -TERM -"$pg" 2>/dev/null; done
  sleep 10
  for s in "${SLOTS[@]}"; do
    pkill -u "$USER" -9 -f "carla-rpc-port=$((16400 + s * 20))" 2>/dev/null
    rm -f "/tmp/.X$((940 + s))-lock"
  done
  log "stopped"; exit 0
fi

build_queue > "${QUEUE}.tmp" 2> "${LOG_DIR}/skipped.txt"; mv "${QUEUE}.tmp" "$QUEUE"
N=$(wc -l < "$QUEUE")

if [ "$MODE" != arm ]; then
  echo
  echo "  frozen eval : ${SWEEP}"
  echo "  eval seeds  : ${EVAL_SEEDS}   (original run keeps 1001-1003)"
  echo "  runs to do  : ${N}"
  echo "  gpus        : ${GPUS[*]}   slots ${SLOTS[*]} -> ports $((16400+40))/$((16400+60)), displays :942/:943"
  echo "  writes      : <run_dir>/run_summary_frozen_eval.json  (run_summary.json untouched)"
  echo "  logs        : ${LOG_DIR}"
  [ -s "${LOG_DIR}/skipped.txt" ] && { echo "  SKIPPED:"; sed 's/^/    /' "${LOG_DIR}/skipped.txt"; }
  [ "$MODE" = status ] || echo "  DRY RUN -- nothing launched. Re-run with --arm."
  echo
  exit 0
fi

next_job() { flock "$LOCK" bash -c 'q="$0"; [ -s "$q" ] || exit 0; head -1 "$q"; sed -i 1d "$q"' "$QUEUE"; }

log "armed: $N runs, eval seeds ${EVAL_SEEDS}, gpus ${GPUS[*]}"
worker() {
  local idx=$1 gpu=$2 slot=${SLOTS[$1]} port=$((16400 + ${SLOTS[$1]} * 20))
  while :; do
    local job; job=$(next_job)
    [ -z "$job" ] && { log "w$idx/gpu$gpu: queue empty"; break; }
    local route ck cseed tseed rdir
    IFS=$'\t' read -r route ck cseed tseed rdir <<< "$job"
    log "w$idx/gpu$gpu: START $route (ckpt $(basename "$ck"))"
    local rlog="${LOG_DIR}/${route}.log"
    EVAL_SEEDS="$EVAL_SEEDS" AGENT_CFG="$AGENT_CFG" RUN_GROUP="$SWEEP" \
    OGBENCH_SAVE_DIR="$OGBENCH_SAVE_DIR" ONLINE_STEPS="$ONLINE_STEPS" \
      setsid ./.run_carla/frozen_eval_run.sh "$route" "$gpu" "$slot" "$ck" "$cseed" "$tseed" "$rdir" \
      > "$rlog" 2>&1 &
    local rc=$!
    ( sleep 600; stalls=0
      while kill -0 "$rc" 2>/dev/null; do
        if [ -f "$rlog" ]; then
          age=$(( $(date +%s) - $(stat -c %Y "$rlog" 2>/dev/null || date +%s) ))
          if [ "$age" -ge "$STALL_SECS" ]; then
            stalls=$((stalls+1))
            [ "$stalls" -ge "$STALL_STRIKES" ] && {
              log "w$idx/gpu$gpu: !! $route stalled ${age}s — aborting"
              kill -TERM -"$rc" 2>/dev/null; sleep 15; kill -9 -"$rc" 2>/dev/null
              pkill -u "$USER" -9 -f "carla-rpc-port=${port}" 2>/dev/null
              rm -f "/tmp/.X$((940 + slot))-lock" 2>/dev/null; break; }
          else stalls=0; fi
        fi
        sleep 60
      done ) &
    local wd=$!
    wait "$rc"; local code=$?
    kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
    local got="no summary"
    [ -f "${rdir}/run_summary_frozen_eval.json" ] && got="wrote run_summary_frozen_eval.json"
    log "w$idx/gpu$gpu: DONE  $route (exit $code, $got)"
    RESULTS_DIR="$RESULTS_DIR" OGBENCH_SAVE_DIR="$OGBENCH_SAVE_DIR" SWEEP="$SWEEP" \
      ./.run_carla/sweep_results.sh >/dev/null 2>&1 || true
  done
}
for i in "${!GPUS[@]}"; do worker "$i" "${GPUS[$i]}" & sleep 8; done
wait
log "frozen eval sweep complete"
