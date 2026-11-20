#!/usr/bin/env bash
# mixed_method_sweep.sh -- the mixed-policy Residual-RL + YAY-Robot sweep (16 b2d + 16 f2d routes, seed 0),
# run the way the HARP sweeps were (b2d_subset_sweep.sh / sweep_extra_worker.sh / qwen_zs_bon_sweep.sh):
#
#   * flock'd queues, a resumable done test (run_summary.json), retries for cells that end without one;
#   * after EVERY cell (and at worker start, and on watchdog abort) the slot's leftovers are found through
#     its Xvfb display -- whose parent is that run's main_carla -- and that whole session is killed, plus
#     CARLA on the slot's port. run_carla.sh puts main_carla in its own session, so killing the launcher's
#     process group leaves a hung run holding the GPU (HARP sweep 2026-09-13/17/18; this sweep 2026-11-19);
#   * a cell that dies in < SLOT_ERROR_SECS is a slot problem, requeued without spending its retry; a
#     worker that hits MAX_SLOT_ERRORS in a row stops loudly instead of draining the queue;
#   * watchdog: CARLA gone (pgrep strikes) or log silent (mtime) -> abort + slot cleanup.
#
# Layout: GPU 2 = b2d[0::2] + f2d[1::2], GPU 3 = b2d[1::2] + f2d[0::2] (animals-1076 needs CARLA 0.9.15, which
# crashes on GPU 2). Two workers per GPU: one serves that GPU's YAY queue first, the other its residual
# queue first; each takes from the other queue once its own is empty. Two workers on one GPU start cells
# through a gate (free memory >= the method's need, and >= GATE_HOLD s since the last start on that GPU).
#
#   ./.run_carla/mixed_method_sweep.sh --dry-run | --arm | --status
#   (--arm is meant to run under a systemd --user unit; `systemctl --user stop` then kills every cell.)
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PATH="/home/cglossop/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

S="${SWEEP_DIR:-/raid/users/cglossop/sweep_results/mixed_residual_yay_b2d_f2d}"
GPUS=(${SWEEP_GPUS:-2 3})
LOG_DIR="$S/runner"; QLOCK="$LOG_DIR/queue.lock"; FAILED="$LOG_DIR/failed.txt"
CELL_RETRIES="${CELL_RETRIES:-2}"
SLOT_ERROR_SECS="${SLOT_ERROR_SECS:-120}"
MAX_SLOT_ERRORS="${MAX_SLOT_ERRORS:-5}"
WATCHDOG_GRACE="${WATCHDOG_GRACE:-1200}"; WATCHDOG_STRIKES="${WATCHDOG_STRIKES:-6}"
STALL_SECS="${STALL_SECS:-900}"; STALL_STRIKES="${STALL_STRIKES:-2}"
GATE_HOLD="${GATE_HOLD:-900}"
NEED_MIB_yay="${YAY_FREE_MIB:-80000}"; NEED_MIB_residual="${RES_FREE_MIB:-45000}"
export CARLA_PORT_BASE="${CARLA_PORT_BASE:-41000}" TM_PORT_BASE="${TM_PORT_BASE:-47000}" DISPLAY_BASE="${DISPLAY_BASE:-850}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-$(sed -n 's/^export GEMINI_API_KEY=//p' /home/cglossop/.bashrc | tr -d "\"'")}"

label() { [ "$1" = residual ] && echo mixed-residual-rl-b2d-f2d-cs0 || echo mixed-yay-robot-b2d-f2d-cs0; }
cell_dir() { echo "$S/$1/runs/$(label "$1")/$2__seed-0"; }
is_done() { find "$(cell_dir "$1" "$2")" -name run_summary.json 2>/dev/null | xargs -r grep -l '"eval' 2>/dev/null | grep -q .; }
queue() { echo "$LOG_DIR/queue_$1_gpu$2.txt"; }
mkdir -p "$LOG_DIR/running" "$S/residual/logs/$(label residual)" "$S/yay/logs/$(label yay)"
log() { echo "[mms $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/sweep.log"; }

mapfile -t B2D < "$S/routes_b2d.txt"; mapfile -t F2D < "$S/routes_f2d.txt"
routes_for_gpu() {  # $1 = gpu
  local k=$(( $1 - ${GPUS[0]} )) i
  for ((i = k; i < 16; i += 2)); do echo "${B2D[$i]}"; done
  for ((i = 1 - k; i < 16; i += 2)); do echo "${F2D[$i]}"; done
}

kill_slot_leftovers() {
  local slot=$1 display=$((DISPLAY_BASE + $1)) port=$((CARLA_PORT_BASE + $1 * 20))
  local x mp sid found=0 sids=() xs=()
  for x in $(pgrep -u "$USER" -f "^Xvfb :${display} " 2>/dev/null); do
    xs+=("$x"); found=1
    mp=$(ps -o ppid= -p "$x" 2>/dev/null | tr -d ' ')
    if [ -n "$mp" ] && ps -o args= -p "$mp" 2>/dev/null | grep -qF "impls/main_carla.py"; then
      sid=$(ps -o sid= -p "$mp" 2>/dev/null | tr -d ' ')
      [ -n "$sid" ] && sids+=("$sid")
      pkill -TERM -P "$mp" 2>/dev/null
    fi
  done
  pgrep -u "$USER" -f "carla-rpc-port=${port}( |$)" >/dev/null 2>&1 && found=1
  [ "$found" = 1 ] || { rm -f "/tmp/.X${display}-lock"; return 0; }
  for sid in "${sids[@]}"; do kill -TERM -- -"$sid" 2>/dev/null; done
  for x in "${xs[@]}"; do kill -TERM "$x" 2>/dev/null; done
  pkill -u "$USER" -TERM -f "carla-rpc-port=${port}( |$)" 2>/dev/null
  sleep 10
  for sid in "${sids[@]}"; do kill -9 -- -"$sid" 2>/dev/null; done
  for x in "${xs[@]}"; do kill -9 "$x" 2>/dev/null; done
  pkill -u "$USER" -9 -f "carla-rpc-port=${port}( |$)" 2>/dev/null
  rm -f "/tmp/.X${display}-lock"
  log "slot $slot: cleaned up leftover run processes (sessions: ${sids[*]:-none}, xvfb: ${xs[*]:-none})"
}

pop() { flock 9; local q; q=$(queue "$1" "$2"); local j; j=$(head -n1 "$q" 2>/dev/null); [ -n "$j" ] && sed -i '1d' "$q"; echo "$j"; } 9>>"$QLOCK"
push() { flock 9; printf '%s\n' "$3" >> "$(queue "$1" "$2")"; } 9>>"$QLOCK"

gate() {  # $1 gpu $2 need_mib $3 slot
  local g="$LOG_DIR/gate_gpu$1" free last now ok waited=0
  while :; do
    exec {gfd}>"$g.lock"; flock "$gfd"
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' ')
    last=$(cat "$g" 2>/dev/null || echo 0); now=$(date +%s); ok=0
    if [[ "$free" =~ ^[0-9]+$ ]] && [ "$free" -ge "$2" ] && [ $((now - last)) -ge "$GATE_HOLD" ]; then echo "$now" > "$g"; ok=1; fi
    flock -u "$gfd"; eval "exec ${gfd}>&-"
    [ "$ok" = 1 ] && return 0
    [ $((waited % 3600)) -eq 0 ] && log "slot $3/gpu$1: waiting for ${2} MiB free (have ${free:-?}), last start $((now - last))s ago"
    sleep 60; waited=$((waited + 60))
  done
}

worker() {  # slot gpu primary fallback
  local slot=$1 gpu=$2 primary=$3 fallback=$4 port=$((CARLA_PORT_BASE + $1 * 20)) slot_errors=0
  kill_slot_leftovers "$slot"
  while :; do
    local job method; job=$(pop "$primary" "$gpu"); method=$primary
    [ -z "$job" ] && { job=$(pop "$fallback" "$gpu"); method=$fallback; }
    [ -z "$job" ] && { log "slot $slot/gpu$gpu: queues empty; worker exiting"; break; }
    local route attempt; IFS=$'\t' read -r route attempt <<< "$job"; attempt="${attempt:-0}"
    if is_done "$method" "$route"; then log "slot $slot/gpu$gpu: SKIP $method $route (already has a summary)"; continue; fi
    local need_var="NEED_MIB_${method}"
    gate "$gpu" "${!need_var}" "$slot"
    local tag="${route}__seed-0" out; out=$(cell_dir "$method" "$route")
    local rlog="$S/$method/logs/$(label "$method")/${tag}.log"
    [ "$attempt" -gt 0 ] && rlog="${rlog%.log}.retry${attempt}.log"
    local exp="mixed-$([ "$method" = residual ] && echo residual-rl || echo yay-robot)_sd-0_${route}"
    [ "$attempt" -gt 0 ] && exp="${exp}__a${attempt}"
    log "slot $slot/gpu$gpu: START $method $route$([ "$attempt" -gt 0 ] && echo " (retry $attempt)") -> $rlog"
    local started; started=$(date +%s)
    setsid ./.run_carla/mixed_method_run.sh "$method" "$route" "$gpu" "$slot" "$out" "$exp" > "$rlog" 2>&1 &
    local rc=$!
    echo "$method $route $rc $(date --iso-8601=seconds)" > "$LOG_DIR/running/slot$slot"
    ( sleep "$WATCHDOG_GRACE"; strikes=0; stalls=0
      while kill -0 "$rc" 2>/dev/null; do
        if pgrep -u "$USER" -f "carla-rpc-port=${port}( |$)" >/dev/null 2>&1; then strikes=0; else strikes=$((strikes + 1)); fi
        age=$(( $(date +%s) - $(stat -c %Y "$rlog" 2>/dev/null || date +%s) ))
        if [ "$age" -ge "$STALL_SECS" ]; then stalls=$((stalls + 1)); else stalls=0; fi
        if [ "$strikes" -ge "$WATCHDOG_STRIKES" ] || [ "$stalls" -ge "$STALL_STRIKES" ]; then
          log "slot $slot/gpu$gpu: !! $method $route carla_gone=$strikes stall=${age}s -- aborting"
          kill -TERM -"$rc" 2>/dev/null; sleep 15; kill -9 -"$rc" 2>/dev/null
          kill_slot_leftovers "$slot"
          break
        fi
        sleep 60
      done ) &
    local wd=$!
    wait "$rc"; local code=$?
    kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
    kill -9 -"$rc" 2>/dev/null
    kill_slot_leftovers "$slot"
    rm -f "$LOG_DIR/running/slot$slot"
    local elapsed=$(( $(date +%s) - started ))
    if is_done "$method" "$route"; then
      slot_errors=0
      log "slot $slot/gpu$gpu: DONE  $method $route (exit $code, $((elapsed / 60)) min)"
    elif [ "$elapsed" -lt "$SLOT_ERROR_SECS" ]; then
      slot_errors=$((slot_errors + 1))
      log "slot $slot/gpu$gpu: SLOT-ERROR $method $route (exit $code after ${elapsed}s, #$slot_errors); requeueing"
      push "$method" "$gpu" "${route}"$'\t'"${attempt}"
      if [ "$slot_errors" -ge "$MAX_SLOT_ERRORS" ]; then
        log "slot $slot/gpu$gpu: !! $MAX_SLOT_ERRORS slot errors in a row -- STOPPING this worker so it cannot drain the queue (see $rlog)"
        break
      fi
      sleep 60
    else
      slot_errors=0
      log "slot $slot/gpu$gpu: FAIL  $method $route (exit $code after $((elapsed / 60)) min, no summary; $rlog)"
      echo "$(date --iso-8601=seconds) $method $route attempt=$attempt exit=$code" >> "$FAILED"
      if [ "$attempt" -lt "$CELL_RETRIES" ]; then
        push "$method" "$gpu" "${route}"$'\t'"$((attempt + 1))"
        log "slot $slot/gpu$gpu: requeued $method $route for retry $((attempt + 1))/$CELL_RETRIES"
      fi
    fi
  done
}

MODE="${1:---status}"
if [ "$MODE" = --status ]; then
  for m in residual yay; do
    d=0; for g in "${GPUS[@]}"; do while read -r r; do is_done "$m" "$r" && d=$((d + 1)); done < <(routes_for_gpu "$g"); done
    echo "$m: $d/32 done; queued: $(cat "$LOG_DIR"/queue_${m}_gpu*.txt 2>/dev/null | grep -c .)"
  done
  echo "running:"; cat "$LOG_DIR"/running/slot* 2>/dev/null | sed 's/^/  /'
  echo "failed attempts: $(grep -c . "$FAILED" 2>/dev/null || echo 0)"; exit 0
fi

: > "$QLOCK"
total=0
for g in "${GPUS[@]}"; do
  for m in yay residual; do
    : > "$(queue "$m" "$g")"
    while read -r r; do is_done "$m" "$r" && continue; printf '%s\t0\n' "$r" >> "$(queue "$m" "$g")"; total=$((total + 1)); done < <(routes_for_gpu "$g")
  done
done
log "armed: $total cells to run on GPUs ${GPUS[*]} (retries ${CELL_RETRIES}x; slots ${CARLA_PORT_BASE}+20k / :${DISPLAY_BASE}+k)"
for g in "${GPUS[@]}"; do for m in yay residual; do log "  gpu$g $m: $(grep -c . "$(queue "$m" "$g")") -> $(cut -f1 "$(queue "$m" "$g")" | tr '\n' ' ')"; done; done
[ "$MODE" = --dry-run ] && { rm -f "$LOG_DIR"/queue_*; exit 0; }
[ "$MODE" = --arm ] || { echo "usage: $0 --dry-run | --arm | --status" >&2; exit 2; }

slot=0
for g in "${GPUS[@]}"; do
  worker "$slot" "$g" yay residual & sleep 8
  worker "$((slot + 1))" "$g" residual yay & sleep 8
  slot=$((slot + 2))
done
wait
log "sweep complete"
"$0" --status | tee -a "$LOG_DIR/sweep.log"
