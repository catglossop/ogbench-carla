#!/usr/bin/env bash
# bon_candlog_rerun.sh -- re-run one route of the mixed b2d zero-shot Qwen BoN sweep (protocol of
# qwenzs_mixed_b2d_sources/launch_t1.sh) with --bon_call_log, so every critic call's candidates
# (raw action chunks, HL subtask/reasoning/full output, Qwen scores, selection) and clean camera
# frames are saved for offline rendering. Sequential over CARLA seeds on ONE worker GPU, with its
# own critic on QWEN_GPU. Runs on its own slot/ports/display so it cannot touch the live sweep.
#
#   WORKER_GPU=0 QWEN_GPU=7 setsid nohup ./.run_carla/bon_candlog_rerun.sh > <log> 2>&1 &
#
# Outputs: sweeps/<RUN_GROUP>/.../<run>/bon_calls/{calls.jsonl,steps.jsonl,frames/}
#          sweep_results/<RUN_GROUP>/<route>/carla_seed_N/run_summary_frozen_eval.json
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PATH="/home/cglossop/.local/bin:/home/cglossop/google-cloud-sdk/bin:/usr/local/cuda/bin:/opt/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

ROUTE="${ROUTE:-non-signalized-junction-left-turn-enter-flow-002}"
# The route's end-of-training CAST-relabel HL export -- the checkpoint every original t1 cell used.
CKPT="${CKPT:-/raid/users/cglossop/sweeps/b2dsteervla_simlingo_fixedcarla_kl005_seed0/OGBench-CARLA/b2dsteervla_simlingo_fixedcarla_kl005_seed0/dsrl_cast-gemini-hl+good_critic-none_upd-hl_non-signalized-junction-left-turn-enter-flow-002_seed_0_20260915_083117/checkpoints/10000}"
SEEDS="${SEEDS:-0 1 2}"
WORKER_GPU="${WORKER_GPU:?WORKER_GPU (physical index for policy + CARLA) must be set}"
QWEN_GPU="${QWEN_GPU:?QWEN_GPU (physical index for the critic) must be set}"
QWEN_PORT="${QWEN_PORT:-18860}"
SLOT="${SLOT:-2}"                        # rpc 17440, tm 17540, display :962 -- clear of the sweep's 0/1
WORKER_NEED_MIB="${WORKER_NEED_MIB:-40000}"  # policy ~22 GB + HL worker ~8 GB + CARLA ~5 GB
CELL_RETRIES="${CELL_RETRIES:-1}"
WATCHDOG_GRACE="${WATCHDOG_GRACE:-900}"; WATCHDOG_STRIKES="${WATCHDOG_STRIKES:-6}"
STALL_SECS="${STALL_SECS:-1800}"; STALL_STRIKES="${STALL_STRIKES:-2}"

export BENCH=b2d ACTOR=mixed N_CANDIDATES=4 N_EVAL=1 EVAL_SEED_OFFSET=0
export MIXED_HL_PYTHON="${MIXED_HL_PYTHON:-/raid/users/cglossop/sweep_results/qwenzs_mixed_b2d_sources/hl_python.sh}"
export RUN_GROUP="${RUN_GROUP:-b2dsteervla_simlingo_kl005_qwenzs_bon_mixed_t1_candlog}"
# main_carla also tags the W&B run with the run group, and W&B rejects tags over 64 characters at
# wandb.init -- every cell then dies in seconds (hit 2026-09-21 with a 71-char group).
[ "${#RUN_GROUP}" -le 64 ] || { echo "[candlog] ABORT: RUN_GROUP is ${#RUN_GROUP} chars; W&B tags max 64" >&2; exit 2; }
export OGBENCH_SAVE_DIR="${OGBENCH_SAVE_DIR:-/raid/users/cglossop/sweeps/${RUN_GROUP}}"
export QWEN_URL="http://127.0.0.1:${QWEN_PORT}"
export EXTRA_MAIN_FLAGS="--bon_call_log=true ${EXTRA_MAIN_FLAGS:-}"
RESULTS_DIR="${RESULTS_DIR:-/raid/users/cglossop/sweep_results/${RUN_GROUP}}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
mkdir -p "$LOG_DIR"
log() { echo "[candlog $(date +%H:%M:%S)] $*" | tee -a "${LOG_DIR}/driver.log"; }

DISPLAY_NUM=$((960 + SLOT)); CARLA_PORT=$((17400 + SLOT * 20))
# Scoped cleanup: only this worktree's main_carla (via its Xvfb) and this slot's CARLA port.
cleanup_slot() {
  local x mp sid sids=()
  for x in $(pgrep -u "$USER" -f "^Xvfb :${DISPLAY_NUM} " 2>/dev/null); do
    mp=$(ps -o ppid= -p "$x" 2>/dev/null | tr -d ' ')
    if [ -n "$mp" ] && ps -o args= -p "$mp" 2>/dev/null | grep -qF "${ROOT_DIR}/.venv/bin/python3 impls/main_carla.py"; then
      sid=$(ps -o sid= -p "$mp" 2>/dev/null | tr -d ' '); [ -n "$sid" ] && sids+=("$sid")
    fi
    kill -TERM "$x" 2>/dev/null
  done
  for sid in "${sids[@]}"; do kill -TERM -- -"$sid" 2>/dev/null; done
  pkill -u "$USER" -TERM -f "carla-rpc-port=${CARLA_PORT}( |$)" 2>/dev/null
  sleep 10
  for sid in "${sids[@]}"; do kill -9 -- -"$sid" 2>/dev/null; done
  pkill -u "$USER" -9 -f "carla-rpc-port=${CARLA_PORT}( |$)" 2>/dev/null
  rm -f "/tmp/.X${DISPLAY_NUM}-lock"
}

log "route=$ROUTE seeds=[$SEEDS] worker gpu=$WORKER_GPU slot=$SLOT (rpc $CARLA_PORT, :$DISPLAY_NUM) critic gpu=$QWEN_GPU port=$QWEN_PORT"
log "run group $RUN_GROUP; extra flags: $EXTRA_MAIN_FLAGS"

STARTED_QWEN=0
if ! QWEN_PORT="$QWEN_PORT" ./.run_carla/qwen_zs_critic_server.sh status >/dev/null 2>&1; then
  log "starting critic on gpu $QWEN_GPU (waits until the GPU has the free memory it needs)"
  QWEN_PORT="$QWEN_PORT" QWEN_GPU="$QWEN_GPU" ./.run_carla/qwen_zs_critic_server.sh start 2>&1 | tee -a "${LOG_DIR}/driver.log"
  QWEN_PORT="$QWEN_PORT" ./.run_carla/qwen_zs_critic_server.sh status >/dev/null 2>&1 \
    || { log "critic failed to start; aborting"; exit 1; }
  STARTED_QWEN=1
else
  log "reusing healthy critic on port $QWEN_PORT"
fi

cleanup_slot
for s in $SEEDS; do
  out="${RESULTS_DIR}/${ROUTE}/carla_seed_${s}"
  attempt=0
  while :; do
    [ -f "${out}/run_summary_frozen_eval.json" ] && { log "seed $s already done; skipping"; break; }
    while :; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$WORKER_GPU" | tr -d ' ')
      [ "${free:-0}" -ge "$WORKER_NEED_MIB" ] && break
      log "gpu $WORKER_GPU has ${free} MiB free (< $WORKER_NEED_MIB); waiting before seed $s"; sleep 300
    done
    rlog="${LOG_DIR}/${ROUTE}__cs${s}$([ "$attempt" -gt 0 ] && echo ".retry${attempt}").log"
    log "START seed $s (attempt $attempt) -> $rlog"
    setsid ./.run_carla/qwen_zs_bon_run.sh "$ROUTE" "$WORKER_GPU" "$SLOT" "$CKPT" "$s" "$out" > "$rlog" 2>&1 &
    rc=$!
    ( sleep "$WATCHDOG_GRACE"; strikes=0; stalls=0
      while kill -0 "$rc" 2>/dev/null; do
        if pgrep -u "$USER" -f "carla-rpc-port=${CARLA_PORT}( |$)" >/dev/null 2>&1; then strikes=0; else strikes=$((strikes + 1)); fi
        age=$(( $(date +%s) - $(stat -c %Y "$rlog" 2>/dev/null || date +%s) ))
        if [ "$age" -ge "$STALL_SECS" ]; then stalls=$((stalls + 1)); else stalls=0; fi
        if [ "$strikes" -ge "$WATCHDOG_STRIKES" ] || [ "$stalls" -ge "$STALL_STRIKES" ]; then
          echo "[candlog $(date +%H:%M:%S)] !! seed $s carla_gone=$strikes stall=${age}s -- aborting" >> "${LOG_DIR}/driver.log"
          kill -TERM -"$rc" 2>/dev/null; sleep 15; kill -9 -"$rc" 2>/dev/null
          break
        fi
        sleep 60
      done ) &
    wd=$!
    wait "$rc"; code=$?
    kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
    cleanup_slot
    if [ -f "${out}/run_summary_frozen_eval.json" ]; then
      log "DONE seed $s (exit $code)"; break
    fi
    log "FAIL seed $s (exit $code, no summary)"
    attempt=$((attempt + 1))
    [ "$attempt" -le "$CELL_RETRIES" ] || { log "seed $s out of retries"; break; }
  done
done

if [ "$STARTED_QWEN" = 1 ]; then QWEN_PORT="$QWEN_PORT" ./.run_carla/qwen_zs_critic_server.sh stop; fi
log "all seeds finished"
