#!/usr/bin/env bash
# qwen_zs_bon_sweep.sh -- zero-shot Qwen BoN over the end-of-training checkpoints of a finished
# CAST/HL sweep. One job per (route with a final checkpoint) x (carla seed); each job is a frozen
# eval of N_EVAL episodes (qwen_zs_bon_run.sh), so the default 3 carla seeds give 9 episodes/route.
#
#   ./.run_carla/qwen_zs_bon_sweep.sh                         # dry run: print the plan
#   QWEN_GPU=1 SWEEP_GPUS="5 6" nohup ./.run_carla/qwen_zs_bon_sweep.sh --arm \
#       > .run_carla/jobs/qwen_zs_b2d_arm.log 2>&1 &
#   ./.run_carla/qwen_zs_bon_sweep.sh --status
#   ./.run_carla/qwen_zs_bon_sweep.sh --results               # per-route table + summary.csv
#   ./.run_carla/qwen_zs_bon_sweep.sh --stop                  # workers + their CARLA; critic kept
#
# BENCH=b2d (default) reads b2dsubset_fixedcarla_kl005_seed0. For f2d set BENCH=f2d and
# SOURCE_SWEEP explicitly (two f2d source sweeps exist). GPUs are never defaulted: --arm needs
# QWEN_GPU (critic) and SWEEP_GPUS (one concurrent route per listed GPU, all sharing the critic).
# QWEN_GPU may also appear in SWEEP_GPUS; that worker then waits for GPU_FREE_MIB_SHARED instead.
# Resumable: a cell with run_summary_frozen_eval.json is skipped on re-arm. A cell that ends
# without a summary (e.g. a CARLA segfault) is re-queued from scratch up to CELL_RETRIES times.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BENCH="${BENCH:-b2d}"
case "$BENCH" in
  b2d) SOURCE_SWEEP="${SOURCE_SWEEP:-b2dsubset_fixedcarla_kl005_seed0}"; ROUTES_FILE="${ROUTES_FILE:-b2d_subset.txt}" ;;
  f2d) SOURCE_SWEEP="${SOURCE_SWEEP:?BENCH=f2d needs SOURCE_SWEEP (f2dsubset_fixedcarla_kl005_seed0 or f2dsubset_fixedcarla_kl001_8k_seed0)}"
       ROUTES_FILE="${ROUTES_FILE:-f2d_subset.txt}" ;;
  *) echo "[qzs] BENCH must be b2d or f2d" >&2; exit 2 ;;
esac
SOURCE_ROOT="${SOURCE_ROOT:-/raid/users/cglossop/sweeps/${SOURCE_SWEEP}}"
CARLA_SEEDS="${CARLA_SEEDS:-0 1 2}"
export N_EVAL="${N_EVAL:-3}"
RUN_GROUP="${RUN_GROUP:-${SOURCE_SWEEP}_qwenzs_bon}"
QWEN_PORT="${QWEN_PORT:-18850}"
QWEN_URL="http://127.0.0.1:${QWEN_PORT}"
export OGBENCH_SAVE_DIR="${OGBENCH_SAVE_DIR:-/raid/users/cglossop/sweeps/${RUN_GROUP}}"
RESULTS_DIR="${RESULTS_DIR:-/raid/users/cglossop/sweep_results/${RUN_GROUP}}"
LOG_DIR=".run_carla/jobs/${RUN_GROUP}"
GPU_FREE_MIB="${GPU_FREE_MIB:-20000}"
GPU_FREE_MIB_SHARED="${GPU_FREE_MIB_SHARED:-100000}"
WATCHDOG_GRACE="${WATCHDOG_GRACE:-900}"
WATCHDOG_STRIKES="${WATCHDOG_STRIKES:-6}"
STALL_SECS="${STALL_SECS:-1800}"
STALL_STRIKES="${STALL_STRIKES:-2}"
CELL_RETRIES="${CELL_RETRIES:-1}"
STOP_QWEN_AT_END="${STOP_QWEN_AT_END:-1}"

MODE="dry"
for a in "$@"; do case "$a" in
  --arm) MODE=arm ;; --dry-run) MODE=dry ;; --status) MODE=status ;;
  --results) MODE=results ;; --stop) MODE=stop ;;
  *) echo "[qzs] unknown arg: $a" >&2; exit 1 ;;
esac; done

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
QUEUE="${LOG_DIR}/queue.txt"; LOCK="${LOG_DIR}/queue.lock"; FAILED="${LOG_DIR}/failed.txt"
log() { echo "[qzs $(date +%H:%M:%S)] $*" | tee -a "${LOG_DIR}/sweep.log"; }

# Kill whatever a slot's run left behind. Killing the run script's process group is not enough:
# run_carla.sh starts main_carla (under `uv run`) in its own session, and the CARLA wrapper gives
# Xvfb and CarlaUE4 sessions of their own. After a CARLA segfault that left main_carla blocked
# forever, still holding ~22 GB on the GPU. The slot's Xvfb display is the reliable handle: its
# parent is that run's main_carla, whose session also holds the `uv run` wrapper.
kill_slot_leftovers() {
  local slot=$1 display=$((960 + $1)) port=$((17400 + $1 * 20))
  local x mp sid found=0 sids=() xs=()
  for x in $(pgrep -u "$USER" -f "^Xvfb :${display} " 2>/dev/null); do
    xs+=("$x"); found=1
    mp=$(ps -o ppid= -p "$x" 2>/dev/null | tr -d ' ')
    if [ -n "$mp" ] && ps -o args= -p "$mp" 2>/dev/null | grep -qF "${ROOT_DIR}/.venv/bin/python3 impls/main_carla.py"; then
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

# route <TAB> final checkpoint <TAB> source eval mean, in ROUTES_FILE order. Only runs whose
# run_summary.json names an existing final checkpoint count; anything else has no end of training.
list_checkpoints() {
  python3 - "$SOURCE_ROOT" "$ROUTES_FILE" <<'PY'
import json, os, sys
from pathlib import Path
root, routes = Path(sys.argv[1]), [r for r in Path(sys.argv[2]).read_text().split() if r]
best = {}
for summ in root.rglob("run_summary.json"):
    if "ckpt_evals" in summ.parts:
        continue
    d = json.loads(summ.read_text())
    ck = (d.get("training") or {}).get("final_checkpoint")
    # OpenPI checkpoints hold params/; SimLingo HL exports hold pytorch_model.bin + .hydra/.
    if not ck or not ((Path(ck) / "params").is_dir() or (Path(ck) / "pytorch_model.bin").is_file()):
        continue
    prev = best.get(d["route"])
    if prev is None or summ.stat().st_mtime > prev[0]:
        best[d["route"]] = (summ.stat().st_mtime, ck, d.get("eval_mean_driving_score"))
# A training run that died before writing run_summary.json leaves checkpoints but no record of a
# final one, so the route would be skipped. CKPT_OVERRIDES (route <TAB> checkpoint per line) names
# one by hand; it only fills routes that have no checkpoint of their own.
ov = os.environ.get("CKPT_OVERRIDES", "")
if ov and Path(ov).is_file():
    for line in Path(ov).read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or parts[0] in best:
            continue
        r, ck = parts
        if (Path(ck) / "params").is_dir() or (Path(ck) / "pytorch_model.bin").is_file():
            best[r] = (0.0, ck, None)
for r in routes:
    if r in best:
        _, ck, m = best[r]
        print(f"{r}\t{ck}\t{'' if m is None else f'{m:.2f}'}")
    else:
        print(f"{r}\t-\t", file=sys.stderr)
PY
}

cell_dir() { echo "${RESULTS_DIR}/$1/carla_seed_$2"; }

if [ "$MODE" = results ]; then
  python3 - "$RESULTS_DIR" "$SOURCE_ROOT" "$ROUTES_FILE" "$CARLA_SEEDS" <<'PY'
import csv, json, sys
from pathlib import Path
res, src, routes, seeds = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).read_text().split(), sys.argv[4].split()
src_mean = {}
for s in src.rglob("run_summary.json"):
    if "ckpt_evals" not in s.parts:
        d = json.loads(s.read_text()); src_mean[d["route"]] = d.get("eval_mean_driving_score")
rows, allv = [], []
print(f"{'route':50s} " + " ".join(f"cs{s:>2s}_mean" for s in seeds) + "   n  qwen_mean  train_eval_mean")
for r in routes:
    per, vals = [], []
    for s in seeds:
        f = res / r / f"carla_seed_{s}" / "run_summary_frozen_eval.json"
        if f.exists():
            e = json.loads(f.read_text()).get("eval") or []
            sc = [x["driving_score"] for x in e]
            vals += sc; per.append(sum(sc) / len(sc) if sc else None)
            rows += [dict(route=r, carla_seed=s, eval_seed=x["eval_seed"], driving_score=x["driving_score"]) for x in e]
        else:
            per.append(None)
    if not vals and r not in src_mean:
        continue
    allv += vals
    fmt = lambda v: f"{v:9.2f}" if v is not None else "        -"
    m = sum(vals) / len(vals) if vals else None
    print(f"{r:50s} " + " ".join(fmt(v) for v in per) + f" {len(vals):3d} {fmt(m)}  {fmt(src_mean.get(r))}")
if allv:
    print(f"\noverall mean driving score over {len(allv)} episodes: {sum(allv) / len(allv):.2f}")
with open(res / "summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["route", "carla_seed", "eval_seed", "driving_score"])
    w.writeheader(); w.writerows(rows)
print(f"per-episode rows -> {res / 'summary.csv'}")
PY
  exit 0
fi

if [ "$MODE" = stop ]; then
  if [ -f "${LOG_DIR}/sweep.pgid" ]; then
    kill -TERM -"$(cat "${LOG_DIR}/sweep.pgid")" 2>/dev/null && log "stopped sweep driver pgid $(cat "${LOG_DIR}/sweep.pgid")"
    rm -f "${LOG_DIR}/sweep.pgid"
  fi
  for pf in "${LOG_DIR}"/running/*.pid; do
    [ -f "$pf" ] || continue
    kill -TERM -"$(cat "$pf")" 2>/dev/null; sleep 10; kill -9 -"$(cat "$pf")" 2>/dev/null
    rm -f "$pf" "${pf%.pid}.job"
  done
  # Slots are 0..7 at most (one per GPU); only this worktree's runs are ever touched.
  for slot in 0 1 2 3 4 5 6 7; do kill_slot_leftovers "$slot"; done
  log "stopped. Critic left running: QWEN_PORT=${QWEN_PORT} ./.run_carla/qwen_zs_critic_server.sh stop"
  exit 0
fi

mapfile -t CKPTS < <(list_checkpoints 2>"${LOG_DIR}/no_checkpoint.txt")
JOBS=(); DONE=0
for line in "${CKPTS[@]}"; do
  IFS=$'\t' read -r route ck _ <<< "$line"
  for s in $CARLA_SEEDS; do
    if [ -f "$(cell_dir "$route" "$s")/run_summary_frozen_eval.json" ]; then DONE=$((DONE + 1)); continue; fi
    JOBS+=("${route}"$'\t'"${ck}"$'\t'"${s}"$'\t'"0")
  done
done

if [ "$MODE" = status ]; then
  total=$(( ${#CKPTS[@]} * $(wc -w <<< "$CARLA_SEEDS") ))
  echo "run group : $RUN_GROUP   (source $SOURCE_SWEEP, bench $BENCH)"
  echo "cells     : $DONE / $total done, ${#JOBS[@]} remaining, $([ -f "$QUEUE" ] && wc -l < "$QUEUE" || echo 0) queued"
  for pf in "${LOG_DIR}"/running/*.pid; do [ -f "$pf" ] && echo "running   : slot $(basename "$pf" .pid): $(cat "${pf%.pid}.job" 2>/dev/null)"; done
  [ -s "$FAILED" ] && { echo "failed    :"; sed 's/^/  /' "$FAILED"; }
  QWEN_PORT="$QWEN_PORT" ./.run_carla/qwen_zs_critic_server.sh status 2>&1 | head -1
  exit 0
fi

echo
echo "  run group   : $RUN_GROUP   (W&B entity catherineglossop)"
echo "  bench       : $BENCH   carla: $([ "$BENCH" = b2d ] && echo '0.9.16 /home/cglossop/carla' || echo '0.9.15 /home/cglossop/f2d_carla')"
echo "  source      : $SOURCE_ROOT"
echo "  critic      : zero-shot Qwen3.8-27B (README settings) at $QWEN_URL"
if [ "${ACTOR:-pi05}" = simlingo ]; then
  echo "  actor       : hierarchical SimLingo SteerVLA (HL = route checkpoint, frozen training LL), ${N_CANDIDATES:-8} candidates"
elif [ "${ACTOR:-pi05}" = mixed ]; then
  echo "  actor       : mixed -- InternVL2 HL = route checkpoint, pi05 LL ${MIXED_LL_CHECKPOINT:-ll_heavy_unnormed_matchcrop/6000}, ${N_CANDIDATES:-8} candidates (sequential)"
else
  echo "  actor       : pi05 ${ACTOR_CONFIG:-pi05_steervla_cot_simplified_reasoning_ll_heavy}, ${N_CANDIDATES:-8} candidates"
fi
echo "  carla seeds : $CARLA_SEEDS   x ${N_EVAL} episode(s) each (model seeds carla_seed+${EVAL_SEED_OFFSET:-1001}..)"
echo "  routes      : ${#CKPTS[@]} with a final checkpoint (from $ROUTES_FILE)"
if [ -s "${LOG_DIR}/no_checkpoint.txt" ]; then
  echo "  skipped     : $(cut -f1 "${LOG_DIR}/no_checkpoint.txt" | tr '\n' ' ')(no final checkpoint)"
fi
echo "  cells       : $DONE done, ${#JOBS[@]} to run   (failed cells retried up to ${CELL_RETRIES}x)"
echo "  save_dir    : $OGBENCH_SAVE_DIR"
echo "  results     : $RESULTS_DIR"
echo
printf '  %-52s %-7s %s\n' route ckpt train_eval_mean
for line in "${CKPTS[@]}"; do
  IFS=$'\t' read -r route ck m <<< "$line"; printf '  %-52s %-7s %s\n' "$route" "$(basename "$ck")" "$m"
done
echo

if [ "$MODE" = dry ]; then
  if [ "${#JOBS[@]}" -gt 0 ]; then
    IFS=$'\t' read -r route ck s _ <<< "${JOBS[0]}"
    echo "  first job, as it would launch (GPU/SLOT shown as placeholders 0):"
    BENCH="$BENCH" QWEN_URL="$QWEN_URL" RUN_GROUP="$RUN_GROUP" DRY_RUN=1 \
      ./.run_carla/qwen_zs_bon_run.sh "$route" 0 0 "$ck" "$s" "$(cell_dir "$route" "$s")" | sed 's/^/    /'
  fi
  echo; echo "  DRY RUN -- nothing launched. Arm with QWEN_GPU=<gpu> SWEEP_GPUS=\"<gpus>\" ... --arm"
  exit 0
fi

# --arm
: "${SWEEP_GPUS:?SWEEP_GPUS must be set to arm (space-separated physical GPUs, one route each)}"
GPUS=($SWEEP_GPUS)
# Critics per worker: QWEN_GPUS / QWEN_PORTS line up with SWEEP_GPUS (worker i uses entry i), so one
# worker can share its GPU with its critic while another keeps the critic on a separate card.
# Unset entries fall back to QWEN_GPU / QWEN_PORT -- the original single shared critic.
QGPUS=(${QWEN_GPUS:-}); QPORTS=(${QWEN_PORTS:-})
WQGPU=(); WQPORT=()
for i in "${!GPUS[@]}"; do
  WQGPU[$i]="${QGPUS[$i]:-${QWEN_GPU:-}}"; WQPORT[$i]="${QPORTS[$i]:-$QWEN_PORT}"
  [ -n "${WQGPU[$i]}" ] || { echo "[qzs] no critic GPU for worker $i: set QWEN_GPU or QWEN_GPUS" >&2; exit 2; }
done
# Own process group, so --stop can end the driver and its watchdogs without touching anyone else.
if [ "$(ps -o pgid= $$ | tr -d ' ')" != "$$" ]; then exec setsid "$0" "$@"; fi
echo "$$" > "${LOG_DIR}/sweep.pgid"
mkdir -p "${LOG_DIR}/running"
printf '%s\n' "${JOBS[@]}" | grep -v '^$' > "$QUEUE"; : > "$LOCK"
_plan=""; for i in "${!GPUS[@]}"; do _plan+=" w$i:gpu${GPUS[$i]}->critic gpu${WQGPU[$i]}:${WQPORT[$i]}"; done
log "armed: ${#JOBS[@]} cells;${_plan}"

STARTED_PORTS=()
declare -A _SEEN_PORT=()
for i in "${!GPUS[@]}"; do
  _p="${WQPORT[$i]}"; _g="${WQGPU[$i]}"
  [ -n "${_SEEN_PORT[$_p]:-}" ] && continue
  _SEEN_PORT[$_p]=1
  if ! QWEN_PORT="$_p" ./.run_carla/qwen_zs_critic_server.sh status >/dev/null 2>&1; then
    QWEN_PORT="$_p" QWEN_GPU="$_g" ./.run_carla/qwen_zs_critic_server.sh start 2>&1 | tee -a "${LOG_DIR}/sweep.log"
    QWEN_PORT="$_p" ./.run_carla/qwen_zs_critic_server.sh status >/dev/null 2>&1 || { log "critic on gpu $_g port $_p failed to start; aborting"; exit 1; }
    STARTED_PORTS+=("$_p")
  else
    log "reusing healthy zero-shot critic at port $_p"
  fi
done

next_job() { flock 9; local j; j=$(head -n1 "$QUEUE"); [ -n "$j" ] && sed -i '1d' "$QUEUE"; echo "$j"; } 9>>"$LOCK"
requeue_job() { flock 9; printf '%s\n' "$1" >> "$QUEUE"; } 9>>"$LOCK"

worker() {
  local slot=$1 gpu=$2 qport=$3 qgpu=$4 port=$((17400 + $1 * 20))
  local qurl="http://127.0.0.1:${qport}"
  kill_slot_leftovers "$slot"
  while :; do
    local job; job=$(next_job)
    [ -z "$job" ] && { log "w$slot/gpu$gpu: queue empty"; break; }
    local route ck s attempt; IFS=$'\t' read -r route ck s attempt <<< "$job"
    attempt="${attempt:-0}"
    local tag="${route}__cs${s}" out; out=$(cell_dir "$route" "$s")
    # A worker sharing the critic's GPU starts with the critic's ~75 GB already in use.
    local free_mib="$GPU_FREE_MIB"
    [ "$gpu" = "$qgpu" ] && free_mib="$GPU_FREE_MIB_SHARED"
    while :; do
      local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
      [ "${used:-999999}" -lt "$free_mib" ] && break
      log "w$slot/gpu$gpu: gpu busy (${used} MiB); waiting before $tag"; sleep 300
    done
    until QWEN_PORT="$qport" ./.run_carla/qwen_zs_critic_server.sh status >/dev/null 2>&1; do
      log "w$slot/gpu$gpu: critic unhealthy (port $qport); waiting before $tag"; sleep 120
    done
    log "w$slot/gpu$gpu: START $tag ($(basename "$ck"))$([ "$attempt" -gt 0 ] && echo " retry $attempt")"
    local rlog="${LOG_DIR}/${tag}.log"
    [ "$attempt" -gt 0 ] && rlog="${LOG_DIR}/${tag}.retry${attempt}.log"
    local started; started=$(date +%s)
    BENCH="$BENCH" QWEN_URL="$qurl" RUN_GROUP="$RUN_GROUP" \
      setsid ./.run_carla/qwen_zs_bon_run.sh "$route" "$gpu" "$slot" "$ck" "$s" "$out" > "$rlog" 2>&1 &
    local rc=$!
    echo "$rc" > "${LOG_DIR}/running/${slot}.pid"; echo "$tag" > "${LOG_DIR}/running/${slot}.job"
    # CARLA dying leaves no process; CARLA deadlocking leaves a silent log. Check both.
    ( sleep "$WATCHDOG_GRACE"; strikes=0; stalls=0
      while kill -0 "$rc" 2>/dev/null; do
        if pgrep -u "$USER" -f "carla-rpc-port=${port}( |$)" >/dev/null 2>&1; then strikes=0; else strikes=$((strikes + 1)); fi
        age=$(( $(date +%s) - $(stat -c %Y "$rlog" 2>/dev/null || date +%s) ))
        if [ "$age" -ge "$STALL_SECS" ]; then stalls=$((stalls + 1)); else stalls=0; fi
        if [ "$strikes" -ge "$WATCHDOG_STRIKES" ] || [ "$stalls" -ge "$STALL_STRIKES" ]; then
          log "w$slot/gpu$gpu: !! $tag carla_gone=$strikes stall=${age}s -- aborting"
          kill -TERM -"$rc" 2>/dev/null; sleep 15; kill -9 -"$rc" 2>/dev/null
          kill_slot_leftovers "$slot"
          break
        fi
        sleep 60
      done ) &
    local wd=$!
    wait "$rc"; local code=$?
    kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
    kill_slot_leftovers "$slot"
    rm -f "${LOG_DIR}/running/${slot}.pid" "${LOG_DIR}/running/${slot}.job"
    if [ -f "${out}/run_summary_frozen_eval.json" ]; then
      log "w$slot/gpu$gpu: DONE  $tag (exit $code)"
    elif [ "$(( $(date +%s) - started ))" -lt "${SLOT_ERROR_SECS:-120}" ]; then
      # Died before it could even reach CARLA: a slot problem (port still listening, stale display,
      # unhealthy critic), not a bad cell. Clean the slot and put the cell back WITHOUT spending its
      # retry, or a single stuck port burns the whole queue in seconds (observed 2026-09-19).
      log "w$slot/gpu$gpu: SLOT-ERROR $tag (exit $code after $(( $(date +%s) - started ))s); cleaning slot and requeueing"
      kill_slot_leftovers "$slot"
      requeue_job "${route}"$'\t'"${ck}"$'\t'"${s}"$'\t'"${attempt}"
      sleep 30
    else
      log "w$slot/gpu$gpu: FAIL  $tag (exit $code, no summary; log $rlog)"
      echo "$(date --iso-8601=seconds) $tag attempt=$attempt exit=$code" >> "$FAILED"
      if [ "$attempt" -lt "$CELL_RETRIES" ]; then
        requeue_job "${route}"$'\t'"${ck}"$'\t'"${s}"$'\t'"$((attempt + 1))"
        log "w$slot/gpu$gpu: requeued $tag for retry $((attempt + 1))/${CELL_RETRIES}"
      fi
    fi
  done
}

for i in "${!GPUS[@]}"; do worker "$i" "${GPUS[$i]}" "${WQPORT[$i]}" "${WQGPU[$i]}" & sleep 8; done
wait
log "sweep complete"
./.run_carla/qwen_zs_bon_sweep.sh --results 2>&1 | tee "${RESULTS_DIR}/summary.txt"
if [ "$STOP_QWEN_AT_END" = 1 ]; then
  for _p in "${STARTED_PORTS[@]}"; do QWEN_PORT="$_p" ./.run_carla/qwen_zs_critic_server.sh stop; done
fi
rm -f "${LOG_DIR}/sweep.pgid"
