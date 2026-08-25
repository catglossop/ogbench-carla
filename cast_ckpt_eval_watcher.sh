#!/usr/bin/env bash
# cast_ckpt_eval_watcher.sh — evaluate a live CAST training run's HL checkpoints as they land.
#
# A ``steervla_cast_relabel_config.py`` run exports its HL-fine-tuned backbone to
# ``<run>/checkpoints/<step>/params`` every ``--hl-ckpt-every`` env steps. This watcher polls that
# directory and, for every new step that appears, queues an inference-only rollout of that frozen
# checkpoint (``steervla_rollout_eval_config.py``, no gradients / no VLM) on a route of your choice.
#
# It exists because the interesting comparison — how the policy degrades over *small* training
# increments — needs one eval per checkpoint, and there are more checkpoints than GPUs. The watcher
# is the queue: it holds a fixed pool of GPUs, launches one eval per free GPU, and reaps finished
# jobs to free the slot for the next checkpoint.
#
# Job indices are DERIVED from the checkpoint step (JOB_BASE + step/CKPT_EVERY), not allocated in
# arrival order, so the mapping step -> job/ports/display is stable across watcher restarts and a
# restart never double-launches a step it already ran.
#
# Usage:
#   ./cast_ckpt_eval_watcher.sh --run-group CastWall1095Deg500 --route generalization-wall-1095 \
#       --gpus 1,2,3,6,7 --job-base 160 --ckpt-every 500 --eval-steps 10000
#
#   ./cast_ckpt_eval_watcher.sh --status          # print the queue and exit
#
# Run it with nohup/setsid — it is a long-lived poller:
#   setsid nohup ./cast_ckpt_eval_watcher.sh ... > .run_carla/ckpt_eval/watcher.log 2>&1 &

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

EXP_ROOT="${CAST_EXP_ROOT:-/raid/users/cglossop/exps/OGBench-CARLA}"
RUN_GROUP=""
ROUTE="generalization-wall-1095"
GPUS="1,2,3,6,7"
JOB_BASE=160
CKPT_EVERY=500
EVAL_STEPS=10000
EVAL_CONFIG="impls/configs/steervla_rollout_eval_config.py"
EVAL_GROUP_PREFIX=""
POLL_SECS=120
# Orbax renames its temp dir into place, but a checkpoint is only *safe* to load once nothing has
# touched it for a while. Require the params dir to be quiet for this long before queueing it.
SETTLE_SECS=120
# Refuse to launch when the checkpoint filesystem is this low — a full /raid takes down the
# training run (and every sibling job) far more expensively than a delayed eval.
MIN_FREE_GB=80
# A GPU counts as free below this many MiB in use. An idle card sits at ~2 GB (other jobs' JAX
# stubs); a card running an eval sits at ~110 GB. 20 GB cleanly separates the two.
GPU_FREE_MIB=20000
# How many times to relaunch a step whose eval died without completing.
MAX_ATTEMPTS=3
PID_STUCK_THRESHOLD=150
SEED=0
STATUS_ONLY=0
MAX_STEP=0   # 0 = no cap; otherwise stop queueing past this checkpoint step

usage() { sed -n '2,30p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-group)   RUN_GROUP="$2"; shift 2 ;;
    --route)       ROUTE="$2"; shift 2 ;;
    --gpus)        GPUS="$2"; shift 2 ;;
    --job-base)    JOB_BASE="$2"; shift 2 ;;
    --ckpt-every)  CKPT_EVERY="$2"; shift 2 ;;
    --eval-steps)  EVAL_STEPS="$2"; shift 2 ;;
    --eval-config) EVAL_CONFIG="$2"; shift 2 ;;
    --eval-group-prefix) EVAL_GROUP_PREFIX="$2"; shift 2 ;;
    --poll-secs)   POLL_SECS="$2"; shift 2 ;;
    --settle-secs) SETTLE_SECS="$2"; shift 2 ;;
    --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
    --gpu-free-mib) GPU_FREE_MIB="$2"; shift 2 ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --pid-stuck-threshold) PID_STUCK_THRESHOLD="$2"; shift 2 ;;
    --seed)        SEED="$2"; shift 2 ;;
    --max-step)    MAX_STEP="$2"; shift 2 ;;
    --status)      STATUS_ONLY=1; shift ;;
    -h|--help)     usage 0 ;;
    *) echo "[watcher] unknown flag: $1" >&2; usage 1 ;;
  esac
done

[[ -n "$RUN_GROUP" ]] || { echo "[watcher] --run-group is required" >&2; exit 1; }
[[ -n "$EVAL_GROUP_PREFIX" ]] || EVAL_GROUP_PREFIX="${RUN_GROUP}_eval"

STATE_DIR="$ROOT_DIR/.run_carla/ckpt_eval/$RUN_GROUP"
mkdir -p "$STATE_DIR"
CONF_FILE="$STATE_DIR/watcher.conf"

# --status is a separate invocation with its own flag defaults, so reading them back would report
# whatever the *caller* typed instead of what the running watcher is actually doing (e.g. a GPU
# pool that excludes a reserved card). The live watcher records its resolved settings here and
# --status prefers them.
if [[ "${STATUS_ONLY:-0}" == 1 && -f "$CONF_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONF_FILE"
fi

log() { echo "[watcher $(date -u +%H:%M:%S)] $*"; }

# ── Locate the training run's checkpoints/ directory ────────────────────────────────────────
# The run subdirectory is named by main_carla (wandb run name), so glob rather than hardcode.
# If a group somehow holds more than one run, take the newest.
find_ckpt_dir() {
  local d
  d="$(ls -1dt "$EXP_ROOT/$RUN_GROUP"/*/checkpoints 2>/dev/null | head -1)"
  [[ -n "$d" ]] && echo "$d"
}

job_for_step() { echo "$(( JOB_BASE + $1 / CKPT_EVERY ))"; }
pid_file()     { echo "$ROOT_DIR/.run_carla/jobs/job-$1.pid"; }
mark_file()    { echo "$STATE_DIR/step-$1.launched"; }

job_alive() {
  local pf; pf="$(pid_file "$1")"
  [[ -f "$pf" ]] && kill -0 "$(cat "$pf" 2>/dev/null)" 2>/dev/null
}

free_gb() { df -BG --output=avail "$EXP_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9'; }

# run_carla.sh prints this once the rollout reached its step budget. Its absence in a log whose
# job is no longer running means the eval died partway.
job_succeeded() {
  grep -q 'run completed (exit 0)' "$ROOT_DIR/.run_carla/jobs/job-$1.log" 2>/dev/null
}

# MiB currently allocated on a physical GPU, by anyone. This box is shared: another user can take
# a card between two polls, and an eval that lands on an occupied GPU either dies with a JAX
# RESOURCE_EXHAUSTED or wedges at 0% CPU holding a queue slot forever. A static pool is not enough
# -- check the card is actually free at launch time.
gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -dc '0-9'
}

gpu_is_free() {
  local used; used="$(gpu_used_mib "$1")"
  [[ -n "$used" ]] || return 1
  [[ "$used" -lt "$GPU_FREE_MIB" ]]
}

# A checkpoint is ready when its params/ carries the orbax manifest AND nothing in it has been
# modified for SETTLE_SECS. Loading a half-written checkpoint fails at actor startup, which costs
# a whole job slot and looks like a model bug.
ckpt_ready() {
  local params="$1/params"
  [[ -f "$params/manifest.ocdbt" && -f "$params/_CHECKPOINT_METADATA" ]] || return 1
  local newest now
  newest="$(find "$params" -newermt "-${SETTLE_SECS} seconds" -print -quit 2>/dev/null)"
  [[ -z "$newest" ]]
}

# ── Status view ─────────────────────────────────────────────────────────────────────────────
print_status() {
  local ckpt_dir; ckpt_dir="$(find_ckpt_dir)"
  echo "run group   : $RUN_GROUP"
  echo "checkpoints : ${ckpt_dir:-<none yet>}"
  echo "eval route  : $ROUTE   steps=$EVAL_STEPS   gpus=$GPUS"
  if [[ -n "${WATCHER_PID:-}" ]] && kill -0 "$WATCHER_PID" 2>/dev/null; then
    echo "watcher     : pid $WATCHER_PID (running, started ${STARTED:-?})"
  else
    echo "watcher     : NOT RUNNING${WATCHER_PID:+ (last pid $WATCHER_PID)}"
  fi
  echo "free space  : $(free_gb)G on $EXP_ROOT"
  echo
  printf '%-8s %-6s %-10s %-6s %s\n' STEP JOB STATE GPU CHECKPOINT
  local step job state gpu
  for f in "$STATE_DIR"/step-*.launched; do
    [[ -e "$f" ]] || continue
    step="$(basename "$f" .launched)"; step="${step#step-}"
    job="$(job_for_step "$step")"
    gpu="$(sed -n 's/^GPU=//p' "$f")"
    if job_alive "$job"; then state=running
    elif [[ -f "$STATE_DIR/step-$step.failed" ]]; then state=FAILED
    elif [[ -f "$STATE_DIR/step-$step.done" ]]; then state=done
    else state=finished; fi
    printf '%-8s %-6s %-10s %-6s %s\n' "$step" "$job" "$state" "$gpu" "$(sed -n 's/^CKPT=//p' "$f")"
  done
  [[ -n "$ckpt_dir" ]] && {
    echo
    echo "checkpoints present : $(ls -1 "$ckpt_dir" 2>/dev/null | grep -c '^[0-9]*$')"
  }
}

if [[ "$STATUS_ONLY" == 1 ]]; then print_status; exit 0; fi

# ── Main loop ───────────────────────────────────────────────────────────────────────────────
IFS=',' read -r -a GPU_POOL <<< "$GPUS"
{
  echo "ROUTE='$ROUTE'"; echo "GPUS='$GPUS'"; echo "JOB_BASE=$JOB_BASE"
  echo "CKPT_EVERY=$CKPT_EVERY"; echo "EVAL_STEPS=$EVAL_STEPS"; echo "MAX_STEP=$MAX_STEP"
  echo "WATCHER_PID=$$"; echo "STARTED='$(date -u +%FT%TZ)'"
} > "$CONF_FILE"
log "watching group=$RUN_GROUP route=$ROUTE gpus=${GPU_POOL[*]} job_base=$JOB_BASE eval_steps=$EVAL_STEPS"
log "state dir: $STATE_DIR"

# GPU -> step currently occupying it ("" = free). Rebuilt from state files on every restart, so a
# watcher that is killed and relaunched re-adopts its running jobs instead of stacking new ones
# onto busy GPUs.
declare -A GPU_BUSY
rebuild_gpu_map() {
  local g step job
  for g in "${GPU_POOL[@]}"; do GPU_BUSY[$g]=""; done
  for f in "$STATE_DIR"/step-*.launched; do
    [[ -e "$f" ]] || continue
    step="$(basename "$f" .launched)"; step="${step#step-}"
    job="$(job_for_step "$step")"
    g="$(sed -n 's/^GPU=//p' "$f")"
    if job_alive "$job" && [[ -n "$g" ]]; then
      GPU_BUSY[$g]="$step"
    elif [[ -f "$STATE_DIR/step-$step.done" ]]; then
      :   # already accounted for
    elif job_succeeded "$job"; then
      : > "$STATE_DIR/step-$step.done"
    else
      # Died without finishing -- OOM from losing a GPU race, a CARLA abort run_carla.sh declined
      # to restart, anything. Treat the slot as retryable rather than done: marking it done would
      # silently drop that checkpoint from the sweep, and an all-dashes row reads as "not started"
      # rather than "failed".
      local attempts
      attempts="$(cat "$STATE_DIR/step-$step.attempts" 2>/dev/null || echo 0)"
      attempts=$(( attempts + 1 ))
      echo "$attempts" > "$STATE_DIR/step-$step.attempts"
      if [[ "$attempts" -lt "$MAX_ATTEMPTS" ]]; then
        log "step $step (job $job) died without completing; will retry (attempt $attempts/$MAX_ATTEMPTS)"
        rm -f "$f"
      else
        log "step $step (job $job) FAILED $attempts times; giving up"
        : > "$STATE_DIR/step-$step.done"
        : > "$STATE_DIR/step-$step.failed"
      fi
    fi
  done
}

launch_eval() {
  local step="$1" gpu="$2" ckpt="$3" job
  job="$(job_for_step "$step")"
  log "launching eval: step=$step job=$job gpu=$gpu ckpt=$ckpt"
  if STEERVLA_EVAL_CHECKPOINT="$ckpt" ./carla_job.sh start \
      --job "$job" --train-gpu "$gpu" --render-adapter "$gpu" --route "$ROUTE" -- \
      --agent-config "$EVAL_CONFIG" \
      --online-steps "$EVAL_STEPS" \
      --enable-updates false \
      --pid-stuck-threshold "$PID_STUCK_THRESHOLD" \
      --run-group "${EVAL_GROUP_PREFIX}_ckpt${step}" \
      --seed "$SEED" >>"$STATE_DIR/launch.log" 2>&1; then
    { echo "STEP=$step"; echo "JOB=$job"; echo "GPU=$gpu"; echo "CKPT=$ckpt"; echo "AT=$(date -u +%FT%TZ)"; } \
      > "$(mark_file "$step")"
    GPU_BUSY[$gpu]="$step"
    return 0
  fi
  log "LAUNCH FAILED for step=$step job=$job (see $STATE_DIR/launch.log)"
  return 1
}

training_alive() { job_alive "$JOB_BASE"; }

while true; do
  rebuild_gpu_map
  ckpt_dir="$(find_ckpt_dir)"

  if [[ -n "$ckpt_dir" ]]; then
    # Oldest checkpoint first: the early steps are the ones the study is about, and they are also
    # the ones a full disk or an aborted watcher would otherwise lose.
    for step in $(ls -1 "$ckpt_dir" 2>/dev/null | grep -E '^[0-9]+$' | sort -n); do
      [[ "$MAX_STEP" -gt 0 && "$step" -gt "$MAX_STEP" ]] && continue
      [[ -f "$(mark_file "$step")" ]] && continue
      ckpt_ready "$ckpt_dir/$step" || continue

      avail="$(free_gb)"
      if [[ -n "$avail" && "$avail" -lt "$MIN_FREE_GB" ]]; then
        log "holding: only ${avail}G free on $EXP_ROOT (< ${MIN_FREE_GB}G)"
        break
      fi

      gpu=""
      for g in "${GPU_POOL[@]}"; do
        [[ -n "${GPU_BUSY[$g]}" ]] && continue
        if ! gpu_is_free "$g"; then
          log "skipping GPU $g: $(gpu_used_mib "$g") MiB already in use (another tenant?)"
          continue
        fi
        gpu="$g"; break
      done
      [[ -z "$gpu" ]] && break   # no usable slot; try again next poll

      launch_eval "$step" "$gpu" "$ckpt_dir/$step" || : > "$(mark_file "$step")"
    done
  fi

  # Done when training has exited, every checkpoint on disk has been launched, and none is running.
  if ! training_alive && [[ -n "$ckpt_dir" ]]; then
    pending=0
    for step in $(ls -1 "$ckpt_dir" 2>/dev/null | grep -E '^[0-9]+$' | sort -n); do
      [[ "$MAX_STEP" -gt 0 && "$step" -gt "$MAX_STEP" ]] && continue
      [[ -f "$(mark_file "$step")" ]] || { pending=1; break; }
      job_alive "$(job_for_step "$step")" && { pending=1; break; }
    done
    if [[ "$pending" == 0 ]]; then
      log "training exited and all checkpoints evaluated — watcher done."
      exit 0
    fi
  fi

  sleep "$POLL_SECS"
done
