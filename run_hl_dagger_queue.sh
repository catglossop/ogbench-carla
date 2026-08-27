#!/usr/bin/env bash
# run_hl_dagger_queue.sh — queue the HL-DAgger route study.
#
# For each of 10 Bench2Drive routes where the base policy scored below DS 100, run a pair
# of jobs back to back on the same GPU, with the same starting weights:
#
#   1. BASE   frozen no_ego_history v1 @ 6000, no updates, no CAST observer.
#             W&B group/tag: bench2drive_hl_dagger_base
#   2. TRAIN  HL-DAgger: CAST relabel -> corrected subtasks -> VLM-backbone update.
#             W&B group/tag: bench2drive_hl_dagger_train
#
# The base run always completes before its training run starts, so every route gets a
# same-settings control. Routes are independent and run --slots at a time.
#
# This machine is shared. The queue never launches into a card that lacks headroom, caps
# JAX with XLA_PYTHON_CLIENT_MEM_FRACTION so a job cannot swallow a co-tenanted GPU, and
# only ever stops job indices it started (scoped carla_job.sh stop -- never reset_carla.sh).
#
#   ./run_hl_dagger_queue.sh                 # dry run: print the plan, launch nothing
#   ./run_hl_dagger_queue.sh --check-wandb   # verify the W&B identity, then exit
#   ./run_hl_dagger_queue.sh --arm           # actually queue (detach it with nohup)
#   ./run_hl_dagger_queue.sh --status
#   ./run_hl_dagger_queue.sh --stop
#
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ── the routes ───────────────────────────────────────────────────────────────────────
# All 10 completed their route (RC = 100.00) but lost points to infractions, so the base
# policy produces a full window of video with real, VLM-visible mistakes in it -- which is
# what CAST relabeling needs to have anything to correct. Routes truncated by the
# TickRuntime cap were deliberately excluded: they drive cleanly and simply run out of
# clock, so there are no BAD events to credit. One route per scenario family.
#           route                                             base DS   penalty  infractions
ROUTES=(
  vanilla-signalized-turn-encounter-green-light-005  #  20.40   0.204   layout + vehicle x2 + off-lane
  enter-actor-flow-003                               #  21.14   0.211   layout + vehicle x2 + off-lane
  hazard-at-side-lane-005                            #  21.60   0.216   vehicle x3
  non-signalized-junction-left-turn-003              #  21.60   0.216   vehicle x3
  parking-exit-001                                   #  21.60   0.216   vehicle x3
  pedestrian-crossing-001                            #  28.80   0.288   vehicle x2 + stop sign
  signalized-junction-left-turn-enter-flow-004       #  35.90   0.359   vehicle + red light + off-lane
  highway-exit-005                                   #  36.00   0.360   vehicle x2
  opposite-vehicle-taking-priority-005               #  36.00   0.360   vehicle x2
  static-cut-in-001                                  #  36.00   0.360   vehicle x2
)

# ── defaults ─────────────────────────────────────────────────────────────────────────
JOB_BASE_INDEX=200          # base runs use 200+i, training runs 210+i (rpc 32000+/33000+)
SLOTS=2                     # routes in flight at once
BASE_STEPS=8000
TRAIN_STEPS=20000
SEED=0
GPUS="0,1,2,3,4,5,6,7"
MIN_FREE_BASE=35000         # MiB free needed before starting a BASE run
MIN_FREE_TRAIN=95000        # MiB free needed before starting a TRAIN run (full TrainState)
MEM_FRAC_BASE="0.15"        # XLA_PYTHON_CLIENT_MEM_FRACTION -- inference only
MEM_FRAC_TRAIN="0.60"       # full OpenPI TrainState: params + optimizer + opt_state
POLL_SECONDS=300
BACKOFF_SECONDS=1800      # after every card has failed preflight, wait this long before re-sweeping
# OFF by default, and that is deliberate. A standalone preflight is a worse instrument than
# the real launcher: ogbench/carla/carla_utils.py already sets up Xvfb + DISPLAY, resolves
# VK_ICD_FILENAMES against paths that exist, passes -g.TimeoutForBlockOnRenderFence=300000,
# picks a healthy sim GPU (_pick_healthy_sim_gpu), and retries via --max-retries. A hand-rolled
# proxy that misses any one of those reports failures the real run never hits -- which is
# exactly what happened on 2026-08-25: three preflight false negatives on an idle GPU 3, where
# the real launcher then had "RPC ready after 21s". Enable with --preflight only when you
# suspect the box itself, and read the log before believing it.
PREFLIGHT=0
# Resolved 2026-08-25 from ~/.wandb_school_key: the identity is "catherineglossop" (no
# underscore). ~/.netrc on this box is a DIFFERENT account (catglossop), so the key file
# below is what puts these runs under the right user -- do not drop it.
WANDB_ENTITY_ARG="${WANDB_ENTITY:-catherineglossop}"
WANDB_KEY_FILE="$HOME/.wandb_school_key"
GROUP_BASE="bench2drive_hl_dagger_base"
GROUP_TRAIN="bench2drive_hl_dagger_train"
RUN_NAME=""
STATE_ROOT="$HOME/hl_dagger_queue"
MODE="dry"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) MODE="arm"; shift ;;
    --dry-run) MODE="dry"; shift ;;
    --status) MODE="status"; shift ;;
    --stop) MODE="stop"; shift ;;
    --check-wandb) MODE="checkwandb"; shift ;;
    --slots) SLOTS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --base-steps) BASE_STEPS="$2"; shift 2 ;;
    --train-steps) TRAIN_STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --job-base) JOB_BASE_INDEX="$2"; shift 2 ;;
    --min-free-base) MIN_FREE_BASE="$2"; shift 2 ;;
    --min-free-train) MIN_FREE_TRAIN="$2"; shift 2 ;;
    --mem-frac-base) MEM_FRAC_BASE="$2"; shift 2 ;;
    --mem-frac-train) MEM_FRAC_TRAIN="$2"; shift 2 ;;
    --poll) POLL_SECONDS="$2"; shift 2 ;;
    --backoff) BACKOFF_SECONDS="$2"; shift 2 ;;
    --no-preflight) PREFLIGHT=0; shift ;;
    --preflight) PREFLIGHT=1; shift ;;
    --wandb-entity) WANDB_ENTITY_ARG="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "[hl_queue] unknown arg: $1" >&2; exit 1 ;;
  esac
done

RUN_NAME="${RUN_NAME:-hl_dagger_$(date +%Y%m%d_%H%M%S)}"
STATE_DIR="${STATE_ROOT}/${RUN_NAME}"
LATEST_LINK="${STATE_ROOT}/latest"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

BASE_CFG="impls/configs/steervla_hl_dagger_base_config.py"
TRAIN_CFG="impls/configs/steervla_hl_dagger_train_config.py"

# ── helpers ──────────────────────────────────────────────────────────────────────────
# Logs go to STDERR on purpose: acquire_gpu returns its GPU via stdout, so anything the
# helpers print on stdout would be captured into that value. nohup sends 2>&1 to queue.log,
# so nothing is lost.
log() { echo "[hl_queue $(date +%H:%M:%S)] $*" >&2; }

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 0; }

# Emptiest allowed GPU with at least $1 MiB free; empty string if none qualifies.
# pick_gpu NEED [EXCLUDE_CSV] -- emptiest allowed card with >= NEED MiB free, skipping any
# GPU in EXCLUDE_CSV. The exclude list is what stops a route from re-picking the same card it
# just failed preflight on: free VRAM says nothing about whether the render thread is being
# starved, so "emptiest" alone would retry a doomed GPU forever.
pick_gpu() {
  local need="$1" exclude="${2:-}" best="" best_free=0 f
  for g in "${GPU_ARR[@]}"; do
    [[ ",${exclude}," == *",${g},"* ]] && continue
    f="$(free_mib "$g")"
    [[ -z "$f" ]] && continue
    if (( f >= need && f > best_free )); then best_free="$f"; best="$g"; fi
  done
  echo "$best"
}

# preflight_gpu GPU PORT DISPLAY -- boot a throwaway CARLA on that card and try load_world.
#
# This MUST mirror how ogbench/carla/carla_utils.py actually launches the simulator, or it
# reports failures the real run would never hit. Two things matter and both were missing from
# the first version of this script:
#
#   * -g.TimeoutForBlockOnRenderFence=300000 -- carla_utils.py:663 passes this because on a
#     busy shared host UE4's render thread routinely takes longer than the Linux default of
#     60 s to finish initial world/shader setup. Without it the simulator SIGSEGVs with
#     "GameThread timed out waiting for RenderThread after 60.00 secs" even on an IDLE card,
#     which is a preflight bug, not a box problem.
#   * an Xvfb display -- carla_utils.py runs one per job and exports DISPLAY to CARLA.
#
# VK_ICD_FILENAMES is resolved the same way carla_utils.py does: point it at a manifest that
# exists, or leave it unset. Naming a missing file makes the Vulkan loader skip default
# discovery and find no driver at all.
preflight_gpu() {
  local gpu="$1" port="$2" disp="$3" plog rc=1 vk="" xvfb_pid="" carla_pid=""
  [[ "$PREFLIGHT" == 1 ]] || return 0
  plog="$(mktemp "${STATE_DIR}/preflight_gpu${gpu}_XXXX.log" 2>/dev/null || mktemp)"

  for c in /etc/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.x86_64.json \
           /usr/share/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json; do
    [[ -e "$c" ]] && { vk="$c"; break; }
  done

  Xvfb ":${disp}" -screen 0 1280x1024x24 -ac +extension GLX +render -noreset \
    > /dev/null 2>&1 &
  xvfb_pid=$!
  sleep 2

  # `env` is required, not stylistic: a ${vk:+VAR=val} expansion is NOT re-parsed by bash as
  # an assignment -- it becomes the command word, and the launch dies with
  # "VK_ICD_FILENAMES=...: No such file or directory". env takes it as a real argument.
  env DISPLAY=":${disp}" ${vk:+VK_ICD_FILENAMES="$vk"} \
    "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound \
    -g.TimeoutForBlockOnRenderFence=300000 \
    -carla-rpc-port="$port" -graphicsadapter="$gpu" \
    -carla-streaming-port=$((port + 1)) > "$plog" 2>&1 &
  carla_pid=$!

  # carla_utils.py allows up to CARLA_BOOT_TIMEOUT (default 180 s) for RPC; match it.
  local waited=0
  while (( waited < 240 )); do
    sleep 10; waited=$((waited + 10))
    ss -ltn 2>/dev/null | grep -q ":${port} " && break
    # NOT `kill -0 $carla_pid`: CarlaUE4.sh is a wrapper that exits once it has spawned
    # CarlaUE4-Linux-Shipping, so its pid dying says nothing about whether the simulator
    # is still booting. Match the real process by its own rpc-port argument instead.
    pgrep -u "$USER" -f "carla-rpc-port=${port}" >/dev/null 2>&1 || break
  done

  if ss -ltn 2>/dev/null | grep -q ":${port} "; then
    if .venv/bin/python - "$port" <<'PY' >> "$plog" 2>&1
import carla, sys, time
c = carla.Client('localhost', int(sys.argv[1])); c.set_timeout(180.0)
t = time.time()
try:
    c.load_world('Town12')
    print("PREFLIGHT OK: load_world('Town12') in %.0fs" % (time.time() - t)); sys.exit(0)
except Exception as e:
    print("PREFLIGHT FAIL after %.0fs: %s" % (time.time() - t, e)); sys.exit(1)
PY
    then rc=0; fi
  else
    echo "PREFLIGHT FAIL: RPC never came up on $port (gpu $gpu) after ${waited}s" >> "$plog"
  fi

  # Scoped to this throwaway server's own rpc port and display -- never reset_carla.sh, which
  # would SIGKILL every CARLA on the box including other tenants'.
  kill "$carla_pid" 2>/dev/null
  pkill -u "$USER" -f "carla-rpc-port=${port}" 2>/dev/null
  kill "$xvfb_pid" 2>/dev/null
  rm -f "/tmp/.X${disp}-lock" 2>/dev/null
  sleep 3
  if [[ "$rc" == 0 ]]; then log "preflight OK on gpu ${gpu}"; else log "preflight FAILED on gpu ${gpu} (see $plog)"; fi
  return "$rc"
}

# acquire_gpu NEED PORT LABEL -- returns (echoes) a GPU that has headroom AND passes preflight.
# Walks the cards emptiest-first, excluding ones that just failed, and when every card has been
# tried it backs off before starting a fresh sweep. That backoff matters: a failed preflight
# costs ~2 min and boots a UE4 server, so spinning on a starved box makes the contention worse.
acquire_gpu() {
  local need="$1" port="$2" disp="$3" label="$4"
  local tried="" gpu sweeps=0
  while :; do
    gpu="$(pick_gpu "$need" "$tried")"
    if [[ -n "$gpu" ]]; then
      if preflight_gpu "$gpu" "$port" "$disp"; then echo "$gpu"; return 0; fi
      log "${label}: gpu ${gpu} had headroom but failed preflight; excluding it this sweep"
      tried="${tried:+${tried},}${gpu}"
      continue
    fi
    if [[ -n "$tried" ]]; then
      sweeps=$((sweeps + 1))
      log "${label}: every card with headroom failed preflight (sweep ${sweeps}); backing off ${BACKOFF_SECONDS}s"
      log "${label}: this is GPU compute starvation, not VRAM -- CarlaUE4's render thread is being"
      log "${label}: starved by co-tenants. Nothing will run until the box frees up."
      tried=""
      sleep "$BACKOFF_SECONDS"
    else
      log "${label}: waiting for a card with >=${need} MiB free"
      sleep "$POLL_SECONDS"
    fi
  done
}

job_alive() {
  local pf="$ROOT_DIR/.run_carla/jobs/job-$1.pid"
  [[ -f "$pf" ]] || return 1
  kill -0 "$(cat "$pf" 2>/dev/null)" 2>/dev/null
}

wait_for_job() { while job_alive "$1"; do sleep 60; done; }

# start_arm KIND JOB GPU ROUTE  -> launches and blocks until that job exits
start_arm() {
  local kind="$1" job="$2" gpu="$3" route="$4"
  local cfg group steps memfrac extra=()
  if [[ "$kind" == base ]]; then
    cfg="$BASE_CFG"; group="$GROUP_BASE"; steps="$BASE_STEPS"; memfrac="$MEM_FRAC_BASE"
    extra=(--critic-mode none --enable-updates false)
  else
    cfg="$TRAIN_CFG"; group="$GROUP_TRAIN"; steps="$TRAIN_STEPS"; memfrac="$MEM_FRAC_TRAIN"
    extra=(--hl-gpu "$gpu" --hl-ckpt-keep-last 2)
  fi

  log "start ${kind} job=${job} gpu=${gpu} route=${route} steps=${steps} mem_frac=${memfrac} group=${group}"
  echo "$(date -Is) START ${kind} ${route} job=${job} gpu=${gpu}" >> "${STATE_DIR}/timeline.log"

  # CARLA_RUN_TAG prefixes the W&B run name so base/train are told apart at a glance.
  # WANDB_ENTITY + the tag both flow through run_carla.sh -> main_carla.py -> setup_wandb,
  # which sets tags=[group]; that is why --run-group carries the requested tag.
  # WANDB_API_KEY is exported once in the arm block below -- do NOT re-inject it here as a
  # ${WANDB_API_KEY:+...} prefix. Such an expansion is not re-parsed as an assignment; it
  # becomes the command word, the launch dies with "command not found", and the key is
  # written verbatim into the log.
  CARLA_RUN_TAG="hldagger_${kind}" \
  WANDB_ENTITY="$WANDB_ENTITY_ARG" \
  WANDB_MODE=online \
  XLA_PYTHON_CLIENT_MEM_FRACTION="$memfrac" \
  ./carla_job.sh start --job "$job" --train-gpu "$gpu" --render-adapter "$gpu" \
      --route "$route" -- \
      --agent-config "$cfg" \
      --train-mode rl \
      --online-steps "$steps" \
      --save-buffer false \
      --max-retries 50 \
      --seed "$SEED" \
      --run-group "$group" \
      --wandb-mode online \
      "${extra[@]}" >> "${STATE_DIR}/${route}.${kind}.launch.log" 2>&1

  # A launch that fails (bad flag, port in use, missing config) exits in seconds and would
  # otherwise be indistinguishable from a completed run -- which would let the training arm
  # start as though its control had been collected. Require the job to still be alive.
  sleep 30
  if ! job_alive "$job"; then
    log "ERROR ${kind} job=${job} route=${route} died within 30s of launch -- not a completed run"
    log "ERROR see ${STATE_DIR}/${route}.${kind}.launch.log and ./carla_job.sh logs ${job}"
    echo "$(date -Is) FAILED-TO-START ${kind} ${route} job=${job}" >> "${STATE_DIR}/timeline.log"
    return 1
  fi
  wait_for_job "$job"
  log "finished ${kind} job=${job} route=${route}"
  echo "$(date -Is) DONE  ${kind} ${route} job=${job}" >> "${STATE_DIR}/timeline.log"
  return 0
}

# ── modes that don't launch ──────────────────────────────────────────────────────────
if [[ "$MODE" == status ]]; then
  target="${STATE_DIR}"; [[ -L "$LATEST_LINK" ]] && target="$(readlink -f "$LATEST_LINK")"
  echo "state: $target"
  [[ -f "$target/timeline.log" ]] && tail -40 "$target/timeline.log" || echo "(no timeline yet)"
  echo "--- live jobs in this queue's index band ---"
  ./carla_job.sh list 2>/dev/null | awk -v b="$JOB_BASE_INDEX" 'NR==1||($1+0>=b && $1+0<b+20 && $2=="running")'
  exit 0
fi

if [[ "$MODE" == stop ]]; then
  target="${STATE_DIR}"; [[ -L "$LATEST_LINK" ]] && target="$(readlink -f "$LATEST_LINK")"
  if [[ -f "$target/queue.pid" ]]; then
    p="$(cat "$target/queue.pid")"
    log "killing queue supervisor pid=$p"
    kill "$p" 2>/dev/null || true
  fi
  # The supervisor backgrounds one subshell per route; killing the parent orphans them and
  # they keep polling. Match on "--arm" so this "--stop" invocation cannot match itself.
  sleep 1
  for sp in $(pgrep -u "$USER" -f "run_hl_dagger_queue.sh --arm" 2>/dev/null); do
    log "killing route worker pid=$sp"
    kill "$sp" 2>/dev/null || true
  done
  for ((i = 0; i < ${#ROUTES[@]}; i++)); do
    for j in $((JOB_BASE_INDEX + i)) $((JOB_BASE_INDEX + 10 + i)); do
      if job_alive "$j"; then log "stopping job $j"; ./carla_job.sh stop "$j" || true; fi
    done
  done
  log "stopped. Other tenants' jobs were not touched."
  exit 0
fi

if [[ "$MODE" == checkwandb ]]; then
  key=""
  [[ -f "$WANDB_KEY_FILE" ]] && key="$(cat "$WANDB_KEY_FILE")"
  echo "[hl_queue] entity to be used : ${WANDB_ENTITY_ARG}"
  echo "[hl_queue] key file          : ${WANDB_KEY_FILE} ($([[ -n "$key" ]] && echo present || echo MISSING -- will fall back to ~/.netrc))"
  WANDB_API_KEY="$key" WANDB_ENTITY="$WANDB_ENTITY_ARG" .venv/bin/python - <<'PY'
import os, wandb
ent = os.environ.get("WANDB_ENTITY")
key = os.environ.get("WANDB_API_KEY") or None
api = wandb.Api(api_key=key) if key else wandb.Api()
print("  logged-in identity :", api.viewer.entity)
print("  entities available :", list(api.viewer.teams))
ok = ent in set(api.viewer.teams)
print(f"  requested entity {ent!r} reachable: {ok}")
if not ok:
    print("  -> runs would NOT land under the requested entity. Fix --wandb-entity or the key file.")
PY
  exit 0
fi

# ── plan ─────────────────────────────────────────────────────────────────────────────
echo
echo "  run name        : ${RUN_NAME}"
echo "  routes          : ${#ROUTES[@]} (base then train, per route)"
echo "  slots           : ${SLOTS} routes in flight"
echo "  job indices     : base $((JOB_BASE_INDEX))..$((JOB_BASE_INDEX + ${#ROUTES[@]} - 1)) | train $((JOB_BASE_INDEX + 10))..$((JOB_BASE_INDEX + 10 + ${#ROUTES[@]} - 1))"
echo "  gpus considered : ${GPUS}"
echo "  gate            : base >= ${MIN_FREE_BASE} MiB free, train >= ${MIN_FREE_TRAIN} MiB free"
echo "  jax mem frac    : base ${MEM_FRAC_BASE}, train ${MEM_FRAC_TRAIN}"
echo "  steps           : base ${BASE_STEPS}, train ${TRAIN_STEPS}, seed ${SEED}"
echo "  wandb entity    : ${WANDB_ENTITY_ARG}"
echo "  wandb tags      : ${GROUP_BASE} / ${GROUP_TRAIN}"
echo "  base config     : ${BASE_CFG}"
echo "  train config    : ${TRAIN_CFG}"
echo "  preflight       : $([[ "$PREFLIGHT" == 1 ]] && echo "on (boot CARLA + load_world before committing)" || echo off)"
echo "  state dir       : ${STATE_DIR}"
echo
echo "  current GPU headroom (MiB free):"
for g in "${GPU_ARR[@]}"; do printf "    gpu %s: %s\n" "$g" "$(free_mib "$g")"; done
echo
for ((i = 0; i < ${#ROUTES[@]}; i++)); do
  printf "  %2d. %-50s base job %d -> train job %d\n" \
    "$((i + 1))" "${ROUTES[$i]}" "$((JOB_BASE_INDEX + i))" "$((JOB_BASE_INDEX + 10 + i))"
done
echo

if [[ "$MODE" == dry ]]; then
  echo "  DRY RUN -- nothing launched. Re-run with --arm to queue."
  echo "  Verify the W&B identity first:  ./run_hl_dagger_queue.sh --check-wandb"
  exit 0
fi

# ── arm ──────────────────────────────────────────────────────────────────────────────
[[ -x "$ROOT_DIR/carla_job.sh" ]] || { echo "[hl_queue] carla_job.sh missing" >&2; exit 1; }
[[ -f "$BASE_CFG" && -f "$TRAIN_CFG" ]] || { echo "[hl_queue] agent configs missing" >&2; exit 1; }
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "[hl_queue] GEMINI_API_KEY must be exported -- CAST relabeling is a Gemini client and" >&2
  echo "           the training arm collects no data without it." >&2
  exit 1
fi
if [[ -z "${CARLA_ROOT:-}" ]]; then
  export CARLA_ROOT="${CARLA_ROOT:-/home/cglossop/carla}"
fi
[[ -x "$CARLA_ROOT/CarlaUE4.sh" ]] || { echo "[hl_queue] no CarlaUE4.sh under CARLA_ROOT=$CARLA_ROOT" >&2; exit 1; }

if [[ -f "$WANDB_KEY_FILE" ]]; then
  export WANDB_API_KEY="$(cat "$WANDB_KEY_FILE")"
fi

mkdir -p "$STATE_DIR"
ln -sfn "$STATE_DIR" "$LATEST_LINK"
echo $$ > "${STATE_DIR}/queue.pid"
{
  echo "RUN_NAME=${RUN_NAME}"; echo "STARTED=$(date -Is)"; echo "SLOTS=${SLOTS}"
  echo "JOB_BASE_INDEX=${JOB_BASE_INDEX}"; echo "GPUS=${GPUS}"
  echo "WANDB_ENTITY=${WANDB_ENTITY_ARG}"; echo "SEED=${SEED}"
  echo "BASE_STEPS=${BASE_STEPS}"; echo "TRAIN_STEPS=${TRAIN_STEPS}"
  echo "ROUTES=${ROUTES[*]}"
} > "${STATE_DIR}/manifest.env"
log "armed. state=${STATE_DIR}  pid=$$"
log "stop with: ./run_hl_dagger_queue.sh --stop"

# One FIFO token per slot: a route holds a token for its whole base->train pair.
SEM="${STATE_DIR}/slots.fifo"
mkfifo "$SEM" 2>/dev/null || true
exec 9<> "$SEM"
for ((s = 0; s < SLOTS; s++)); do echo "token" >&9; done

run_route() {
  local i="$1" route="${ROUTES[$1]}"
  local base_job=$((JOB_BASE_INDEX + i)) train_job=$((JOB_BASE_INDEX + 10 + i))
  local gpu=""

  # BASE -- wait for a card with headroom.
  gpu="$(acquire_gpu "$MIN_FREE_BASE" $((12000 + 100 * base_job + 50)) $((700 + base_job)) "base ${route}")"
  if ! start_arm base "$base_job" "$gpu" "$route"; then
    log "skipping train for ${route}: its base-policy control never ran"
    return 1
  fi

  # TRAIN -- re-pick, since the box moves while the base run is going.
  gpu="$(acquire_gpu "$MIN_FREE_TRAIN" $((12000 + 100 * train_job + 50)) $((700 + train_job)) "train ${route}")"
  start_arm train "$train_job" "$gpu" "$route"
}

for ((i = 0; i < ${#ROUTES[@]}; i++)); do
  read -r -u 9 _token
  (
    run_route "$i"
    echo "token" >&9
  ) &
  sleep 5
done

wait
log "all ${#ROUTES[@]} route pairs complete at $(date -Is)"
echo "$(date -Is) QUEUE COMPLETE" >> "${STATE_DIR}/timeline.log"
