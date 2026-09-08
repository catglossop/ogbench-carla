#!/usr/bin/env bash
# run_leaderboard_f2d.sh — faithful Fail2Drive leaderboard evaluation.
#
# The Fail2Drive counterpart to the ad-hoc wrapper used for the Bench2Drive 220-route
# runs (leaderboard_runs/llheavy_matchcrop_6k_b2d220). It drives the same
# ``run_leaderboard.py`` orchestrator — same scoring, same slot/port scheme, same
# resumable ``leaderboard_summary.json`` — and only adds what Fail2Drive needs:
#
#   1. FAIL2DRIVE_ROUTES_DIR    -> ogbench/carla/route_registry.py finds the 200 route XMLs
#   2. FAIL2DRIVE_SCENARIOS_DIR -> ogbench/carla/fail2drive_compat.py registers the F2D
#                                  scenario classes (ImageOnObject, ObscuredStopSign,
#                                  HardBrakeNoLights, RoadBlocked, ...) with the
#                                  leaderboard's discovery
#   3. a CARLA install that can actually serve the Fail2Drive assets  (see below)
#   4. a stall watchdog + the watch_leaderboard.py invocation to monitor with
#
# ---------------------------------------------------------------------------
# Which CARLA install?  (read this before passing --carla-root)
# ---------------------------------------------------------------------------
# Fail2Drive ships its own simulator build, typically at ~/f2d_carla. On this box that
# build is **CARLA 0.9.15.2**, while the repo's Python client (uv, .venv) is **0.9.16**.
# They are not wire-compatible: connecting a 0.9.16 client to the 0.9.15 server prints
#
#   WARNING: Version mismatch detected ... Simulator API version = 0.9.15.2-dirty
#
# and then **segfaults the client**. There is no 3.11-compatible 0.9.15 client to pair
# with it (f2d_carla ships only a cp310 wheel/egg, and this repo is pinned to 3.11).
#
# The vanilla 0.9.16 install ($CARLA_ROOT, /home/cglossop/carla) already carries every
# Fail2Drive asset pack — WallAssets, ImageAssets, StopOcclusions, AnimalVarietyPack,
# FarmAnimalsPack, AfricanAnimalsPack — installed via install_f2d_content.sh. Its Content
# tree is a strict superset of f2d_carla's (40833 vs 40475 files), and all 42
# ``static.prop.*`` ids the route XMLs reference resolve on it. So the version-matched
# 0.9.16 install is the correct default, and this script refuses a version mismatch
# rather than letting you discover it as a core dump 20 minutes in.
#
# Known gap: ``walker.animal.*`` (10 of 200 routes, Generalization_Animals_10*). Walker
# ids are registered by Content/Carla/Blueprints/Walkers/WalkerFactory.uasset, which is a
# cooked asset, not a *.Package.json — so install_f2d_content.sh's loose-file copy does
# not register them. f2d_carla's WalkerFactory does reference BP_Zebra/BP_Fox/... ; the
# 0.9.16 one does not. This script probes for that statically and warns.  See
# --check-assets output; the run still proceeds (those 10 routes simply fail to spawn
# their animal).
#
# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
#   # See the plan without launching anything
#   ./run_leaderboard_f2d.sh --dry-run
#
#   # Smoke test on two routes, in the foreground
#   ./run_leaderboard_f2d.sh --routes base-wall-0090,base-image-on-object-0040 \
#       --tag f2d_smoke --foreground
#
#   # The full 200, one route per GPU on GPUs 5 and 6, detached (the canonical run)
#   ./run_leaderboard_f2d.sh --slots 5:5,6:6 --tag f2d_llheavy_matchcrop_6k
#
#   # Resume it after an interruption
#   ./run_leaderboard_f2d.sh --slots 5:5,6:6 --tag f2d_llheavy_matchcrop_6k --resume
#
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
SLOTS="5:5,6:6"                 # TRAIN_GPU:RENDER_ADAPTER, one per concurrent route
ROUTES="fail2drive"             # all 200 F2D routes
TAG=""                          # -> leaderboard_runs/<TAG>; default derived below
OUT_DIR=""
AGENT_CONFIG="impls/configs/steervla_ll_heavy_unnormed_matchcrop_eval_config.py"
CHECKPOINT="/raid/users/cglossop/steervla_pi_ckpts/ll_heavy_unnormed_matchcrop/6000"
ACTOR_CONFIG=""
SEED="0"
XLA_MEM_FRACTION="0.30"         # 2 slots/GPU-pair; matches the b2d220 2-slot run
ONLINE_STEPS="6000"
ROUTE_TIMEOUT="7200"
RETRIES="1"
COT_TEMPERATURE="0.0"
STALL_S="600"                   # watchdog: kill a worker silent this long w/ CARLA dead
DISPLAY_BASE="440"              # :440+k — well clear of the stale :30-:90 sockets on this box
RPC_BASE="12000"
TM_BASE="18000"
WANDB_MODE="disabled"
RUN_GROUP="leaderboard-f2d"
# The repo's uv venv. When this script runs from a git worktree (.claude/worktrees/*),
# that worktree has no .venv of its own -- fall back to the main checkout's.
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python"
else
  _MAIN_ROOT="$(git -C "$ROOT_DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
  _MAIN_ROOT="${_MAIN_ROOT%/.git}"
  PYTHON="${_MAIN_ROOT}/.venv/bin/python"
fi
RESUME="false"
DRY_RUN="false"
FOREGROUND="false"
SAVE_VIDEO="false"
NO_WATCHDOG="false"
ALLOW_VERSION_MISMATCH="false"
EXTRA_ARGS=()

# ── Fail2Drive repo layout (mirrors run_simlingo_fail2drive.sh) ────────────────
_F2D_ROOT="${FAIL2DRIVE_ROOT:-${HOME}/fail2drive}"
FAIL2DRIVE_ROUTES_DIR="${FAIL2DRIVE_ROUTES_DIR:-${_F2D_ROOT}/fail2drive_split}"
FAIL2DRIVE_SCENARIOS_DIR="${FAIL2DRIVE_SCENARIOS_DIR:-${_F2D_ROOT}/scenario_runner/srunner/scenarios}"
# Deliberately the version-matched vanilla install, NOT ~/f2d_carla — see the header.
CARLA_ROOT_ARG="${CARLA_ROOT:-${HOME}/carla}"

usage() {
  sed -n '2,60p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<EOF

Options:
  --slots PAIRS             TRAIN_GPU:RENDER_ADAPTER pairs, one per concurrent route.
                            Default: ${SLOTS}
  --routes SPEC             'fail2drive' | 'bench2drive' | 'all' | comma-list | @file.
                            Default: ${ROUTES}
  --tag NAME                Run name -> leaderboard_runs/NAME. Default: derived from the
                            checkpoint + date.
  --out-dir DIR             Explicit run directory (overrides --tag).
  --agent-config PATH       Base agent config. Default: ${AGENT_CONFIG}
  --checkpoint PATH         SteerVLA checkpoint (gs:// or local).
                            Default: ${CHECKPOINT}
  --actor-config NAME       Override config.steervla.actor_config.
  --seed N                  TM + agent seed. Default: ${SEED}
  --xla-mem-fraction F      XLA_PYTHON_CLIENT_MEM_FRACTION per worker. Default: ${XLA_MEM_FRACTION}
  --online-steps N          Worker env-step budget (> the 4000-tick guard). Default: ${ONLINE_STEPS}
  --route-timeout S         Hard kill for a hung worker. Default: ${ROUTE_TIMEOUT}
  --retries N               Requeue a route that produced no record. Default: ${RETRIES}
  --cot-temperature F       steervla.cot_temperature. Default: ${COT_TEMPERATURE} (greedy)
  --stall S                 Watchdog staleness threshold. Default: ${STALL_S}
  --display-base N          X display base (:N+k). Default: ${DISPLAY_BASE}
  --rpc-base N              CARLA rpc port base. Default: ${RPC_BASE}
  --tm-base N               Traffic-manager port base. Default: ${TM_BASE}
  --carla-root DIR          CARLA install to launch. Default: ${CARLA_ROOT_ARG}
  --fail2drive-root DIR     F2D repo root. Default: ${_F2D_ROOT}
  --routes-dir DIR          F2D route XMLs. Default: \$FAIL2DRIVE_ROOT/fail2drive_split
  --scenarios-dir DIR       F2D srunner scenarios. Default: \$FAIL2DRIVE_ROOT/scenario_runner/srunner/scenarios
  --wandb-mode MODE         online|offline|disabled. Default: ${WANDB_MODE}
  --run-group NAME          Default: ${RUN_GROUP}
  --python PATH             Interpreter. Default: ${PYTHON}
  --save-video              Keep each route's MP4 (200 routes is a lot).
  --resume                  Skip routes that already have a record.
  --foreground              Run attached with the live dashboard (default: nohup + no-ui).
  --no-watchdog             Don't start the stall watchdog.
  --allow-version-mismatch  Proceed even if the CARLA server/client versions differ.
                            (They will segfault. Only for a matched client you installed.)
  --dry-run                 Print the plan and exit.
  -h | --help               This text.
  -- ...                    Everything after a bare -- goes to impls/main_carla.py.
EOF
}

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --slots)                  SLOTS="$2"; shift 2 ;;
    --routes)                 ROUTES="$2"; shift 2 ;;
    --tag)                    TAG="$2"; shift 2 ;;
    --out-dir)                OUT_DIR="$2"; shift 2 ;;
    --agent-config)           AGENT_CONFIG="$2"; shift 2 ;;
    --checkpoint)             CHECKPOINT="$2"; shift 2 ;;
    --actor-config)           ACTOR_CONFIG="$2"; shift 2 ;;
    --seed)                   SEED="$2"; shift 2 ;;
    --xla-mem-fraction)       XLA_MEM_FRACTION="$2"; shift 2 ;;
    --online-steps)           ONLINE_STEPS="$2"; shift 2 ;;
    --route-timeout)          ROUTE_TIMEOUT="$2"; shift 2 ;;
    --retries)                RETRIES="$2"; shift 2 ;;
    --cot-temperature)        COT_TEMPERATURE="$2"; shift 2 ;;
    --stall)                  STALL_S="$2"; shift 2 ;;
    --display-base)           DISPLAY_BASE="$2"; shift 2 ;;
    --rpc-base)               RPC_BASE="$2"; shift 2 ;;
    --tm-base)                TM_BASE="$2"; shift 2 ;;
    --carla-root)             CARLA_ROOT_ARG="$2"; shift 2 ;;
    --fail2drive-root)        _F2D_ROOT="$2"
                              FAIL2DRIVE_ROUTES_DIR="${_F2D_ROOT}/fail2drive_split"
                              FAIL2DRIVE_SCENARIOS_DIR="${_F2D_ROOT}/scenario_runner/srunner/scenarios"
                              shift 2 ;;
    --routes-dir)             FAIL2DRIVE_ROUTES_DIR="$2"; shift 2 ;;
    --scenarios-dir)          FAIL2DRIVE_SCENARIOS_DIR="$2"; shift 2 ;;
    --wandb-mode)             WANDB_MODE="$2"; shift 2 ;;
    --run-group)              RUN_GROUP="$2"; shift 2 ;;
    --python)                 PYTHON="$2"; shift 2 ;;
    --save-video)             SAVE_VIDEO="true"; shift ;;
    --resume)                 RESUME="true"; shift ;;
    --foreground)             FOREGROUND="true"; shift ;;
    --no-watchdog)            NO_WATCHDOG="true"; shift ;;
    --allow-version-mismatch) ALLOW_VERSION_MISMATCH="true"; shift ;;
    --dry-run)                DRY_RUN="true"; shift ;;
    -h|--help)                usage; exit 0 ;;
    --)                       shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "[f2d] unknown option: $1" >&2; echo "try --help" >&2; exit 2 ;;
  esac
done

say()  { echo "[f2d] $*"; }
die()  { echo "[f2d] ERROR: $*" >&2; exit 1; }

# ── Fail2Drive sources ────────────────────────────────────────────────────────
[[ -d "$FAIL2DRIVE_ROUTES_DIR" ]] \
  || die "FAIL2DRIVE_ROUTES_DIR=$FAIL2DRIVE_ROUTES_DIR not found. Clone catglossop/fail2drive or pass --fail2drive-root."
[[ -d "$FAIL2DRIVE_SCENARIOS_DIR" ]] \
  || die "FAIL2DRIVE_SCENARIOS_DIR=$FAIL2DRIVE_SCENARIOS_DIR not found (scenario classes won't resolve)."
export FAIL2DRIVE_ROUTES_DIR FAIL2DRIVE_SCENARIOS_DIR

N_ROUTE_XML="$(find "$FAIL2DRIVE_ROUTES_DIR" -maxdepth 1 -name '*.xml' | wc -l)"
say "routes_dir=$FAIL2DRIVE_ROUTES_DIR (${N_ROUTE_XML} route XMLs)"
say "scenarios_dir=$FAIL2DRIVE_SCENARIOS_DIR"

# ── CARLA install + version guard ─────────────────────────────────────────────
[[ -x "$CARLA_ROOT_ARG/CarlaUE4.sh" ]] || die "no runnable CarlaUE4.sh under --carla-root=$CARLA_ROOT_ARG"
export CARLA_ROOT="$CARLA_ROOT_ARG"
export CARLA_PYTHON_API_ROOT="$CARLA_ROOT/PythonAPI/carla"

SERVER_VER="$(tr -d '[:space:]' < "$CARLA_ROOT/VERSION" 2>/dev/null || echo unknown)"
CLIENT_VER="$("$PYTHON" -c 'import importlib.metadata as m; print(m.version("carla"))' 2>/dev/null || echo unknown)"
say "carla_root=$CARLA_ROOT (server $SERVER_VER) | python client $CLIENT_VER"

# Compare on major.minor.patch — f2d_carla reports e.g. "0.9.15.2-dirty".
_srv_mmp="$(echo "$SERVER_VER" | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+' || true)"
_cli_mmp="$(echo "$CLIENT_VER" | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+' || true)"
if [[ -n "$_srv_mmp" && -n "$_cli_mmp" && "$_srv_mmp" != "$_cli_mmp" ]]; then
  if [[ "$ALLOW_VERSION_MISMATCH" == "true" ]]; then
    say "WARNING: server $_srv_mmp != client $_cli_mmp — proceeding because --allow-version-mismatch."
  else
    cat >&2 <<EOF
[f2d] ERROR: CARLA version mismatch — server $_srv_mmp vs Python client $_cli_mmp.
      A 0.9.16 client connecting to a 0.9.15 server segfaults (verified on this box);
      you would lose the run partway in rather than get an error.

      The Fail2Drive simulator build (~/f2d_carla) is 0.9.15.x. Its assets are already
      installed into the version-matched 0.9.16 tree, so use that instead:
          --carla-root \$HOME/carla
      Override only if you have installed a matching client: --allow-version-mismatch
EOF
    exit 1
  fi
fi

# ── Asset probe: walker.animal.* registration (10 of 200 routes) ──────────────
WALKER_FACTORY="$CARLA_ROOT/CarlaUE4/Content/Carla/Blueprints/Walkers/WalkerFactory.uasset"
if [[ -f "$WALKER_FACTORY" ]] && ! strings -a "$WALKER_FACTORY" 2>/dev/null | grep -q "BP_Zebra"; then
  cat >&2 <<EOF
[f2d] WARNING: $CARLA_ROOT does not register walker.animal.* blueprints.
      WalkerFactory.uasset is a cooked asset, so install_f2d_content.sh's loose-file copy
      cannot add them. The 10 Generalization_Animals_10** routes will fail to spawn their
      animal; the other 190 routes are unaffected.
      To fix, copy Fail2Drive's factory in (a .stock-0.9.16.bak is kept beside it):
        cp \$HOME/f2d_carla/CarlaUE4/Content/Carla/Blueprints/Walkers/WalkerFactory.uasset \\
           \$HOME/f2d_carla/CarlaUE4/Content/Carla/Blueprints/Walkers/WalkerFactory.uexp \\
           $CARLA_ROOT/CarlaUE4/Content/Carla/Blueprints/Walkers/
      then re-run those 10 routes with --resume after deleting their records.
EOF
fi

# ── Run directory ─────────────────────────────────────────────────────────────
if [[ -z "$OUT_DIR" ]]; then
  [[ -n "$TAG" ]] || TAG="f2d_$(basename "$(dirname "$CHECKPOINT")")_$(basename "$CHECKPOINT")_$(date +%Y%m%d)"
  OUT_DIR="${ROOT_DIR}/leaderboard_runs/${TAG}"
fi
CONSOLE_LOG="${OUT_DIR}.console.log"
RUN_NAME="$(basename "$OUT_DIR")"

# ── Port / display collision check ────────────────────────────────────────────
N_SLOTS="$(awk -F, '{print NF}' <<<"$SLOTS")"
for ((k = 0; k < N_SLOTS; k++)); do
  rpc=$((RPC_BASE + 100 * k)); tm=$((TM_BASE + 100 * k)); dpy=$((DISPLAY_BASE + k))
  for p in "$rpc" "$((rpc + 1))" "$tm"; do
    if ss -ltn 2>/dev/null | grep -qE "[:.]${p}\b"; then
      die "slot $k: port $p is already in use. Shift with --rpc-base / --tm-base."
    fi
  done
  if [[ -e "/tmp/.X${dpy}-lock" || -e "/tmp/.X11-unix/X${dpy}" ]]; then
    die "slot $k: X display :$dpy already has a lock/socket. Shift with --display-base."
  fi
done

export CARLA_RPC_BASE="$RPC_BASE" CARLA_TM_BASE="$TM_BASE" CARLA_DISPLAY_BASE="$DISPLAY_BASE"

# ── Assemble the orchestrator command ─────────────────────────────────────────
CMD=("$PYTHON" "${ROOT_DIR}/run_leaderboard.py"
     --slots "$SLOTS"
     --routes "$ROUTES"
     --out-dir "$OUT_DIR"
     --agent-config "$AGENT_CONFIG"
     --seed "$SEED"
     --carla-root "$CARLA_ROOT"
     --xla-mem-fraction "$XLA_MEM_FRACTION"
     --online-steps "$ONLINE_STEPS"
     --route-timeout "$ROUTE_TIMEOUT"
     --retries "$RETRIES"
     --cot-temperature "$COT_TEMPERATURE"
     --wandb-mode "$WANDB_MODE"
     --run-group "$RUN_GROUP")
[[ -n "$CHECKPOINT"   ]] && CMD+=(--steervla-checkpoint "$CHECKPOINT")
[[ -n "$ACTOR_CONFIG" ]] && CMD+=(--steervla-actor-config "$ACTOR_CONFIG")
[[ "$RESUME"     == "true" ]] && CMD+=(--resume)
[[ "$SAVE_VIDEO" == "true" ]] && CMD+=(--save-video)
[[ "$FOREGROUND" != "true" ]] && CMD+=(--no-ui)
[[ "$DRY_RUN"    == "true" ]] && CMD+=(--dry-run)
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && CMD+=(-- "${EXTRA_ARGS[@]}")

say "run=$RUN_NAME out_dir=$OUT_DIR"
say "slots=$SLOTS xla=$XLA_MEM_FRACTION stall=${STALL_S}s ckpt=$CHECKPOINT"
for ((k = 0; k < N_SLOTS; k++)); do
  say "  slot$k rpc=$((RPC_BASE + 100 * k)) stream=$((RPC_BASE + 100 * k + 1)) tm=$((TM_BASE + 100 * k)) display=:$((DISPLAY_BASE + k))"
done

if [[ "$DRY_RUN" == "true" ]]; then
  say "dry run:"; printf '  %q' "${CMD[@]}"; echo
  exec "${CMD[@]}"
fi

mkdir -p "$OUT_DIR"

if [[ "$FOREGROUND" == "true" ]]; then
  say "running in the foreground (Ctrl-C kills the workers immediately)"
  exec "${CMD[@]}"
fi

# ── Detached launch ───────────────────────────────────────────────────────────
{
  echo "[$RUN_NAME] $(date +%F' '%T) starting: slots=$SLOTS xla=$XLA_MEM_FRACTION stall=${STALL_S}s ckpt=$CHECKPOINT"
  echo "[$RUN_NAME] carla_root=$CARLA_ROOT (server $SERVER_VER, client $CLIENT_VER)"
  echo "[$RUN_NAME] routes=$ROUTES routes_dir=$FAIL2DRIVE_ROUTES_DIR"
} >> "$CONSOLE_LOG"

nohup "${CMD[@]}" >> "$CONSOLE_LOG" 2>&1 &
ORCH_PID=$!
echo "[$RUN_NAME] orchestrator pid=$ORCH_PID  log=$CONSOLE_LOG" >> "$CONSOLE_LOG"
say "orchestrator pid=$ORCH_PID"
say "console log: $CONSOLE_LOG"

# ── Stall watchdog ────────────────────────────────────────────────────────────
# Kill a worker whose CARLA server has died so the orchestrator requeues its route
# immediately instead of waiting out --route-timeout. Deliberately conservative: a worker
# is killed only when its log has been silent > STALL_S *and* nothing is listening on that
# slot's rpc port. A slow-but-alive route is never touched.
if [[ "$NO_WATCHDOG" != "true" ]]; then
  WATCHDOG_LOG="${OUT_DIR}.watchdog.log"
  nohup bash -c '
    OUT_DIR="$1"; NSLOTS="$2"; STALE_S="$3"; RPC_BASE="$4"; DISPLAY_BASE="$5"; ORCH_PID="$6"
    log() { echo "[$(date +%F" "%T)] $*"; }
    log "watchdog start: out_dir=$OUT_DIR slots=$NSLOTS stale=${STALE_S}s orch=$ORCH_PID"
    while true; do
      kill -0 "$ORCH_PID" 2>/dev/null || { log "orchestrator $ORCH_PID gone; exiting"; exit 0; }
      for ((k = 0; k < NSLOTS; k++)); do
        pid=$(pgrep -f "slot${k}_agent\.py" | head -1)
        [ -z "$pid" ] && continue
        route=$(tr "\0" "\n" < "/proc/$pid/cmdline" 2>/dev/null | sed -n "s/^--route=//p")
        [ -z "$route" ] && continue
        logf="$OUT_DIR/logs/${route}.log"
        [ -f "$logf" ] || continue
        age=$(( $(date +%s) - $(stat -c %Y "$logf") ))
        [ "$age" -lt "$STALE_S" ] && continue
        rpc=$((RPC_BASE + 100 * k))
        if ss -ltn "sport = :$rpc" 2>/dev/null | grep -q ":$rpc"; then
          log "slot$k ($route) stale ${age}s but CARLA rpc $rpc is UP -- leaving alone"
          continue
        fi
        log "slot$k ($route) STALE ${age}s and CARLA rpc $rpc DOWN -- killing worker $pid"
        kill -TERM "$pid" 2>/dev/null; sleep 10; kill -9 "$pid" 2>/dev/null
        pkill -9 -f "carla-rpc-port=$rpc" 2>/dev/null
        pkill -9 -f "Xvfb :$((DISPLAY_BASE + k)) -screen" 2>/dev/null
        rm -f "/tmp/.X$((DISPLAY_BASE + k))-lock"
        log "slot$k cleaned; orchestrator should requeue $route"
      done
      sleep 60
    done
  ' _ "$OUT_DIR" "$N_SLOTS" "$STALL_S" "$RPC_BASE" "$DISPLAY_BASE" "$ORCH_PID" \
    > "$WATCHDOG_LOG" 2>&1 &
  say "watchdog pid=$! log=$WATCHDOG_LOG"
fi

cat <<EOF

[f2d] monitor with:
    ${PYTHON} ${ROOT_DIR}/watch_leaderboard.py ${OUT_DIR} --log ${CONSOLE_LOG}

[f2d] stop with:
    kill ${ORCH_PID}
EOF
