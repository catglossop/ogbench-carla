#!/usr/bin/env bash
# carla_job.sh — run several CARLA jobs side-by-side without port/display/GPU
# collisions, and stop one without touching the others.
#
# Each job is identified by an integer --job INDEX. All of its ports, its Xvfb
# display, and its bookkeeping are derived deterministically from that index, so
# two different indices can never overlap:
#
#   rpc port       = RPC_BASE   + PORT_STRIDE * INDEX          (default 12000 + 100*k)
#   streaming port = rpc port + 1
#   tm port        = TM_BASE    + PORT_STRIDE * INDEX          (default 18000 + 100*k)
#   x display      = DISPLAY_BASE + INDEX                      (default 30 + k)
#
# GPUs are NOT derived (there's no safe default) — pass --train-gpu / --render-adapter.
#
# The wrapper (ogbench/carla/carla_utils.py) launches its own Xvfb + CarlaUE4
# server on these ports and, at startup, only kills stale processes matching its
# OWN rpc port / display. So distinct indices are fully isolated. `stop` here
# mirrors that same scoped kill — it never touches another job.
#
# Usage:
#   ./carla_job.sh start --job 0 --train-gpu 2 --render-adapter 4 \
#       --route parking-cut-in-001 [-- <extra args forwarded to run_carla.sh>]
#   ./carla_job.sh list
#   ./carla_job.sh logs 0            # tail this job's stdout/stderr
#   ./carla_job.sh stop 0            # stop ONLY job 0 (scoped kill)
#   ./carla_job.sh stop-all          # stop every job this script started
#
# Anything after `--` is passed straight through to run_carla.sh, e.g.:
#   ./carla_job.sh start --job 1 --train-gpu 3 --render-adapter 5 -- \
#       --critic-mode delta-lang --train-mode dagger --online-steps 50000

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

JOBS_DIR="$ROOT_DIR/.run_carla/jobs"
mkdir -p "$JOBS_DIR"

# Port/display layout. Override via env if these clash with other tenants.
RPC_BASE="${CARLA_RPC_BASE:-12000}"
TM_BASE="${CARLA_TM_BASE:-18000}"
DISPLAY_BASE="${CARLA_DISPLAY_BASE:-30}"
PORT_STRIDE="${CARLA_PORT_STRIDE:-100}"

die() { echo "[carla_job] $*" >&2; exit 1; }

rpc_port_for()     { echo "$(( RPC_BASE + PORT_STRIDE * $1 ))"; }
stream_port_for()  { echo "$(( RPC_BASE + PORT_STRIDE * $1 + 1 ))"; }
tm_port_for()      { echo "$(( TM_BASE  + PORT_STRIDE * $1 ))"; }
display_for()      { echo "$(( DISPLAY_BASE + $1 ))"; }

meta_file()  { echo "$JOBS_DIR/job-$1.env"; }
pid_file()   { echo "$JOBS_DIR/job-$1.pid"; }
log_file()   { echo "$JOBS_DIR/job-$1.log"; }

port_in_use() {
  # Returns 0 if something is already listening on the TCP port.
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :$1" 2>/dev/null | grep -q ":$1 "
  else
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3>&- 3<&-; return 0; } || return 1
  fi
}

cmd_start() {
  local job="" train_gpu="" render_adapter="" route="parking-cut-in-001"
  local passthrough=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --job)            job="$2"; shift 2 ;;
      --train-gpu)      train_gpu="$2"; shift 2 ;;
      --render-adapter|--sim-gpu) render_adapter="$2"; shift 2 ;;
      --route)          route="$2"; shift 2 ;;
      --) shift; passthrough+=("$@"); break ;;
      *)  passthrough+=("$1"); shift ;;   # forward unknowns to run_carla.sh
    esac
  done

  [[ -n "$job" ]] || die "start requires --job INDEX"
  [[ "$job" =~ ^[0-9]+$ ]] || die "--job must be a non-negative integer (got '$job')"
  [[ -n "$train_gpu" ]] || die "start requires --train-gpu N"
  [[ -n "$render_adapter" ]] || die "start requires --render-adapter N (CARLA -graphicsadapter GPU)"

  local rpc stream tm disp
  rpc="$(rpc_port_for "$job")"
  stream="$(stream_port_for "$job")"
  tm="$(tm_port_for "$job")"
  disp="$(display_for "$job")"

  # Refuse to stomp a job that's already running under this index.
  if [[ -f "$(pid_file "$job")" ]] && kill -0 "$(cat "$(pid_file "$job")")" 2>/dev/null; then
    die "job $job already running (pid $(cat "$(pid_file "$job")")). Use 'stop $job' first."
  fi
  # Refuse to launch onto ports another process is already holding.
  for p in "$rpc" "$stream" "$tm"; do
    port_in_use "$p" && die "port $p already in use — another tenant? pick a different --job index or set CARLA_RPC_BASE."
  done

  cat > "$(meta_file "$job")" <<EOF
JOB=$job
RPC_PORT=$rpc
STREAMING_PORT=$stream
TM_PORT=$tm
X_DISPLAY_NUM=$disp
TRAIN_GPU=$train_gpu
RENDER_ADAPTER=$render_adapter
ROUTE=$route
EOF

  echo "[carla_job] starting job $job: route=$route rpc=$rpc stream=$stream tm=$tm display=:$disp train_gpu=$train_gpu render_adapter=$render_adapter"
  echo "[carla_job] extra run_carla.sh args: ${passthrough[*]:-<none>}"

  # setsid → own process group, so 'stop' can signal the whole tree.
  setsid bash "$ROOT_DIR/run_carla.sh" \
    --route "$route" \
    --carla-port "$rpc" \
    --carla-streaming-port "$stream" \
    --tm-port "$tm" \
    --x-display-num "$disp" \
    --train-gpu "$train_gpu" \
    --render-adapter "$render_adapter" \
    "${passthrough[@]}" \
    > "$(log_file "$job")" 2>&1 &

  local pid=$!
  echo "$pid" > "$(pid_file "$job")"
  echo "[carla_job] job $job pid=$pid  log=$(log_file "$job")"
  echo "[carla_job] tail with:  ./carla_job.sh logs $job"
}

cmd_list() {
  shopt -s nullglob
  local metas=("$JOBS_DIR"/job-*.env)
  if [[ ${#metas[@]} -eq 0 ]]; then echo "[carla_job] no jobs recorded."; return; fi
  printf "%-5s %-8s %-8s %-8s %-8s %-9s %-9s %-7s %s\n" \
    JOB STATUS PID RPC STREAM TM DISPLAY GPUt/r ROUTE
  for m in "${metas[@]}"; do
    ( source "$m"
      local pid="-" status="stopped"
      if [[ -f "$(pid_file "$JOB")" ]]; then
        pid="$(cat "$(pid_file "$JOB")")"
        kill -0 "$pid" 2>/dev/null && status="running" || { status="dead"; pid="-"; }
      fi
      printf "%-5s %-8s %-8s %-8s %-8s %-9s :%-8s %s/%s   %s\n" \
        "$JOB" "$status" "$pid" "$RPC_PORT" "$STREAMING_PORT" "$TM_PORT" \
        "$X_DISPLAY_NUM" "$TRAIN_GPU" "$RENDER_ADAPTER" "$ROUTE"
    )
  done
}

cmd_logs() {
  local job="${1:-}"; [[ -n "$job" ]] || die "logs requires a job index"
  local lf; lf="$(log_file "$job")"
  [[ -f "$lf" ]] || die "no log for job $job at $lf"
  exec tail -n 200 -f "$lf"
}

cmd_stop() {
  local job="${1:-}"; [[ -n "$job" ]] || die "stop requires a job index"
  local mf; mf="$(meta_file "$job")"
  [[ -f "$mf" ]] || die "no such job $job (no $mf)"
  # shellcheck disable=SC1090
  source "$mf"

  echo "[carla_job] stopping job $job (rpc=$RPC_PORT display=:$X_DISPLAY_NUM)"

  # 1) SIGTERM the run_carla process group so main_carla can clean up gracefully.
  if [[ -f "$(pid_file "$job")" ]]; then
    local pid; pid="$(cat "$(pid_file "$job")")"
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -9 "-$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    fi
  fi

  # 2) Scoped kill of THIS job's CARLA + Xvfb only (mirrors carla_utils.py:1465).
  pkill -9 -f "CarlaUE4.*-carla-rpc-port=${RPC_PORT}" 2>/dev/null || true
  pkill -9 -f "Xvfb :${X_DISPLAY_NUM}\b" 2>/dev/null || true
  rm -f "/tmp/.X${X_DISPLAY_NUM}-lock" 2>/dev/null || true

  rm -f "$(pid_file "$job")"
  echo "[carla_job] job $job stopped (other jobs untouched)."
}

cmd_stop_all() {
  shopt -s nullglob
  local metas=("$JOBS_DIR"/job-*.env)
  [[ ${#metas[@]} -gt 0 ]] || { echo "[carla_job] no jobs to stop."; return; }
  for m in "${metas[@]}"; do
    ( source "$m"; echo "$JOB" )
  done | sort -n | while read -r j; do cmd_stop "$j" || true; done
}

sub="${1:-}"; shift || true
case "$sub" in
  start)     cmd_start "$@" ;;
  list|ls)   cmd_list ;;
  logs|log)  cmd_logs "$@" ;;
  stop)      cmd_stop "$@" ;;
  stop-all)  cmd_stop_all ;;
  ""|-h|--help)
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed '1d'
    ;;
  *) die "unknown subcommand '$sub' (try: start|list|logs|stop|stop-all)" ;;
esac
