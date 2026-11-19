#!/usr/bin/env bash
# qwen_zs_critic_server.sh start|stop|status
#
# The ZERO-SHOT Qwen BoN critic, launched with exactly the settings in qwen-critic README.md
# (main): Qwen/Qwen3.8-27B, no adapter, BF16 eager (torch.compile off), scene_criteria_v2 with the
# original traffic wording, per-term risk thresholds 0.8, utility weights
# 0.5*goal + progress + 0.5*correctness - crash - 0.5*offroad - 0.5*traffic.
#
#   QWEN_GPU=1 ./.run_carla/qwen_zs_critic_server.sh start
#   ./.run_carla/qwen_zs_critic_server.sh status
#   ./.run_carla/qwen_zs_critic_server.sh stop
#
# The GPU is deliberately not defaulted: pass QWEN_GPU (physical index) at start.
# Weights are read offline from Celine's HF cache (readable, 52 GB) instead of re-downloading.
set -uo pipefail

MODE="${1:-status}"
QWEN_ROOT="${QWEN_ROOT:-/home/cglossop/qwen-critic}"
QWEN_PORT="${QWEN_PORT:-18850}"
QWEN_HF_HOME="${QWEN_HF_HOME:-/raid/users/celine/qwen-critic/huggingface}"
# 27B BF16 weights are ~54 GB; Celine's identical zero-shot server sits at ~78 GB in use.
QWEN_NEED_MIB="${QWEN_NEED_MIB:-80000}"
QWEN_STARTUP_SECS="${QWEN_STARTUP_SECS:-1800}"
QWEN_LOG_DIR="${QWEN_LOG_DIR:-/raid/users/cglossop/sweep_results/qwen_zs_critic}"
CAPTURE_REQUESTS="${CAPTURE_REQUESTS:-}"
URL="http://127.0.0.1:${QWEN_PORT}"
PID_FILE="${QWEN_LOG_DIR}/server_${QWEN_PORT}.pid"
LOG="${QWEN_LOG_DIR}/server_${QWEN_PORT}.log"
mkdir -p "$QWEN_LOG_DIR"

say() { echo "[qwen_zs $(date +%H:%M:%S)] $*"; }

# Exit 0 only if the service on QWEN_PORT is the zero-shot README configuration. Anything else
# (an adapter, another prompt profile, compile on) must never silently score a sweep.
verify_health() {
  local body
  body=$(curl -fsS --max-time 10 "${URL}/health" 2>/dev/null) || return 2
  python3 - "$body" <<'PY'
import json, sys
h = json.loads(sys.argv[1])
want = {
    "adapter": None,
    "prompt_profile": "scene_criteria_v2",
    "traffic_rule_context": "original",
}
bad = [f"{k}={h.get(k)!r} (want {v!r})" for k, v in want.items() if h.get(k) != v]
if (h.get("torch_compile") or {}).get("enabled") is not False:
    bad.append(f"torch_compile={h.get('torch_compile')!r} (want enabled=False)")
if bad:
    print("health mismatch: " + "; ".join(bad), file=sys.stderr)
    sys.exit(1)
PY
}

case "$MODE" in
  status)
    if verify_health; then
      say "OK: zero-shot critic healthy at $URL"
      curl -fsS "${URL}/health"; echo
    else
      say "no healthy zero-shot critic at $URL"; exit 1
    fi
    ;;

  start)
    : "${QWEN_GPU:?QWEN_GPU (physical GPU index) must be set to start the critic}"
    if verify_health; then say "already running and verified at $URL"; exit 0; fi
    if ss -ltn 2>/dev/null | grep -q ":${QWEN_PORT} "; then
      say "ABORT: port ${QWEN_PORT} is taken by something that is not the zero-shot critic"; exit 1
    fi
    waited=0
    while :; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$QWEN_GPU" 2>/dev/null | tr -d ' ')
      [ "${free:-0}" -ge "$QWEN_NEED_MIB" ] && break
      [ "$waited" -eq 0 ] && say "gpu ${QWEN_GPU} has ${free} MiB free (< ${QWEN_NEED_MIB}); waiting"
      waited=$((waited + 1)); sleep 60
    done
    say "starting on gpu ${QWEN_GPU}, port ${QWEN_PORT}; log ${LOG}"
    git -C "$QWEN_ROOT" log -1 --format='qwen-critic %H %s' > "${QWEN_LOG_DIR}/server_${QWEN_PORT}.commit" 2>/dev/null
    # vlcritic is not installed into the venv, so the repo root goes on PYTHONPATH.
    CUDA_VISIBLE_DEVICES="$QWEN_GPU" HF_HOME="$QWEN_HF_HOME" HF_HUB_OFFLINE=1 \
    PYTHONPATH="$QWEN_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      setsid nohup "$QWEN_ROOT/.venv/bin/python" "$QWEN_ROOT/scripts/serve_bon.py" \
        --model Qwen/Qwen3.8-27B --port "$QWEN_PORT" \
        --no-torch-compile --image-size 512 \
        --prompt-profile scene_criteria_v2 \
        --action-representation native_delta_xy_t_delta_xy_space \
        --no-include-current-speed --context-prefix '' \
        --risk-threshold 1.01 --risk-weight 1 \
        --crash-threshold 0.8 --offroad-threshold 0.8 --traffic-threshold 0.8 \
        --crash-weight 1 --offroad-weight 0.5 --traffic-weight 0.5 \
        --goal-weight 0.5 --progress-weight 1 --correctness-weight 0.5 \
        ${CAPTURE_REQUESTS:+--capture-requests "$CAPTURE_REQUESTS"} \
        > "$LOG" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$PID_FILE"
    for _ in $(seq 1 $((QWEN_STARTUP_SECS / 10))); do
      if verify_health 2>/dev/null; then
        curl -fsS "${URL}/health" > "${QWEN_LOG_DIR}/server_${QWEN_PORT}.health.json"
        say "healthy (pid $pid): $(cat "${QWEN_LOG_DIR}/server_${QWEN_PORT}.health.json")"
        exit 0
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        say "ABORT: server exited during startup"; tail -50 "$LOG" >&2; exit 1
      fi
      sleep 10
    done
    say "ABORT: not healthy after ${QWEN_STARTUP_SECS}s"; verify_health; exit 1
    ;;

  stop)
    [ -f "$PID_FILE" ] || { say "no pid file for port ${QWEN_PORT}; nothing of ours to stop"; exit 0; }
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid"
      say "sent SIGTERM to critic pgid $pid"
    fi
    rm -f "$PID_FILE"
    ;;

  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
