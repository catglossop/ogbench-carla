#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /raid/.../step-N/adapter [server_gpu route1_gpu route2_gpu]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER="$(realpath "$1")"
SERVER_GPU="${2:-0}"
ROUTE1_GPU="${3:-2}"
ROUTE2_GPU="${4:-3}"
QWEN_ROOT="/home/celine/qwen-critic"
RUN_ROOT="/raid/users/celine/qwen-bon-online-20k"
SERVER_PORT="${QWEN_SERVER_PORT:-18766}"
REUSE_QWEN_SERVER="${REUSE_QWEN_SERVER:-0}"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/server" "$RUN_ROOT/enter-actor-flow-004" \
  "$RUN_ROOT/generalization-wall-1095"

if [[ "$REUSE_QWEN_SERVER" == "1" ]]; then
  curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >"$RUN_ROOT/server/health.json"
  SERVER_PID="reused"
else
  HF_HOME=/raid/users/celine/qwen-critic/huggingface \
  CUDA_VISIBLE_DEVICES="$SERVER_GPU" \
  nohup "$QWEN_ROOT/.venv/bin/python" "$QWEN_ROOT/scripts/serve_bon.py" \
    --adapter "$ADAPTER" --port "$SERVER_PORT" --online-train \
    --online-lr 2e-6 --online-update-every 20 --online-batch-size 1 \
    --online-max-updates 300 --online-save-every 25 \
    --online-output-dir "$RUN_ROOT/qwen-online-checkpoints" \
    --wandb-name qwen38-step60-online-20k \
    >"$RUN_ROOT/logs/qwen-server.log" 2>&1 < /dev/null &
  SERVER_PID=$!
  echo "$SERVER_PID" >"$RUN_ROOT/server/server.pid"

  for _attempt in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >"$RUN_ROOT/server/health.json"; then
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Qwen server exited during startup" >&2
      tail -100 "$RUN_ROOT/logs/qwen-server.log" >&2
      exit 1
    fi
    sleep 2
  done
fi
curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null

COMMON=(
  --agent-config impls/configs/steervla_bon_online_critic_noexp_waypoints_config.py
  --train-mode rl
  --online-steps 20000
  --eval-only false
  --enable-updates true
  --bon-online-critic true
  --bon-qwen-select true
  --qwen-bon-url "http://127.0.0.1:${SERVER_PORT}"
  --bon-num-candidates 8
  --bon-max-sample-attempts 1
  --bon-qwen-cadence 5
  --qwen-online-train true
  --qwen-online-warmup-episodes 2
  --bon-candidates-log-every 5
  --save-buffer true
  --save-video-local true
  --wandb-mode online
  --max-retries 50
  --seed 0
)

cd "$ROOT_DIR"
OGBENCH_SAVE_DIR="$RUN_ROOT/enter-actor-flow-004" \
nohup ./run_carla.sh "${COMMON[@]}" \
  --route enter-actor-flow-004 --train-gpu "$ROUTE1_GPU" --render-adapter "$ROUTE1_GPU" \
  --carla-port 2140 --carla-streaming-port 2141 --tm-port 8140 --x-display-num 44 \
  >"$RUN_ROOT/logs/enter-actor-flow-004.log" 2>&1 < /dev/null &
ROUTE1_PID=$!
echo "$ROUTE1_PID" >"$RUN_ROOT/enter-actor-flow-004/launcher.pid"

OGBENCH_SAVE_DIR="$RUN_ROOT/generalization-wall-1095" \
nohup ./run_carla.sh "${COMMON[@]}" \
  --route generalization-wall-1095 --train-gpu "$ROUTE2_GPU" --render-adapter "$ROUTE2_GPU" \
  --carla-port 2160 --carla-streaming-port 2161 --tm-port 8160 --x-display-num 46 \
  >"$RUN_ROOT/logs/generalization-wall-1095.log" 2>&1 < /dev/null &
ROUTE2_PID=$!
echo "$ROUTE2_PID" >"$RUN_ROOT/generalization-wall-1095/launcher.pid"

cat <<EOF
Qwen server PID: $SERVER_PID (GPU $SERVER_GPU)
enter-actor-flow-004 PID: $ROUTE1_PID (GPU $ROUTE1_GPU)
generalization-wall-1095 PID: $ROUTE2_PID (GPU $ROUTE2_GPU)
Logs: $RUN_ROOT/logs
EOF
