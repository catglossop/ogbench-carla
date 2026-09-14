#!/usr/bin/env bash
# qwen_zs_bon_run.sh ROUTE GPU SLOT CHECKPOINT CARLA_SEED OUT_DIR
#
# One (route, carla seed) cell of the zero-shot Qwen BoN sweep: load a finished run's
# end-of-training SteerVLA checkpoint and roll it out frozen for N_EVAL episodes, selecting every
# action with the zero-shot Qwen critic. Actor/BoN flags are the qwen-critic README launch, with
# three deliberate differences:
#   * the actor is the per-route fine-tuned checkpoint with its training config
#     (pi05_steervla_cot_simplified_reasoning_ll_heavy), not the base 6000 checkpoint;
#   * --frozen-eval replaces --max-episodes 1: N_EVAL episodes replaying CARLA_SEED, one model
#     seed per episode, summary written to OUT_DIR/run_summary_frozen_eval.json;
#   * seeds follow the eval-subset contract (carla seed N -> model seeds N+1001, N+1002, ...), and
#     --train-seed is set to the FIRST eval seed. Frozen eval never re-seeds episode 1 (it starts
#     from train_seed; eval seeds are applied only at later resets), so this is what makes every
#     episode actually sample with the seed recorded for it.
#
# Env: BENCH=b2d|f2d (selects the simulator), QWEN_URL, RUN_GROUP, OGBENCH_SAVE_DIR.
#   b2d -> CARLA 0.9.16 at /home/cglossop/carla (what the b2d checkpoints trained on)
#   f2d -> Fail2Drive CARLA 0.9.15 at /home/cglossop/f2d_carla via CARLA_0915_ROOT
# DRY_RUN=1 prints the resolved command instead of running it.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ROUTE="${1:?route}"; GPU="${2:?gpu}"; SLOT="${3:?slot}"
CKPT="${4:?checkpoint}"; CARLA_SEED="${5:?carla seed}"; OUT_DIR="${6:?out dir}"

BENCH="${BENCH:?BENCH must be b2d or f2d}"
QWEN_URL="${QWEN_URL:?QWEN_URL must be set}"
RUN_GROUP="${RUN_GROUP:?RUN_GROUP must be set}"
: "${OGBENCH_SAVE_DIR:?OGBENCH_SAVE_DIR must be set}"
ACTOR_CONFIG="${ACTOR_CONFIG:-pi05_steervla_cot_simplified_reasoning_ll_heavy}"
N_EVAL="${N_EVAL:-3}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-4000}"
# Frozen eval stops at --online-steps even mid-episode, so budget every episode to its cap.
ONLINE_STEPS="${ONLINE_STEPS:-$((N_EVAL * MAX_EPISODE_STEPS + 1000))}"

FIRST_EVAL_SEED=$((CARLA_SEED + 1001))
EVAL_SEEDS=$(seq -s, "$FIRST_EVAL_SEED" $((FIRST_EVAL_SEED + N_EVAL - 1)))
TRAIN_SEED="$FIRST_EVAL_SEED"

# Own port/display range: clear of the training sweeps (16400+/940+), the README defaults
# (27780/835) and Celine's critic servers (1876x/1883x).
CARLA_PORT=$((17400 + SLOT * 20)); TM_PORT=$((17500 + SLOT * 20)); DISPLAY_NUM=$((960 + SLOT))

case "$BENCH" in
  b2d)
    export CARLA_ROOT="${B2D_CARLA_ROOT:-/home/cglossop/carla}"
    unset CARLA_0915_ROOT
    ;;
  f2d)
    export CARLA_0915_ROOT="${F2D_CARLA_0915_ROOT:-/home/cglossop/f2d_carla}"
    ;;
  *) echo "[qwen_zs_run] ABORT: BENCH must be b2d or f2d, got '$BENCH'" >&2; exit 2 ;;
esac

export OGBENCH_SAVE_DIR
export UV_NO_SYNC=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export QWEN_VIDEO_HISTORY=0 QWEN_SPEED_HISTORY=0 QWEN_RECORD_PDM_PLAN=0
export CARLA_DISABLE_RENDER_THREAD_TIMEOUT=1
# W&B identity: the school account, never the ~/.netrc (catglossop) fallthrough.
export WANDB_API_KEY="$(cat /home/cglossop/.wandb_school_key)"
export WANDB_ENTITY=catherineglossop

EXP_NAME="${ROUTE}-cs${CARLA_SEED}-qwenzs_$(date +%Y%m%d_%H%M%S)"
CMD=(
  bash run_carla.sh
  --route "$ROUTE" --carla-config impls/configs/carla_config.yaml
  --online-steps "$ONLINE_STEPS" --max-episode-steps "$MAX_EPISODE_STEPS"
  --seed "$CARLA_SEED" --carla-seed "$CARLA_SEED" --train-seed "$TRAIN_SEED"
  --eval-seeds "$EVAL_SEEDS" --eval-mode --frozen-eval true
  --frozen-eval-out "$OUT_DIR" --post-stop-eval-episodes "$N_EVAL"
  --exp-name "$EXP_NAME" --run-group "$RUN_GROUP"
  --save-buffer false --wandb-mode online
  --train-gpu 0 --render-adapter "$GPU"
  --carla-port "$CARLA_PORT" --carla-streaming-port $((CARLA_PORT + 1))
  --tm-port "$TM_PORT" --x-display-num "$DISPLAY_NUM"
  --critic-mode none --train-mode rl
  --enable-updates false --base-only true --max-retries 0
  --steervla-checkpoint "$CKPT" --actor-config "$ACTOR_CONFIG"
  --bon-num-candidates 8 --bon-max-sample-attempts 1
  --bon-batch-policy-candidates true --bon-cot-temperature 1.0
  --bon-candidates-log-every 20 --bon-candidates-wandb false
  --bon-qwen-select true --qwen-bon-url "$QWEN_URL"
  --bon-qwen-cadence 3 --qwen-online-train false
  --terminate-on-collision false --save-video-local true --eval-only true --
  --bon_qwen_label_source=subtask --bon_include_brake_candidate=false
  --agent.steervla.actions_per_cot=5
  --agent.steervla.actions_per_model_query=3
  --agent.steervla.proprio_norm=false
)

echo "[qwen_zs_run] bench=$BENCH route=$ROUTE gpu=$GPU slot=$SLOT rpc=$CARLA_PORT tm=$TM_PORT display=:$DISPLAY_NUM"
echo "[qwen_zs_run] carla_root=${CARLA_0915_ROOT:-$CARLA_ROOT} ckpt=$CKPT actor_config=$ACTOR_CONFIG"
echo "[qwen_zs_run] carla_seed=$CARLA_SEED train_seed=$TRAIN_SEED eval_seeds=$EVAL_SEEDS online_steps=$ONLINE_STEPS"
echo "[qwen_zs_run] critic=$QWEN_URL group=$RUN_GROUP summary -> $OUT_DIR/run_summary_frozen_eval.json"

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf 'CUDA_VISIBLE_DEVICES=%s' "$GPU"; printf ' %q' "${CMD[@]}"; echo
  exit 0
fi

[ -d "$CKPT/params" ] || { echo "[qwen_zs_run] ABORT: no params/ under checkpoint $CKPT" >&2; exit 1; }
if ! QWEN_PORT="${QWEN_URL##*:}" "$ROOT_DIR/.run_carla/qwen_zs_critic_server.sh" status >/dev/null 2>&1; then
  QWEN_PORT="${QWEN_URL##*:}" "$ROOT_DIR/.run_carla/qwen_zs_critic_server.sh" status >&2
  echo "[qwen_zs_run] ABORT: $QWEN_URL is not a healthy zero-shot critic" >&2; exit 1
fi
for p in "$CARLA_PORT" $((CARLA_PORT + 1)) "$TM_PORT"; do
  if ss -ltn 2>/dev/null | grep -q ":${p} "; then
    echo "[qwen_zs_run] ABORT: port $p already listening" >&2; exit 1; fi
done
LOCK="/tmp/.X${DISPLAY_NUM}-lock"
if [ -e "$LOCK" ]; then
  if pgrep -u "$USER" -f "Xvfb :${DISPLAY_NUM}\b" >/dev/null 2>&1 \
     || pgrep -u "$USER" -f "carla-rpc-port=${CARLA_PORT}\b" >/dev/null 2>&1; then
    echo "[qwen_zs_run] ABORT: display :$DISPLAY_NUM genuinely in use" >&2; exit 1
  fi
  if [ -O "$LOCK" ]; then
    echo "[qwen_zs_run] reclaiming stale lock $LOCK" >&2; rm -f "$LOCK"
  else
    echo "[qwen_zs_run] ABORT: $LOCK held by another user" >&2; exit 1
  fi
fi

mkdir -p "$OUT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" exec "${CMD[@]}"
