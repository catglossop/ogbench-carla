#!/usr/bin/env bash
# mixed_method_run.sh METHOD ROUTE GPU SLOT OUT_DIR EXP_NAME
#
# One reportable cell of the mixed-policy Residual-RL / YAY-Robot sweep, in the style of
# eval_subset_run.sh (the HARP sweep's per-cell runner): seed contract, school W&B identity, stale-lock
# reclaim, port check, one GPU masked in. The run itself is exactly what Surya's run_residual_eval.sh /
# run_yay_robot_eval.sh pass to run_carla.sh (10k online steps, 4000-step episodes, 3 post-stop eval
# episodes, --eval-mode, --max-retries 50), with the mixed-policy agent configs.
#
#   seed N -> carla_seed = N, train_seed = N, eval_seeds = N+1001, N+1002, N+1003   (sweep: N = 0)
#
# METHOD: residual | yay.  Simulator: CARLA_ROOT (0.9.16 + f2d content) for every route except
# generalization-animals-*, which needs the Fail2Drive 0.9.15 build (F2D_CARLA_0915_ROOT).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

METHOD="${1:?usage: mixed_method_run.sh METHOD ROUTE GPU SLOT OUT_DIR EXP_NAME}"
ROUTE="${2:?route}"; GPU="${3:?gpu}"; SLOT="${4:?slot}"; OUT_DIR="${5:?out dir}"; EXP_NAME="${6:?exp name}"

SEED="${SEED:-0}"
EVAL_SEEDS="$((SEED + 1001)),$((SEED + 1002)),$((SEED + 1003))"
CARLA_PORT=$((${CARLA_PORT_BASE:-41000} + SLOT * 20))
TM_PORT=$((${TM_PORT_BASE:-47000} + SLOT * 20))
DISPLAY_NUM=$((${DISPLAY_BASE:-850} + SLOT))

export CARLA_ROOT="${CARLA_ROOT:-/home/cglossop/carla}"
export CARLA_PYTHON_API_ROOT="${CARLA_ROOT}/PythonAPI/carla"
export OGBENCH_SAVE_DIR="$OUT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# The interpreter the sweep runs on, pinned (sweep_extra_worker.sh note 1: a clean env made uv build a
# bare worktree .venv and every route died on `import flax`).
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
# W&B identity: the school account, never ~/.netrc (catglossop). Key never echoed.
export WANDB_API_KEY="$(cat /home/cglossop/.wandb_school_key)"
export WANDB_ENTITY=catherineglossop
# YAY corrector model; --eval-mode forces EVAL_MODE_GEMINI_MODEL, which follows this.
export OGBENCH_GEMINI_MODEL="${OGBENCH_GEMINI_MODEL:-gemini-3.8-flash}"

unset CARLA_0915_ROOT CARLA_0915_PYTHON FAIL2DRIVE_CARLA_ROOT
if [[ "$ROUTE" == generalization-animals-* ]]; then
  export CARLA_0915_ROOT="${F2D_CARLA_0915_ROOT:-/home/cglossop/f2d_carla}"
  export CARLA_0915_PYTHON="${F2D_CARLA_0915_PYTHON:-$PWD/.venv-carla-0915/bin/python}"
  echo "[mixed_run] $ROUTE: CARLA 0.9.15 (CARLA_0915_ROOT=$CARLA_0915_ROOT)"
fi

case "$METHOD" in
  residual)
    AGENT_CFG=impls/configs/steervla_mixed_residual_sweep_config.py
    RUN_GROUP=mixed_residual_rl_b2d_f2d_cs0
    METHOD_ARGS=(--save-buffer true)
    # Crash-resume (resume_state.json + agent + buffer, committed every resume_interval = 1000 steps): a retry
    # continues the same run dir / W&B run from its last commit instead of re-training from step 0. The
    # Bench2Drive BackgroundActivity junction-topology hang at an episode start otherwise costs the whole cell.
    # Retries therefore keep the first attempt's exp name (the sweep driver appends __aN).
    EXP_NAME="${EXP_NAME%%__a[0-9]*}"
    export OGBENCH_RESUME=true   # run_carla.sh passes --resume last, so set it there, not via `--`
    TAIL_ARGS=()
    ;;
  yay)
    : "${GEMINI_API_KEY:?GEMINI_API_KEY must be exported -- the YAY corrector is a Gemini client}"
    AGENT_CFG=impls/configs/steervla_mixed_yay_robot_train_config.py
    RUN_GROUP=mixed_yay_robot_b2d_f2d_cs0
    # Surya's single-GPU YAY setting: JAX (pi05) capped so the InternVL2 worker + CARLA fit beside it.
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${YAY_JAX_MEM_FRACTION:-0.30}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    METHOD_ARGS=(--hl-gpu 0 --train-mode rl --critic-mode none --save-buffer false)
    TAIL_ARGS=()
    ;;
  *) echo "[mixed_run] ABORT: METHOD must be residual or yay, got $METHOD" >&2; exit 2 ;;
esac

[ -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ] || { echo "[mixed_run] ABORT: no python in $UV_PROJECT_ENVIRONMENT" >&2; exit 1; }
for p in "$CARLA_PORT" $((CARLA_PORT + 1)) "$TM_PORT"; do
  if ss -ltn 2>/dev/null | grep -q ":${p} "; then
    echo "[mixed_run] ABORT: port $p already listening" >&2; exit 1; fi
done
# A lock is only a real conflict if something holds the display (eval_subset_run.sh, 2026-09-10).
LOCK="/tmp/.X${DISPLAY_NUM}-lock"
if [ -e "$LOCK" ]; then
  if pgrep -u "$USER" -f "Xvfb :${DISPLAY_NUM}( |$)" >/dev/null 2>&1 \
     || pgrep -u "$USER" -f "carla-rpc-port=${CARLA_PORT}( |$)" >/dev/null 2>&1; then
    echo "[mixed_run] ABORT: display :$DISPLAY_NUM genuinely in use" >&2; exit 1
  fi
  if [ -O "$LOCK" ]; then
    echo "[mixed_run] reclaiming stale lock $LOCK" >&2; rm -f "$LOCK"
  else
    echo "[mixed_run] ABORT: $LOCK held by another user" >&2; exit 1
  fi
fi

mkdir -p "$OUT_DIR"
echo "[mixed_run] tail args: ${TAIL_ARGS[*]:-none}"
echo "[mixed_run] method=$METHOD route=$ROUTE gpu=$GPU slot=$SLOT exp=$EXP_NAME"
echo "[mixed_run] seed=$SEED -> carla_seed=$SEED train_seed=$SEED eval_seeds=$EVAL_SEEDS"
echo "[mixed_run] rpc=$CARLA_PORT tm=$TM_PORT display=:$DISPLAY_NUM group=$RUN_GROUP gemini=$OGBENCH_GEMINI_MODEL"

# CUDA_VISIBLE_DEVICES masks to one card; --train-gpu/--hl-gpu index into it, --render-adapter is physical.
CUDA_VISIBLE_DEVICES="$GPU" exec bash run_carla.sh \
  --agent-config "$AGENT_CFG" --route "$ROUTE" \
  --carla-seed "$SEED" --train-seed "$SEED" --eval-seeds "$EVAL_SEEDS" --seed "$SEED" \
  --exp-name "$EXP_NAME" --online-steps 10000 --max-episode-steps 4000 \
  --train-gpu 0 --render-adapter "$GPU" \
  --carla-port "$CARLA_PORT" --carla-streaming-port $((CARLA_PORT + 1)) \
  --tm-port "$TM_PORT" --x-display-num "$DISPLAY_NUM" \
  --post-stop-eval-episodes 3 --run-group "$RUN_GROUP" --eval-mode \
  --save-video-local true --wandb-mode online --max-retries 50 \
  "${METHOD_ARGS[@]}" "${TAIL_ARGS[@]}"
