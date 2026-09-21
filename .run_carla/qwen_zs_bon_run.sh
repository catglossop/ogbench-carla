#!/usr/bin/env bash
# qwen_zs_bon_run.sh ROUTE GPU SLOT CHECKPOINT CARLA_SEED OUT_DIR
#
# One (route, carla seed) cell of the zero-shot Qwen BoN sweep: load a finished run's
# end-of-training checkpoint and roll it out frozen for N_EVAL episodes, selecting every action
# with the zero-shot Qwen critic. Two actors:
#
#   ACTOR=pi05 (default)  OpenPI SteerVLA checkpoint (<ckpt>/params) with
#                         pi05_steervla_cot_simplified_reasoning_ll_heavy; the original sweep.
#   ACTOR=simlingo        hierarchical SimLingo SteerVLA (qwen-critic README, 2026-09-18): the
#                         checkpoint is the route's fine-tuned HL export (<ckpt>/pytorch_model.bin
#                         + .hydra/); the LL stays the frozen training LL. Agent config
#                         simlingo_steervla_qwen_bon_eval_config.py (CoT temperature 1.0).
#
# Seeds: model seeds are CARLA_SEED+EVAL_SEED_OFFSET, +1, ... (N_EVAL of them) and --train-seed is
# the first of them, because frozen eval samples episode 1 with train_seed. EVAL_SEED_OFFSET=1001
# is the eval-subset contract; EVAL_SEED_OFFSET=0 N_EVAL=1 is the README protocol (one episode,
# carla_seed = train_seed = SEED).
#
# Env: BENCH=b2d|f2d (selects the simulator), QWEN_URL, RUN_GROUP, OGBENCH_SAVE_DIR.
#   b2d -> CARLA 0.9.16 at /home/cglossop/carla
#   f2d -> Fail2Drive CARLA 0.9.15 at /home/cglossop/f2d_carla (CARLA_ROOT and CARLA_0915_ROOT)
# DRY_RUN=1 prints the resolved command instead of running it.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ROUTE="${1:?route}"; GPU="${2:?gpu}"; SLOT="${3:?slot}"
CKPT="${4:?checkpoint}"; CARLA_SEED="${5:?carla seed}"; OUT_DIR="${6:?out dir}"

# Hand-queued extra evals (e.g. an intermediate checkpoint) pass CHECKPOINT as "<dir>#<tag>". The
# tag keeps them out of the route's own cell: results go to <route>__<tag>/carla_seed_N instead of
# overwriting <route>/carla_seed_N, and the tag is added to the W&B run name.
EXTRA_TAG=""
case "$CKPT" in
  *'#'*)
    EXTRA_TAG="${CKPT##*#}"; CKPT="${CKPT%%#*}"
    OUT_DIR="$(dirname "$(dirname "$OUT_DIR")")/${ROUTE}__${EXTRA_TAG}/$(basename "$OUT_DIR")"
    ;;
esac

BENCH="${BENCH:?BENCH must be b2d or f2d}"
QWEN_URL="${QWEN_URL:?QWEN_URL must be set}"
RUN_GROUP="${RUN_GROUP:?RUN_GROUP must be set}"
: "${OGBENCH_SAVE_DIR:?OGBENCH_SAVE_DIR must be set}"
ACTOR="${ACTOR:-pi05}"
ACTOR_CONFIG="${ACTOR_CONFIG:-pi05_steervla_cot_simplified_reasoning_ll_heavy}"
N_EVAL="${N_EVAL:-3}"
N_CANDIDATES="${N_CANDIDATES:-8}"
EVAL_SEED_OFFSET="${EVAL_SEED_OFFSET:-1001}"
# Leaderboard-faithful episode ends: no wrapper step cap (0) -- the route ends only on the
# leaderboard's own criteria (AgentBlockedTest 60 s, InRouteTest, completion, ...). Frozen eval
# still stops at --online-steps even mid-episode, so that budget is a far-off safety net.
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-0}"
if [ "$MAX_EPISODE_STEPS" -gt 0 ]; then
  ONLINE_STEPS="${ONLINE_STEPS:-$((N_EVAL * MAX_EPISODE_STEPS + 1000))}"
else
  ONLINE_STEPS="${ONLINE_STEPS:-$((N_EVAL * 20000 + 1000))}"
fi

FIRST_EVAL_SEED=$((CARLA_SEED + EVAL_SEED_OFFSET))
EVAL_SEEDS=$(seq -s, "$FIRST_EVAL_SEED" $((FIRST_EVAL_SEED + N_EVAL - 1)))
TRAIN_SEED="$FIRST_EVAL_SEED"

# Own port/display range: clear of the training sweeps (16400+/940+), the README defaults
# (18920/992, 27780/835) and Celine's critic servers (1876x/1883x).
CARLA_PORT=$((17400 + SLOT * 20)); TM_PORT=$((17500 + SLOT * 20)); DISPLAY_NUM=$((960 + SLOT))

case "$BENCH" in
  b2d)
    export CARLA_ROOT="${B2D_CARLA_ROOT:-/home/cglossop/carla}"
    unset CARLA_0915_ROOT
    ;;
  f2d)
    # README: for Fail2Drive both roots point at the Fail2Drive simulator.
    export CARLA_ROOT="${F2D_CARLA_0915_ROOT:-/home/cglossop/f2d_carla}"
    export CARLA_0915_ROOT="$CARLA_ROOT"
    [ -n "${F2D_CARLA_0915_PYTHON:-}" ] && export CARLA_0915_PYTHON="$F2D_CARLA_0915_PYTHON"
    ;;
  *) echo "[qwen_zs_run] ABORT: BENCH must be b2d or f2d, got '$BENCH'" >&2; exit 2 ;;
esac

export OGBENCH_SAVE_DIR
export UV_NO_SYNC=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export QWEN_VIDEO_HISTORY=0 QWEN_SPEED_HISTORY=0
export CARLA_DISABLE_RENDER_THREAD_TIMEOUT=1 CARLA_DISABLE_RHI_THREAD=1
# W&B identity: the school account, never the ~/.netrc (catglossop) fallthrough.
export WANDB_API_KEY="$(cat /home/cglossop/.wandb_school_key)"
export WANDB_ENTITY=catherineglossop

EXP_NAME="${ROUTE}-cs${CARLA_SEED}${EXTRA_TAG:+-$EXTRA_TAG}-qwenzs_$(date +%Y%m%d_%H%M%S)"
# Per-job CARLA config with the wrapper's post-collision stuck cutoff disabled, so a blocked ego
# gets the leaderboard's own AgentBlockedTest (0.1 m/s for 60 s). main_carla's --eval_crash_stuck_steps
# does this in-process, but cannot reach the 0.9.15 env subprocess; the yaml is read by both.
LB_CARLA_CONFIG="$OUT_DIR/carla_config_leaderboard_eval.yaml"
# Batched candidate sampling; the mixed actor must sample sequentially (see its case below).
BON_BATCH="${BON_BATCH:-true}"
[ "$ACTOR" = mixed ] && BON_BATCH=false
COMMON=(
  --route "$ROUTE" --carla-config "$LB_CARLA_CONFIG"
  --online-steps "$ONLINE_STEPS" --max-episode-steps "$MAX_EPISODE_STEPS"
  --seed "$CARLA_SEED" --carla-seed "$CARLA_SEED" --train-seed "$TRAIN_SEED"
  --eval-seeds "$EVAL_SEEDS" --eval-mode --frozen-eval true
  --frozen-eval-out "$OUT_DIR" --post-stop-eval-episodes "$N_EVAL"
  --exp-name "$EXP_NAME" --run-group "$RUN_GROUP"
  --save-buffer false --wandb-mode online
  --carla-port "$CARLA_PORT" --carla-streaming-port $((CARLA_PORT + 1))
  --tm-port "$TM_PORT" --x-display-num "$DISPLAY_NUM"
  --critic-mode none --train-mode rl
  --enable-updates false --max-retries 0
  --bon-num-candidates "$N_CANDIDATES" --bon-max-sample-attempts 1
  --bon-batch-policy-candidates "$BON_BATCH" --bon-cot-temperature 1.0
  --bon-candidates-log-every 20 --bon-candidates-wandb false
  --bon-qwen-select true --qwen-bon-url "$QWEN_URL"
  --bon-qwen-cadence 3 --qwen-online-train false
  --terminate-on-collision false --save-video-local true
)

case "$ACTOR" in
  pi05)
    [ -d "$CKPT/params" ] || { echo "[qwen_zs_run] ABORT: no params/ under checkpoint $CKPT" >&2; exit 1; }
    export QWEN_RECORD_PDM_PLAN=0
    # The actor restores inference-only params (SteerVLAActor default load_trainable_params=False).
    CMD=(bash run_carla.sh "${COMMON[@]}" --train-gpu 0 --render-adapter "$GPU"
         --base-only true --eval-only true
         --steervla-checkpoint "$CKPT" --actor-config "$ACTOR_CONFIG" --
         --bon_qwen_label_source=subtask --bon_include_brake_candidate=false
         --agent.steervla.actions_per_cot=5 --agent.steervla.actions_per_model_query=3
         --agent.steervla.proprio_norm=false)
    GPU_ENV=(CUDA_VISIBLE_DEVICES="$GPU")
    ;;
  simlingo)
    [ -f "$CKPT/pytorch_model.bin" ] || { echo "[qwen_zs_run] ABORT: no pytorch_model.bin under $CKPT" >&2; exit 1; }
    # The HL loads from steervla.hl_checkpoint (from SIMLINGO_HL_CHECKPOINT); --steervla-checkpoint
    # is ignored for SimLingo, so the route's fine-tuned HL export goes in through the env.
    export SIMLINGO_HL_CHECKPOINT="$CKPT"
    export SIMLINGO_LL_CHECKPOINT="${SIMLINGO_LL_CHECKPOINT:-/raid/users/celine/steervla-ckpts/2026_05_23_21_39_41_simlingo_ll_vla_meta_conditioned/checkpoints/epoch=029.ckpt}"
    export SIMLINGO_SOURCE_ROOT="${SIMLINGO_SOURCE_ROOT:-/home/cglossop/simlingo-steervla}"
    export PYTHONPATH="$ROOT_DIR:${SIMLINGO_DEPS:-/raid/users/cglossop/ogbench-simlingo-deps}"
    export QWEN_RECORD_PDM_PLAN="${QWEN_RECORD_PDM_PLAN:-1}"
    # README: keep every GPU visible so --train-gpu is the physical index.
    CMD=(bash run_carla.sh "${COMMON[@]}" --train-gpu "$GPU" --render-adapter "$GPU"
         --agent-config impls/configs/simlingo_steervla_qwen_bon_eval_config.py
         --cot-temperature 1.0 --
         --bon_qwen_label_source=subtask --bon_include_brake_candidate=false)
    GPU_ENV=()
    unset CUDA_VISIBLE_DEVICES
    ;;
  mixed)
    # steervla_mixed_eval_config.py: frozen InternVL2 HL (run as a torch worker process by
    # MixedSteerVLAActor) + pi05 vision backbone / action expert. The HL is the route's fine-tuned
    # CAST-relabel export; the LL defaults to the ll_heavy 6000 checkpoint with its training config.
    # The mixed actor only swaps the HL on the batch-1 rollout path (_sample_cot_checked); batched
    # BoN goes through pi05's own _sample_cot and would bypass InternVL2, so candidates must be
    # sampled sequentially. The HL worker's CUDA_VISIBLE_DEVICES is set to training_gpu_rank, so
    # every GPU stays visible and --train-gpu is the physical index.
    [ -f "$CKPT/pytorch_model.bin" ] && [ -f "$CKPT/.hydra/config.yaml" ] \
      || { echo "[qwen_zs_run] ABORT: $CKPT is not a SimLingo HL export" >&2; exit 1; }
    # Default LL = steervla_mixed_eval_config's own pi05 (no_ego_history v1 @ 6000), from the local
    # OpenPI cache; its 200/64/64 prompt/subtask/reasoning budget does not truncate InternVL2 text.
    MIXED_LL_CHECKPOINT="${MIXED_LL_CHECKPOINT:-/raid/users/cglossop/openpi/cat-logs/pi05_steervla_cot_simplified_reasoning_no_ego_history/pi05_steervla_simplified_reasoning_no_ego_history_v1/pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000}"
    MIXED_LL_ACTOR_CONFIG="${MIXED_LL_ACTOR_CONFIG:-pi05_steervla_cot_simplified_reasoning_no_ego_history}"
    MIXED_HL_PYTHON="${MIXED_HL_PYTHON:?MIXED_HL_PYTHON must point at a python with the SimLingo deps}"
    SIMLINGO_SOURCE_ROOT="${SIMLINGO_SOURCE_ROOT:-/home/cglossop/simlingo-steervla}"
    [ -d "$MIXED_LL_CHECKPOINT/params" ] || { echo "[qwen_zs_run] ABORT: no params/ under $MIXED_LL_CHECKPOINT" >&2; exit 1; }
    export QWEN_RECORD_PDM_PLAN=0
    CMD=(bash run_carla.sh "${COMMON[@]}"
         --train-gpu "$GPU" --render-adapter "$GPU"
         --agent-config impls/configs/steervla_mixed_eval_config.py
         --steervla-checkpoint "$MIXED_LL_CHECKPOINT" --actor-config "$MIXED_LL_ACTOR_CONFIG" --
         --bon_qwen_label_source=subtask --bon_include_brake_candidate=false
         --agent.steervla.hl_checkpoint="$CKPT"
         --agent.steervla.simlingo_source_root="$SIMLINGO_SOURCE_ROOT"
         --agent.steervla.hl_python="$MIXED_HL_PYTHON"
         --agent.steervla.actions_per_cot=5 --agent.steervla.actions_per_model_query=3
         --agent.steervla.proprio_norm=false)
    GPU_ENV=()
    unset CUDA_VISIBLE_DEVICES
    ;;
  *) echo "[qwen_zs_run] ABORT: ACTOR must be pi05, simlingo or mixed, got '$ACTOR'" >&2; exit 2 ;;
esac

# Extra main_carla flags, appended after the `--` every actor's CMD ends with
# (e.g. EXTRA_MAIN_FLAGS="--bon_call_log=true").
if [ -n "${EXTRA_MAIN_FLAGS:-}" ]; then
  read -r -a _extra_main_flags <<< "$EXTRA_MAIN_FLAGS"
  CMD+=("${_extra_main_flags[@]}")
fi

echo "[qwen_zs_run] actor=$ACTOR bench=$BENCH route=$ROUTE gpu=$GPU slot=$SLOT rpc=$CARLA_PORT tm=$TM_PORT display=:$DISPLAY_NUM"
echo "[qwen_zs_run] carla_root=$CARLA_ROOT carla_0915_root=${CARLA_0915_ROOT:-<unset>} ckpt=$CKPT candidates=$N_CANDIDATES"
[ "$ACTOR" = simlingo ] && echo "[qwen_zs_run] simlingo hl=$SIMLINGO_HL_CHECKPOINT ll=$SIMLINGO_LL_CHECKPOINT src=$SIMLINGO_SOURCE_ROOT"
echo "[qwen_zs_run] carla_seed=$CARLA_SEED train_seed=$TRAIN_SEED eval_seeds=$EVAL_SEEDS online_steps=$ONLINE_STEPS"
echo "[qwen_zs_run] critic=$QWEN_URL group=$RUN_GROUP summary -> $OUT_DIR/run_summary_frozen_eval.json"

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf '%s ' "${GPU_ENV[@]}"; printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

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
sed -E -e 's/^crash_stuck_steps:.*/crash_stuck_steps: 1000000000/' \
       -e "s/^max_episode_steps:.*/max_episode_steps: ${MAX_EPISODE_STEPS}/" \
       impls/configs/carla_config.yaml > "$LB_CARLA_CONFIG"
grep -q '^crash_stuck_steps: 1000000000$' "$LB_CARLA_CONFIG" \
  && grep -q "^max_episode_steps: ${MAX_EPISODE_STEPS}\$" "$LB_CARLA_CONFIG" \
  || { echo "[qwen_zs_run] ABORT: could not write leaderboard eval config $LB_CARLA_CONFIG" >&2; exit 1; }
if [ "${#GPU_ENV[@]}" -gt 0 ]; then exec env "${GPU_ENV[@]}" "${CMD[@]}"; else exec "${CMD[@]}"; fi
