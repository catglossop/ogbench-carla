#!/usr/bin/env bash
# Paired experiment: train CAST/HL-DAgger, export that policy, then train residual
# SAC from the exported frozen policy. The two stages are separate by design so the
# residual result measures transfer from a CAST checkpoint, not joint CAST+RL updates.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PHYSICAL_GPU="${1:-${CAST_RESIDUAL_GPU:-5}}"
RUN_ROOT="${2:-${CAST_RESIDUAL_ROOT:-/raid/users/${USER}/carla_exps/cast_to_residual}}"
EXPERIMENT_LABEL="${CAST_RESIDUAL_LABEL:-cast-to-residual-20260908}"
if [[ -n "${CAST_RESIDUAL_NONCE:-}" ]]; then
  RUN_NONCE="$CAST_RESIDUAL_NONCE"
else
  printf -v RUN_NONCE '%05d%05d' "$RANDOM" "$RANDOM"
fi
[[ "$RUN_NONCE" =~ ^[0-9]+$ ]] || { echo 'CAST_RESIDUAL_NONCE must be numeric.' >&2; exit 2; }

# Pilot defaults; pass CAST_RESIDUAL_ROUTES="route-a route-b" and/or
# CAST_RESIDUAL_SEEDS="0 1 2" to expand the study without changing the script.
CAST_RESIDUAL_ROUTES="${CAST_RESIDUAL_ROUTES:-construction-obstacle-002}"
CAST_RESIDUAL_SEEDS="${CAST_RESIDUAL_SEEDS:-0}"
CAST_RUN_GROUP="${CAST_RUN_GROUP:-CastRelabelHL16Adaptive}"
RESIDUAL_RUN_GROUP="${RESIDUAL_RUN_GROUP:-ResidualRLFromCastRelabel}"
MAX_RETRIES="${MAX_RETRIES:-50}"
WANDB_MODE="${WANDB_MODE:-online}"
SAVE_VIDEO_LOCAL="${SAVE_VIDEO_LOCAL:-true}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-true}"
DRY_RUN="${DRY_RUN:-0}"

# Dedicated range, intentionally separate from the ordinary sweep/eval ranges.
CARLA_PORT="${CAST_RESIDUAL_CARLA_PORT:-16200}"
CARLA_STREAMING_PORT="${CAST_RESIDUAL_STREAMING_PORT:-$((CARLA_PORT + 1))}"
TM_PORT="${CAST_RESIDUAL_TM_PORT:-$((CARLA_PORT + 100))}"
X_DISPLAY_NUM="${CAST_RESIDUAL_X_DISPLAY_NUM:-780}"

read -r -a ROUTES <<< "$CAST_RESIDUAL_ROUTES"
read -r -a SEEDS <<< "$CAST_RESIDUAL_SEEDS"
(( ${#ROUTES[@]} > 0 && ${#SEEDS[@]} > 0 )) || { echo 'Route and seed lists must be non-empty.' >&2; exit 2; }

mkdir -p "$RUN_ROOT/logs/$EXPERIMENT_LABEL" "$RUN_ROOT/status/$EXPERIMENT_LABEL" "$RUN_ROOT/runs/$EXPERIMENT_LABEL"
cat > "$RUN_ROOT/experiment_spec_${EXPERIMENT_LABEL}.txt" <<SPEC
created=$(date --iso-8601=seconds)
experiment_label=$EXPERIMENT_LABEL
run_nonce=$RUN_NONCE
routes=${ROUTES[*]}
seeds=${SEEDS[*]}
cast_agent_config=impls/configs/steervla_cast_relabel_hl16_adaptive_config.py
residual_agent_config=impls/configs/steervla_residual_from_cast_config.py
cast_online_steps=10000
residual_online_steps=10000
max_episode_steps=4000
cast_hl_checkpoint_every_steps=1000
cast_max_hl_updates=150
cast_stop_on_driving_score=100
cast_updates_after_driving_score=20
residual_accel_scale=0.2
residual_steer_scale=0.2
residual_bc_beta=0.1
residual_warmup_steps=1000
residual_ramp_steps=1500
SPEC

total=$(( ${#ROUTES[@]} * ${#SEEDS[@]} ))
index=0
for route in "${ROUTES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    index=$((index + 1))
    tag="${route}__seed-${seed}"
    complete_file="$RUN_ROOT/status/$EXPERIMENT_LABEL/${tag}.done"
    failure_file="$RUN_ROOT/status/$EXPERIMENT_LABEL/${tag}.failed"
    cast_log="$RUN_ROOT/logs/$EXPERIMENT_LABEL/${tag}__cast.log"
    residual_log="$RUN_ROOT/logs/$EXPERIMENT_LABEL/${tag}__residual.log"
    cast_dir="$RUN_ROOT/runs/$EXPERIMENT_LABEL/${tag}/cast"
    policy_dir="$RUN_ROOT/runs/$EXPERIMENT_LABEL/${tag}/cast_policy"
    residual_dir="$RUN_ROOT/runs/$EXPERIMENT_LABEL/${tag}/residual"
    if [[ "$SKIP_COMPLETED" == "1" && -f "$complete_file" ]]; then
      echo "[$index/$total] SKIP completed: $tag"
      continue
    fi

    cast_name="cast-relabel_${EXPERIMENT_LABEL}_sd-${seed}_${route}_${RUN_NONCE}"
    residual_name="residual-from-cast_${EXPERIMENT_LABEL}_sd-${seed}_${route}_${RUN_NONCE}"
    echo "[$index/$total] START CAST: $tag (wandb_name=$cast_name)"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[$index/$total] DRY RUN residual_wandb_name=$residual_name"
      continue
    fi

    mkdir -p "$cast_dir" "$policy_dir" "$residual_dir"
    set +e
    CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" OGBENCH_SAVE_DIR="$cast_dir" \
      ./run_carla.sh \
        --route "$route" --seed "$seed" --exp-name "$cast_name" \
        --train-gpu 0 --hl-gpu 0 --render-adapter "$PHYSICAL_GPU" \
        --carla-port "$CARLA_PORT" --carla-streaming-port "$CARLA_STREAMING_PORT" \
        --tm-port "$TM_PORT" --x-display-num "$X_DISPLAY_NUM" \
        --agent-config impls/configs/steervla_cast_relabel_hl16_adaptive_config.py \
        --train-mode rl --critic-mode none --online-steps 10000 --max-episode-steps 4000 \
        --hl-ckpt-dir "$policy_dir" --hl-ckpt-every 1000 --hl-ckpt-keep-last 1 \
        --save-buffer false --save-video-local "$SAVE_VIDEO_LOCAL" \
        --run-group "$CAST_RUN_GROUP" --wandb-mode "$WANDB_MODE" --max-retries "$MAX_RETRIES" \
        -- --max_hl_updates=150 --stop_on_driving_score=100 --updates_after_driving_score=20 \
        2>&1 | tee "$cast_log"
    cast_code=${PIPESTATUS[0]} tee_code=${PIPESTATUS[1]}
    set -e
    if [[ "$cast_code" -ne 0 || "$tee_code" -ne 0 ]]; then
      printf 'stage=cast\ntimestamp=%s\nrun_exit_code=%s\ntee_exit_code=%s\n' "$(date --iso-8601=seconds)" "$cast_code" "$tee_code" > "$failure_file"
      echo "[$index/$total] CAST FAILED: $tag" >&2
      [[ "$CONTINUE_ON_FAILURE" == "true" || "$CONTINUE_ON_FAILURE" == "1" ]] && continue || exit "$cast_code"
    fi

    latest_step="$(find "$policy_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | awk '/^[0-9]+$/' | sort -n | tail -1)"
    cast_checkpoint="$policy_dir/$latest_step"
    if [[ -z "$latest_step" || ! -d "$cast_checkpoint/params" ]]; then
      printf 'stage=cast_checkpoint\ntimestamp=%s\npolicy_dir=%s\n' "$(date --iso-8601=seconds)" "$policy_dir" > "$failure_file"
      echo "[$index/$total] CAST produced no usable params-only checkpoint: $policy_dir" >&2
      [[ "$CONTINUE_ON_FAILURE" == "true" || "$CONTINUE_ON_FAILURE" == "1" ]] && continue || exit 1
    fi

    echo "[$index/$total] START RESIDUAL: $tag checkpoint=$cast_checkpoint (wandb_name=$residual_name)"
    set +e
    CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" OGBENCH_SAVE_DIR="$residual_dir" \
      ./run_carla.sh \
        --route "$route" --seed "$seed" --exp-name "$residual_name" \
        --train-gpu 0 --render-adapter "$PHYSICAL_GPU" \
        --carla-port "$CARLA_PORT" --carla-streaming-port "$CARLA_STREAMING_PORT" \
        --tm-port "$TM_PORT" --x-display-num "$X_DISPLAY_NUM" \
        --agent-config impls/configs/steervla_residual_from_cast_config.py \
        --steervla-checkpoint "$cast_checkpoint" \
        --online-steps 10000 --max-episode-steps 4000 \
        --save-buffer true --save-video-local "$SAVE_VIDEO_LOCAL" \
        --run-group "$RESIDUAL_RUN_GROUP" --wandb-mode "$WANDB_MODE" --max-retries "$MAX_RETRIES" \
        2>&1 | tee "$residual_log"
    residual_code=${PIPESTATUS[0]} tee_code=${PIPESTATUS[1]}
    set -e
    if [[ "$residual_code" -eq 0 && "$tee_code" -eq 0 ]]; then
      touch "$complete_file"
      echo "[$index/$total] DONE: $tag"
      continue
    fi
    printf 'stage=residual\ntimestamp=%s\nrun_exit_code=%s\ntee_exit_code=%s\n' "$(date --iso-8601=seconds)" "$residual_code" "$tee_code" > "$failure_file"
    echo "[$index/$total] RESIDUAL FAILED: $tag" >&2
    [[ "$CONTINUE_ON_FAILURE" == "true" || "$CONTINUE_ON_FAILURE" == "1" ]] && continue || exit "$residual_code"
  done
done

echo "CAST-to-residual experiment complete: $RUN_ROOT"
