#!/usr/bin/env bash
# Bench2Drive + Fail2Drive checkpoint-evaluation matrix.
# For each train seed: visit every route; train once; then run three frozen
# checkpoint evaluations with the simulator seed held at the training seed.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU="${1:-${F2D_EVAL_GPU:-0}}"
METHOD="${F2D_METHOD:-residual}"
[[ "$METHOD" == residual || "$METHOD" == cast_to_residual ]] || { echo "F2D_METHOD must be residual or cast_to_residual." >&2; exit 2; }
ROOT="${B2D_F2D_EVAL_ROOT:-/raid/users/${USER}/carla_exps/evals/b2d_f2d_${METHOD}}"
LABEL="${B2D_F2D_EVAL_LABEL:-b2d-f2d-${METHOD}-checkpoint-eval-20260911}"
TRAIN_SEEDS="${F2D_TRAIN_SEEDS:-0 1 2}"
EVAL_SEEDS="${F2D_MODEL_EVAL_SEEDS:-0 1 2}"
WANDB_MODE="${WANDB_MODE:-online}"
DRY_RUN="${DRY_RUN:-0}"

# Earlier 23-route Bench2Drive subset plus the supplied 18-route Fail2Drive subset.
B2D_ROUTES="signalized-junction-left-turn-001 enter-actor-flow-004 non-signalized-junction-right-turn-001 non-signalized-junction-left-turn-enter-flow-002 signalized-junction-left-turn-enter-flow-003 non-signalized-junction-left-turn-002 signalized-junction-right-turn-004 parked-obstacle-004 accident-two-ways-002 construction-obstacle-003 highway-exit-002 pedestrian-crossing-004 parking-exit-002 static-cut-in-001 vanilla-signalized-turn-encounter-red-light-002 accident-005 crossing-bicycle-flow-004 vanilla-signalized-turn-encounter-green-light-004 merger-into-slow-traffic-001 vehicle-turning-route-pedestrian-005 vehicle-opens-door-two-ways-005 sequential-lane-change-005 interurban-actor-flow-004"
F2D_ROUTES="generalization-construction-permutations-1019 generalization-custom-obstacles-1020 generalization-pedestrians-on-road-1085 generalization-construction-pedestrian-1011 generalization-pedestrian-crowd-1069 generalization-custom-obstacles-1024 generalization-fully-blocked-1032 generalization-hard-brake-1036 generalization-bad-parking-1009 generalization-pedestrian-other-blocker-1072 generalization-right-construction-1093 generalization-right-of-way-1056 generalization-wall-1095 generalization-image-on-object-1041 generalization-obscured-stop-1048 generalization-bad-parking-1004 generalization-animals-1083 generalization-wall-1097"

ROUTES="$B2D_ROUTES $F2D_ROUTES"
read -r -a TRAIN_LIST <<< "$TRAIN_SEEDS"
read -r -a EVAL_LIST <<< "$EVAL_SEEDS"
read -r -a ROUTE_LIST <<< "$ROUTES"
mkdir -p "$ROOT/logs/$LABEL" "$ROOT/status/$LABEL" "$ROOT/runs/$LABEL"

CARLA_PORT="${F2D_EVAL_CARLA_PORT:-$((15000 + GPU * 20))}"
STREAM_PORT="${F2D_EVAL_STREAMING_PORT:-$((CARLA_PORT + 1))}"
TM_PORT="${F2D_EVAL_TM_PORT:-$((21000 + GPU * 20))}"
DISPLAY="${F2D_EVAL_X_DISPLAY_NUM:-$((700 + GPU))}"

run_logged() {
  local log="$1"; shift
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY RUN:' | tee -a "$log"; printf ' %q' "$@" | tee -a "$log"; printf '\n' | tee -a "$log"; return 0
  fi
  "$@" 2>&1 | tee "$log"
}

index=0
# Seed is outermost, then route; each route gets train + its three evals before next route.
for train_seed in "${TRAIN_LIST[@]}"; do
  for route in "${ROUTE_LIST[@]}"; do
    index=$((index + 1))
    tag="${route}__train-${train_seed}"
    done_file="$ROOT/status/$LABEL/${tag}.done"
    [[ -f "$done_file" ]] && { echo "SKIP completed: $tag"; continue; }
    nonce="${train_seed}$(printf '%03d' "$index")"

    if [[ "$METHOD" == residual ]]; then
      train_group="ResidualRLB2DF2DEvalTrain"
      eval_config=impls/configs/steervla_residual_eval_config.py
      eval_group="ResidualRLB2DF2DEvalRollout"
      train_name="residual-train_${LABEL}_train-${train_seed}_${route}_${nonce}"
      train_root="$ROOT/runs/$LABEL/$tag/train"
      checkpoint_root="$train_root/OGBench-CARLA-Residual/$train_group/$train_name"
      checkpoint_epoch=10000
      base_args=()
      run_logged "$ROOT/logs/$LABEL/${tag}__train.log" env CUDA_VISIBLE_DEVICES="$GPU" OGBENCH_SAVE_DIR="$train_root" ./run_carla.sh --agent-config impls/configs/steervla_residual_eval_config.py --route "$route" --seed "$train_seed" --carla-seed "$train_seed" --train-seed "$train_seed" --eval-mode --exp-name "$train_name" --online-steps 10000 --max-episode-steps 4000 --train-gpu 0 --render-adapter "$GPU" --carla-port "$CARLA_PORT" --carla-streaming-port "$STREAM_PORT" --tm-port "$TM_PORT" --x-display-num "$DISPLAY" --run-group "$train_group" --save-buffer true --save-video-local true --wandb-mode "$WANDB_MODE" --max-retries 50
    else
      train_group="CastToResidualB2DF2DEvalTrain"
      eval_group="CastToResidualB2DF2DEvalRollout"
      pair_label="${LABEL}__${route}__train-${train_seed}"
      eval_config=impls/configs/steervla_residual_from_cast_config.py
      pair_root="$ROOT/runs/$LABEL/runs/$pair_label/${route}__seed-${train_seed}"
      run_logged "$ROOT/logs/$LABEL/${tag}__train.log" env CUDA_VISIBLE_DEVICES="$GPU" CAST_RESIDUAL_ROOT="$ROOT/runs/$LABEL" CAST_RESIDUAL_LABEL="$pair_label" CAST_RESIDUAL_ROUTES="$route" CAST_RESIDUAL_SEEDS="$train_seed" CAST_RESIDUAL_NONCE="$nonce" CAST_RUN_GROUP="$train_group" RESIDUAL_RUN_GROUP="$train_group" ./run_cast_to_residual.sh "$GPU"
      cast_step="$(find "$pair_root/cast_policy" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | awk '/^[0-9]+$/' | sort -n | tail -1)"
      base_args=(--steervla-checkpoint "$pair_root/cast_policy/$cast_step")
      checkpoint_root="$(find "$pair_root/residual" -name 'params_*.pkl' -printf '%h\n' | sort -u | tail -1)"
      checkpoint_epoch="$(find "$checkpoint_root" -name 'params_*.pkl' -printf '%f\n' | sed -n 's/^params_\([0-9][0-9]*\)\.pkl$/\1/p' | sort -n | tail -1)"
    fi

    [[ "$DRY_RUN" == 1 || ( -d "$checkpoint_root" && "$checkpoint_epoch" =~ ^[1-9][0-9]*$ ) ]] || { echo "Missing final residual checkpoint for $tag" >&2; exit 1; }
    for eval_seed in "${EVAL_LIST[@]}"; do
      eval_name="${METHOD}-checkpoint-eval_${LABEL}_train-${train_seed}_eval-${eval_seed}_${route}_${nonce}"
      eval_root="$ROOT/runs/$LABEL/$tag/evals/model-seed-${eval_seed}"
      mkdir -p "$eval_root"
      run_logged "$ROOT/logs/$LABEL/${tag}__eval-${eval_seed}.log" env CUDA_VISIBLE_DEVICES="$GPU" OGBENCH_SAVE_DIR="$eval_root" ./run_carla.sh --agent-config "$eval_config" --route "$route" --seed "$eval_seed" --carla-seed "$train_seed" --train-seed "$eval_seed" --eval-only true --exp-name "$eval_name" --online-steps 4000 --max-episodes 1 --max-episode-steps 4000 --train-gpu 0 --render-adapter "$GPU" --carla-port "$CARLA_PORT" --carla-streaming-port "$STREAM_PORT" --tm-port "$TM_PORT" --x-display-num "$DISPLAY" --run-group "$eval_group" --save-buffer false --save-video-local true --wandb-mode "$WANDB_MODE" --max-retries 50 "${base_args[@]}" -- --restore_path="$checkpoint_root" --restore_epoch="$checkpoint_epoch"
    done
    touch "$done_file"
  done
done

echo "Bench2Drive + Fail2Drive checkpoint matrix complete: $ROOT"
