#!/usr/bin/env bash
# Sequential residual-SAC sweep launcher. One physical GPU is used at a time.
#
# Sweep matrices live in impls/configs/residual_rl_sweeps.yaml. The active default
# is coarse_grid_v3; choose a preserved profile with SWEEP_PROFILE=<name>.
#
# Examples:
#   ./run_residual_sweep.sh 3
#   SWEEP_PROFILE=coarse_grid_v2 ./run_residual_sweep.sh 3
#   SWEEP_PROFILE=coarse_grid_v1 ./run_residual_sweep.sh 3
#   SWEEP_PROFILE=initial_scale ./run_residual_sweep.sh 3
#   DRY_RUN=1 ./run_residual_sweep.sh 3
#   SEEDS="0 1 2" ./run_residual_sweep.sh 3 /raid/users/$USER/carla_exps/residual_sweep
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PHYSICAL_GPU="${1:-${SWEEP_GPU:-0}}"
RUN_ROOT="${2:-${SWEEP_ROOT:-/raid/users/${USER}/carla_exps/residual_scale_sweep}}"
SWEEP_CONFIG="${SWEEP_CONFIG:-$ROOT_DIR/impls/configs/residual_rl_sweeps.yaml}"
SWEEP_PROFILE="${SWEEP_PROFILE:-coarse_grid_v3}"
MAX_RETRIES="${MAX_RETRIES:-50}"
WANDB_MODE="${WANDB_MODE:-online}"
SAVE_BUFFER="${SAVE_BUFFER:-true}"
SAVE_VIDEO_LOCAL="${SAVE_VIDEO_LOCAL:-true}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
# Empty -> selected YAML profile's launcher policy.
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-}"
RETRY_FAILED="${RETRY_FAILED:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$SWEEP_CONFIG" ]]; then
  echo "Sweep YAML not found: $SWEEP_CONFIG" >&2
  exit 2
fi

# Parse and validate the selected YAML profile once, then expose only shell-quoted
# scalar/list values. The YAML is versioned repo configuration, not user input.
eval "$("$ROOT_DIR/.venv/bin/python" - "$SWEEP_CONFIG" "$SWEEP_PROFILE" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
profile_name = sys.argv[2]
data = yaml.safe_load(path.read_text())
if not isinstance(data, dict):
    raise SystemExit(f"{path}: expected a top-level mapping")
common = data.get("common", {})
profiles = data.get("profiles", {})
if not isinstance(common, dict) or not isinstance(profiles, dict):
    raise SystemExit(f"{path}: common and profiles must be mappings")
profile = profiles.get(profile_name)
if not isinstance(profile, dict):
    available = ", ".join(sorted(map(str, profiles)))
    raise SystemExit(f"unknown sweep profile {profile_name!r}; available: {available}")
merged = {**common, **profile}
required = ("label", "run_group", "routes", "betas", "encoders", "seeds", "online_steps", "fixed_agent_overrides", "steervla_overrides", "launcher")
missing = [key for key in required if key not in merged]
if missing:
    raise SystemExit(f"{path}:{profile_name}: missing required keys: {', '.join(missing)}")
for key in ("routes", "betas", "encoders", "seeds"):
    value = merged[key]
    if not isinstance(value, list) or not value:
        raise SystemExit(f"{path}:{profile_name}: {key} must be a non-empty list")
if "scale_pairs" in merged:
    raw_pairs = merged["scale_pairs"]
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise SystemExit(f"{path}:{profile_name}: scale_pairs must be a non-empty list")
    scale_pairs = []
    for pair in raw_pairs:
        if not isinstance(pair, dict) or set(pair) != {"accel", "steer"}:
            raise SystemExit(f"{path}:{profile_name}: each scale_pairs item must have accel and steer")
        scale_pairs.append((pair["accel"], pair["steer"]))
else:
    for key in ("accel_scales", "steer_scales"):
        value = merged.get(key)
        if not isinstance(value, list) or not value:
            raise SystemExit(f"{path}:{profile_name}: {key} must be a non-empty list when scale_pairs is absent")
    scale_pairs = [(accel, steer) for accel in merged["accel_scales"] for steer in merged["steer_scales"]]
fixed = merged["fixed_agent_overrides"]
if not isinstance(fixed, dict):
    raise SystemExit(f"{path}:{profile_name}: fixed_agent_overrides must be a mapping")
for key in ("expo", "best_of_n", "otf_td_backup", "residual_bc_normalize"):
    if key not in fixed:
        raise SystemExit(f"{path}:{profile_name}: fixed_agent_overrides missing {key}")
steervla = merged["steervla_overrides"]
if not isinstance(steervla, dict):
    raise SystemExit(f"{path}:{profile_name}: steervla_overrides must be a mapping")
for key in ("actions_per_model_query", "actions_per_cot"):
    if int(steervla.get(key, 0)) < 1:
        raise SystemExit(f"{path}:{profile_name}: steervla_overrides.{key} must be >= 1")
launcher = merged["launcher"]
if not isinstance(launcher, dict):
    raise SystemExit(f"{path}:{profile_name}: launcher must be a mapping")
for key in ("continue_on_failure", "retry_failed_on_resume"):
    if not isinstance(launcher.get(key), bool):
        raise SystemExit(f"{path}:{profile_name}: launcher.{key} must be boolean")

def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")

def emit_list(name, values):
    emit(name, " ".join(str(value) for value in values))

emit("YAML_LABEL", merged["label"])
emit("YAML_RUN_GROUP", merged["run_group"])
emit("YAML_ONLINE_STEPS", merged["online_steps"])
emit_list("YAML_ROUTES", merged["routes"])
emit_list("YAML_SEEDS", merged["seeds"])
emit_list("YAML_SCALE_PAIRS", [f"{accel}:{steer}" for accel, steer in scale_pairs])
emit_list("YAML_BETAS", merged["betas"])
emit_list("YAML_ENCODERS", merged["encoders"])
for key in ("expo", "best_of_n", "otf_td_backup", "residual_bc_normalize"):
    emit(f"YAML_AGENT_{key.upper()}", str(fixed[key]).lower())
for key in ("actions_per_model_query", "actions_per_cot"):
    emit(f"YAML_STEERVLA_{key.upper()}", steervla[key])
for key in ("continue_on_failure", "retry_failed_on_resume"):
    emit(f"YAML_LAUNCHER_{key.upper()}", str(launcher[key]).lower())
PY
)"

# Environment values intentionally override YAML for one-off variants. Use a new
# SWEEP_LABEL when an override should have its own artifact/status namespace.
ONLINE_STEPS="${ONLINE_STEPS:-$YAML_ONLINE_STEPS}"
RUN_GROUP="${SWEEP_RUN_GROUP:-$YAML_RUN_GROUP}"
SWEEP_LABEL="${SWEEP_LABEL:-$YAML_LABEL}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-$YAML_LAUNCHER_CONTINUE_ON_FAILURE}"
RETRY_FAILED="${RETRY_FAILED:-$YAML_LAUNCHER_RETRY_FAILED_ON_RESUME}"
read -r -a ROUTES <<< "${ROUTES:-$YAML_ROUTES}"
read -r -a SEEDS <<< "${SEEDS:-$YAML_SEEDS}"
read -r -a SCALE_PAIRS <<< "${SCALE_PAIRS:-$YAML_SCALE_PAIRS}"
read -r -a BETAS <<< "${BETAS:-$YAML_BETAS}"
read -r -a ENCODERS <<< "${ENCODERS:-$YAML_ENCODERS}"

if (( ${#ROUTES[@]} == 0 || ${#SEEDS[@]} == 0 || ${#SCALE_PAIRS[@]} == 0 || ${#BETAS[@]} == 0 || ${#ENCODERS[@]} == 0 )); then
  echo 'All sweep lists must be non-empty.' >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs/$SWEEP_LABEL" "$RUN_ROOT/status/$SWEEP_LABEL" "$RUN_ROOT/runs/$SWEEP_LABEL"
# The profile label versions artifact/status namespaces. Keep a profile's label stable
# to resume it; add a new label/profile for an intentionally different matrix.
cat > "$RUN_ROOT/sweep_spec_${SWEEP_LABEL}.txt" <<SPEC
created=$(date --iso-8601=seconds)
sweep_config=$SWEEP_CONFIG
sweep_profile=$SWEEP_PROFILE
sweep_label=$SWEEP_LABEL
physical_gpu=$PHYSICAL_GPU
train_gpu=0
render_adapter=$PHYSICAL_GPU
online_steps=$ONLINE_STEPS
routes=${ROUTES[*]}
seeds=${SEEDS[*]}
scale_pairs=${SCALE_PAIRS[*]}
betas=${BETAS[*]}
encoders=${ENCODERS[*]}
actions_per_model_query=$YAML_STEERVLA_ACTIONS_PER_MODEL_QUERY
actions_per_cot=$YAML_STEERVLA_ACTIONS_PER_COT
continue_on_failure=$CONTINUE_ON_FAILURE
retry_failed=$RETRY_FAILED
fixed_config: expo=$YAML_AGENT_EXPO, best_of_n=$YAML_AGENT_BEST_OF_N, otf_td_backup=$YAML_AGENT_OTF_TD_BACKUP, residual_bc_normalize=$YAML_AGENT_RESIDUAL_BC_NORMALIZE
SPEC

total=$(( ${#ROUTES[@]} * ${#SEEDS[@]} * ${#SCALE_PAIRS[@]} * ${#BETAS[@]} * ${#ENCODERS[@]} ))
index=0

for encoder in "${ENCODERS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for scale_pair in "${SCALE_PAIRS[@]}"; do
      IFS=: read -r accel_scale steer_scale <<< "$scale_pair"
      if [[ -z "$accel_scale" || -z "$steer_scale" ]]; then
        echo "Invalid SCALE_PAIRS entry: $scale_pair (expected accel:steer)" >&2
        exit 2
      fi
      for route in "${ROUTES[@]}"; do
          for seed in "${SEEDS[@]}"; do
            index=$((index + 1))
            tag="${route}__enc-${encoder}__a-${accel_scale}__s-${steer_scale}__beta-${beta}__seed-${seed}"
            tag="${tag//\//_}"
            done_file="$RUN_ROOT/status/$SWEEP_LABEL/${tag}.done"
            failure_file="$RUN_ROOT/status/$SWEEP_LABEL/${tag}.failed"
            log_file="$RUN_ROOT/logs/$SWEEP_LABEL/${tag}.log"
            if [[ "$SKIP_COMPLETED" == "1" && -f "$done_file" ]]; then
              echo "[$index/$total] SKIP completed: $tag"
              continue
            fi

            echo "[$index/$total] START: $tag"
            if [[ "$DRY_RUN" == "1" ]]; then
              echo "[$index/$total] DRY RUN: profile=$SWEEP_PROFILE label=$SWEEP_LABEL route=$route encoder=$encoder accel=$accel_scale steer=$steer_scale beta=$beta seed=$seed"
              continue
            fi
            run_dir="$RUN_ROOT/runs/$SWEEP_LABEL/$tag"
            # Put all sweep dimensions before the route, so the stable W&B id's
            # 128-character safety cap never elides a distinguishing parameter.
            exp_name="residual-rl-hyperparam-sweep_${SWEEP_LABEL}_a-${accel_scale}_s-${steer_scale}_b-${beta}_enc-${encoder}_sd-${seed}_${route}"
            mkdir -p "$run_dir"
            set +e
            CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" OGBENCH_SAVE_DIR="$run_dir" \
              ./run_carla.sh \
                --agent-config impls/configs/steervla_residual_config.py \
                --route "$route" \
                --seed "$seed" \
                --exp-name "$exp_name" \
                --online-steps "$ONLINE_STEPS" \
                --train-gpu 0 \
                --render-adapter "$PHYSICAL_GPU" \
                --run-group "$RUN_GROUP" \
                --save-buffer "$SAVE_BUFFER" \
                --save-video-local "$SAVE_VIDEO_LOCAL" \
                --wandb-mode "$WANDB_MODE" \
                --max-retries "$MAX_RETRIES" \
                -- \
                --agent.state_encoder="$encoder" \
                --agent.residual_accel_scale="$accel_scale" \
                --agent.residual_steer_scale="$steer_scale" \
                --agent.residual_bc_beta="$beta" \
                --agent.residual_bc_normalize="$YAML_AGENT_RESIDUAL_BC_NORMALIZE" \
                --agent.expo="$YAML_AGENT_EXPO" \
                --agent.best_of_n="$YAML_AGENT_BEST_OF_N" \
                --agent.otf_td_backup="$YAML_AGENT_OTF_TD_BACKUP" \
                --agent.steervla.actions_per_model_query="$YAML_STEERVLA_ACTIONS_PER_MODEL_QUERY" \
                --agent.steervla.actions_per_cot="$YAML_STEERVLA_ACTIONS_PER_COT" \
                2>&1 | tee "$log_file"
            # Expand both pipeline statuses before either assignment resets PIPESTATUS.
            run_code=${PIPESTATUS[0]} tee_code=${PIPESTATUS[1]}
            set -e
            if [[ "$run_code" -eq 0 && "$tee_code" -eq 0 ]]; then
              touch "$done_file"
              echo "[$index/$total] DONE: $tag"
              continue
            fi
            # Ctrl-C is an operator-requested stop, not a failed experiment. Do not
            # write a failure marker or silently advance to the next queued setting.
            if [[ "$run_code" -eq 130 ]]; then
              echo "[$index/$total] interrupted; stopping sweep queue." >&2
              exit 130
            fi
            printf 'timestamp=%s\nrun_exit_code=%s\ntee_exit_code=%s\n' "$(date --iso-8601=seconds)" "$run_code" "$tee_code" > "$failure_file"
            echo "[$index/$total] FAILED: $tag (run=$run_code tee=$tee_code; marker=$failure_file)" >&2
            if [[ "$CONTINUE_ON_FAILURE" == "1" || "$CONTINUE_ON_FAILURE" == "true" ]]; then
              echo "[$index/$total] continuing after recorded failure." >&2
              continue
            fi
            exit "$run_code"
          done
      done
    done
  done
done

echo "Sweep complete: $RUN_ROOT"
