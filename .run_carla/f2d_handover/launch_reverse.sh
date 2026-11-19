#!/usr/bin/env bash
# f2d mixed-policy zero-shot Qwen BoN sweep -- SECOND MACHINE, routes in REVERSE order.
#
# bellman runs the f2d routes front to back; this runs the remaining ones back to front, and the two
# meet in the middle (see README.md, "Meeting in the middle"). Same protocol as bellman's run, same
# RUN_GROUP, so results merge into one table afterwards.
#
#   ./launch_reverse.sh --dry-run     check everything and print the first job; launches nothing
#   ./launch_reverse.sh --arm         start the sweep (run it under tmux or systemd-run --user)
#
# Every path below is this machine's; edit the block to match, or export the variables first.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:---dry-run}"

# ---------------------------------------------------------------- this machine's paths (EDIT) ---
: "${REPO:=$(cd "$HERE/../.." && pwd)}"                                     # the ogbench-carla clone
: "${HL_CKPT_ROOT:=/home/celinet/steervla_harp_ckpts/f2dsteervla_simlingo_fixedcarla_kl005_seed0}"
: "${MIXED_LL_CHECKPOINT:?set MIXED_LL_CHECKPOINT to the local no_ego_history 6000 checkpoint dir (from GCS)}"
: "${F2D_CARLA_0915_ROOT:?set F2D_CARLA_0915_ROOT to the Fail2Drive CARLA 0.9.15 install}"
: "${SIMLINGO_SOURCE_ROOT:?set SIMLINGO_SOURCE_ROOT to the simlingo-steervla checkout}"
: "${MIXED_HL_PYTHON:?set MIXED_HL_PYTHON to your hl_python.sh (see hl_python.sh.template)}"
: "${QWEN_ROOT:?set QWEN_ROOT to the qwen-critic checkout}"
: "${QWEN_HF_HOME:?set QWEN_HF_HOME to the HF cache holding Qwen/Qwen3.8-27B}"
: "${WANDB_KEY_FILE:?set WANDB_KEY_FILE to a copy of the catherine_glossop school W&B key}"
: "${WORK_ROOT:?set WORK_ROOT to a writable dir for outputs, e.g. /home/celinet/f2d_bon}"
: "${SWEEP_GPUS:?set SWEEP_GPUS, one route per GPU, e.g. \"0 1\"}"
: "${QWEN_GPUS:=$SWEEP_GPUS}"
# Not 18850/18851: those are the scripts' defaults, and another session reusing them on the same GPU
# is exactly how bellman's critic got killed mid-episode on 2026-09-21.
: "${QWEN_PORTS:=18870 18871}"

# ---------------------------------------------------------- protocol (identical to bellman's) ---
export BENCH=f2d
# Required by the driver for f2d. Only names the sweep here: SOURCE_ROOT below overrides where its
# run_summary.json files are looked for.
export SOURCE_SWEEP=f2dsteervla_simlingo_fixedcarla_kl005_seed0
export RUN_GROUP=f2dsteervla_kl005_seed0_qwenzs_bon_mixed
export ROUTES_FILE="$HERE/routes_f2d_reverse.txt"
export CARLA_SEEDS="0 1 2" N_EVAL=1 EVAL_SEED_OFFSET=0
export ACTOR=mixed N_CANDIDATES=4
export MIXED_LL_CHECKPOINT F2D_CARLA_0915_ROOT SIMLINGO_SOURCE_ROOT MIXED_HL_PYTHON
export QWEN_ROOT QWEN_HF_HOME WANDB_KEY_FILE SWEEP_GPUS QWEN_GPUS QWEN_PORTS
export QWEN_LOG_DIR="$WORK_ROOT/qwen_zs_critic"
export OGBENCH_SAVE_DIR="$WORK_ROOT/sweeps/$RUN_GROUP"
export RESULTS_DIR="$WORK_ROOT/sweep_results/$RUN_GROUP"
# Checkpoints come only from CKPT_OVERRIDES: bellman's run_summary.json files hold bellman paths, so
# point run_summary discovery at an empty dir and let every route fall through to the overrides.
export SOURCE_ROOT="$WORK_ROOT/empty_source"
export CKPT_OVERRIDES="$WORK_ROOT/ckpt_overrides_f2d.tsv"
unset HL_KV_CACHE   # the cached decode path is not bit-reproducible; bellman's cells ran without it

fail=0
check() { if eval "$2"; then echo "  ok    $1"; else echo "  FAIL  $1"; fail=1; fi; }
echo "== preflight"
check "repo at $REPO"                          '[ -x "$REPO/.run_carla/qwen_zs_bon_sweep.sh" ]'
check "main .venv"                             '[ -x "$REPO/.venv/bin/python" ]'
check "f2d env (py3.10 + carla 0.9.15)"        '[ -x "${CARLA_0915_PYTHON:-$REPO/.venv-carla-0915/bin/python}" ]'
check "f2d env imports carla, srunner, fail2drive" \
  '"${CARLA_0915_PYTHON:-$REPO/.venv-carla-0915/bin/python}" -c "import carla, srunner, fail2drive" 2>/dev/null'
check "CARLA 0.9.15 CarlaUE4.sh"               '[ -x "$F2D_CARLA_0915_ROOT/CarlaUE4.sh" ]'
check "Fail2Drive animal assets installed"     '[ -d "$F2D_CARLA_0915_ROOT/CarlaUE4/Content/AnimalVarietyPack" ]'
check "LL checkpoint has params/"              '[ -d "$MIXED_LL_CHECKPOINT/params" ]'
check "simlingo-steervla source"               '[ -d "$SIMLINGO_SOURCE_ROOT/simlingo_training" ]'
check "hl_python.sh executable"                '[ -x "$MIXED_HL_PYTHON" ]'
check "qwen-critic serve_bon.py"               '[ -f "$QWEN_ROOT/scripts/serve_bon.py" ]'
check "Qwen3.8-27B in the HF cache"            'ls -d "$QWEN_HF_HOME"/hub/models--Qwen--Qwen3.8-27B >/dev/null 2>&1'
check "W&B key file non-empty"                 '[ -s "$WANDB_KEY_FILE" ]'
check "HL_KV_CACHE unset"                      '[ -z "${HL_KV_CACHE:-}" ]'

mkdir -p "$SOURCE_ROOT" "$RESULTS_DIR" "$QWEN_LOG_DIR"
echo "== checkpoints (last per route; INVALID runs skipped)"
python3 "$HERE/build_ckpt_overrides.py" "$HL_CKPT_ROOT" "$ROUTES_FILE" "$CKPT_OVERRIDES" || fail=1

# Global rule: log only as catherine_glossop. Check the identity the runs will actually use.
echo "== W&B identity"
viewer=$(WANDB_API_KEY="$(cat "$WANDB_KEY_FILE" 2>/dev/null)" "$REPO/.venv/bin/python" -c \
  'import wandb; print(wandb.Api().viewer.username)' 2>/dev/null | tail -1)
if [ "$viewer" = catherine_glossop ]; then echo "  ok    viewer is catherine_glossop"
else echo "  FAIL  W&B viewer is '${viewer:-<none>}', not catherine_glossop -- will not launch"; fail=1; fi

done_n=$(ls "$RESULTS_DIR"/*/carla_seed_[0-9]/run_summary_frozen_eval.json 2>/dev/null | wc -l)
echo "== bellman results seeded here: $done_n cell(s) (copy them in first; see README step 5)"

[ "$fail" = 0 ] || { echo; echo "preflight FAILED -- nothing launched"; exit 1; }
cd "$REPO" && exec ./.run_carla/qwen_zs_bon_sweep.sh "$MODE"
