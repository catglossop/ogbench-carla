#!/usr/bin/env bash
# f2d mixed CAST-relabel sweep -- BELLMAN half (the /raid machine).
#
# Same sweep as the one running on dgx6 (same SWEEP_NAME, same recipe, same HL checkpoint), on the
# routes handed to this machine. The two share no disk, so the split is by route list: ROUTES_FILE
# here must not overlap the other machine's queue. README.md walks through the handoff and merge.
#
# Two GPUs per route: policy + CARLA on SWEEP_GPUS[i], the InternVL2 HL worker on SWEEP_HL_GPUS[i].
#
#   ROUTES_FILE=.run_carla/f2d_mixed_bellman/routes_bellman.txt SWEEP_GPUS="3 5" SWEEP_HL_GPUS="6 7" \
#     ./.run_carla/f2d_mixed_bellman/launch_bellman.sh --dry-run
#   ... --arm     (run it under tmux or systemd-run --user; the sweep dies with its shell)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
MODE="${1:---dry-run}"

# --------------------------------------------------------------------------- bellman paths (EDIT) ---
: "${MIXED_LL_CHECKPOINT:=/raid/users/cglossop/openpi/cat-logs/pi05_steervla_cot_simplified_reasoning_no_ego_history/pi05_steervla_simplified_reasoning_no_ego_history_v1/pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000}"
: "${MIXED_HL_CHECKPOINT:=/raid/users/celine/steervla-ckpts/2026_05_24_06_52_33_simlingo_seed1_bellman/checkpoints/epoch=019.ckpt}"
: "${SIMLINGO_SOURCE_ROOT:=/home/cglossop/simlingo-steervla}"
: "${MIXED_HL_PYTHON:=$REPO/.venv/bin/python}"        # needs the simlingo extra; see hl_python.sh.template
: "${HL_REPLAY_ROOT:=/raid/users/cglossop/simlingo_hl_pools}"
: "${HL_REPLAY_POOL:=simlingo_hl_simplified}"
: "${CARLA_ROOT:=/home/cglossop/carla}"               # 0.9.16 + f2d content pack, 15 of the 16 routes
: "${F2D_CARLA_0915_ROOT:=/home/cglossop/f2d_carla}"  # animals-1076 only
: "${F2D_CARLA_0915_PYTHON:=$REPO/.venv-f2d-eval/bin/python}"
: "${WANDB_KEY_FILE:=/home/cglossop/.wandb_school_key}"
: "${SWEEP_GPUS:?set SWEEP_GPUS, one policy/sim GPU per parallel route, e.g. \"3 5\"}"
: "${SWEEP_HL_GPUS:?set SWEEP_HL_GPUS, one HL GPU per entry in SWEEP_GPUS, e.g. \"6 7\"}"
# Which routes this machine owns. No default on purpose: an accidental full list would duplicate
# every route the other machine is already running.
: "${ROUTES_FILE:?set ROUTES_FILE to the route list for this machine -- it must not overlap the other machine}"
export MIXED_LL_CHECKPOINT MIXED_HL_CHECKPOINT SIMLINGO_SOURCE_ROOT MIXED_HL_PYTHON
export HL_REPLAY_ROOT HL_REPLAY_POOL CARLA_ROOT F2D_CARLA_0915_ROOT F2D_CARLA_0915_PYTHON
export SWEEP_GPUS SWEEP_HL_GPUS ROUTES_FILE
# run_carla.sh's cache defaults (/raid/users/$USER/...) are already right on this machine, so unlike
# the dgx6 launcher there is nothing to redirect.
export UV_PROJECT_ENVIRONMENT="$REPO/.venv"
export PYTHONPATH="$REPO"

# ------------------------------------------------------- protocol (identical to the dgx6 half) ---
# Same name on both machines: the halves merge into one table.
export SWEEP_NAME="${SWEEP_NAME:-f2dmixed_internvl2hl_pi05ll_fixedcarla_kl005_seed0}"
export AGENT_CFG=impls/configs/mixed_steervla_cast_relabel_train_bellman_config.py
export SEED=0
export ONLINE_STEPS=10000 MAX_HL_UPDATES=500
export STOP_ON_SCORE=95 STOP_SCORE_STREAK=3 UPDATES_AFTER_SCORE=0
export FIXED_CARLA_SEED=true
export EVAL_EVERY=                    # empty = no periodic eval; scored once, at the end
export POST_STOP_EVAL_EPISODES=3      # the 3-seed frozen eval of the stop checkpoint
export RESULT_TRAIN_STEP=10000
# Bellman has hosted other sweeps on 17400/17500 and :960-:994. Override if these are taken.
export CARLA_PORT_BASE="${CARLA_PORT_BASE:-18200}" TM_PORT_BASE="${TM_PORT_BASE:-18300}" DISPLAY_BASE="${DISPLAY_BASE:-920}"
export OGBENCH_SAVE_DIR="${OGBENCH_SAVE_DIR:-/raid/users/cglossop/sweeps/$SWEEP_NAME}"
export RESULTS_DIR="${RESULTS_DIR:-/raid/users/cglossop/sweep_results/$SWEEP_NAME}"
export EXTRA_ARGS="--cot-temperature 0.1 --hl-kl-coef 0.05 --hl-ckpt-keep-last 5 \
  --steervla-checkpoint $MIXED_LL_CHECKPOINT \
  --actor-config pi05_steervla_cot_simplified_reasoning_no_ego_history"

# CAST relabel is a Gemini client; the key lives in the shell profile, not the repo.
if [ -z "${GEMINI_API_KEY:-}" ]; then
  eval "$(grep -m1 '^export GEMINI_API_KEY=' ~/.bashrc 2>/dev/null)"
fi

fail=0
check() { if eval "$2"; then echo "  ok    $1"; else echo "  FAIL  $1"; fail=1; fi; }
echo "== preflight"
check "main .venv"                            '[ -x "$REPO/.venv/bin/python" ]'
check "agent config"                          '[ -f "$REPO/$AGENT_CFG" ]'
check "routes file"                           '[ -s "$ROUTES_FILE" ]'
check "HL checkpoint"                         'ls -d "$MIXED_HL_CHECKPOINT" >/dev/null 2>&1'
check "pi05 LL checkpoint has params/"        '[ -d "$MIXED_LL_CHECKPOINT/params" ]'
check "simlingo-steervla source"              '[ -d "$SIMLINGO_SOURCE_ROOT/simlingo_training" ]'
check "HL python has the simlingo deps"       '"$MIXED_HL_PYTHON" -c "import peft, hydra, pytorch_lightning, timm, torch" 2>/dev/null'
check "CARLA 0.9.16 CarlaUE4.sh"              '[ -x "$CARLA_ROOT/CarlaUE4.sh" ]'
check "CARLA 0.9.15 (animals route)"          '[ -x "$F2D_CARLA_0915_ROOT/CarlaUE4.sh" ]'
check "0.9.15 animal assets"                  '[ -d "$F2D_CARLA_0915_ROOT/CarlaUE4/Content/AnimalVarietyPack" ]'
check "0.9.15 python imports fail2drive"      '"$F2D_CARLA_0915_PYTHON" -c "import carla, srunner, fail2drive" 2>/dev/null'
check "HL replay pool ($HL_REPLAY_POOL)"      '[ -s "$HL_REPLAY_ROOT/$HL_REPLAY_POOL/hl_samples.json" ]'
check "GEMINI_API_KEY exported"               '[ -n "${GEMINI_API_KEY:-}" ]'
check "W&B school key"                        '[ -s "$WANDB_KEY_FILE" ]'
check "one HL GPU per policy GPU"             '[ "$(echo $SWEEP_GPUS | wc -w)" = "$(echo $SWEEP_HL_GPUS | wc -w)" ]'
check "no GPU serves two roles"               '[ "$(echo $SWEEP_GPUS $SWEEP_HL_GPUS | tr " " "\n" | sort -u | wc -l)" = "$(echo $SWEEP_GPUS $SWEEP_HL_GPUS | wc -w)" ]'
check "CARLA ports free"                      '! ss -ltn 2>/dev/null | grep -qE ":($CARLA_PORT_BASE|$TM_PORT_BASE) "'
echo "== W&B identity"
viewer=$(WANDB_API_KEY="$(cat "$WANDB_KEY_FILE" 2>/dev/null)" "$REPO/.venv/bin/python" -c \
  'import wandb; print(wandb.Api().viewer.username)' 2>/dev/null | tail -1)
if [ "$viewer" = catherine_glossop ]; then echo "  ok    viewer is catherine_glossop"
else echo "  FAIL  W&B viewer is '${viewer:-<none>}', not catherine_glossop"; fail=1; fi

[ "$fail" = 0 ] || { echo; echo "preflight FAILED -- nothing launched"; exit 1; }
mkdir -p "$OGBENCH_SAVE_DIR" "$RESULTS_DIR"
echo
echo "== plan: $(grep -c . "$ROUTES_FILE") route(s) from $ROUTES_FILE"
echo "         policy/sim gpus [$SWEEP_GPUS], HL gpus [$SWEEP_HL_GPUS], sweep $SWEEP_NAME"
if [ "$MODE" = --arm ]; then
  exec ./.run_carla/b2d_subset_sweep.sh --arm
fi
exec ./.run_carla/b2d_subset_sweep.sh
