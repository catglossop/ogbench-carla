#!/usr/bin/env bash
# f2d CAST-relabel HL sweep for the MIXED stack: original SimLingo InternVL2 HL + pi05 no-ego-history LL.
#
# Same recipe as the SimLingo sweep in F2D_STEERVLA_SIMLINGO_SWEEP_HANDOVER.md (§2/§3) -- same 16
# routes, same stop rule, same HL optimizer and replay mixture, no periodic evals, one frozen
# 3-seed eval at the end -- with SimLingo's waypoint LL replaced by pi05, so the HL trains against
# the policy that will be evaluated.
#
# Two GPUs per route: the policy + CARLA on SWEEP_GPUS[i], the InternVL2 HL worker on SWEEP_HL_GPUS[i].
#
#   ./.run_carla/f2d_mixed_cast_launch.sh --dry-run    # preflight + the driver's plan; launches nothing
#   ./.run_carla/f2d_mixed_cast_launch.sh --arm        # start it (run under tmux/systemd-run)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"
MODE="${1:---dry-run}"

# ------------------------------------------------------------------ this machine's paths (EDIT) ---
: "${WORK_ROOT:=/data/local/cglossop/f2d_mixed_cast}"
: "${MIXED_LL_CHECKPOINT:=/data/local/cglossop/f2d_bon/ll/6000}"
: "${MIXED_HL_CHECKPOINT:=/data/local/cglossop/2026_05_24_06_52_33_simlingo_seed1_bellman/checkpoints/epoch=019.ckpt}"
: "${SIMLINGO_SOURCE_ROOT:=/home/celinet/simlingo-steervla}"
: "${HL_REPLAY_ROOT:=/data/local/cglossop}"
: "${HL_REPLAY_POOL:=simlingo_hl_simplified_img}"   # frames bundled as .npz beside the manifest
: "${CARLA_ROOT:=/home/cglossop/carla}"                      # 0.9.16 + f2d content, 15 of 16 routes
: "${F2D_CARLA_0915_ROOT:=/home/cglossop/f2d_carla}"         # animals-1076 only
: "${F2D_CARLA_0915_PYTHON:=/home/cglossop/ogbench-carla/.claude/worktrees/qwen-zs-bon-sweep/.venv-carla-0915/bin/python}"
: "${SWEEP_GPUS:=0 2}"
: "${SWEEP_HL_GPUS:=1 3}"
export MIXED_LL_CHECKPOINT MIXED_HL_CHECKPOINT SIMLINGO_SOURCE_ROOT HL_REPLAY_ROOT HL_REPLAY_POOL
export CARLA_ROOT F2D_CARLA_0915_ROOT F2D_CARLA_0915_PYTHON SWEEP_GPUS SWEEP_HL_GPUS
# run_carla.sh defaults these to /raid/users/$USER, which dgx6 does not have.
: "${OPENPI_DATA_HOME:=/home/cglossop/.cache/openpi}"
: "${HF_HOME:=/data/local/cglossop/f2d_bon/cache/hf}"
: "${TORCH_HOME:=/data/local/cglossop/f2d_bon/cache/torch}"
: "${XDG_CACHE_HOME:=/data/local/cglossop/f2d_bon/cache/xdg}"
export OPENPI_DATA_HOME HF_HOME TORCH_HOME XDG_CACHE_HOME
export UV_PROJECT_ENVIRONMENT="$REPO/.venv"
# The simlingo extra is installed in this venv, so the checkout alone is enough on PYTHONPATH.
export PYTHONPATH="$REPO"

# ------------------------------------------------- protocol (identical to the SimLingo sweep) ---
export SWEEP_NAME="${SWEEP_NAME:-f2dmixed_internvl2hl_pi05ll_fixedcarla_kl005_seed0}"
export ROUTES_FILE=f2d_steervla_subset.txt
export AGENT_CFG=impls/configs/mixed_steervla_cast_relabel_train_config.py
export SEED=0
export ONLINE_STEPS=10000 MAX_HL_UPDATES=500
export STOP_ON_SCORE=95 STOP_SCORE_STREAK=3 UPDATES_AFTER_SCORE=0
export FIXED_CARLA_SEED=true
export EVAL_EVERY=                    # empty = no periodic eval; the run is scored once, at the end
export POST_STOP_EVAL_EPISODES=3      # the 3-seed frozen eval of the stop checkpoint
export RESULT_TRAIN_STEP=10000
# Distinct from the BoN sweep's 17400/17500/960 so the two never share a port or an X display.
export CARLA_PORT_BASE=17600 TM_PORT_BASE=17700 DISPLAY_BASE=970
export OGBENCH_SAVE_DIR="$WORK_ROOT/sweeps/$SWEEP_NAME"
export RESULTS_DIR="$WORK_ROOT/sweep_results/$SWEEP_NAME"
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
check "routes file (16 routes)"               '[ "$(grep -c . "$REPO/$ROUTES_FILE")" = 16 ]'
check "original HL checkpoint"                'ls -d "$MIXED_HL_CHECKPOINT" >/dev/null 2>&1'
check "pi05 LL checkpoint has params/"        '[ -d "$MIXED_LL_CHECKPOINT/params" ]'
check "simlingo-steervla source"              '[ -d "$SIMLINGO_SOURCE_ROOT/simlingo_training" ]'
check "CARLA 0.9.16 CarlaUE4.sh"              '[ -x "$CARLA_ROOT/CarlaUE4.sh" ]'
check "CARLA 0.9.15 (animals route)"          '[ -x "$F2D_CARLA_0915_ROOT/CarlaUE4.sh" ]'
check "0.9.15 animal assets"                  '[ -d "$F2D_CARLA_0915_ROOT/CarlaUE4/Content/AnimalVarietyPack" ]'
check "0.9.15 python imports fail2drive"      '"$F2D_CARLA_0915_PYTHON" -c "import carla, srunner, fail2drive" 2>/dev/null'
check "HL replay pool ($HL_REPLAY_POOL, 4000 samples)"  '[ -s "$HL_REPLAY_ROOT/$HL_REPLAY_POOL/hl_samples.json" ]'
check "GEMINI_API_KEY exported"               '[ -n "${GEMINI_API_KEY:-}" ]'
check "W&B school key"                        '[ -s /home/cglossop/.wandb_school_key ]'
check "one HL GPU per policy GPU"             '[ "$(echo $SWEEP_GPUS | wc -w)" = "$(echo $SWEEP_HL_GPUS | wc -w)" ]'
echo "== W&B identity"
viewer=$(WANDB_API_KEY="$(cat /home/cglossop/.wandb_school_key 2>/dev/null)" "$REPO/.venv/bin/python" -c \
  'import wandb; print(wandb.Api().viewer.username)' 2>/dev/null | tail -1)
if [ "$viewer" = catherine_glossop ]; then echo "  ok    viewer is catherine_glossop"
else echo "  FAIL  W&B viewer is '${viewer:-<none>}', not catherine_glossop"; fail=1; fi

[ "$fail" = 0 ] || { echo; echo "preflight FAILED -- nothing launched"; exit 1; }
mkdir -p "$OGBENCH_SAVE_DIR" "$RESULTS_DIR"
echo
echo "== plan: policy/sim gpus [$SWEEP_GPUS], HL gpus [$SWEEP_HL_GPUS], sweep $SWEEP_NAME"
if [ "$MODE" = --arm ]; then
  exec ./.run_carla/b2d_subset_sweep.sh --arm
fi
exec ./.run_carla/b2d_subset_sweep.sh
