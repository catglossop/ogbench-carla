#!/usr/bin/env bash
# fixedseed_run.sh ROUTE GPU SLOT EXP
# Same recipe as kl01_streak3 (cot_temperature 0.1, hl_kl_coef 0.1, stop on 3 consecutive 100 DS,
# 500-update budget) plus --fixed-carla-seed: TRAINING episodes now replay carla_seed=0 instead of
# walking it, so training and eval run the SAME scenario and differ only in the model sampling seed
# (training fixed at train_seed, eval at 1001/1002/1003).
#
# The point: until now a train->eval drop confounded two things, because training walked the
# scenario while eval pinned it. With the scenario held equal on both sides, a remaining gap can
# only come from the inference seed. Scenario coverage is recovered by running more carla seeds.
set -uo pipefail
cd /home/cglossop/ogbench-carla
ROUTE="${1:?route}"; GPU="${2:?gpu}"; SLOT="${3:?slot}"; EXP="${4:?exp}"
KL_COEF="${KL_COEF:-0.1}"   # KL weight tethering the HL policy to its start checkpoint
LOG_DIR=".run_carla/jobs/${EXP}"; mkdir -p "$LOG_DIR"
say() { echo "[fixedseed $(date +%H:%M:%S)] $*" | tee -a "${LOG_DIR}/run.log"; }
say "waiting for gpu ${GPU}"
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
  [ "${used:-999999}" -lt 20000 ] && break
  sleep 180
done
say "gpu ${GPU} free (${used} MiB); starting ${ROUTE} (kl=${KL_COEF})"
rm -rf "/raid/users/cglossop/experiments/${EXP}" 2>/dev/null
SEED=0 \
AGENT_CFG=impls/configs/steervla_cast_relabel_hl200x10_adaptive_config.py \
RUN_GROUP="$EXP" \
OGBENCH_SAVE_DIR="/raid/users/cglossop/experiments/${EXP}" \
UPDATES_AFTER_SCORE=0 \
MAX_HL_UPDATES=500 \
STOP_ON_SCORE=100 \
STOP_SCORE_STREAK=3 \
FIXED_CARLA_SEED=true \
ONLINE_STEPS=20000 \
  ./.run_carla/eval_subset_run.sh "$ROUTE" "$GPU" "$SLOT" \
    --cot-temperature 0.1 \
    --hl-kl-coef "$KL_COEF" \
  > "${LOG_DIR}/${ROUTE}.log" 2>&1
say "exited ($?)"
