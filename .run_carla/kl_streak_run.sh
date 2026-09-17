#!/usr/bin/env bash
# kl_streak_run.sh ROUTE GPU SLOT EXP
# cot_temperature 0.1, hl_kl_coef 0.10, 500-update budget, and a stop condition that requires
# THREE CONSECUTIVE episodes at DS 100 rather than one.
#
# Why the streak: on this subset eval scores swing between ~30 and 100 on the same weights, so a
# single 100 is one lucky draw, not evidence of a converged policy -- five routes in the seed-0
# sweep stopped on a single 100 and then evaluated in the 30s-50s. Three in a row is a much
# stronger signal, and the KL penalty is what should make it reachable without the policy
# drifting off the base behaviour that produced the first one.
set -uo pipefail
cd /home/cglossop/ogbench-carla
ROUTE="${1:?route}"; GPU="${2:?gpu}"; SLOT="${3:?slot}"; EXP="${4:?exp}"
LOG_DIR=".run_carla/jobs/${EXP}"; mkdir -p "$LOG_DIR"
say() { echo "[klstreak $(date +%H:%M:%S)] $*" | tee -a "${LOG_DIR}/run.log"; }

say "waiting for gpu ${GPU}"
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
  [ "${used:-999999}" -lt 20000 ] && break
  sleep 180
done
say "gpu ${GPU} free (${used} MiB); starting ${ROUTE}"
rm -rf "/raid/users/cglossop/experiments/${EXP}" 2>/dev/null
SEED=0 \
AGENT_CFG=impls/configs/steervla_cast_relabel_hl200x10_adaptive_config.py \
RUN_GROUP="$EXP" \
OGBENCH_SAVE_DIR="/raid/users/cglossop/experiments/${EXP}" \
UPDATES_AFTER_SCORE=0 \
MAX_HL_UPDATES=500 \
STOP_ON_SCORE=100 \
STOP_SCORE_STREAK=3 \
ONLINE_STEPS=20000 \
  ./.run_carla/eval_subset_run.sh "$ROUTE" "$GPU" "$SLOT" \
    --cot-temperature 0.1 \
    --hl-kl-coef 0.1 \
  > "${LOG_DIR}/${ROUTE}.log" 2>&1
say "exited ($?)"
