#!/usr/bin/env bash
# Full Bench2Drive evaluation-subset version of run_cast_to_residual.sh.
# It deliberately delegates all stage/checkpoint/resume logic to that paired runner.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Same 23-route evaluation subset used by run_residual_eval.sh. Two sheet labels
# are already canonicalized for the route registry: static-cut-in / merger-into.
SUBSET_ROUTES="signalized-junction-left-turn-001 enter-actor-flow-004 non-signalized-junction-right-turn-001 non-signalized-junction-left-turn-enter-flow-002 signalized-junction-left-turn-enter-flow-003 non-signalized-junction-left-turn-002 signalized-junction-right-turn-004 parked-obstacle-004 accident-two-ways-002 construction-obstacle-003 highway-exit-002 pedestrian-crossing-004 parking-exit-002 static-cut-in-001 vanilla-signalized-turn-encounter-red-light-002 accident-005 crossing-bicycle-flow-004 vanilla-signalized-turn-encounter-green-light-004 merger-into-slow-traffic-001 vehicle-turning-route-pedestrian-005 vehicle-opens-door-two-ways-005 sequential-lane-change-005 interurban-actor-flow-004"

export CAST_RESIDUAL_LABEL="${CAST_RESIDUAL_LABEL:-cast-to-residual-eval-subset-20260908}"
export CAST_RESIDUAL_ROUTES="${CAST_RESIDUAL_ROUTES:-$SUBSET_ROUTES}"
export CAST_RESIDUAL_SEEDS="${CAST_RESIDUAL_SEEDS:-0 1 2}"

exec "$ROOT_DIR/run_cast_to_residual.sh" "$@"
