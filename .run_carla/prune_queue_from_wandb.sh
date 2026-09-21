#!/usr/bin/env bash
# prune_queue_from_wandb.sh QUEUE_FILE RUN_GROUP
#
# For a qwen_zs_bon_sweep.sh driver that is ALREADY running (it pops a queue it built at arm time
# and never re-checks it): every INTERVAL seconds, drop from its queue.txt each cell another
# machine has claimed on W&B (wandb_claimed_cells.py). Edits happen under the same flock the
# sweep's workers take to pop jobs, so a pop and a prune never interleave. Exits when the queue
# is empty. Only other hosts' claims count, so the local sweep's own retries are left alone.
#
#   INTERVAL=600 setsid nohup ./.run_carla/prune_queue_from_wandb.sh \
#     <sweep worktree>/.run_carla/jobs/<RUN_GROUP>/queue.txt <RUN_GROUP> > prune.log 2>&1 &
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE="${1:?queue.txt of the running sweep}"; GROUP="${2:?W&B run group}"
LOCK="$(dirname "$QUEUE")/queue.lock"
INTERVAL="${INTERVAL:-600}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/cglossop/.wandb_school_key}"
PY="${PY:-${ROOT_DIR}/.venv/bin/python}"
[ -f "$QUEUE" ] && [ -f "$LOCK" ] || { echo "[prune] ABORT: $QUEUE or its queue.lock missing" >&2; exit 1; }
log() { echo "[prune $(date +%H:%M:%S)] $*"; }

tmp="$(mktemp)"; trap 'rm -f "$tmp" "$tmp.q" "$tmp.d"' EXIT
while :; do
  if ! WANDB_API_KEY="$(cat "$WANDB_KEY_FILE")" "$PY" "${ROOT_DIR}/.run_carla/wandb_claimed_cells.py" \
       "$GROUP" --list > "$tmp"; then
    log "W&B query failed; retrying in 120 s"; sleep 120; continue
  fi
  {
    flock 9
    # queue line: route<TAB>checkpoint<TAB>seed<TAB>attempt ; claim line: route<TAB>seed<TAB>state<TAB>host
    awk -F'\t' -v drops="$tmp.d" 'NR == FNR { c[$1 "\t" $2] = $4; next }
                ($1 "\t" $3) in c { printf "dropped %s cs%s (claimed on %s)\n", $1, $3, c[$1 "\t" $3] > drops; next }
                { print }' "$tmp" "$QUEUE" > "$tmp.q"
    cat "$tmp.q" > "$QUEUE"   # rewrite in place: keeps the inode the sweep holds open
  } 9>>"$LOCK"
  [ -s "$tmp.d" ] && while read -r l; do log "$l"; done < "$tmp.d"
  : > "$tmp.d"
  left=$(grep -c . "$QUEUE" || true)
  log "queue has ${left} cell(s) left; $(wc -l < "$tmp") claimed elsewhere"
  [ "$left" -eq 0 ] && { log "queue empty; exiting"; exit 0; }
  sleep "$INTERVAL"
done
