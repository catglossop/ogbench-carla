#!/usr/bin/env python3
"""Drop from THIS machine's queue every cell the other machine has already finished.

This is how the two sweeps meet in the middle without running a cell twice: bellman works the
routes front to back, the second machine back to front, and each periodically removes what the
other has done. When a machine's queue empties its workers log "queue empty" and stop.

Only queued cells are removed; a cell already running is left alone. Takes the queue lock the
driver uses, so it is safe while the sweep runs.

    python3 prune_queue.py JOBS_DIR done_on_other_machine.tsv
"""
import fcntl
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit(__doc__)
jobs, done_file = Path(sys.argv[1]), Path(sys.argv[2])
done = {tuple(ln.split("\t")[:2]) for ln in done_file.read_text().splitlines() if ln.strip()}

with open(jobs / "queue.lock", "a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    q = jobs / "queue.txt"
    lines = [ln for ln in q.read_text().splitlines() if ln.strip()] if q.exists() else []
    keep = [ln for ln in lines if (ln.split("\t")[0], ln.split("\t")[2]) not in done]
    q.write_text("".join(ln + "\n" for ln in keep))

for ln in lines:
    if ln not in keep:
        f = ln.split("\t")
        print(f"  removed {f[0]} seed {f[2]} (done on the other machine)")
print(f"queue: {len(lines)} -> {len(keep)}")
