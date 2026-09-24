#!/usr/bin/env python3
"""Hand routes from the dgx6 half of the sweep to bellman, without duplicating work.

Removes the named routes from the *live* queue of the running sweep (holding the driver's own lock,
so a worker cannot pop mid-edit) and writes them to a route file for the other machine. Only routes
still QUEUED can be handed over: one already running or finished is reported and skipped, because
the other machine would redo work that is already paid for.

  # see what would move
  ./handoff_routes.py --jobs-dir <dgx6 .run_carla/jobs/<SWEEP>> --out routes_bellman.txt \\
      --routes generalization-bad-parking-1004 generalization-obscured-stop-1046 --dry-run
  # then for real (drop --dry-run), copy routes_bellman.txt to bellman, and launch there
"""
from __future__ import annotations

import argparse
import fcntl
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-dir", required=True, help=".run_carla/jobs/<SWEEP_NAME> of the running sweep")
    ap.add_argument("--out", required=True, help="route file to write for the other machine")
    ap.add_argument("--routes", nargs="+", default=[], help="routes to hand over (by name)")
    ap.add_argument("--tail", type=int, default=0, help="instead: hand over the LAST N queued routes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.routes and not args.tail:
        ap.error("give --routes <name> ... or --tail N")

    jobs = Path(args.jobs_dir)
    queue, lock = jobs / "queue.txt", jobs / "queue.lock"
    if not queue.is_file():
        print(f"no queue at {queue}", file=sys.stderr)
        return 2

    with open(lock, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)  # the driver's next_route() holds this too
        queued = [ln.strip() for ln in queue.read_text().splitlines() if ln.strip()]
        wanted = queued[-args.tail:] if args.tail else list(dict.fromkeys(args.routes))
        move = [r for r in wanted if r in queued]
        missing = [r for r in wanted if r not in queued]
        for r in missing:
            print(f"  SKIP {r}: not in the queue (already running, done, or misspelled)")
        if not move:
            print("nothing to hand over")
            return 1
        remaining = [r for r in queued if r not in set(move)]
        print(f"hand over {len(move)} route(s):")
        for r in move:
            print(f"  -> {r}")
        print(f"queue: {len(queued)} -> {len(remaining)} routes")
        if args.dry_run:
            print("dry run: queue untouched, no route file written")
            return 0
        Path(args.out).write_text("\n".join(move) + "\n", encoding="utf-8")
        # Write the queue in place, under the lock, so the running workers see the shorter list.
        queue.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
    print(f"wrote {args.out}; copy it to the other machine and use it as ROUTES_FILE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
