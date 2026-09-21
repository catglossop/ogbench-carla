#!/usr/bin/env python3
"""Print the f2d cells this machine has finished, one `route<TAB>seed` per line.

Run on each machine and hand the output to the other one's prune_queue.py, so the two sweeps stop
before running anything the other already has.

    python3 cells_done.py RESULTS_DIR > done_here.tsv
"""
import sys
from pathlib import Path

res = Path(sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__))
for f in sorted(res.glob("*/carla_seed_[0-9]/run_summary_frozen_eval.json")):
    print(f"{f.parent.parent.name}\t{f.parent.name.removeprefix('carla_seed_')}")
