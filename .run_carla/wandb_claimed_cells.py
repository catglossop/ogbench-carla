#!/usr/bin/env python3
"""Which (route, carla seed) cells of a Qwen BoN sweep are already claimed on W&B.

A cell is claimed when a run in the W&B group is ``running``, ``finished`` or ``crashed``; only
``failed``/``killed`` runs leave it free. ``crashed`` counts because W&B marks a run crashed when
its heartbeat drops, which also happens to cells that did write run_summary_frozen_eval.json
(seen 2026-09-21 on pedestrian-crowd-1069 cs0/cs1), so the state cannot tell a finished cell
from a dead one. W&B is the only state two machines share, so it is the claim signal; the
per-machine summaries stay the record of what actually finished.

  wandb_claimed_cells.py GROUP --list             # route<TAB>seed<TAB>state<TAB>host per claim
  wandb_claimed_cells.py GROUP ROUTE SEED         # exit 0 if claimed, 1 if free, 2 on API error

Only claims from OTHER hosts count (W&B records the host of every run): a machine's own crashed
cells are its own sweep's retries to manage, and must not be skipped or pruned on that account.
Set CLAIM_INCLUDE_OWN_HOST=1 to count this host's runs too (e.g. to inspect everything).

Run names come from main_carla: ``dsrl_critic-none_noupd_<route>_seed_<seed>_<timestamp>``.
Needs WANDB_API_KEY (the catherine_glossop school key) in the environment.
"""
import os
import re
import socket
import sys

NAME_RE = re.compile(r"_(?P<route>[a-z0-9-]+)_seed_(?P<seed>\d+)_\d{8}_\d{6}$")
CLAIMING = {"running", "finished", "crashed"}


def claims(group: str) -> list[tuple[str, int, str, str]]:
    import wandb

    entity = os.environ.get("WANDB_ENTITY", "catherineglossop")
    project = os.environ.get("WANDB_PROJECT", "OGBench-CARLA")
    own = None if os.environ.get("CLAIM_INCLUDE_OWN_HOST") == "1" else socket.gethostname()
    out = []
    for run in wandb.Api(timeout=60).runs(f"{entity}/{project}", {"group": group}):
        m = NAME_RE.search(run.name or "")
        if not m or run.state not in CLAIMING:
            continue
        host = (run.metadata or {}).get("host") or "?"
        if host == own:
            continue
        out.append((m["route"], int(m["seed"]), run.state, host))
    return out


def main() -> int:
    args = sys.argv[1:]
    try:
        if len(args) == 2 and args[1] == "--list":
            for route, seed, state, host in sorted(claims(args[0])):
                print(f"{route}\t{seed}\t{state}\t{host}")
            return 0
        if len(args) == 3:
            group, route, seed = args[0], args[1], int(args[2])
            return 0 if any(r == route and s == seed for r, s, _, _ in claims(group)) else 1
    except Exception as exc:  # network / auth: let the caller decide, never guess "free"
        print(f"[wandb_claimed_cells] W&B query failed: {exc!r}", file=sys.stderr)
        return 2
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
