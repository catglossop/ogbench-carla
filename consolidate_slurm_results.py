#!/usr/bin/env python3
"""Consolidate a SLURM-array leaderboard eval into a run_leaderboard-style run directory.

`fail2drive/slurm_evaluate.py` fans a route set out over a SLURM array and drops one
`<route_id>_res.json` per route into `<run>/res/`. Each of those is an ordinary leaderboard
`StatisticsManager` checkpoint -- the same artefact `run_leaderboard.py` writes -- but they
are named by numeric route id and there is no aggregate over the set.

This turns such a directory into the layout the rest of this repo expects:

    <out>/records/<route-name>.json     each route's record, renamed to its scenario name
    <out>/leaderboard_summary.json      one aggregate, schema-identical to a normal run

so `watch_leaderboard.py`, `merge_leaderboard_summaries.py` and `report_f2d_status.py` all
work against it unchanged. Scores are **not** recomputed: records are parsed with
`run_leaderboard.read_record` and aggregated with `run_leaderboard.aggregate`.

Usage
-----
    ./consolidate_slurm_results.py --routes fail2drive \
        --run f2d_simlingo_celine=/raid/users/celine/results/simlingo \
        --run f2d_steervla_celine=/raid/users/celine/results/steervla

`--run NAME=PATH` is repeatable; PATH is the directory containing `res/`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import run_leaderboard as rl  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME=PATH",
                   help="output name and the SLURM result dir holding res/ (repeatable)")
    p.add_argument("--routes", default="fail2drive",
                   help="route source the ids belong to. Default: fail2drive")
    p.add_argument("--out-root", default=str(REPO_ROOT / "leaderboard_runs"))
    p.add_argument("--seed", type=int, default=None,
                   help="seed to record; omitted (null) when unknown rather than guessed")
    p.add_argument("--note", default=None, help="free-text provenance note")
    args = p.parse_args()

    from ogbench.carla.route_registry import list_routes
    by_id = {e.route_id: e.scenario_name for e in list_routes(source=args.routes)}

    out_root = Path(args.out_root).expanduser().resolve()
    rc = 0

    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run expects NAME=PATH, got {spec!r}")
        name, src = spec.split("=", 1)
        src_dir = Path(src).expanduser().resolve()
        res_dir = src_dir / "res"
        if not res_dir.is_dir():
            raise SystemExit(f"no res/ under {src_dir}")

        out_dir = out_root / name
        rec_out = out_dir / "records"
        rec_out.mkdir(parents=True, exist_ok=True)

        results, pending, unknown = [], [], []
        for path in sorted(res_dir.glob("*_res.json")):
            rid = path.name.split("_")[0].lstrip("0") or "0"
            route = by_id.get(rid)
            if route is None:
                unknown.append(path.name)
                continue
            rec = rl.read_record(path, route)
            if rec is None:                     # started but never scored
                pending.append(route)
                continue
            results.append(rec)
            shutil.copy2(path, rec_out / f"{route}.json")

        results.sort(key=lambda r: r.route)
        pending.sort()
        missing = sorted(set(by_id.values()) - {r.route for r in results} - set(pending))
        agg = rl.aggregate(results)

        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
            "routes_spec": args.routes,
            "agent_config": None,
            "steervla_checkpoint": None,
            "n_completed": len(results),
            "n_pending": len(pending) + len(missing),
            "pending": pending + missing,
            "aggregate": agg,
            "_source": {
                "note": args.note or (
                    "Consolidated from a SLURM-array leaderboard eval by "
                    "consolidate_slurm_results.py. Scores are read from each route's "
                    "StatisticsManager record and aggregated by run_leaderboard.aggregate "
                    "-- not recomputed. 'wall_s' is the record's duration_system (the "
                    "route's own system time), not the worker's total wall time."),
                "source_dir": str(src_dir),
                "n_res_files": len(list(res_dir.glob("*_res.json"))),
                "unscored_routes": pending,
                "ids_not_in_registry": unknown,
            },
            "routes": [
                {
                    "route": r.route,
                    "status": r.status,
                    "driving_score": r.score_composed,
                    "route_completion": r.score_route,
                    "infraction_penalty": r.score_penalty,
                    "num_infractions": r.num_infractions,
                    "infractions": r.infractions,
                    "route_length_m": r.route_length,
                    "duration_game_s": r.duration_game,
                    "duration_system_s": r.duration_system,
                    "wall_s": round(r.duration_system, 1),
                }
                for r in results
            ],
        }
        (out_dir / "leaderboard_summary.json").write_text(json.dumps(summary, indent=2))

        print(f"{name}: {len(results)} scored -> {out_dir/'leaderboard_summary.json'}")
        print(f"  DS={agg['driving_score']:.2f} RC={agg['route_completion']:.2f} "
              f"IP={agg['infraction_penalty']:.3f} success={agg['success_rate']:.1f}% "
              f"km={agg['total_km']:.1f}")
        if pending:
            print(f"  WARNING: {len(pending)} route(s) started but never scored: {pending}")
            rc = 1
        if missing:
            print(f"  WARNING: {len(missing)} route(s) absent entirely: {missing[:10]}")
            rc = 1
        if unknown:
            print(f"  WARNING: {len(unknown)} file(s) whose id is not in the "
                  f"'{args.routes}' registry: {unknown[:5]}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
