#!/usr/bin/env python3
"""Merge several ``run_leaderboard.py`` run directories into one leaderboard_summary.json.

A run directory's ``leaderboard_summary.json`` only ever describes the routes of the
*invocation that wrote it* -- a ``--resume`` pass that scores 3 routes rewrites the file with
just those 3. The per-route records under ``records/`` are the durable artefact. This script
rebuilds a single summary over the union of several runs' records.

That matters for Fail2Drive, which cannot be run as one job: 10 routes need Fail2Drive's own
CARLA 0.9.15 build (for ``walker.animal.*``) and the other 190 run on vanilla 0.9.16, so the
sweep is necessarily split across two run directories with two different simulators.

Scores are **not** recomputed. Records are parsed with ``run_leaderboard.read_record`` and
aggregated with ``run_leaderboard.aggregate``, i.e. exactly the same code that writes a
normal single-run summary, so the output is schema-identical and comparable.

Usage
-----
    ./merge_leaderboard_summaries.py --out leaderboard_runs/f2d_llheavy_matchcrop_6k_seed0 \
        --source leaderboard_runs/f2d_llheavy_matchcrop_6k \
        --source leaderboard_runs/f2d_animals_eval \
        --expect fail2drive --seed 0 \
        --checkpoint /raid/users/cglossop/steervla_pi_ckpts/ll_heavy_unnormed_matchcrop/6000

The merged directory gets a ``records/`` copy of every record, so it is self-contained:
``watch_leaderboard.py`` renders it and a later ``--resume`` against it skips everything.
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

import run_leaderboard as rl  # noqa: E402  (reuse its record parsing + aggregation)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="merged run directory to create")
    p.add_argument("--source", action="append", required=True, metavar="RUN_DIR",
                   help="a run directory to merge (repeatable; order sets precedence)")
    p.add_argument("--expect", default=None,
                   help="route source to check coverage against ('fail2drive'/'bench2drive')")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--agent-config", default=None)
    p.add_argument("--no-copy-records", action="store_true",
                   help="write only the summary; do not copy records into --out")
    args = p.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    sources = [Path(s).expanduser().resolve() for s in args.source]
    for s in sources:
        if not (s / "records").is_dir():
            raise SystemExit(f"no records/ under {s}")

    # Collect records. First source wins a duplicate, but say so loudly -- for Fail2Drive a
    # duplicate means the same route was scored on both simulators, and which one is correct
    # is a judgement call the operator has to make.
    results, origin, duplicates = [], {}, []
    for src in sources:
        for path in sorted((src / "records").glob("*.json")):
            route = path.stem
            rec = rl.read_record(path, route)
            if rec is None:
                continue                      # stub/unfinished record
            if route in origin:
                duplicates.append((route, origin[route], src.name))
                continue
            origin[route] = src.name
            results.append(rec)

    results.sort(key=lambda r: r.route)
    agg = rl.aggregate(results)

    missing = []
    if args.expect:
        from ogbench.carla.route_registry import list_routes
        expected = {e.scenario_name for e in list_routes(source=args.expect)}
        missing = sorted(expected - set(origin))
        extra = sorted(set(origin) - expected)
        if extra:
            print(f"WARNING: {len(extra)} record(s) not in the '{args.expect}' registry: {extra[:5]}")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "routes_spec": args.expect or "merged",
        "agent_config": args.agent_config,
        "steervla_checkpoint": args.checkpoint,
        "n_completed": len(results),
        "n_pending": len(missing),
        "pending": missing,
        "aggregate": agg,
        "_merge": {
            "note": "Merged from several run directories by merge_leaderboard_summaries.py. "
                    "Scores are read from each route's StatisticsManager record and "
                    "aggregated by run_leaderboard.aggregate -- not recomputed. Per-route "
                    "'wall_s' is the record's duration_system (the route's own system time), "
                    "not the worker's total wall time, which is not stored in a record.",
            "sources": [str(s) for s in sources],
            "route_counts": {s.name: sum(1 for v in origin.values() if v == s.name)
                             for s in sources},
            "duplicates": [{"route": r, "kept_from": a, "ignored_from": b}
                           for r, a, b in duplicates],
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
                "source_run": origin[r.route],
            }
            for r in results
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_copy_records:
        rec_dir = out_dir / "records"
        rec_dir.mkdir(exist_ok=True)
        for src in sources:
            for path in (src / "records").glob("*.json"):
                if origin.get(path.stem) == src.name:
                    shutil.copy2(path, rec_dir / path.name)

    (out_dir / "leaderboard_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"merged {len(results)} routes -> {out_dir / 'leaderboard_summary.json'}")
    for s in sources:
        print(f"  {s.name}: {summary['_merge']['route_counts'][s.name]}")
    print(f"  DS={agg['driving_score']:.2f} RC={agg['route_completion']:.2f} "
          f"IP={agg['infraction_penalty']:.3f} success={agg['success_rate']:.1f}% "
          f"km={agg['total_km']:.1f}")
    if duplicates:
        print(f"  WARNING: {len(duplicates)} duplicate route(s): "
              f"{[d[0] for d in duplicates][:5]}")
    if missing:
        print(f"  WARNING: {len(missing)} expected route(s) missing: {missing[:10]}")
    return 1 if (missing or duplicates) else 0


if __name__ == "__main__":
    raise SystemExit(main())
