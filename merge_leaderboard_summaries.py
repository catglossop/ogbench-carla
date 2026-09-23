"""Merge several leaderboard_summary.json files into one.

Re-aggregates with the exact same logic as run_leaderboard.py :: aggregate(),
so the merged numbers are directly comparable to the per-run summaries.

    python merge_leaderboard_summaries.py <out.json> <in1.json> <in2.json> ...
"""
import json
import sys
from datetime import datetime
from pathlib import Path

STATUS_SUCCESS = ("Completed", "Perfect")

out = Path(sys.argv[1])
srcs = [Path(p) for p in sys.argv[2:]]

routes, seen, specs = [], {}, []
meta = None
for s in srcs:
    blob = json.loads(s.read_text())
    if meta is None:
        meta = blob
    for key in ("seed", "agent_config", "steervla_checkpoint"):
        if blob.get(key) != meta.get(key):
            sys.exit(f"mismatch on {key!r}: {meta.get(key)!r} vs {blob.get(key)!r} ({s})")
    specs.append(blob.get("routes_spec"))
    if blob.get("n_pending"):
        sys.exit(f"{s} still has {blob['n_pending']} pending routes")
    for r in blob["routes"]:
        if r["route"] in seen:
            sys.exit(f"duplicate route {r['route']} in {s} and {seen[r['route']]}")
        seen[r["route"]] = s
        routes.append(r)

n = len(routes)
total_km = sum(r.get("route_length_m", 0.0) for r in routes) / 1000.0
infraction_totals: dict[str, int] = {}
for r in routes:
    for k, v in (r.get("infractions") or {}).items():
        infraction_totals[k] = infraction_totals.get(k, 0) + len(v)

agg = {
    "n": n,
    "driving_score": sum(r["driving_score"] for r in routes) / n,
    "route_completion": sum(r["route_completion"] for r in routes) / n,
    "infraction_penalty": sum(r["infraction_penalty"] for r in routes) / n,
    "success_rate": 100.0 * sum(r["status"] in STATUS_SUCCESS for r in routes) / n,
    "total_km": total_km,
    "infraction_totals": infraction_totals,
    "infractions_per_km": {
        k: (v / total_km if total_km > 0 else 0.0) for k, v in infraction_totals.items()
    },
}

summary = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "seed": meta["seed"],
    "routes_spec": specs,
    "merged_from": [str(s.parent) for s in srcs],
    "agent_config": meta["agent_config"],
    "steervla_checkpoint": meta["steervla_checkpoint"],
    "n_completed": n,
    "n_pending": 0,
    "pending": [],
    "aggregate": agg,
    "routes": sorted(routes, key=lambda x: x["route"]),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2))
print(
    f"wrote {out}  n={n}  DS={agg['driving_score']:.2f}  RC={agg['route_completion']:.2f}  "
    f"SR={agg['success_rate']:.2f}%  km={total_km:.2f}"
)
