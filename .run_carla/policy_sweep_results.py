"""Aggregate the Qwen zero-shot BoN policy sweeps into one readable results file.

Reads every run_summary_frozen_eval.json the four sweeps have written so far and prints, per sweep:

  * the driving score of each (route, carla seed) cell,
  * the mean over seeds for each route,
  * the average over routes for each seed, and the overall average,
  * a running average over routes, in the sweep's own route order, so partial sweeps still read.

Cells that have not finished show as "--" and are left out of every average; each average says how
many cells it covers, so a partial column is never mistaken for a complete one.

Usage:  policy_sweep_results.py [--out PATH]      (default: the sweep root's RESULTS.md)
"""

import argparse
import json
import statistics
from pathlib import Path

P = Path("/raid/users/cglossop/sweep_results/qwenzs_policy_sweeps")
SWEEPS = [
    ("b2d_steervla", "Bench2Drive, SteerVLA base (hierarchical SimLingo)"),
    ("f2d_steervla", "Fail2Drive, SteerVLA base (hierarchical SimLingo)"),
    ("b2d_llheavy", "Bench2Drive, ll_heavy base (pi05)"),
    ("f2d_llheavy", "Fail2Drive, ll_heavy base (pi05)"),
]
SEEDS = (0, 1, 2)


def score(name: str, route: str, seed: int) -> float | None:
    """Driving score of one cell, or None if it has not finished."""
    f = Path(f"/raid/users/cglossop/sweep_results/qwenzs_bon_{name}/{route}/carla_seed_{seed}/run_summary_frozen_eval.json")
    if not f.is_file():
        return None
    try:
        return float(json.load(f.open())["eval_mean_driving_score"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None  # a summary still being written


def fmt(v: float | None, width: int = 6) -> str:
    return f"{v:>{width}.2f}" if v is not None else f"{'--':>{width}}"


def routes_of(name: str) -> list[str]:
    f = P / f"routes_{name}.txt"
    return [r.strip() for r in f.read_text().splitlines() if r.strip()] if f.is_file() else []


def render(name: str, title: str) -> list[str]:
    routes = routes_of(name)
    if not routes:
        return [f"## {name}", "", "_no route list; sweep not built_", ""]

    rows = {r: [score(name, r, s) for s in SEEDS] for r in routes}
    done = sum(1 for r in routes for v in rows[r] if v is not None)
    out = [
        f"## {name} — {title}",
        "",
        f"{done} of {len(routes) * len(SEEDS)} cells complete "
        f"({len(routes)} routes x {len(SEEDS)} carla seeds)",
        "",
        "| route | seed 0 | seed 1 | seed 2 | route mean | running mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    seen: list[float] = []
    for r in routes:
        vals = [v for v in rows[r] if v is not None]
        seen.extend(vals)
        mean = statistics.fmean(vals) if vals else None
        run = statistics.fmean(seen) if seen else None
        cells = "".join(f" {fmt(v)} |" for v in rows[r])
        out.append(f"| {r} |{cells} {fmt(mean)} | {fmt(run)} |")

    # Per-seed average across routes, then the overall average over every completed cell.
    per_seed = []
    for i, s in enumerate(SEEDS):
        vals = [rows[r][i] for r in routes if rows[r][i] is not None]
        per_seed.append((s, statistics.fmean(vals) if vals else None, len(vals)))
    allv = [v for r in routes for v in rows[r] if v is not None]
    overall = statistics.fmean(allv) if allv else None
    sd = statistics.stdev(allv) if len(allv) > 1 else None

    cells = "".join(f" {fmt(m)} |" for _, m, _ in per_seed)
    out += [
        f"| **average over routes** |{cells} **{fmt(overall).strip()}** | |",
        "",
        "Cells behind each average: "
        + ", ".join(f"seed {s}: {n}/{len(routes)}" for s, _, n in per_seed)
        + f", overall: {len(allv)}/{len(routes) * len(SEEDS)}"
        + (f"; sd over cells {sd:.2f}" if sd is not None else ""),
        "",
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(P / "RESULTS.md"))
    args = ap.parse_args()

    import datetime

    lines = [
        "# Qwen zero-shot BoN policy sweeps — driving scores",
        "",
        f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. Regenerate with "
        "`.run_carla/policy_sweep_results.py`.",
        "",
        "Frozen eval of the final CAST/DAgger checkpoint per route, scored by the leaderboard; "
        "3 carla seeds per route; zero-shot Qwen3.8-27B BoN critic (`scene_criteria_v2`, no adapter), "
        "4 candidates per decision. Episodes cap at 4000 wrapper steps.",
        "",
    ]
    for name, title in SWEEPS:
        lines += render(name, title)

    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
