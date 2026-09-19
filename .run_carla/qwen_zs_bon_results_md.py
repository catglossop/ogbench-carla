#!/usr/bin/env python3
"""Write results.md for a zero-shot Qwen BoN sweep (qwen_zs_bon_sweep.sh).

  python3 .run_carla/qwen_zs_bon_results_md.py                      # original pi05 b2d sweep
  python3 .run_carla/qwen_zs_bon_results_md.py --actor mixed \
      --source-sweep b2dsteervla_simlingo_fixedcarla_kl005_seed0 \
      --run-group b2dsteervla_simlingo_fixedcarla_kl005_seed0_qwenzs_bon_mixed \
      --routes-file /raid/users/cglossop/sweep_results/qwenzs_mixed_b2d_sources/routes_b2d.txt \
      --carla-seeds "0 1"

Per route: every finished episode (one row per CARLA seed x eval seed), the route's Qwen BoN mean and
sample std (n-1) over those episodes, and -- as a baseline -- the same final checkpoint's own
post-training eval from the source sweep (no BoN). Summary: mean of route means and std averaged over
routes, for Qwen BoN and for the baseline on the same routes. With --jobs-dir, a timing table per
worker GPU from the job logs. Re-run any time; unfinished routes show as pending.
"""
import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

PROTOCOL = {
    "pi05": [
        "**Actor:** each route's end-of-training SteerVLA checkpoint (`run_summary.json` -> `final_checkpoint`), "
        "`pi05_steervla_cot_simplified_reasoning_ll_heavy`, inference-only params.",
        "**BoN:** 8 batched candidates, CoT temperature 1.0, re-query every 3 env steps, no brake candidate.",
    ],
    "simlingo": [
        "**Actor:** hierarchical SimLingo SteerVLA: the route's end-of-training InternVL2 HL export, frozen training LL.",
        "**BoN:** 4 batched candidates, HL temperature 1.0, re-query every 3 env steps, no brake candidate.",
    ],
    "mixed": [
        "**Actor:** mixed SteerVLA (`steervla_mixed_eval_config.py`): the route's end-of-training InternVL2 HL export "
        "(CAST-relabel fine-tuned) generates reasoning + subtask; the pi05 `ll_heavy_unnormed_matchcrop/6000` "
        "checkpoint (`pi05_steervla_cot_simplified_reasoning_ll_heavy`) conditions on that text with its own vision "
        "backbone and action expert.",
        "**HL decoding is greedy** (forced in `internvl2_hl_worker.py`), so all candidates share one subtask; "
        "candidates differ only in the pi05 action samples.",
        "**BoN:** 4 candidates sampled **sequentially** (the batched path would bypass the InternVL2 HL), re-query "
        "every 3 env steps, no brake candidate.",
        "**Critics:** worker on GPU 5 used a critic on GPU 5 (shared card); worker on GPU 6 used a critic on GPU 7.",
    ],
}


def mean_std(xs):
    if not xs:
        return None, None
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else None)


def fmt(v, nd=2):
    return "–" if v is None else f"{v:.{nd}f}"


def pm(m, s):
    if m is None:
        return "–"
    return f"{m:.2f} ± {s:.2f}" if s is not None else f"{m:.2f}"


def job_timing(jobs_dir: Path):
    """Per worker GPU: mean BoN decision time ('VLA sample took') and mean s/step over job logs."""
    per = {}
    for log in sorted(jobs_dir.glob("*__cs*.log")):
        text = log.read_text(errors="replace").replace("\r", "\n")
        m = re.search(r"\[qwen_zs_run\] actor=\S+ bench=\S+ route=\S+ gpu=(\d+)", text)
        if not m:
            continue
        gpu = m.group(1)
        took = [float(x) for x in re.findall(r"VLA sample took ([0-9.]+)s", text)]
        rate = re.findall(r"\d+/\d+ \[[^\]]*?([0-9.]+)s/it\]", text)
        d = per.setdefault(gpu, {"decisions": [], "rates": [], "jobs": 0})
        d["decisions"] += took
        if rate:
            d["rates"].append(float(rate[-1]))
        d["jobs"] += 1
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-sweep", default="b2dsubset_fixedcarla_kl005_seed0")
    ap.add_argument("--routes-file", default="b2d_subset.txt")
    ap.add_argument("--run-group", default=None)
    ap.add_argument("--carla-seeds", default="0 1 2")
    ap.add_argument("--actor", default="pi05", choices=sorted(PROTOCOL))
    ap.add_argument("--jobs-dir", default=None, help="job log dir for the timing table (default: .run_carla/jobs/<group>)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    group = args.run_group or f"{args.source_sweep}_qwenzs_bon"
    src_root = Path("/raid/users/cglossop/sweeps") / args.source_sweep
    res_root = Path("/raid/users/cglossop/sweep_results") / group
    out = Path(args.out) if args.out else res_root / "results.md"
    routes_path = Path(args.routes_file)
    routes = [r for r in (routes_path if routes_path.is_absolute() else root / routes_path).read_text().split() if r]
    seeds = [int(s) for s in args.carla_seeds.split()]
    jobs_dir = Path(args.jobs_dir) if args.jobs_dir else root / ".run_carla" / "jobs" / group

    # Source (training) summaries: newest run_summary.json per route with a loadable final checkpoint.
    src = {}
    for p in src_root.rglob("run_summary.json"):
        if "ckpt_evals" in p.parts:
            continue
        d = json.loads(p.read_text())
        ck = (d.get("training") or {}).get("final_checkpoint")
        if not ck or not ((Path(ck) / "params").is_dir() or (Path(ck) / "pytorch_model.bin").is_file()):
            continue
        if d["route"] not in src or p.stat().st_mtime > src[d["route"]][0]:
            src[d["route"]] = (p.stat().st_mtime, d)
    src = {r: d for r, (_, d) in src.items()}

    cells = {}
    for r in routes:
        for s in seeds:
            f = res_root / r / f"carla_seed_{s}" / "run_summary_frozen_eval.json"
            if f.exists():
                d = json.loads(f.read_text())
                cells[(r, s)] = [(e["eval_seed"], e["driving_score"]) for e in d.get("eval") or []]

    running, queued = [], 0
    if (jobs_dir / "running").is_dir():
        running = [j.read_text().strip() for j in sorted((jobs_dir / "running").glob("*.job"))]
    if (jobs_dir / "queue.txt").exists():
        queued = sum(1 for ln in (jobs_dir / "queue.txt").read_text().splitlines() if ln.strip())

    eval_routes = [r for r in routes if r in src]
    excluded = [r for r in routes if r not in src]
    n_cells = len(eval_routes) * len(seeds)

    L = [f"# Zero-shot Qwen BoN ({args.actor}) — {args.source_sweep}", ""]
    L.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from `{res_root}`._")
    L.append("")
    L.append(f"**Status:** {len(cells)}/{n_cells} cells finished ({len(eval_routes)} routes x CARLA seeds "
             f"{', '.join(map(str, seeds))})"
             + (f"; running: {', '.join(running)}" if running else "")
             + (f"; {queued} queued" if queued else "") + ".")
    L.append("")

    L.append("## Per route")
    L.append("")
    L.append("Qwen BoN mean ± std over every finished episode of the route (sample std, n−1). Baseline = the same "
             "final checkpoint's post-training eval in the source sweep (no BoN; CARLA seed 0, model seeds "
             "1001–1003), so it is a reference, not a paired comparison.")
    L.append("")
    L.append("| Route | Ckpt | " + " | ".join(f"CARLA seed {s}" for s in seeds)
             + " | Qwen BoN mean ± std (n) | Baseline episodes | Baseline mean ± std | Δ mean |")
    L.append("|---|---:|" + "---:|" * len(seeds) + "---:|---|---:|---:|")
    bon_m, bon_s, base_m, base_s = [], [], [], []
    for r in eval_routes:
        d = src[r]
        ck = Path(d["training"]["final_checkpoint"]).name
        base = [e["driving_score"] for e in d.get("eval") or []]
        bm, bs = mean_std(base)
        cols, allx = [], []
        for s in seeds:
            if (r, s) in cells:
                xs = [x for _, x in cells[(r, s)]]
                allx += xs
                cols.append(", ".join(fmt(x) for x in xs))
            else:
                cols.append("_running_" if f"{r}__cs{s}" in running else "_pending_")
        qm, qs = mean_std(allx)
        if allx:
            bon_m.append(qm); base_m.append(bm)
            if qs is not None:
                bon_s.append(qs)
            if bs is not None:
                base_s.append(bs)
        delta = f"{qm - bm:+.2f}" if allx and bm is not None else "–"
        L.append(f"| {r} | {ck} | " + " | ".join(cols)
                 + f" | {pm(qm, qs)} ({len(allx)}) | {', '.join(fmt(x) for x in base)} | {pm(bm, bs)} | {delta} |")
    L.append("")
    if bon_m:
        L.append(f"**Summary over the {len(bon_m)} routes with at least one finished episode**")
        L.append("")
        L.append("| | Mean of route means | Std averaged over routes |")
        L.append("|---|---:|---:|")
        L.append(f"| Qwen BoN | {statistics.fmean(bon_m):.2f} | {fmt(statistics.fmean(bon_s) if bon_s else None)}"
                 + (f" ({len(bon_s)} routes with ≥2 episodes)" if len(bon_s) != len(bon_m) else "") + " |")
        L.append(f"| Baseline (same routes) | {statistics.fmean(base_m):.2f} | "
                 f"{fmt(statistics.fmean(base_s) if base_s else None)} |")
        allv = [x for v in cells.values() for _, x in v]
        L.append("")
        L.append(f"Episode-level Qwen BoN mean over all {len(allv)} finished episodes: {statistics.fmean(allv):.2f}.")
        L.append("")

    timing = job_timing(jobs_dir) if jobs_dir.is_dir() else {}
    if timing:
        L.append("## Timing per worker GPU")
        L.append("")
        L.append("From the job logs: `VLA sample took` is the whole BoN decision (HL + LL candidates + Qwen critic); "
                 "s/step is the final tqdm rate of each job, averaged.")
        L.append("")
        L.append("| Worker GPU | Jobs | BoN decisions | Mean decision time | Mean s/step |")
        L.append("|---:|---:|---:|---:|---:|")
        for gpu in sorted(timing):
            t = timing[gpu]
            L.append(f"| {gpu} | {t['jobs']} | {len(t['decisions'])} | "
                     f"{fmt(statistics.fmean(t['decisions']) if t['decisions'] else None, 1)} s | "
                     f"{fmt(statistics.fmean(t['rates']) if t['rates'] else None, 2)} |")
        L.append("")

    if excluded:
        L.append("## Not evaluated")
        L.append("")
        L.append("No end-of-training checkpoint in the source sweep: " + ", ".join(f"`{r}`" for r in excluded) + ".")
        L.append("")

    L.append("## Protocol")
    L.append("")
    for line in PROTOCOL[args.actor]:
        L.append(f"- {line}")
    L.append("- **Critic:** zero-shot `Qwen/Qwen3.8-27B`, no adapter, `scene_criteria_v2` prompt (original traffic "
             "wording), BF16 eager; risk thresholds 0.8; utility = 0.5·goal + progress + 0.5·correctness − crash "
             "− 0.5·off-road − 0.5·traffic.")
    L.append(f"- **Eval:** frozen eval, CARLA seeds {', '.join(map(str, seeds))}; see the per-route columns for the "
             "episodes per seed. 4000-step episode cap.")
    L.append("")

    out.write_text("\n".join(L))
    print(out)


if __name__ == "__main__":
    main()
