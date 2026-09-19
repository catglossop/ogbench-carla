#!/usr/bin/env python3
"""Write results.md for a zero-shot Qwen BoN sweep (qwen_zs_bon_sweep.sh).

  python3 .run_carla/qwen_zs_bon_results_md.py                      # b2d defaults
  python3 .run_carla/qwen_zs_bon_results_md.py --source-sweep ... --routes-file f2d_subset.txt

Std is the SAMPLE std (n-1) over a cell's eval seeds (model seeds carla_seed+1001..), i.e. the
spread of the policy on a fixed scenario. "Std averaged over routes" is the mean of the per-route
stds. The baseline is the same checkpoint's own post-training eval from the source sweep's
run_summary.json: carla seed 0, eval seeds 1001-1003, no BoN, so it pairs with the carla-seed-0 cell.
Re-run any time; rows for unfinished cells show as pending.
"""
import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-sweep", default="b2dsubset_fixedcarla_kl005_seed0")
    ap.add_argument("--routes-file", default="b2d_subset.txt")
    ap.add_argument("--run-group", default=None)
    ap.add_argument("--carla-seeds", default="0 1 2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    group = args.run_group or f"{args.source_sweep}_qwenzs_bon"
    src_root = Path("/raid/users/cglossop/sweeps") / args.source_sweep
    res_root = Path("/raid/users/cglossop/sweep_results") / group
    out = Path(args.out) if args.out else res_root / "results.md"
    routes = [r for r in (root / args.routes_file).read_text().split() if r]
    seeds = [int(s) for s in args.carla_seeds.split()]
    jobs_dir = root / ".run_carla" / "jobs" / group

    # Source (training) summaries: newest run_summary.json per route with a final checkpoint.
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

    running = []
    for jf in sorted((jobs_dir / "running").glob("*.job")) if (jobs_dir / "running").is_dir() else []:
        running.append(jf.read_text().strip())
    queued = []
    if (jobs_dir / "queue.txt").exists():
        queued = [ln.split("\t")[0] + f"__cs{ln.split(chr(9))[2]}" for ln in (jobs_dir / "queue.txt").read_text().splitlines() if ln.strip()]

    eval_routes = [r for r in routes if r in src]
    excluded = [r for r in routes if r not in src]

    L = []
    L.append(f"# Zero-shot Qwen BoN — {args.source_sweep}")
    L.append("")
    L.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from `{res_root}`._")
    n_done0 = sum((r, 0) in cells for r in eval_routes)
    L.append("")
    L.append(f"**Status:** {n_done0}/{len(eval_routes)} routes have their CARLA-seed-0 cell; "
             f"{len(cells)} cells finished in total"
             + (f"; running: {', '.join(running)}" if running else "")
             + (f"; {len(queued)} queued." if queued else "."))
    L.append("")

    # ---- Headline: carla seed 0, paired with the training eval on the same seeds ----
    L.append("## CARLA seed 0 (eval seeds 1001–1003) vs. the checkpoint's own eval")
    L.append("")
    L.append("Std = sample std (n−1) over the 3 eval seeds. Baseline = the same final checkpoint's "
             "post-training eval from the source sweep (same CARLA seed and model seeds, no BoN).")
    L.append("")
    L.append("| Route | Ckpt | Qwen BoN episodes | Qwen BoN mean ± std | Baseline episodes | Baseline mean ± std | Δ mean |")
    L.append("|---|---:|---|---:|---|---:|---:|")
    bon_m, bon_s, base_m, base_s = [], [], [], []
    for r in eval_routes:
        d = src[r]
        ck = Path(d["training"]["final_checkpoint"]).name
        base = [e["driving_score"] for e in d.get("eval") or []]
        bm, bs = mean_std(base)
        if (r, 0) in cells:
            q = [x for _, x in cells[(r, 0)]]
            qm, qs = mean_std(q)
            bon_m.append(qm); base_m.append(bm)
            if qs is not None and bs is not None:
                bon_s.append(qs); base_s.append(bs)
            L.append(f"| {r} | {ck} | {', '.join(fmt(x) for x in q)} | {pm(qm, qs)} | "
                     f"{', '.join(fmt(x) for x in base)} | {pm(bm, bs)} | {qm - bm:+.2f} |")
        else:
            state = "running" if f"{r}__cs0" in running else "pending"
            L.append(f"| {r} | {ck} | _{state}_ | – | {', '.join(fmt(x) for x in base)} | {pm(bm, bs)} | – |")
    L.append("")
    if bon_m:
        L.append(f"**Over the {len(bon_m)} completed routes:**")
        L.append("")
        L.append("| | Mean of route means | Std averaged over routes |")
        L.append("|---|---:|---:|")
        L.append(f"| Qwen BoN | {statistics.fmean(bon_m):.2f} | {fmt(statistics.fmean(bon_s) if bon_s else None)} |")
        L.append(f"| Baseline (same routes) | {statistics.fmean(base_m):.2f} | {fmt(statistics.fmean(base_s) if base_s else None)} |")
        L.append("")

    # ---- Routes with more than one CARLA seed ----
    multi = [r for r in eval_routes if sum((r, s) in cells for s in seeds) > 1]
    if multi:
        L.append("## Routes with multiple CARLA seeds")
        L.append("")
        L.append("Per-seed columns are mean ± std over that seed's 3 eval seeds. “All episodes” pools every "
                 "finished episode for the route; “mean per-seed std” averages the per-seed stds.")
        L.append("")
        L.append("| Route | " + " | ".join(f"CARLA seed {s}" for s in seeds) + " | All episodes (n) | Mean per-seed std |")
        L.append("|---|" + "---:|" * len(seeds) + "---:|---:|")
        for r in multi:
            cols, allx, stds = [], [], []
            for s in seeds:
                if (r, s) in cells:
                    xs = [x for _, x in cells[(r, s)]]
                    m, sd = mean_std(xs)
                    cols.append(pm(m, sd)); allx += xs
                    if sd is not None:
                        stds.append(sd)
                else:
                    cols.append("_running_" if f"{r}__cs{s}" in running else "–")
            am, asd = mean_std(allx)
            L.append(f"| {r} | " + " | ".join(cols) + f" | {pm(am, asd)} ({len(allx)}) | "
                     f"{fmt(statistics.fmean(stds) if stds else None)} |")
        L.append("")

    if excluded:
        L.append("## Not evaluated")
        L.append("")
        L.append("No end-of-training checkpoint in the source sweep: " + ", ".join(f"`{r}`" for r in excluded) + ".")
        L.append("")

    L.append("## Protocol")
    L.append("")
    L.append("- **Critic:** zero-shot `Qwen/Qwen3.8-27B`, no adapter, `scene_criteria_v2` prompt (original traffic wording), "
             "BF16 eager; risk thresholds 0.8 (crash/off-road/traffic); utility = 0.5·goal + progress + 0.5·correctness "
             "− crash − 0.5·off-road − 0.5·traffic.")
    L.append("- **Actor:** each route's end-of-training SteerVLA checkpoint (`run_summary.json` → `final_checkpoint`), "
             "`pi05_steervla_cot_simplified_reasoning_ll_heavy`, inference-only params.")
    L.append("- **BoN:** 8 batched candidates, CoT temperature 1.0, re-query every 3 env steps, no brake candidate, "
             "`actions_per_cot=5`, `actions_per_model_query=3`.")
    L.append("- **Eval:** frozen eval, 3 episodes per CARLA seed, model seeds CARLA seed + 1001..1003 "
             "(`--train-seed` = first eval seed), 4000-step episode cap, CARLA 0.9.16.")
    L.append("- **Baseline:** the source sweep's post-training frozen eval of the same checkpoint: single policy sample "
             "(no BoN, no critic), CoT temperature 0.1, CARLA seed 0, eval seeds 1001–1003.")
    L.append("- **Scope change (2026-09-14):** routes after `non-signalized-junction-right-turn-001` run CARLA seed 0 only.")
    L.append("")

    out.write_text("\n".join(L))
    print(out)


if __name__ == "__main__":
    main()
