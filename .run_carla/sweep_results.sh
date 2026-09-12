#!/usr/bin/env bash
# sweep_results.sh — collect a sweep's per-route results into one reviewable folder.
#
# Reads every run_summary.json under $OGBENCH_SAVE_DIR and writes, into $RESULTS_DIR:
#   RESULTS.md        a table, sorted by eval mean
#   results.csv       the same, for plotting
#   summaries/        a copy of each route's run_summary.json
#   strategies.md     the episode strategy sentences each route produced, with their scores
#
# Re-runnable at any time; the sweep calls it after every route so the folder is current mid-run.
set -uo pipefail
SAVE_DIR="${OGBENCH_SAVE_DIR:?set OGBENCH_SAVE_DIR}"
RESULTS_DIR="${RESULTS_DIR:?set RESULTS_DIR}"
SWEEP="${SWEEP:-$(basename "$SAVE_DIR")}"
mkdir -p "$RESULTS_DIR/summaries"

SAVE_DIR="$SAVE_DIR" RESULTS_DIR="$RESULTS_DIR" SWEEP="$SWEEP" python3 - <<'PY'
import json, os, re, shutil
from pathlib import Path

save, out, sweep = Path(os.environ["SAVE_DIR"]), Path(os.environ["RESULTS_DIR"]), os.environ["SWEEP"]
def _avg(entries):
    vals = [float(e.get("driving_score")) for e in entries if e.get("driving_score") is not None]
    return sum(vals) / len(vals) if vals else None


rows = []
for p in sorted(save.rglob("run_summary.json")):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    shutil.copy2(p, out / "summaries" / f"{d.get('route','unknown')}.json")
    t, ev = d.get("training", {}), list(d.get("eval", []))
    # Extra eval seeds run later against the same final checkpoint (frozen_eval_sweep.sh) live
    # beside the original summary. Pool them: they evaluate the same weights with the same
    # carla_seed, differing only in model seed, so 1001-1003 and 1004-1006 are interchangeable
    # draws and the pooled mean is simply a better-estimated version of the same quantity.
    fe = p.parent / "run_summary_frozen_eval.json"
    if fe.exists():
        try:
            ev += list(json.loads(fe.read_text()).get("eval", []))
            shutil.copy2(fe, out / "summaries" / f"{d.get('route','unknown')}.frozen_eval.json")
        except Exception:
            pass
    seen_sd, ev_u = set(), []
    for e in ev:
        sd = e.get("eval_seed")
        if sd in seen_sd:
            continue
        seen_sd.add(sd); ev_u.append(e)
    ev = ev_u
    n_orig = len(d.get("eval", []))
    ev_old, ev_new = ev[:n_orig], ev[n_orig:]
    rows.append(dict(
        route=d.get("route", "?"),
        seeds=f"{d.get('seeds',{}).get('carla_seed','?')}/{d.get('seeds',{}).get('train_seed','?')}",
        stop="score" if "past driving_score" in str(t.get("stop_reason", "")) else "cap",
        grad=t.get("hl_updates_applied"), env=t.get("env_steps"),
        train_ds=t.get("final_driving_score"),
        evals=[e.get("driving_score") for e in ev],
        eval_seeds=[e.get("eval_seed") for e in ev],
        evals_old=[e.get("driving_score") for e in ev_old],
        evals_new=[e.get("driving_score") for e in ev_new],
        mean_old=_avg(ev_old),
        mean_new=_avg(ev_new),
        eval_mean=_avg(ev),
        ckpt=str(t.get("final_checkpoint") or "").rsplit("/", 1)[-1],
    ))

rows.sort(key=lambda r: (r["eval_mean"] is None, -(r["eval_mean"] or 0)))

# Routes held out of the headline number. These are excluded from the eval split by request;
# they still run, still get checkpoints, and are still reported below -- just scored separately
# so one hard/degenerate route can't drag the split mean around.
EXCLUDED = {
    "parked-obstacle-004",
    "accident-two-ways-002",
    "construction-obstacle-003",
    "highway-exit-002",
    "accident-005",
    "parking-exit-002",
    "static-cut-in-001",
}

def mean_of(rs):
    vals = [r["eval_mean"] for r in rs if r["eval_mean"] is not None]
    return sum(vals) / len(vals) if vals else 0.0

split = [r for r in rows if r["route"] not in EXCLUDED]
excl = [r for r in rows if r["route"] in EXCLUDED]

HDR = ("| route | seeds c/t | stop | grad | env | train DS | "
       "eval 1001-1003 | mean old | eval 1004-1006 | mean new | **pooled** | n |\n"
       "|---|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|")

def table(rs):
    out_lines = [HDR]
    def fmt(vals):
        return " / ".join(f"{v:.2f}" if v is not None else "-" for v in vals) or "_pending_"

    def num(v):
        return f"{v:.2f}" if v is not None else "-"

    for r in rs:
        out_lines.append(
            f"| `{r['route']}` | {r['seeds']} | {r['stop']} | {r['grad']} | {r['env']} | "
            f"{r['train_ds']:.2f} | {fmt(r['evals_old'])} | {num(r['mean_old'])} | "
            f"{fmt(r['evals_new'])} | {num(r['mean_new'])} | **{num(r['eval_mean'])}** | "
            f"{len(r['evals'])} |")
    return out_lines

def mean_key(rs, key):
    vals = [r[key] for r in rs if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


n_new = sum(1 for r in rows if r["evals_new"])
md = [f"# {sweep}", "",
      f"**Eval split (pooled): {mean_of(split):.2f}** over {len(split)} routes  ",
      f"&nbsp;&nbsp;seeds 1001-1003 only: {mean_key(split, 'mean_old'):.2f}  ",
      (f"&nbsp;&nbsp;seeds 1004-1006 only: {mean_key(split, 'mean_new'):.2f} "
       f"({sum(1 for r in split if r['evals_new'])}/{len(split)} routes have them)  "
       if any(r['evals_new'] for r in split)
       else f"&nbsp;&nbsp;seeds 1004-1006 only: _not run yet_ (0/{len(split)} routes)  "),
      f"All {len(rows)} routes complete (pooled): {mean_of(rows):.2f}  ",
      f"Excluded ({len(excl)} of {len(EXCLUDED)} run so far): {mean_of(excl):.2f}",
      "",
      f"Extra eval seeds run so far: {n_new}/{len(rows)} routes. Same final checkpoint, same "
      f"carla_seed, different model seed -- pooled with the originals as interchangeable draws.",
      "", "## Eval split", ""]
md += table(split)
md += ["", "## Excluded from the eval split", "",
       "Reported for completeness; not counted in the headline number.", ""]
md += table(excl) if excl else ["_(none complete yet)_"]
(out / "RESULTS.md").write_text("\n".join(md) + "\n")

with open(out / "results.csv", "w") as f:
    f.write("route,carla_seed,train_seed,stop,grad_steps,env_steps,train_ds,n_evals,eval_scores,eval_seeds,mean_old,mean_new,eval_mean_pooled,in_eval_split,final_ckpt\n")
    for r in rows:
        c, t = (r["seeds"].split("/") + ["", ""])[:2]
        sc = " ".join(f"{v:.2f}" for v in r["evals"] if v is not None)
        sd = " ".join(str(v) for v in r["eval_seeds"])
        mo = "" if r["mean_old"] is None else f"{r['mean_old']:.4f}"
        mn = "" if r["mean_new"] is None else f"{r['mean_new']:.4f}"
        f.write(f"{r['route']},{c},{t},{r['stop']},{r['grad']},{r['env']},{r['train_ds']},"
                f"{len(r['evals'])},{sc},{sd},{mo},{mn},{r['eval_mean']},{0 if r['route'] in EXCLUDED else 1},{r['ckpt']}\n")

# Strategy sentences: the episode-level memory each route accumulated.
lines = [f"# {sweep} — episode strategies", ""]
for sm in sorted(save.rglob("strategy_memory.json")):
    route = "?"
    m = re.search(r"upd-hl_(.+?)_seed_", str(sm))
    if m:
        route = m.group(1)
    try:
        entries = json.loads(sm.read_text()).get("strategies", [])
    except Exception:
        continue
    if not entries:
        continue
    lines.append(f"## {route}")
    for e in entries:
        lines.append(f"- **ep {e.get('episode')}** (score {float(e.get('driving_score',0)):.1f}): {e.get('sentence','')}")
    lines.append("")
(out / "strategies.md").write_text("\n".join(lines) + "\n")

print(f"[sweep_results] {len(rows)} routes -> {out}/RESULTS.md   "
      f"eval split {mean_of(split):.2f} over {len(split)}   all {mean_of(rows):.2f}   "
      f"extra-seed routes {n_new}/{len(rows)}")
PY
