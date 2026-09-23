#!/usr/bin/env bash
# sweep_results.sh — collect a sweep's per-route results into one reviewable folder.
#
# Reads every run_summary.json under $OGBENCH_SAVE_DIR and writes, into $RESULTS_DIR:
#   RESULTS.md        a table, sorted by eval mean
#   results.csv       the same, for plotting
#   summaries/        a copy of each route's run_summary.json
#   strategies.md     the episode strategy sentences each route produced, with their scores
#   periodic_evals.csv  frozen evals run DURING training (main_carla --eval_every_env_steps), if any;
#                       also appended to RESULTS.md as a per-route table of mean score by train step
#
# Optional ROUTES_FILE (one route per line, e.g. f2d_subset.txt): report only those routes. Runs of
# routes dropped from the subset stay on disk untouched; they just stop being reported.
#
# A route whose training run died before writing run_summary.json, but which has a checkpoint eval
# (ckpt_evals/<step>/run_summary_frozen_eval.json), is reported from its LATEST such checkpoint and
# marked as such -- those are different weights from every other row's final checkpoint.
#
# Optional RESULT_TRAIN_STEP (e.g. 10000): a run that trained past that many train steps is reported
# by the frozen eval of its checkpoint at that step (from periodic_evals.jsonl), not its final one.
#
# Re-runnable at any time; the sweep calls it after every route so the folder is current mid-run.
set -uo pipefail
SAVE_DIR="${OGBENCH_SAVE_DIR:?set OGBENCH_SAVE_DIR}"
RESULTS_DIR="${RESULTS_DIR:?set RESULTS_DIR}"
SWEEP="${SWEEP:-$(basename "$SAVE_DIR")}"
ROUTES_FILE="${ROUTES_FILE:-}"
mkdir -p "$RESULTS_DIR/summaries"

SAVE_DIR="$SAVE_DIR" RESULTS_DIR="$RESULTS_DIR" SWEEP="$SWEEP" ROUTES_FILE="$ROUTES_FILE" \
RESULT_TRAIN_STEP="${RESULT_TRAIN_STEP:-}" python3 - <<'PY'
import json, os, re, shutil
from pathlib import Path

save, out, sweep = Path(os.environ["SAVE_DIR"]), Path(os.environ["RESULTS_DIR"]), os.environ["SWEEP"]
_rf = os.environ.get("ROUTES_FILE", "").strip()
ALLOWED = None
if _rf:
    if not Path(_rf).is_file():
        raise SystemExit(f"[sweep_results] ROUTES_FILE {_rf} does not exist")
    ALLOWED = {l.strip() for l in open(_rf) if l.strip() and not l.lstrip().startswith("#")}
RESULT_TRAIN_STEP = int(os.environ.get("RESULT_TRAIN_STEP", "") or 0)


def _eval_cutoff(run_dir):
    """'60s' if the run's frozen evals used the leaderboard AgentBlockedTest rule, '1s' if they used the
    wrapper's training cutoff (runs launched before main_carla --eval_crash_stuck_steps existed, or <= 0),
    None if unknown."""
    f = Path(run_dir) / "flags.json"
    try:
        v = json.loads(f.read_text()).get("eval_crash_stuck_steps")
    except Exception:
        return None
    return "60s" if (v is not None and int(v) > 0) else "1s"


def _eval_at_train_step(run_dir, training):
    """With RESULT_TRAIN_STEP set and a run whose final checkpoint is past it: the frozen eval of its
    RESULT_TRAIN_STEP checkpoint, i.e. the first periodic_evals.jsonl record at or after that train step
    (updates stay frozen from a checkpoint until its periodic eval). None when not applicable."""
    if not RESULT_TRAIN_STEP:
        return None
    last = str(training.get("final_checkpoint") or "").rsplit("/", 1)[-1]
    if not last.isdigit() or int(last) <= RESULT_TRAIN_STEP:
        return None
    pj = Path(run_dir) / "periodic_evals.jsonl"
    if not pj.exists():
        return None
    for line in pj.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if int(r.get("train_env_step", -1)) >= RESULT_TRAIN_STEP:
            return dict(
                eval=[{"eval_seed": s, "driving_score": v}
                      for s, v in zip(r.get("eval_seeds", []), r.get("driving_scores", []))],
                hl_updates=r.get("hl_updates_applied"),
            )
    return None


def _sd(vals):
    vals = [float(v) for v in vals if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _avg(entries):
    vals = [float(e.get("driving_score")) for e in entries if e.get("driving_score") is not None]
    return sum(vals) / len(vals) if vals else None


rows = []
for p in sorted(save.rglob("run_summary.json")):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    if ALLOWED is not None and d.get("route") not in ALLOWED:
        continue
    shutil.copy2(p, out / "summaries" / f"{d.get('route','unknown')}.json")
    t, ev = d.get("training", {}), list(d.get("eval", []))
    _at = _eval_at_train_step(p.parent, t)
    if _at is not None:
        ev = list(_at["eval"])
    # Extra eval seeds run later against the same final checkpoint (frozen_eval_sweep.sh) live
    # beside the original summary. Pool them: they evaluate the same weights with the same
    # carla_seed, differing only in model seed, so 1001-1003 and 1004-1006 are interchangeable
    # draws and the pooled mean is simply a better-estimated version of the same quantity.
    fe = p.parent / "run_summary_frozen_eval.json"
    # Extra seeds evaluate the final checkpoint, so they never pool into a RESULT_TRAIN_STEP row.
    if fe.exists() and _at is None:
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
    n_orig = len(ev) if _at is not None else len(d.get("eval", []))
    ev_old, ev_new = ev[:n_orig], ev[n_orig:]
    # Intermediate-checkpoint eval (ckpt_eval_at_step.sh). Reported ALONGSIDE the headline, never
    # pooled into it: these are different weights, so averaging them with the final checkpoint's
    # scores would silently mix two policies into one number.
    ev_mid, mid_step = [], ""
    mid_root = p.parent / "ckpt_evals"
    if mid_root.is_dir():
        for sd in sorted(mid_root.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 0):
            f = sd / "run_summary_frozen_eval.json"
            if f.exists():
                try:
                    ev_mid = list(json.loads(f.read_text()).get("eval", []))
                    mid_step = sd.name
                except Exception:
                    pass
                break
    rows.append(dict(
        route=d.get("route", "?"),
        seeds=f"{d.get('seeds',{}).get('carla_seed','?')}/{d.get('seeds',{}).get('train_seed','?')}",
        stop=(f"@{RESULT_TRAIN_STEP}" if _at is not None
              else "score" if "past driving_score" in str(t.get("stop_reason", ""))
              else "budget" if "training budget" in str(t.get("stop_reason", ""))
              else "manual" if "stopped manually" in str(t.get("stop_reason", ""))
              else "cap"),
        grad=_at["hl_updates"] if _at is not None else t.get("hl_updates_applied"), env=t.get("env_steps"),
        train_ds=t.get("final_driving_score"),
        evals=[e.get("driving_score") for e in ev],
        eval_seeds=[e.get("eval_seed") for e in ev],
        evals_old=[e.get("driving_score") for e in ev_old],
        evals_new=[e.get("driving_score") for e in ev_new],
        mean_old=_avg(ev_old),
        mean_new=_avg(ev_new),
        eval_mean=_avg(ev),
        mean_mid=_avg(ev_mid),
        mid_step=mid_step,
        # Bench2Drive success == DS 100 exactly (RC x IP with no infraction). Counted over the
        # pooled eval episodes of the FINAL checkpoint, so it matches the headline mean.
        n_success=sum(1 for e in ev if (e.get("driving_score") or 0) >= 99.999),
        # Sample SD (ddof=1) across this route's eval seeds -- the scenario is identical across
        # them under --fixed_train_carla_seed, so this is spread attributable to the model's
        # sampling seed alone. Needs >= 2 episodes.
        eval_sd=_sd([e.get("driving_score") for e in ev]),
        ckpt=str(RESULT_TRAIN_STEP) if _at is not None else str(t.get("final_checkpoint") or "").rsplit("/", 1)[-1],
        from_ckpt="",
        eval_cutoff=_eval_cutoff(p.parent),
    ))

# Routes with no final summary but a checkpoint eval: report the latest checkpoint, marked. A run
# dir that DOES have a run_summary.json already reported its ckpt_evals above as the 8k column.
_have = {r["route"] for r in rows}
_fallback = {}
for f in save.rglob("run_summary_frozen_eval.json"):
    if "ckpt_evals" not in f.parts or not f.parent.name.isdigit():
        continue
    if (f.parent.parent.parent / "run_summary.json").exists():
        continue
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    route, step = d.get("route", "?"), int(f.parent.name)
    if route in _have or (ALLOWED is not None and route not in ALLOWED):
        continue
    if route not in _fallback or step > _fallback[route][0]:
        _fallback[route] = (step, d)
for route, (step, d) in _fallback.items():
    ev = list(d.get("eval", []))
    rows.append(dict(
        route=route,
        seeds=f"{d.get('seeds',{}).get('carla_seed','?')}/{d.get('seeds',{}).get('train_seed','?')}",
        stop=f"died@{step}", grad=None, env=None, train_ds=None,
        evals=[e.get("driving_score") for e in ev],
        eval_seeds=[e.get("eval_seed") for e in ev],
        evals_old=[e.get("driving_score") for e in ev], evals_new=[],
        mean_old=_avg(ev), mean_new=None, eval_mean=_avg(ev),
        mean_mid=None, mid_step="",
        n_success=sum(1 for e in ev if (e.get("driving_score") or 0) >= 99.999),
        eval_sd=_sd([e.get("driving_score") for e in ev]),
        ckpt=str(step), from_ckpt=str(step), eval_cutoff=None,
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

def mean_key(rs, key):
    vals = [r[key] for r in rs if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


split = [r for r in rows if r["route"] not in EXCLUDED]
excl = [r for r in rows if r["route"] in EXCLUDED]

HDR = ("| route | stop | grad | env | train DS | "
       "eval 1001-1003 | mean old | eval 1004-1006 | mean new | **pooled** | n | "
       "sd | success | 8k ckpt | 8k-vs-final |\n"
       "|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")

def num(v):
    return f"{v:.2f}" if v is not None else "-"

def dash(v):
    return "-" if v is None else str(v)

def table(rs):
    out_lines = [HDR]
    def fmt(vals):
        return " / ".join(f"{v:.2f}" if v is not None else "-" for v in vals) or "_pending_"

    for r in rs:
        delta = ("-" if (r["mean_mid"] is None or r["eval_mean"] is None)
                 else f"{r['eval_mean'] - r['mean_mid']:+.2f}")
        mark = " †" if r["from_ckpt"] else ""
        out_lines.append(
            f"| `{r['route']}`{mark} | {r['stop']} | {dash(r['grad'])} | {dash(r['env'])} | "
            f"{num(r['train_ds'])} | {fmt(r['evals_old'])} | {num(r['mean_old'])} | "
            f"{fmt(r['evals_new'])} | {num(r['mean_new'])} | **{num(r['eval_mean'])}** | "
            f"{len(r['evals'])} | {num(r['eval_sd'])} | "
            f"{r['n_success']}/{len(r['evals'])} = {100.0*r['n_success']/max(1,len(r['evals'])):.0f}% | "
            f"{num(r['mean_mid'])} | {delta} |")
    notes = [f"† `{r['route']}`: the training run died before a final checkpoint; this row is the "
             f"frozen eval of its step-{r['from_ckpt']} checkpoint, not a final one."
             for r in rs if r["from_ckpt"]]
    if notes:
        out_lines += [""] + [n + "  " for n in notes]
    return out_lines

n_new = sum(1 for r in rows if r["evals_new"])
md = [f"# {sweep}", "",
      (f"**Eval split (pooled): {mean_of(split):.2f}** "
       f"(avg per-route SD {mean_key(split, 'eval_sd'):.2f}) over {len(split)} routes  "),
      f"&nbsp;&nbsp;seeds 1001-1003 only: {mean_key(split, 'mean_old'):.2f}  ",
      (f"&nbsp;&nbsp;seeds 1004-1006 only: {mean_key(split, 'mean_new'):.2f} "
       f"({sum(1 for r in split if r['evals_new'])}/{len(split)} routes have them)  "
       if any(r['evals_new'] for r in split)
       else f"&nbsp;&nbsp;seeds 1004-1006 only: _not run yet_ (0/{len(split)} routes)  "),
      (f"All {len(rows)} routes complete (pooled): {mean_of(rows):.2f} "
       f"(avg per-route SD {mean_key(rows, 'eval_sd'):.2f})  "),
      f"Excluded ({len(excl)} of {len(EXCLUDED)} run so far): {mean_of(excl):.2f}",
      "",
      f"Extra eval seeds run so far: {n_new}/{len(rows)} routes. Same final checkpoint, same "
      f"carla_seed, different model seed -- pooled with the originals as interchangeable draws.",
      "", ]
if RESULT_TRAIN_STEP:
    md += [f"Scored at {RESULT_TRAIN_STEP} train steps: a run that trained past it is reported by the frozen "
           f"eval of its {RESULT_TRAIN_STEP} checkpoint (stop `@{RESULT_TRAIN_STEP}`; grad = HL updates at that "
           f"checkpoint), not by its final checkpoint.", ""]
if ALLOWED is not None:
    md += [f"Routes restricted to `{_rf}` ({len(ALLOWED)} listed).", ""]
def _succ(rs):
    """(route-averaged success rate, successes, episodes). Route-averaged weights each route
    equally regardless of how many eval seeds it has; the episode-level ratio is given too."""
    rs = [r for r in rs if r["evals"]]
    if not rs:
        return 0.0, 0, 0
    per = sum(r["n_success"] / len(r["evals"]) for r in rs) / len(rs)
    return per, sum(r["n_success"] for r in rs), sum(len(r["evals"]) for r in rs)


_sp, _sk, _sn = _succ(split)
_ap, _ak, _an = _succ(rows)
md += [f"**Success rate (DS = 100)**: eval split **{_sp*100:.1f}%** "
       f"(route-averaged; {_sk}/{_sn} episodes = {100.0*_sk/max(1,_sn):.1f}%)  ",
       f"&nbsp;&nbsp;all {len(rows)} routes: {_ap*100:.1f}% ({_ak}/{_an} episodes)  ",
       f"&nbsp;&nbsp;routes that succeed at least once: "
       f"{sum(1 for r in split if r['n_success'] > 0)}/{len(split)} in the split", ""]

_mid = [r for r in split if r["mean_mid"] is not None and r["eval_mean"] is not None]
if _mid:
    m8 = sum(r["mean_mid"] for r in _mid) / len(_mid)
    mf = sum(r["eval_mean"] for r in _mid) / len(_mid)
    better = sum(1 for r in _mid if r["mean_mid"] > r["eval_mean"])
    step = next((r["mid_step"] for r in _mid if r["mid_step"]), "8k")
    md += [f"**Checkpoint {step} vs final**, over the {len(_mid)} split routes with both: "
           f"{step} mean {m8:.2f} vs final {mf:.2f} ({mf - m8:+.2f}); "
           f"the earlier checkpoint wins on {better}/{len(_mid)}. "
           f"Reported alongside, never pooled -- they are different weights.", ""]
md += ["## Eval split", ""]
md += table(split)
md += ["", "## Excluded from the eval split", "",
       "Reported for completeness; not counted in the headline number.", ""]
md += table(excl) if excl else ["_(none complete yet)_"]
# Eval blocked-agent cutoff. Only reported when a sweep MIXES runs from before and after main_carla's
# --eval_crash_stuck_steps, so reports of uniform sweeps are unchanged.
_old_cut = sorted(r["route"] for r in rows if r.get("eval_cutoff") == "1s")
if _old_cut and any(r.get("eval_cutoff") == "60s" for r in rows):
    md += ["", "## Eval blocked-agent cutoff", "",
           "These routes' frozen evals (periodic and final) used the wrapper's 1 s post-collision stuck "
           "cutoff (`crash_stuck_steps: 20`); every other route used the leaderboard's AgentBlockedTest "
           "(60 s), as in run_leaderboard.py. A 1 s eval episode that wedges after a collision ends early, "
           "so these scores can be lower than the 60 s rule would give.", ""]
    md += [f"- `{r}`" for r in _old_cut]
# Periodic frozen evals (main_carla --eval_every_env_steps): one periodic_evals.jsonl per run dir,
# read directly so routes still training show their curve too. Absent -> nothing is added.
pe_rows = []
for pj in sorted(save.rglob("periodic_evals.jsonl")):
    for line in pj.read_text().splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        e["run"] = pj.parent.name
        # Same ROUTES_FILE restriction the headline table uses: a route dropped from the subset
        # keeps its run dir on disk but stops being reported, here as everywhere else.
        if ALLOWED is not None and (e.get("route") or e["run"]) not in ALLOWED:
            continue
        pe_rows.append(e)
if pe_rows:
    by_route = {}
    for e in pe_rows:
        by_route.setdefault(e.get("route") or e["run"], []).append(e)
    md += ["", "## Periodic evals during training", "",
           "Frozen eval episodes run every `--eval_every_env_steps` training steps (carla_seed replayed, "
           "model seeds = eval seeds). Cells are the mean driving score at that train step "
           "(HL grad steps in parentheses); the final column is the end-of-training eval mean.", ""]
    final_by_route = {r["route"]: r["eval_mean"] for r in rows}
    md.append("| route | evals (train step: mean (HL grad steps)) | final eval |")
    md.append("|---|---|---:|")
    for route in sorted(by_route):
        es = sorted(by_route[route], key=lambda e: e["train_env_step"])
        cells = ", ".join(f"{e['train_env_step']}: **{e['mean_driving_score']:.1f}** ({e.get('hl_updates_applied', '?')})" for e in es)
        fm = final_by_route.get(route)
        md.append(f"| `{route}` | {cells} | {fm:.2f} |" if fm is not None else f"| `{route}` | {cells} | _training_ |")
    with open(out / "periodic_evals.csv", "w") as f:
        f.write("route,run,train_env_step,env_step,hl_updates_applied,carla_seed,eval_seeds,driving_scores,mean_driving_score\n")
        for e in sorted(pe_rows, key=lambda e: ((e.get("route") or e["run"]), e["train_env_step"])):
            f.write(f"{e.get('route', '')},{e['run']},{e['train_env_step']},{e.get('env_step', '')},"
                    f"{e.get('hl_updates_applied', '')},{e.get('carla_seed', '')},"
                    f"{' '.join(str(x) for x in e.get('eval_seeds', []))},"
                    f"{' '.join(f'{x:.2f}' for x in e.get('driving_scores', []))},{e['mean_driving_score']:.4f}\n")
(out / "RESULTS.md").write_text("\n".join(md) + "\n")

def _c(v):
    return "" if v is None else v

with open(out / "results.csv", "w") as f:
    f.write("route,carla_seed,train_seed,stop,grad_steps,env_steps,train_ds,n_evals,n_success,success_rate,eval_scores,eval_seeds,mean_old,mean_new,eval_mean_pooled,eval_sd,in_eval_split,final_ckpt\n")
    for r in rows:
        c, t = (r["seeds"].split("/") + ["", ""])[:2]
        sc = " ".join(f"{v:.2f}" for v in r["evals"] if v is not None)
        sd = " ".join(str(v) for v in r["eval_seeds"])
        mo = "" if r["mean_old"] is None else f"{r['mean_old']:.4f}"
        mn = "" if r["mean_new"] is None else f"{r['mean_new']:.4f}"
        esd = "" if r["eval_sd"] is None else f"{r['eval_sd']:.4f}"
        f.write(f"{r['route']},{c},{t},{r['stop']},{_c(r['grad'])},{_c(r['env'])},{_c(r['train_ds'])},"
                f"{len(r['evals'])},{r['n_success']},{r['n_success']/max(1,len(r['evals'])):.4f},{sc},{sd},{mo},{mn},{_c(r['eval_mean'])},{esd},{0 if r['route'] in EXCLUDED else 1},{r['ckpt']}\n")

# Strategy sentences: the episode-level memory each route accumulated.
lines = [f"# {sweep} — episode strategies", ""]
for sm in sorted(save.rglob("strategy_memory.json")):
    route = "?"
    m = re.search(r"upd-hl_(.+?)_seed_", str(sm))
    if m:
        route = m.group(1)
    if ALLOWED is not None and route not in ALLOWED:
        continue
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
      f"eval split {mean_of(split):.2f} (avg SD {mean_key(split, 'eval_sd'):.2f}) over {len(split)}   "
      f"all {mean_of(rows):.2f}   extra-seed routes {n_new}/{len(rows)}   "
      f"from-checkpoint rows {len(_fallback)}")
PY
