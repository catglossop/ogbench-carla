#!/usr/bin/env bash
# Collect the checkpoint ladder into one table: eval score as a function of training checkpoint.
set -uo pipefail
EXP="${EXP:-ckptladder_seqlane005}"
SAVE_DIR="/raid/users/cglossop/experiments/${EXP}"
OUT="/raid/users/cglossop/experiment_results/${EXP}"
mkdir -p "$OUT"
SAVE_DIR="$SAVE_DIR" OUT="$OUT" EXP="$EXP" python3 - <<'PY'
import json, os
from pathlib import Path
save, out, exp = Path(os.environ["SAVE_DIR"]), Path(os.environ["OUT"]), os.environ["EXP"]

base = None
for p in save.rglob("run_summary.json"):
    if "ckpt_evals" in p.parts:
        continue
    base = json.loads(p.read_text()); run_dir = p.parent; break

rows = []
if base is not None:
    for d in sorted((run_dir / "ckpt_evals").glob("*"), key=lambda x: int(x.name) if x.name.isdigit() else 0):
        f = d / "run_summary_frozen_eval.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        sc = [e["driving_score"] for e in s.get("eval", [])]
        rows.append((int(d.name), sc, sum(sc) / len(sc) if sc else None))

md = [f"# {exp} — checkpoint ladder", ""]
if base is None:
    md += ["_training has not produced a run_summary.json yet_"]
else:
    t = base["training"]
    md += [f"Route `{base['route']}` · cot_temperature 0.1 · no updates after first 100 DS  ",
           f"Training stopped: {t['stop_reason']}  ",
           f"Final training DS {t['final_driving_score']:.2f} at {t['hl_updates_applied']} gradient "
           f"steps / {t['env_steps']} env steps  ",
           "Stop-checkpoint eval (seeds 1001-1003): "
           + " / ".join("%.2f" % e["driving_score"] for e in base.get("eval", []))
           + " → mean **%.2f**" % (base.get("eval_mean_driving_score") or 0), "",
           "## Eval score by checkpoint", "",
           "| checkpoint (env step) | eval 1001 | eval 1002 | eval 1003 | mean |",
           "|---:|---:|---:|---:|---:|"]
    if rows:
        best = max(r[2] for r in rows if r[2] is not None)
        for st, sc, m in rows:
            cells = " | ".join(f"{v:.2f}" for v in sc) if sc else "- | - | -"
            star = " **←best**" if m is not None and abs(m - best) < 1e-9 else ""
            md.append(f"| {st} | {cells} | **{m:.2f}**{star} |")
        md += ["", f"Best checkpoint scores {best:.2f}; the run stopped at "
                   f"{t['env_steps']} env steps with an eval mean of "
                   f"{base.get('eval_mean_driving_score') or 0:.2f}."]
    else:
        md += ["| _pending_ | | | | |"]
(out / "CKPT_LADDER.md").write_text("\n".join(md) + "\n")

with open(out / "ckpt_ladder.csv", "w") as f:
    f.write("checkpoint_env_step,eval1,eval2,eval3,mean\n")
    for st, sc, m in rows:
        e = (sc + [None, None, None])[:3]
        f.write(f"{st},{e[0]},{e[1]},{e[2]},{m}\n")
print(f"[ckpt_ladder] {len(rows)} checkpoints -> {out}/CKPT_LADDER.md")
PY
