"""Generate the Fail2Drive leaderboard status report as Markdown, from live run state."""
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/home/cglossop/ogbench-carla")
from ogbench.carla.route_registry import list_routes  # noqa: E402

LB = Path("/home/cglossop/ogbench-carla/leaderboard_runs")
MAIN, ANIM = LB / "f2d_llheavy_matchcrop_6k", LB / "f2d_animals_eval"
OUT = LB / "F2D_LEADERBOARD_STATUS.md"
CKPT = "/raid/users/cglossop/steervla_pi_ckpts/ll_heavy_unnormed_matchcrop/6000"

# Formerly blocked by the 20-tick build_scenarios deadlock (any route with a 2nd scenario).
# Fixed in carla_utils.py (commit ff7af76), so nothing is blocked any more -- these are just
# ordinary routes now. Kept as a named list because they are the ones to re-check first if
# that deadlock ever regresses.
WAS_BLOCKED = ["base-animals-0076", "base-animals-0079", "base-animals-0083",
               "generalization-animals-1076", "generalization-animals-1079",
               "generalization-animals-1083"]
BLOCKED = []
NEEDS_F2D = {f"generalization-animals-10{n}" for n in range(75, 85)}

all_routes = [e.scenario_name for e in list_routes(source="fail2drive")]
main_list = set((LB / "f2d_routes_main.txt").read_text().split())
anim_list = set((LB / "f2d_routes_animals.txt").read_text().split())


def load(run_dir):
    rows = {}
    rec = run_dir / "records"
    for p in sorted(rec.glob("*.json")) if rec.is_dir() else []:
        try:
            r = json.loads(p.read_text())["_checkpoint"]["records"][0]
        except Exception:
            continue
        if not r.get("scores"):
            continue
        rows[p.stem] = dict(
            ds=r["scores"]["score_composed"], rc=r["scores"]["score_route"],
            ip=r["scores"]["score_penalty"], status=str(r.get("status", "")),
            km=(r.get("meta") or {}).get("route_length", 0) / 1000.0,
            infr=r.get("infractions") or {},
        )
    return rows


def agg(rows):
    if not rows:
        return None
    n = len(rows)
    km = sum(v["km"] for v in rows.values())
    tot = Counter()
    for v in rows.values():
        for k, lst in v["infr"].items():
            tot[k] += len(lst) if isinstance(lst, list) else 0
    succ = sum(1 for v in rows.values() if v["status"] in ("Completed", "Perfect"))
    return dict(n=n, ds=sum(v["ds"] for v in rows.values()) / n,
                rc=sum(v["rc"] for v in rows.values()) / n,
                ip=sum(v["ip"] for v in rows.values()) / n,
                success=100.0 * succ / n, km=km, infr=tot)


def alive(run_dir):
    out = subprocess.run(["pgrep", "-af", "run_leaderboard.py"], capture_output=True, text=True).stdout
    return str(run_dir) in out


main_rows, anim_rows = load(MAIN), load(ANIM)
main_agg, anim_agg = agg(main_rows), agg(anim_rows)
scored = set(main_rows) | set(anim_rows)
remaining = sorted((main_list | anim_list) - scored)
main_alive, anim_alive = alive(MAIN), alive(ANIM)

L = []
w = L.append
w("# Fail2Drive leaderboard — status report")
w("")
w(f"_Generated {datetime.now():%Y-%m-%d %H:%M}_ · checkpoint `ll_heavy_unnormed_matchcrop/6000`")
w("")
w("## Headline")
w("")
w(f"| | routes |")
w("|---|---|")
w(f"| Fail2Drive routes total | **{len(all_routes)}** |")
w(f"| Scored | **{len(scored)}** |")
w(f"| **Remaining to run** | **{len(remaining)}** |")
w(f"| Blocked (cannot run) | **{len(BLOCKED)}** |")
w("")
w(f"`{len(scored)} scored + {len(remaining)} remaining + {len(BLOCKED)} blocked = {len(all_routes)}`")
w("")
if BLOCKED:
    w(f"Runnable universe is **{len(all_routes) - len(BLOCKED)} of {len(all_routes)}**; the "
      f"other {len(BLOCKED)} hang deterministically (see *Blocked routes*).")
else:
    w(f"**All {len(all_routes)} routes are runnable.** The 6 that used to hang at 20 ticks "
      "were unblocked by the `build_scenarios` deadlock fix (commit `ff7af76`).")
w("")

w("## Scores so far")
w("")
w("| Job | CARLA | env | n | DS | RC | IP | success | km |")
w("|---|---|---|---:|---:|---:|---:|---:|---:|")
if main_agg:
    w(f"| main (non-animal) | 0.9.16 `~/carla` | `.venv` (3.11) | {main_agg['n']} | "
      f"{main_agg['ds']:.2f} | {main_agg['rc']:.2f} | {main_agg['ip']:.3f} | "
      f"{main_agg['success']:.1f}% | {main_agg['km']:.1f} |")
if anim_agg:
    w(f"| animals | 0.9.15 `~/f2d_carla` | `.venv-f2d-eval` (3.10) | {anim_agg['n']} | "
      f"{anim_agg['ds']:.2f} | {anim_agg['rc']:.2f} | {anim_agg['ip']:.3f} | "
      f"{anim_agg['success']:.1f}% | {anim_agg['km']:.2f} |")
w("")
w("DS/RC/IP come straight from the leaderboard `StatisticsManager` records; nothing is recomputed.")
w("**The main job's numbers are partial and will move** until the remaining routes land.")
w("")

if anim_rows:
    w("### Animal routes (all 7 runnable, complete)")
    w("")
    w("| route | animal | DS | RC | IP | status |")
    w("|---|---|---:|---:|---:|---|")
    for name in sorted(anim_rows):
        v = anim_rows[name]
        w(f"| `{name}` | yes | {v['ds']:.2f} | {v['rc']:.2f} | {v['ip']:.3f} | {v['status'][:28]} |")
    w("")
    w("These are the only routes needing Fail2Drive's own simulator build — they reference "
      "`walker.animal.*` blueprints that **only** `~/f2d_carla` registers. Verified spawning: "
      "`requested=walker.animal.1009 spawned=[...] matched=True`.")
    w("")

w("## Route accounting")
w("")
w(f"### Blocked — not run, not queued ({len(BLOCKED)})")
w("")
if BLOCKED:
    w("| route | needs f2d_carla | blocker |")
    w("|---|---|---|")
    for r in BLOCKED:
        w(f"| `{r}` | {'yes' if r in NEEDS_F2D else 'no'} | build_scenarios deadlock |")
else:
    w("**None.** Six routes previously hung at exactly 20 ticks / 1.0 s game time with their "
      "CARLA server alive and nothing logged, burning the full `--route-timeout`. Cause: "
      "`RouteScenario.__init__` sets `runtime_init_mode(True)` after building its first batch "
      "of scenarios, which makes `BasicScenario.__init__` call `world.wait_for_tick()`. "
      "Upstream builds scenarios on a separate thread while the main thread ticks; this "
      "wrapper builds on the main thread, so it waited for a tick only its own blocked call "
      "stack could produce. Fixed in `ff7af76` by clearing the flag around the wrapper's own "
      "`build_scenarios` call.")
    w("")
    w("The discriminator was **\"route has a second scenario\"**, not the scenario type — so "
      "this also affected 4 `Generalization_PedestrianCrowd` routes, and applies to "
      "**Bench2Drive** wherever a route carries more than one scenario.")
    w("")
    w(f"Formerly blocked, now ordinary routes: {', '.join('`' + r + '`' for r in WAS_BLOCKED)}.")
w("")

w(f"### Remaining to run ({len(remaining)})")
w("")
if remaining:
    w(f"All in the main job, which is **{'running' if main_alive else 'NOT running'}**.")
    w("")
    w("```")
    for i in range(0, len(remaining), 3):
        w("  " + "  ".join(x.ljust(42) for x in remaining[i:i + 3]).rstrip())
    w("```")
else:
    w("None — every runnable route has been scored.")
w("")

w("## How these were run")
w("")
w("Two jobs, because they need different CARLA versions:")
w("")
w("```bash")
w("# 1) the 187 non-animal routes, vanilla 0.9.16")
w("./run_leaderboard_f2d.sh --slots 5:5,6:6 \\")
w("  --routes @leaderboard_runs/f2d_routes_main.txt \\")
w("  --out-dir leaderboard_runs/f2d_llheavy_matchcrop_6k \\")
w("  --resume --xla-mem-fraction 0.30 --stall 600 --route-timeout 2400")
w("")
w("# 2) the 7 runnable animal routes, Fail2Drive's own 0.9.15 build + matching client")
w("./run_leaderboard_f2d.sh --slots 2:2 \\")
w("  --routes @leaderboard_runs/f2d_routes_animals.txt \\")
w("  --carla-root ~/f2d_carla \\")
w("  --python /home/cglossop/ogbench-carla/.venv-f2d-eval/bin/python \\")
w("  --out-dir leaderboard_runs/f2d_animals_eval \\")
w("  --wandb-mode online --run-group f2d-animals-eval \\")
w("  --rpc-base 13000 --tm-base 19000 --display-base 450 --setup-timeout 1200")
w("```")
w("")
w("Both use leaderboard-faithful settings (`crash_stuck_steps` disabled, `max_episode_steps=0`, "
  "`terminate_on_infraction=false`, greedy CoT, no gradient updates), identical to the "
  "Bench2Drive 220-route run — including the same absent OpenPI norm stats, so the numbers are "
  "directly comparable.")
w("")
w("The animal job logs to W&B (`catherineglossop/OGBench-CARLA`, group `f2d-animals-eval`), one "
  "run per route. Build its env with `build_f2d_eval_env.sh`.")
w("")

w("## Why two environments")
w("")
w("- `~/f2d_carla` is CARLA **0.9.15.2**; the repo's client is **0.9.16**. Mixing them segfaults "
  "the worker (`rc=139`). carla 0.9.15 has no cp311 wheel, so its client needs Python **3.10**.")
w("- The vanilla 0.9.16 install already carries every other Fail2Drive asset (all 42 "
  "`static.prop.*` ids resolve; its Content tree is a superset of f2d_carla's), so only the "
  "10 animal routes need the second env.")
w("- Copying f2d_carla's `WalkerFactory.uasset` into the 0.9.16 tree **does not work** — the "
  "0.9.15-cooked blueprint bytecode kills every server boot with "
  "`LowLevelFatalError: Unknown code token 30`. Do not retry it.")
w("")

w("## Bugs found and fixed along the way")
w("")
w("| # | Bug | Effect | Fix |")
w("|---|---|---|---|")
w("| 1 | `carla_utils` passed the render-fence timeout only as `-g.TimeoutForBlockOnRenderFence=N`, "
  "which 0.9.15 ignores | f2d_carla died at 60 s loading Town13 | also pass "
  "`-ExecCmds=g.TimeoutForBlockOnRenderFence 300000` |")
w("| 2 | 300 s `apply_settings` timeout hardcoded | cold shader cache on Town13 blew through it | "
  "`CARLA_SETUP_ATTEMPT_TIMEOUT` / `--setup-timeout` |")
w("| 3 | Animal lifecycle monitor read `self._spawn_transform` unconditionally; "
  "`DynamicObjectCrossing` never defines it | leaderboard swallowed it as *Skipping scenario*, so "
  "the animal spawned, the scenario was **dropped**, and the route scored with **no hazard** | "
  "skip only the monitor |")
w("| 4 | Worktree runs silently used the main checkout's `ogbench/` via `ogbench.pth` | patches "
  "had no effect | export `PYTHONPATH=$ROOT_DIR` |")
w("")
w("Bug 3 is the one worth remembering: it produced *optimistic* scores rather than a visible "
  "failure.")
w("")
w("## Caveat on this run's history")
w("")
w("An attempt to register the animal blueprints by swapping `WalkerFactory` into the 0.9.16 tree "
  "crashed every CARLA boot for ~7 minutes. Because the crash lands ~14 s into boot, the "
  "orchestrator churned through and burned the retry budget of **23 routes**. They were "
  "recovered by a `--resume` pass and are included above; no score was corrupted (a crashed route "
  "writes no record), but it cost wall-clock.")
w("")
w("## Files")
w("")
w("```")
w(f"{MAIN}/                 main run (records/, logs/, leaderboard_summary.json)")
w(f"{ANIM}/                        animal run")
w(f"{LB}/f2d_routes_main.txt                    187 non-animal routes")
w(f"{LB}/f2d_routes_animals.txt                 7 runnable animal routes")
w(f"{LB}/f2d_routes_hang_priorityatjunction.txt 6 blocked routes")
w(f"{LB}/route_id_map.txt                       route name <-> id for all 420 routes")
w("```")
w("")

OUT.write_text("\n".join(L) + "\n")
print(f"wrote {OUT} ({len(L)} lines)")
print(f"scored={len(scored)} remaining={len(remaining)} blocked={len(BLOCKED)}")
