# Handoff — Fail2Drive leaderboard evaluation

Written 2026-09-08. Everything below is verified against the code or against a real run
unless it says otherwise. If you are picking this up on a different machine, read
**§2 Prerequisites** first — several things here are machine-specific.

Branch: `worktree-f2d-leaderboard` on `git@github.com:catglossop/ogbench-carla.git`
(7 commits ahead of `dev`, all pushed).

---

## 1. What this is

Score the SteerVLA policy (`ll_heavy_unnormed_matchcrop/6000`) on all **200 Fail2Drive
routes**, using the same faithful-leaderboard harness as the existing Bench2Drive 220-route
run (`leaderboard_runs/llheavy_matchcrop_6k_b2d220`), so the two are directly comparable.

`run_leaderboard.py` (the orchestrator) and `watch_leaderboard.py` (the read-only dashboard)
already existed and are unchanged in spirit — this work added a Fail2Drive launcher on top
and fixed five bugs that made F2D routes unrunnable.

**Read `LEADERBOARD_EVAL.md` first** — it explains the scoring harness itself (why
`crash_stuck_steps`/`max_episode_steps`/`terminate_on_infraction` are overridden, the slot
and port scheme, the persistent XLA cache). This document only covers the Fail2Drive delta.

---

## 2. Prerequisites (machine-specific — check every one)

| Thing | On the original box | Notes |
|---|---|---|
| Repo | `/home/cglossop/ogbench-carla` | editable install; `ogbench.pth` points at the **main** checkout — see §7 gotcha 1 |
| Vanilla CARLA **0.9.16** | `/home/cglossop/carla` (`$CARLA_ROOT`) | runs 190 of the 200 routes |
| Fail2Drive CARLA **0.9.15.2** | `/home/cglossop/f2d_carla` | needed by 10 routes only |
| Fail2Drive repo | `/home/cglossop/fail2drive` | supplies route XMLs + scenario classes |
| Checkpoint | `/raid/users/cglossop/steervla_pi_ckpts/ll_heavy_unnormed_matchcrop/6000` | |
| Main env (3.11) | `.venv` — carla **0.9.16** | drives the vanilla install |
| F2D env (3.10) | `.venv-f2d-eval` — carla **0.9.15** | build with `build_f2d_eval_env.sh` |
| W&B key | `~/.wandb_school_key` | school account; **never** the key in `~/.netrc` |

Fail2Drive assets: the six content packs (WallAssets, ImageAssets, StopOcclusions,
AnimalVarietyPack, FarmAnimalsPack, AfricanAnimalsPack) are already installed into the
**0.9.16** tree via `install_f2d_content.sh`. All 42 `static.prop.*` ids the route XMLs
reference resolve there. Verify on a new box with a blueprint-library dump before assuming.

---

## 3. Why there are two environments

This is the single most important thing to understand, and it cost a lot of time to learn.

- 10 of the 200 routes (`generalization-animals-1075..1084`) reference `walker.animal.*`
  blueprints. **Only `~/f2d_carla` registers those.** The 0.9.16 tree has the animal meshes
  and even the `BP_*` assets, but walker ids come from the *cooked*
  `Content/Carla/Blueprints/Walkers/WalkerFactory.uasset`, which the loose-file content
  install cannot add.
- **Do not copy that WalkerFactory into the 0.9.16 tree.** Tested: the 0.9.15-cooked
  blueprint bytecode is not loadable by a 0.9.16 engine and kills *every* server boot with
  `LowLevelFatalError: Unknown code token 30 ... WalkerFactory_C:GenerateDefinitions` →
  SIGSEGV. Because that crash lands ~14 s into boot, a running orchestrator churns routes at
  ~3/min and burns their retry budgets — a 7-minute exposure destroyed **23 routes**. A
  `.stock-0.9.16.bak` sits beside the file; restore from it if anyone tries again.
- f2d_carla therefore needs a **matching 0.9.15 client**. A 0.9.16 client segfaults the
  worker (`rc=139`) shortly after printing the version-mismatch warning. carla 0.9.15 has no
  cp311 wheel (PyPI stops at cp310), so that client needs **Python 3.10**.
- openpi pins `requires-python = ">=3.11"`, but that is *nearly* conservative: the tree
  byte-compiles under 3.10 and every dependency allows 3.10. There is exactly **one** real
  3.11 API — a `datetime.UTC` in `openpi/shared/download.py`. `build_f2d_eval_env.sh`
  installs openpi from a staged copy with the pin relaxed and that line rewritten. The
  upstream clone is never modified.

---

## 4. How to run

```bash
# 190 routes on vanilla 0.9.16 (everything except the 10 animal routes)
./run_leaderboard_f2d.sh --slots 5:5,6:6 \
  --routes @leaderboard_runs/f2d_routes_main.txt \
  --out-dir leaderboard_runs/f2d_llheavy_matchcrop_6k \
  --resume --xla-mem-fraction 0.30 --stall 600 --route-timeout 2400

# 10 animal routes on Fail2Drive's own build, with the matching 0.9.15 client
./run_leaderboard_f2d.sh --slots 2:2 \
  --routes @leaderboard_runs/f2d_routes_animals.txt \
  --carla-root ~/f2d_carla \
  --python /home/cglossop/ogbench-carla/.venv-f2d-eval/bin/python \
  --out-dir leaderboard_runs/f2d_animals_eval \
  --resume --wandb-mode online --run-group f2d-animals-eval \
  --rpc-base 13000 --tm-base 19000 --display-base 450 \
  --xla-mem-fraction 0.30 --stall 1800 --route-timeout 3600 --setup-timeout 1200
```

Both are resumable and safe to Ctrl-C; already-scored routes are skipped. The launcher
refuses a CARLA server/client version mismatch up front rather than letting you discover it
as a core dump 20 minutes in — `--allow-version-mismatch` overrides, but you almost never
want that.

Monitor: `.venv/bin/python watch_leaderboard.py <out-dir> --log <out-dir>.console.log`
Report: `FAIL2DRIVE_ROUTES_DIR=~/fail2drive/fail2drive_split .venv/bin/python report_f2d_status.py`
→ regenerates `leaderboard_runs/F2D_LEADERBOARD_STATUS.md` with live scored/remaining/blocked.

---

## 5. State at handoff

- **158 / 200 scored.** Main job alive with **36 remaining**; animal job **complete, 7/7**.
- Animal results: DS 82.14 / RC 100.00 / IP 0.821, 100% success (five of seven at DS 100).
- Main job (151 routes at last sample): DS ~60 / RC ~95 / IP ~0.61, ~93% success. Partial.
- W&B: `catherineglossop/OGBench-CARLA`, group `f2d-animals-eval`, one run per route.

### The one thing left to do

The 20-tick deadlock (§6, bug 5) was **fixed and verified after both jobs were launched**.
Because every route runs in a fresh process, the running main job picks the fix up
automatically for each new route — but the 6 two-scenario routes were *excluded* from its
route list back when they were believed unrunnable. They are now runnable. Once the main
job finishes:

```bash
# 3 base-animals routes — they are in f2d_routes_no_animals.txt (190) but not the 187 list
./run_leaderboard_f2d.sh --slots 5:5,6:6 \
  --routes base-animals-0076,base-animals-0079,base-animals-0083 \
  --out-dir leaderboard_runs/f2d_llheavy_matchcrop_6k --resume ...

# 3 generalization-animals routes — need f2d_carla + the 3.10 env
./run_leaderboard_f2d.sh --slots 2:2 \
  --routes generalization-animals-1076,generalization-animals-1079,generalization-animals-1083 \
  --carla-root ~/f2d_carla --python .../.venv-f2d-eval/bin/python \
  --out-dir leaderboard_runs/f2d_animals_eval --resume ...
```

That takes the sweep to **200/200**. Do not start these while the main job is still writing
to the same `--out-dir` — per-route records are safe but the two orchestrators would race on
`leaderboard_summary.json`.

---

## 6. Bugs found and fixed (do not re-investigate)

| # | Commit | Bug | Symptom |
|---|---|---|---|
| 1 | `2154cd7` | render-fence timeout passed only as `-g.TimeoutForBlockOnRenderFence=N`, which **0.9.15 ignores** | f2d_carla died at 60 s loading Town13. Fix: also pass `-ExecCmds=g.TimeoutForBlockOnRenderFence 300000` |
| 2 | `0d3f2df` | 300 s `apply_settings` timeout hardcoded | cold shader cache on Town13 blew through it. Now `CARLA_SETUP_ATTEMPT_TIMEOUT` / `--setup-timeout` |
| 3 | `ff14dd8` | animal lifecycle monitor read `self._spawn_transform`, which `DynamicObjectCrossing` never defines | leaderboard swallowed it as *"Skipping scenario … setup error"*: the animal spawned, the scenario was **dropped**, and the route scored with **no hazard**. Optimistic scores, not a visible failure |
| 4 | `2154cd7` | worktree runs silently used the main checkout's `ogbench/` via `ogbench.pth` | patches had no effect. Fix: launcher exports `PYTHONPATH=$ROOT_DIR` |
| 5 | `ff7af76` | **deadlock building a route's second scenario** | see below |

### Bug 5 in detail — the 20-tick hang

Any route with more than one `<scenario>` froze at exactly 20 ticks / 1.0 s game time, with
its CARLA server alive and nothing logged, burning the full `--route-timeout`. Stack, via a
`faulthandler` SIGUSR1 dump (ptrace is restricted on that box, so py-spy could not attach —
see §7 gotcha 4):

```
basic_scenario.py:67        world.wait_for_tick()
green_traffic_light.py:40   PriorityAtJunction.__init__ -> super().__init__
route_scenario.py:322       build_scenarios
carla_utils.py:1182         _tick_scenario_locked
```

`RouteScenario.__init__` builds its first batch of scenarios and *then* sets
`runtime_init_mode(True)`. With that on, `BasicScenario.__init__` calls
`world.wait_for_tick()` instead of `world.tick()`. Upstream can afford that because it
builds scenarios on a **separate thread** while the main thread ticks; this wrapper
deliberately builds on the main thread to keep CARLA RPCs single-threaded — so it waits for
a tick only its own blocked stack could produce. Single-scenario routes never hit it because
their only batch is built *before* the flag is set.

The fix clears `runtime_init_mode` around the wrapper's own `build_scenarios` call.
Verified: `base-animals-0076` froze at 20 ticks in two independent runs and now drives past
100.

**This affects Bench2Drive too**, wherever a route carries more than one scenario — worth
checking whether any b2d results were silently truncated by it.

---

## 7. Gotchas that will bite you

1. **`ogbench.pth` pins `import ogbench` to the main checkout.** Editing `ogbench/` in a git
   worktree does nothing at runtime unless `PYTHONPATH` puts your checkout first. The
   launcher does this; ad-hoc `python` invocations do not.
2. **Fail2Drive route ids collide with Bench2Drive's.** They are small integers reused across
   files. Use the kebab name or the `f2d:<id>` alias. `leaderboard_runs/route_id_map.txt` maps
   all 420 routes (name / id / alias / file / town / scenario type).
3. **Stale X debris.** A CARLA run that exits leaves `/tmp/.X<N>-lock` and
   `/tmp/.X11-unix/X<N>`; the next Xvfb on that display then dies with "server already
   running". The launcher clears debris when no Xvfb is actually live.
4. **`ptrace_scope=1`** on that box blocks py-spy/gdb from attaching to a non-descendant. To
   debug a hang, inject a `sitecustomize.py` on `PYTHONPATH` that does
   `faulthandler.register(signal.SIGUSR1, all_threads=True)`, then `kill -USR1 <worker>` —
   the traceback lands in the route log. This is how bug 5 was found.
5. **Shared GPU box.** Co-tenants routinely fill GPUs 1–4. A starved renderer shows up as
   `GameThread timed out waiting for RenderThread` → SIGSEGV, which reads like a code bug but
   is contention. Check `nvidia-smi` utilization before blaming the stack.
6. **Do not `pkill -f run_leaderboard.py`** from a shell whose own command line contains that
   string — it matches and kills your own shell. Kill by PID.

---

## 8. Files

```
run_leaderboard_f2d.sh                        the Fail2Drive launcher (this work)
build_f2d_eval_env.sh                         builds .venv-f2d-eval (py3.10 + carla 0.9.15)
report_f2d_status.py                          regenerates the status report from live records
run_leaderboard.py / watch_leaderboard.py     pre-existing orchestrator + dashboard
LEADERBOARD_EVAL.md                           the scoring harness (read this first)

leaderboard_runs/
  f2d_llheavy_matchcrop_6k/                   main run (records/, logs/, summary)
  f2d_animals_eval/                           animal run (f2d_carla)
  f2d_routes_main.txt                         187 routes (the 190 minus 3 base-animals)
  f2d_routes_no_animals.txt                   190 non-animal routes
  f2d_routes_animals.txt                      7 animal routes run so far
  f2d_routes_hang_priorityatjunction.txt      the 6 formerly-blocked routes
  route_id_map.txt                            name <-> id for all 420 routes
  F2D_LEADERBOARD_STATUS.md                   generated status report
```

Note `f2d_routes_main.txt` (187) excludes the 3 `base-animals` two-scenario routes because
they were unrunnable at the time. With bug 5 fixed, `f2d_routes_no_animals.txt` (190) is the
correct list for a fresh full run.
