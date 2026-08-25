# Faithful leaderboard evaluation — `run_leaderboard.py`

Session handoff, 2026-08-20. Everything below is verified against the code unless marked
otherwise. Read "Start here after the reboot" first if you just want to launch a run.

---

## 1. Why this script exists

`run_carla.sh` is built for **training** runs. Three wrapper-level policies in
`CarlaBench2DriveWrapper` end routes earlier than the Bench2Drive leaderboard would, and
each one deflates `score_route`, so training-loop numbers are a systematic underestimate
of leaderboard performance:

| Knob | Training default | Leaderboard behaviour |
|---|---|---|
| `crash_stuck_steps` | 20 ticks (**1 s**) below 0.1 m/s after any collision → route ends | `AgentBlockedTest(min_speed=0.1, max_time=60.0, terminate_on_failure=True)` — **60 s** |
| `max_episode_steps` | 4000 (code default is 250!) | no wrapper cap; only Bench2Drive's `tick_count > 4000` guard |
| `terminate_on_infraction` | `False` (already correct) | `False` — infractions are scored, not fatal |

`run_leaderboard.py` sets these to leaderboard semantics, runs every route, and aggregates.

**Scores are not recomputed.** Each worker's `_finalize_route` calls the real
`leaderboard.utils.statistics_manager.StatisticsManager.compute_route_statistics()`, which
writes a standard leaderboard checkpoint JSON. The script just points each route's
`checkpoint` at its own file and reads them back. `score_composed = score_route *
score_penalty` is computed purely from criteria events — the `crash_message` string only
affects the `status` field, never the number.

### Is the CARLA side faithful? (verified earlier this session)

Yes. The installed `bench2drive @ catglossop/Bench2Drive` leaderboard is near-identical to
simlingo's vendored copy — `agent_wrapper.py`, `envs/sensor_interface.py`,
`utils/statistics_manager.py` are byte-identical; `scenario_manager.py` differs only in
that simlingo **comments out** the `tick_count > 4000` guard that this fork leaves enabled.
`SteppableScenarioManager._tick_scenario_locked` is a verbatim copy of upstream's tick with
the agent's control swapped for `pending_control`, and `_load_route_and_begin_stepping` is
a line-for-line port of `_load_and_run_scenario`'s setup half. The TM seed *is* applied
(`_load_and_wait_for_world` → `set_random_device_seed`). Scenario-init cadence is a
non-issue: `RouteScenario.INIT_THRESHOLD = 500` m, so the 20-tick `build_scenarios` polling
is irrelevant.

What is **not** faithful to simlingo is the *inference* side (camera at `(0.7,0,1.6)`
FOV 90 vs simlingo's `(-1.5,0,2.0)` FOV 110, no JPEG/crop preprocessing, 4 Hz re-query vs
simlingo's every-tick, no 40-tick initial brake). That is a deliberate divergence and does
not affect whether the *leaderboard scoring* is trustworthy.

---

## 2. Start here after the reboot

```bash
cd /home/carla/ogbench-carla

# Sanity: no leftover CARLA, both GPUs near 0 MiB
nvidia-smi --query-gpu=index,memory.used --format=csv

# See the plan without launching anything
CARLA_ROOT=/home/carla/carla-0-9-16 .venv/bin/python run_leaderboard.py \
  --slots 1:1 --routes bench2drive \
  --agent-config impls/configs/steervla_rollout_job2_eaf004_no_ego_history.py \
  --dry-run

# Two routes first — confirms the stack works and warms the XLA cache
CARLA_ROOT=/home/carla/carla-0-9-16 .venv/bin/python run_leaderboard.py \
  --slots 1:1 --routes accident-001,accident-002 \
  --agent-config impls/configs/steervla_rollout_job2_eaf004_no_ego_history.py \
  --xla-mem-fraction 0.60 --out-dir /tmp/lb_smoke

# Then the full 220 (resumable; safe to Ctrl-C)
CARLA_ROOT=/home/carla/carla-0-9-16 nohup .venv/bin/python run_leaderboard.py \
  --slots 1:1 --routes bench2drive \
  --agent-config impls/configs/steervla_rollout_job2_eaf004_no_ego_history.py \
  --xla-mem-fraction 0.60 \
  --out-dir leaderboard_runs/b2d_seed0 --no-ui > /tmp/lb.log 2>&1 &
```

Resume after any interruption — already-scored routes are skipped:

```bash
... --out-dir leaderboard_runs/b2d_seed0 --resume
```

### The config to evaluate

`impls/configs/steervla_rollout_job2_eaf004_no_ego_history.py` is the **base policy from
cast_relabel**: it is `steervla_cast_relabel_config.py` with updates, the CAST observer,
and the trainable restore switched off (via `steervla_rollout_base.py`), pinned to

```
gs://cat-logs/pi05_steervla_cot_simplified_reasoning_no_ego_history
  /pi05_steervla_simplified_reasoning_no_ego_history_v1
  /pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000
```

with `actor_config="pi05_steervla_cot_simplified_reasoning_no_ego_history"` and
`include_ego_history=False`. That checkpoint is already cached at
`~/.cache/openpi/cat-logs/...` (116 G of cache present), so no GCS download is needed.

Its hardcoded `gpu_rank=1` is overridden per slot by the generated wrapper.

---

## 3. What the script does

Orchestrator + one subprocess per route. Each worker is

```
impls/main_carla.py --eval_only=true --max_episodes=1 --route=<name>
    --agent=<generated slot config> --carla_config=<generated slot yaml>
```

run with `cwd=REPO_ROOT` (required — `get_weather_id()` reads `data/weather.xml`
relative to CWD).

### Generated per-slot agent config

`runpy`-loads your `--agent-config`, then forces:

- `training_gpu_rank` / `siglip_device` from the slot
- `enable_updates`, `enable_updates_rl/bc/bc_hl` → `False`
- `debug_task` → `False` (it swaps env reward for `-ego_speed`)
- `debug_noise` → `False` (see §6 — this one bit us)
- `steervla.cot_temperature` → `0.0` (greedy; `--cot-temperature` to change)
- optional `steervla.checkpoint` / `actor_config` overrides

### Generated per-slot CARLA yaml

Base `impls/configs/carla_config.yaml` plus the faithful overrides, the slot's ports, and
a per-route `checkpoint` path. `debug` is pinned to `0` — see §5.

### Slots, ports, GPUs

`--slots` takes `TRAIN_GPU:RENDER_ADAPTER` pairs, one per concurrent worker.
Ports derive from the slot index using **carla_job.sh's scheme**, so they can never
collide with a running `carla_job.sh` job at a different index:

```
rpc     = 12000 + 100*k      streaming = rpc + 1
tm      = 18000 + 100*k      display   = :30 + k
```

**Confirmed empirically this session:** CARLA's `-graphicsadapter=1` renders on
**physical GPU 0**, and `-graphicsadapter=0` on **physical GPU 1**. The adapter ordering is
swapped relative to `nvidia-smi`, exactly as `carla_config.yaml` warns.

So `--slots 1:1` = JAX/SigLIP on physical GPU 1, CARLA renderer on physical GPU 0.

### Outputs

```
<out-dir>/
  records/<route>.json        # the real leaderboard StatisticsManager checkpoint
  records/<route>.live.txt    # per-route debug_checkpoint (empty at debug=0)
  logs/<route>.log            # full worker stdout+stderr
  logs/<route>.attemptN.log   # log from a failed attempt that was retried
  configs/slot<k>_agent.py    # generated, regenerated each run
  configs/slot<k>_carla.yaml
  runs/<route>/               # main_carla's own save_dir
  leaderboard_summary.json    # rewritten after every route completes
```

`leaderboard_summary.json` carries per-route DS/RC/IP/status/infractions/durations plus an
`aggregate` block (mean driving score, route completion, infraction penalty, success rate,
total km, infraction totals and per-km rates) and a `pending` list for `--resume`.

### UI

`rich` live dashboard: header (routes done, queued, elapsed, ETA), running
DS/RC/IP/success, a per-slot worker table (GPU pair, rpc port, route, phase, ticks, m/s,
elapsed), the last 12 results, and infraction totals. Progress within a route comes from
counting the worker's per-tick `[RC-PID] Steer:` lines. `--no-ui` (or a non-TTY) falls back
to a progress line every 30 s.

---

## 4. Flags worth knowing

| Flag | Default | Notes |
|---|---|---|
| `--slots` | `0:1` | `TRAIN_GPU:RENDER_ADAPTER` pairs, one per worker |
| `--routes` | `bench2drive` | `all` / `bench2drive` / `fail2drive` / comma-list / `@file` |
| `--out-dir` | `leaderboard_runs/<timestamp>` | |
| `--resume` | off | skip routes with an existing record |
| `--retries` | `1` | requeue a route that produced **no record**; a scored route is never retried |
| `--route-timeout` | `5400` s | kill a hung worker (**see §6 — leaves a zombie**) |
| `--xla-mem-fraction` | `0.60` | 0.45 OOMs on the OpenPI restore (needs one 4.83 GB alloc) |
| `--jax-cache-dir` | `~/.cache/jax_leaderboard` | persistent XLA cache; `''` disables |
| `--openpi-data-home` | `~/.cache/openpi` | |
| `--crash-stuck-steps` | `10**9` | sentinel; **`0` is rejected**, see below |
| `--online-steps` | `6000` | must exceed the 4000-tick guard |
| `--save-video` | off | 220 MP4s is a lot |
| `--seed` | `0` | traffic-manager seed + agent seed |

**`--crash-stuck-steps 0` does not disable the check.** The test is
`self._crash_stuck_ticks >= self._crash_stuck_steps`, so `0 >= 0` is true on the first
step of every route. The script exits with an error if you pass 0.

---

## 5. Deliberate design decisions

- **`debug` pinned to 0.** `debug > 1` would give per-tick live scores in the UI, but the
  same value is passed to `RouteScenario(debug_mode=...)`, which calls `_draw_waypoints` →
  `world.debug.draw_point` over the whole route. Those primitives render into the scene and
  land in the RGB camera the policy reads — a nicer dashboard would silently change the
  observations being scored. Not worth it.
- **One process per route, not one per slot.** `CarlaBench2DriveWrapper` is built around a
  single route (`_get_single_route_config`) and `main_carla.py` takes one `--route`. The
  cost is a per-process XLA compile, which the persistent cache solves.
- **Interrupted routes are not scored.** On Ctrl-C, an in-flight route goes to `pending`
  rather than being recorded as 0, so it can't drag the mean down.

---

## 6. Known issues (found by smoke-testing, in priority order)

### 6.1 Zombie workers hold GPU memory until reboot — the reason you're rebooting

Killing a `main_carla` worker mid-CUDA-work leaves it as a `Zsl` (zombie, session leader,
multi-threaded) process reparented to init. Its threads are stuck inside the CUDA driver,
so **the process never fully dies and its VRAM is never released**. Observed twice this
session: 16.1 GB stuck on GPU 0, 20.3 GB on GPU 1. `reset_carla.sh` documents this — a
reboot is the only fix.

Practical consequence: **every `--route-timeout` kill over a long run may leak a worker's
VRAM.** Workers that exit normally do *not* leak. If you see free VRAM shrinking across a
220-route run, check `ps -eo pid,stat,cmd | grep defunct`. Mitigation for now is a generous
`--route-timeout` so kills are rare.

### 6.2 Intermittent openpi crash on the first VLA query

`accident-001` died 25 s in with:

```
File "openpi/models/gemma.py", line 123, in RMSNorm.__call__
    normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
TypeError: unsupported operand type(s) for +: 'function' and 'list'
```

via `steervla.py :: _sample_best_of_random_noises → _forward_pi0 → _sample_or_reuse_cot →
_sample_cot_checked`. `accident-002` got past the same point on the same checkpoint, so it
is **intermittent**. The failing path is the `debug_noise` one, so the script now forces
`config.debug_noise = False` — which is correct for a scoring run regardless (it draws 8
extra noise vectors per query purely to log a plot, and
`steervla_dsrl_config.py` ships it `True` with `log_every_n_steps=1`).

**Not confirmed fixed** — the override went in after that run. If it recurs with
`debug_noise=False`, it's a genuine openpi bug on the normal inference path and
`--retries` will paper over it; full traceback is in the git history of this session's
`/tmp/lb_smoke/logs/accident-001.log` (now deleted — reproduce and capture it).

### 6.3 Two concurrent slots probably will not fit on 2×24 GB

Measured per slot: JAX ~14.5 GB (at `--xla-mem-fraction 0.60`) + SigLIP ~5 GB + CarlaUE4
~7 GB ≈ **26 GB**. Two slots need ~52 GB against 48 GB total. `--xla-mem-fraction 0.45`
(~11 GB) is *not* enough — the OpenPI restore needs a single 4.83 GB allocation and OOMs.

Options if you want 2-way parallelism:

1. **Drop SigLIP** — frees ~5 GB, giving ~21 GB/slot which fits two slots comfortably.
   `config.image_encoder = "siglip"` (set in `steervla_dsrl_config.py:78`) is what builds
   it; `main_carla.py` only constructs it when `obs_mode == "image" and image_encoder ==
   "siglip"`. In a frozen rollout with no critic, nothing consumes the encoding — but this
   is a behavioural change to the agent config and has **not been tested**, so it was left
   alone deliberately.
2. Accept 1 slot. Serial over 220 routes.

### 6.4 Startup cost

First route pays a **~17-minute XLA compile** of the Pi0-CoT graph. With one process per
route and no cache that is ~62 hours over 220 routes, so the script sets
`JAX_COMPILATION_CACHE_DIR` (+ `MIN_COMPILE_TIME_SECS=1.0`,
`MIN_ENTRY_SIZE_BYTES=-1`). The cache was observed being populated
(`~/.cache/jax_leaderboard`, 4 entries) but **the cache-hit speedup was never confirmed** —
the second route was still compiling when the run was stopped for the reboot. **Verify this
first after rebooting**: run two routes and check that route 2 reaches its first
`[RC-PID] Steer:` line much faster than route 1.

---

## 7. Bugs already fixed in the script

Each of these killed a real run during smoke-testing:

1. **`PermissionError: '/raid'`** — OpenPI's `get_cache_dir()` mkdirs eagerly at actor
   startup and several configs point at `/raid`. Fixed by setting `OPENPI_DATA_HOME`.
2. **`RESOURCE_EXHAUSTED` on param restore** — JAX preallocates ~75% of the card. Fixed by
   `XLA_PYTHON_CLIENT_MEM_FRACTION` (`--xla-mem-fraction`).
3. **Leaked CarlaUE4 after a kill** — the wrapper `setsid`s CARLA and Xvfb, so `killpg` on
   the worker's group never reaches them and ~7 GB leaks per killed route. Fixed with a
   port/display-scoped `pkill` mirroring `carla_job.sh stop`
   (`carla-rpc-port=<port>`, `Xvfb :<display> -screen`).
4. **Ctrl-C hung** waiting out in-flight routes (up to an hour each). Fixed: workers are
   killed immediately on SIGINT/SIGTERM.
5. **Per-process XLA compile** — fixed with the persistent cache (§6.4).

---

## 8. Manual cleanup crib sheet

```bash
# Scoped — only slot k's CARLA. Safe alongside other jobs.
pkill -9 -f "carla-rpc-port=$((12000 + 100*k))"
pkill -9 -f "Xvfb :$((30 + k)) -screen"

# Nuclear — kills EVERY CARLA on the box
./reset_carla.sh

# Never `pkill -f run_leaderboard.py` from a shell whose own command line contains that
# string — it matches and kills your own shell. Kill by PID instead.
```

---

## 9. Unrelated findings from this session

- **`ogbench/carla/carla.py` is dead code**: a 916-line near-verbatim copy of
  `simlingo/team_code/agent_steervla.py` (9 diff lines, all comments), imported by nothing
  (`ogbench/carla/__init__.py` is empty). `CLAUDE.md` describes it as the home of
  `CarlaBench2DriveWrapper`, which actually lives in `carla_utils.py:1388`. Worth fixing
  the doc.
- **`carla_utils.py :: run_leaderboard()` is broken.** Line 1142 does
  `os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible_devices` immediately after the
  branch that `del`s it → `TypeError` when the var wasn't already set. Nothing calls it, so
  it has never been exercised. One-line fix: delete that line.
- **Latent async-mode hazard**: `_stop_active_scenario` (`carla_utils.py:1791`) calls
  `evaluator._reset_world_settings()`, which sets `synchronous_mode = False`. Upstream calls
  that exactly once at the end of `run()`. Nothing re-asserts synchronous mode per episode,
  and `load_world(town, reset_settings=False)` preserves it. Currently dormant — every
  episode ends via `_finalize_route`, which sets `_scenario_active = False`, so the
  reset-time call no-ops, and `truncated` is hardcoded `False` (`carla_utils.py:3252`). But
  if anything ever resets mid-episode, every later route runs asynchronously and the
  results are silently garbage. Worth a defensive re-assert in
  `_load_route_and_begin_stepping`.
