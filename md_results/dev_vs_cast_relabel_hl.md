# dev vs. `cast_relabel` — HL-only CAST-relabel runs on 5 routes

**Status: running.** Launched 2026-08-02 01:13 PT. 10 runs × 20 000 env steps, 6 at a time
(one per GPU). Results tables below are filled in as runs finish.

Two matched sets of runs, same config, same routes, same checkpoint:

| set | code | W&B run-name prefix |
|---|---|---|
| A | `dev` branch working tree (`62ecc90` + the uncommitted fixes described below) | `dev_…` |
| B | `cast_relabel` branch (`fefb061`), checked out in a git worktree | `castrelabelbranch_…` |

Purpose: check that the merged `dev` reproduces the `cast_relabel` branch's behaviour on the
CAST-relabel high-level (VLM-backbone) training path.

---

## Configuration

Agent config: **`impls/configs/steervla_cast_relabel_train_config.py`** on both branches —
the two files are **byte-identical** (`git diff cast_relabel dev` on that file and on the
`steervla_cast_relabel_config.py` it inherits from is empty), so the two sets really do run
the same configuration.

| knob | value | source |
|---|---|---|
| checkpoint | `gs://cat-logs/…/pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000` | config (same on both branches) |
| actor config | `pi05_steervla_cot_simplified_reasoning_no_ego_history` | config |
| `load_trainable_params` | `True` (full OpenPI `TrainState`) | train config |
| env steps | 20 000 | `--online-steps` |
| CAST relabel window | 150 env steps, `provider=gemini`, `gemini-3.5-flash`, `debug=True` | config |
| HL dataset | `store_hl_dataset=True`, `store_good_chunks=True` | config |
| HL update cadence | `hl_update_every=8`, `hl_update_batch_size=2`, `hl_update_num_steps=1` | **train** config (overrides the `5`/`64`/`2` in the base config) |
| HL replay mix | online 0.7 + `simlingo_dataset_all_img512_1116` 0.2 + `simplified_reasoning_dataset` 0.1 | config; pools present at `/raid/users/cglossop/steervla_hl_pools` |
| HL freeze | `[".*img.*", ".*embedder.*"]` → 2412.0M/3353.4M trainable (71.9 %) | config |
| critic feedback | `none` (`language_feedback.source=expert`, `expert_mode=none`) | config + `--critic-mode none` |
| online training mode | `rl` | `--train-mode rl` (**must be passed** — `run_carla.sh` defaults to `dagger`) |
| seed | 0 | default |

### Only the HL updates are on

This was the explicit requirement, and it is enforced twice — in the config and again on the
command line, so neither can silently drift:

```
config.enable_updates       = True
config.enable_updates_rl    = False   # DSRL critic/actor
config.enable_updates_bc    = False   # DAgger / BC imitation
config.enable_updates_bc_hl = True    # VLM-backbone update on relabeled subtasks
```
plus `--enable_updates=true --enable_updates_rl=false --enable_updates_bc=false
--enable_updates_bc_hl=true`.

Verified in the logs of **every** run:

```
[main_carla] updates enabled -> rl=False bc=False hl=True
```

Both branches gate identically (`rl_updates_on / bc_updates_on / hl_updates_on`, each ANDed
with the master switch) and both dispatch through
`update_with_vla(batch, run_rl=False, run_hl=True)`, which calls
`SteerVLAActor.update_hl` on the CAST-relabel HL samples. Confirmed live:

```
[cast_relabel] window 1: reviewing 75 frames (150 env steps, offset=0) -> …/ep0001_win0001
[cast_relabel] wrote 15 high-level samples (total 15) -> …/cast_relabel_hl_dataset/ep0001_win0001
[steervla.update_hl] no HL update: waiting for the online HL pool to start: 15/20 samples
```

(`hl_min_online_samples=20`, so the first gradient step lands during the second relabel
window, around env step 300.)

### One GPU per run

Requirement: the renderer, the learner and the HL update all on the same card. Three
different index conventions had to be reconciled — GPUs 0 and 6 are
`compute_mode=Prohibited` on this box:

| setting | convention | value for physical GPU *p* |
|---|---|---|
| `--train-gpu` / `--hl-gpu` | index into `jax.devices("gpu")` = `[1,2,3,4,5,7]` | rank from the map below |
| `--render-adapter` → `gpu_rank` in `carla_config.yaml` | CARLA `-graphicsadapter` = `nvidia-smi` index | *p* |
| `siglip_device` | **torch** index = `nvidia-smi` index | `cuda:p` |

`jax rank → physical: 0→1, 1→2, 2→3, 3→4, 4→5, 5→7` (verified by enumerating
`jax.devices("gpu")`).

`run_carla.sh` hardcodes `config.siglip_device = "cuda:${TRAIN_GPU_RANK}"` — i.e. it assumes
the JAX rank and the torch index coincide, which is false here (it would put SigLIP on the
prohibited GPU 0). Every run therefore carries an explicit
`--agent.siglip_device=cuda:<physical>` override. Confirmed per run:

```
[main_carla] SigLIP encoder … device=cuda:1
[main_carla] JAX default device -> cuda:1 (training_gpu_rank=0)
[run_carla.sh] train_gpu_rank=0 render_adapter=1 hl_gpu_rank=0
```

Measured footprint: **~85 GB of the 143 GB card per run** (JAX learner + HL train state
~78 GB, CARLA UE4 ~7 GB), so exactly one run fits per GPU — hence 6 concurrent, not 10.
Every JAX process also opens a 528 MiB context on each *other* visible GPU; that is JAX
device enumeration, not work.

---

## How the runs were launched

`start_run.sh` (below) wraps `carla_job.sh`, which derives every port and X display from the
integer job index (`rpc = 12000 + 100k`, `stream = rpc+1`, `tm = 18000 + 100k`,
`display = :30+k`) so no two runs can collide. Indices 10-14 are the `dev` set and 15-19 the
`cast_relabel` set; 0-9 were used by earlier work.

```bash
#!/usr/bin/env bash
# start_run.sh BRANCH ROUTE JOB PHYS_GPU
set -euo pipefail
BRANCH="$1"; ROUTE="$2"; JOB="$3"; PHYS="$4"

case "$PHYS" in
  1) RANK=0 ;; 2) RANK=1 ;; 3) RANK=2 ;; 4) RANK=3 ;; 5) RANK=4 ;; 7) RANK=5 ;;
esac
case "$BRANCH" in
  dev)  ROOT=/home/cglossop/ogbench-carla ;;
  cast) ROOT=/home/cglossop/ogbench-carla-castrelabel ;;
esac
cd "$ROOT"

# Both worktrees share the dev venv (pyproject.toml + uv.lock are identical on the two
# branches). PYTHONPATH must lead so the worktree's own `ogbench` beats the editable
# ogbench.pth, which always points at the dev checkout.
export UV_PROJECT_ENVIRONMENT=/home/cglossop/ogbench-carla/.venv
export UV_NO_SYNC=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Neither run_carla.sh sets this; without it each JAX process grabs 75% of the card it
# shares with a CARLA UE4 server.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=online
case "$BRANCH" in
  dev)  export CARLA_RUN_TAG="dev" ;;
  cast) export CARLA_RUN_TAG="castrelabelbranch" ;;
esac

# --max-retries (the crash-resume supervisor) only exists on dev's run_carla.sh.
RETRY_FLAG=(); [[ "$BRANCH" == "dev" ]] && RETRY_FLAG=(--max-retries 10)

./carla_job.sh start --job "$JOB" --train-gpu "$RANK" --render-adapter "$PHYS" --route "$ROUTE" -- \
  --agent-config impls/configs/steervla_cast_relabel_train_config.py \
  --hl-gpu "$RANK" \
  --online-steps 20000 \
  --train-mode rl \
  --critic-mode none \
  --run-group DevVsCastRelabel \
  --save-buffer false \
  "${RETRY_FLAG[@]}" \
  -- \
  --agent.siglip_device="cuda:${PHYS}" \
  --save_dir="${ROOT}/exp" \
  --enable_updates=true \
  --enable_updates_rl=false \
  --enable_updates_bc=false \
  --enable_updates_bc_hl=true
```

A small scheduler daemon keeps all six GPUs busy: it launches the first six, then starts a
queued run on whichever GPU frees up first.

### Allocation

| # | set | route | job | GPU | rpc / tm / display | wave |
|---|---|---|---|---|---|---|
| 1 | dev | `enter-actor-flow-004` | 10 | 1 | 13000 / 19000 / :40 | 1 |
| 2 | cast | `enter-actor-flow-004` | 15 | 2 | 13500 / 19500 / :45 | 1 |
| 3 | dev | `generalization-wall-1095` | 14 | 3 | 13400 / 19400 / :44 | 1 |
| 4 | cast | `generalization-wall-1095` | 19 | 4 | 13900 / 19900 / :49 | 1 |
| 5 | dev | `signalized-junction-left-turn-001` | 11 | 5 | 13100 / 19100 / :41 | 1 |
| 6 | cast | `signalized-junction-left-turn-001` | 16 | 7 | 13600 / 19600 / :46 | 1 |
| 7 | dev | `opposite-vehicle-running-red-light-001` | 12 | — | 13200 / 19200 / :42 | 2 |
| 8 | cast | `opposite-vehicle-running-red-light-001` | 17 | — | 13700 / 19700 / :47 | 2 |
| 9 | dev | `merger-into-slow-traffic-v2-005` | 13 | — | 13300 / 19300 / :43 | 2 |
| 10 | cast | `merger-into-slow-traffic-v2-005` | 18 | — | 13800 / 19800 / :48 | 2 |

Each route's two arms run **concurrently on identical hardware**, so a dev/cast pair is never
compared across different machine load. The `fail2drive` route was put in wave 1 so any
asset/scenario problem would surface early rather than 15 h in.

Observed rate is ~0.36 env steps/s, so a 20 k-step run is **~15 h**; two waves ≈ **31 h**
wall clock.

---

## Route note

`opposite-vehicle-running-red-light` is not a route id — the registry has `-001` … `-005`.
Initially launched as **`-001`**; **changed to `-004` on request** (2026-08-03). Both are
Town12 / `OppositeVehicleRunningRedLight`, so the swap is a different route instance of the
same scenario type.

The `-001` pair (dev job 12, cast job 17) had **already completed 20 000/20 000** before the
swap, so it is kept as extra data rather than discarded; `-004` runs as jobs **20** (dev) and
**25** (cast), queued behind the six runs currently holding the GPUs. The results table below
lists `-004` as the requested route and `-001` as a completed extra.

`generalization-wall-1095` is a **fail2drive** route (Town13, `RoadBlocked`), not
bench2drive. Checked before launching: Town12/Town13 are installed, the `fail2drive` package
is importable in the shared venv alongside the bench2drive `srunner`, and the route's prop
(`static.prop.brickwall`) resolves to an installed asset
(`CarlaUE4/Content/WallAssets/Static/Wall/Brickwall`). The other four routes are
bench2drive.

---

## Differences between the two branches that affect this comparison

These are **not** controlled for — they are what the comparison measures. Listing them so the
numbers are not over-read.

### 1. The rollout policy differs when RL is off (biggest one)

With `enable_updates_rl=False`, the two branches pick the flow latent differently:

| branch | rollout latent |
|---|---|
| `cast_relabel` | `agent.sample_actions_with_vla(obs, seed=…)` — the learned noise actor, unconditionally |
| `dev` | `tanh(N(0,1)) * vla_noise_scale`, a bounded fixed latent |

On `dev` the learned-actor path is now gated behind `rl_updates_on`. Since RL updates are off
in this experiment, the noise actor never trains, so on `cast_relabel` the executed policy is
driven by an actor **frozen at random init**, while `dev` uses a bounded random latent. Both
are "untrained noise", but they are different distributions, so **route-completion differences
between the two sets are not purely attributable to the relabel pipeline.**

This gate is a change I made on 2026-08-01 (see `dev_testrun.md`, follow-up fix 6): the merge
had replaced the learned-actor call with the fixed latent *unconditionally*, leaving
`sample_actions_with_vla` called nowhere and making `enable_updates_rl` a no-op on behaviour.
The gate restores the learned actor for real RL runs while keeping the bounded latent for
rollout-only/eval — which is what this HL-only experiment is.

### 2. The ×7 action denormalization is applied at a different layer

Both branches end up correct, but not in the same place:

| branch | who applies `denormalize_actions` (×7 on the speed-waypoint deltas) |
|---|---|
| `cast_relabel` | the **actor** — `SteerVLAActor._postprocess_action_trajectory` calls `steervla_physical_denormalize_actions`, so the env is set to `action_input_space="policy_output"` |
| `dev` | the **env** — the actor returns raw model output, so `action_input_space="normalized"` |

The merge combined `cast_relabel`'s env-side `policy_output` with the other branch's
non-denormalizing actor, dropping the ×7 entirely; that is the crawl bug fixed on
2026-08-01. Both arms should now decode to the same physical units.

### 3. `dev` has a crash-resume supervisor; `cast_relabel` does not

`dev`'s `run_carla.sh` has the `--max-retries` supervisor loop that relaunches with
`--resume=true` after a CARLA segfault. `cast_relabel`'s `run_carla.sh` has neither the loop
nor a `--resume` flag, so a segfault ends that run where it stands. Any `cast_relabel` run
that stops short of 20 000 steps is reported as such below rather than silently restarted.

### 4. Cosmetic patch applied to both trees

The DSRL entrypoint named every run `{agent}_{route}_seed_N_{ts}`, which made ten concurrent
runs indistinguishable in the W&B sidebar. Both trees got the **same** patch so the run name
also carries the experiment arm (`CARLA_RUN_TAG`), the coach form and which updates are on:

```
dev_dsrl_cast-gemini-hl+good_critic-none_upd-hl_enter-actor-flow-004_seed_0_20260802_011306
castrelabelbranch_dsrl_cast-gemini-hl+good_critic-none_upd-hl_generalization-wall-1095_seed_0_20260802_011554
```

It only changes the string handed to `setup_wandb` / `save_dir`; no behaviour is affected.

### 5. `dev`'s working tree is not clean

Set A runs the `dev` **working tree**, not committed `62ecc90`. Uncommitted at launch:
the `action_input_space` fix, the noise-actor gate, `debug_task=False`, the restored video
violation banners, and the naming patch above. Nothing is committed yet.

---

## Incident — 4 runs stalled by the kernel OOM killer (2026-08-02 ~22:00-23:30)

Four of the six live runs stopped making progress. **Cause: the host ran out of RAM and the
kernel SIGKILLed four `CarlaUE4` servers.** Evidence, in order of decisiveness:

```
$ cat /sys/fs/cgroup/user.slice/user-1010.slice/memory.events
oom_kill 4                 # exactly four kills — matches the four dead servers
$ cat …/memory.peak
2127058423808              # 1.93 TiB of the box's 2.0 TiB
$ cat /tmp/carla_rpc13000.log
Killed                     # and identically for 13100, 13500, 13600
```

`memory.max` is unset, so this was genuine system-wide exhaustion, not a cgroup cap.

With its server gone, each affected python process blocks on CARLA's RPC timeout — which is
**7 200 000 ms (2 h) per call**:

```
ERROR: failed to destroy actor 626 : time-out of 7200000ms while waiting for the simulator,
make sure the simulator is ready and connected to localhost:13000
```

so the runs neither progress nor exit. Note `dev`'s `--max-retries` supervisor does **not**
rescue this: it only relaunches on an exit code ≥ 128 (a crash signal), and a process hung in
a 2-hour RPC never exits at all.

| job | set | route | steps reached | outcome |
|---|---|---|---|---|
| 10 | dev | enter-actor-flow-004 | 4 752 | CARLA killed 22:01 → **restarted from 0** |
| 11 | dev | signalized-junction-left-turn-001 | 3 474 | CARLA killed ~23:27 → **restarted from 0** |
| 15 | cast | enter-actor-flow-004 | 13 179 | CARLA killed 22:55 → **restarted from 0** |
| 16 | cast | signalized-junction-left-turn-001 | 8 205 | CARLA killed 22:12 → **restarted from 0** |
| 12 | dev | opposite-vehicle-running-red-light-001 | 15 159 | unaffected, still running |
| 17 | cast | opposite-vehicle-running-red-light-001 | 16 110 | unaffected, still running |
| 14 | dev | generalization-wall-1095 | 20 000 | **completed** before the incident |
| 19 | cast | generalization-wall-1095 | 20 000 | **completed** before the incident |

**Restarted from 0, not resumed.** The DSRL entrypoint ignores `--exp_name` and builds a
timestamped name, so a relaunch always lands in a fresh `save_dir` and `--resume` finds no
prior state; the `cast_relabel` branch has no `--resume` flag at all. Worth fixing if these
runs are going to be re-run often.

**What actually consumed 1.9 TiB is not yet pinned down.** The steady per-process leak is only
~2.5 GB/h (the two oldest processes sit at 28.7 GB after 10.8 h and 22.1 GB after 9.6 h), and
six of those plus six ~8 GB CARLA servers accounts for well under 300 GB. So the peak was
transient. Suspicion falls on episode-end video handling — episodes here reach 2001 captured
frames and both a W&B video encode and a local mp4 + per-frame JPEG dump happen at once — but
that is a hypothesis, not a measurement. A watchdog now samples total RAM, the cgroup
`oom_kill` counter and per-process RSS every 30 s, and dumps the full process table above
1.4 TB, so a recurrence will be caught in the act.

Secondary observation from the same incident: the **JAX arena grows too**. Runs start at
~78 GB of VRAM and three of the six had grown to 113 GB after ~22 h (GPU 4 reached 124 GB of
143 GB including CARLA). `XLA_PYTHON_CLIENT_PREALLOCATE=false` grows on demand and never
returns memory, so a long run plus a co-resident CARLA server is not comfortably within one
card forever.

Recovery: stopped the four hung jobs with the scoped `carla_job.sh stop`, relaunched them on
the same GPUs, restarted the queue scheduler with corrected state, and added the RAM
watchdog. All six GPUs are busy again.

---

## Root cause — the ego stalls, traffic jams behind it, and CARLA leaks to 1.4 TB

The reported "other vehicles aren't moving" and the OOM stalls above turn out to be **the same
failure**, and the entry point is the ego.

**1. The ego stops and never restarts.** Mean speed per decile of the run, from the
`[RC-PID] Current speed` trace (dev arm; the `cast_relabel` branch has no such print):

| decile | dev 10 `enter-actor-flow-004` | dev 12 `opposite-vehicle-…-001` | dev 14 `generalization-wall-1095` |
|---|---|---|---|
| 10 % | 2.95 m/s (57 % stopped) | 4.84 (24 %) | 4.87 (30 %) |
| 30 % | 2.03 (65 %) | 1.89 (59 %) | 1.64 (68 %) |
| 50 % | **0.03 (100 %)** | 3.87 (30 %) | 1.75 (65 %) |
| 80 % | **0.00 (100 %)** | 2.01 (71 %) | 3.72 (49 %) |
| 100 % | **0.00 (100 %)** | 1.32 (79 %) | 2.59 (57 %) |

On `enter-actor-flow-004` the ego is in a permanent standstill from ~40 % of the run onward.
Every route also shows a **monotone decay** from ~4.9 m/s at the start.

**2. Traffic is not frozen — it is queued.** Probing the live `enter-actor-flow-004` world
(cast job 15, rpc 13500) over 40 s wallclock, ego at **0.00 m/s**:

```
moving=22   frozen_at_red_light=6   frozen_within_30m_of_ego=5   frozen_far_from_ego=10
```

The 22 moving vehicles are all 38-75 m away and doing up to 8.1 m/s, so the traffic manager is
working (`set_traffic_manager_port` *is* reached — our `IsolatedLeaderboardEvaluator` only
overrides `_setup_simulation`, and the inherited `_load_and_wait_for_world` still sets it from
`args.traffic_manager_port`). What the ego camera sees is the stopped queue the ego itself
created, plus vehicles legitimately held at red lights.

Also visible in the same probe: the sim advanced **22 ticks (1.10 s of sim) in 40 s of
wallclock**, a ~36× slowdown, because the world only ticks once per env step and an env step
costs ~1-2.7 s of VLA inference.

**3. The jam causes a spawn storm.** `enter-actor-flow` continuously spawns a *flow* of
vehicles at fixed source points. With the junction blocked, the flow never drains, the source
points stay occupied and every subsequent spawn fails against the vehicle already parked there
— the same handful of `Location(...)` values repeating ~14 000 times each:

| run | `Cannot spawn actor` | spawn-guard give-ups |
|---|---|---|
| dev 10 `enter-actor-flow-004` | 71 272 | 23 813 |
| cast 15 `enter-actor-flow-004` | 248 868 | 83 115 |
| dev 12 `opposite-vehicle-…-001` | 193 070 | 64 729 |
| cast 19 `generalization-wall-1095` | 13 850 | 4 714 |

Every run on every route on both branches does this, so it is systemic, not a merge
regression.

**4. That retry loop leaks memory inside the CARLA server.** The watchdog caught it: RSS of a
single `CarlaUE4-Linux-Shipping` climbing ~150 GB/h until the kernel kills it, then the next
server starting the same climb.

```
01:17 used= 325GB  biggest proc  140GB
02:47 used=1246GB                843GB   (CarlaUE4 rpc-port=13600)
03:47 used=1828GB               1061GB
04:18 used=1265GB oom_kill 4->5          <- kernel kills it, next server starts climbing
05:48 used=1001GB oom_kill 5->6
09:20 used=1516GB               1426GB   (CarlaUE4 rpc-port=13100)
```

So the OOM incident is not a mysterious transient — it is the simulator accumulating state
from hundreds of thousands of failed spawns. **Python is not the leaker**; the client
processes sit at a steady 16-22 GB.

### Two things making it worse

**The anti-stall creep almost never fires.** `SimlingoStyleWaypointDecoder` has exactly the
right mechanism — after `stuck_threshold` consecutive near-stopped PID calls it forces
`creep_throttle` for `creep_duration` calls to break static friction — but the default
`stuck_threshold` is **800** calls. At 1-2.7 s per env step that is 15-35 minutes of wallclock
before the first creep attempt. `run_carla.sh` already exposes `--pid-stuck-threshold`,
`--pid-creep-duration` and `--pid-creep-throttle`; none were set.

**The HL updates are being trained to stand still.** Of the 590 HL samples stored by dev 10,
the subtask distribution is overwhelmingly stationary:

```
 91  "The vehicle remains stopped normally due to a red traffic light."
 90  "The vehicle remains stopped …"
 62  "The vehicle remains stopped normally …"
 60  "The vehicle remains stopped normally."
 50  "The vehicle remains stopped steadily …"
 45  "The vehicle accelerates steadily to follow the route …"
```

With `store_good_chunks=True`, chunks that are not labelled BAD are stored with the model's
*own* subtask — so once the ego is stuck, the HL dataset fills with "remains stopped" and the
VLM backbone is fine-tuned to reinforce it. That is a plausible mechanism for the monotone
speed decay in the table above, and it is self-reinforcing.

**Data bug spotted alongside:** some stored subtasks carry raw location tokens through into the
text, e.g. `"<loc1022>The vehicle remains stopped normally.;<loc1021>"` (about 1 in 12 of the
samples inspected). Those are being used as CoT supervision targets as-is.

### Action taken

Stopped dev job 11, whose CARLA server had reached **1426 GB** and would have OOM'd the box
within the hour, plus dev 10 and cast 16, both already dead-server zombies. Host RAM went
1.5 TB → 346 GB and the three remaining servers are back to a normal 4-6 GB. The three runs
still progressing (dev 13, cast 15, cast 18) were left alone and keep their original config.

**Creep threshold set to 150, not 800 — and not 40 either.** Sizing it from the measured
distribution of consecutive-stopped (< 0.1 m/s) streaks rather than by guess:

| run | n streaks | median | p90 | p99 | max |
|---|---|---|---|---|---|
| dev 12 `opposite-vehicle-…-001` | 786 | 3 | 15 | 97 | 333 |
| dev 14 `generalization-wall-1095` | 1156 | 3 | 16 | 61 | 809 |
| dev 10 `enter-actor-flow-004` | 111 | 6 | 99 | 353 | 565 |

Ordinary stops (red lights, yielding) are almost all under ~100 steps; the pathological ones
are the 333/565/809 tails. **800 was too high to ever fire** — dev 10's longest streak was 565
and its creep fired *0* times in 4758 steps, as did dev 12's in a full 20 000 (dev 14's fired
23 times). But the ~40 I originally floated sits between p90 and p99, so it would fire inside
2-5 % of *legitimate* stops and creep the ego through red lights — corrupting exactly the
traffic-violation metric being measured. 150 clears the p99 of normal stops on every route
while still catching the deadlock tails. Note 1 env step = 1 tick = 0.05 s sim, so 150 steps
is 7.5 s of stationary sim time.

**The fix is dev-only.** The `cast_relabel` branch has **no creep mechanism at all** — its
`SimlingoStyleWaypointDecoder.__init__` takes no `stuck_threshold` / `creep_*` kwargs, and the
force-move block in `ogbench/carla/carla.py:723-726` is commented out. There is no flag to
set. So the cast arm cannot unstick itself, and cast 16 was relaunched unchanged; expect it to
deadlock the same way. This is itself a real difference between the branches.

**Guard added.** `carla_guard.sh` polls every 60 s and issues a scoped `carla_job.sh stop` on
any job whose CarlaUE4 server passes **250 GB** RSS (~40× a healthy 4-6 GB server). The point
is to confine the damage to the one broken run instead of letting the kernel OOM-kill
whichever process is biggest and take healthy neighbours with it.

---

## Spawn-guard fix + `cot_temperature=0.5` re-run (2026-08-03)

The uncommitted `ogbench/carla/carla_utils.py` change found the actual origin of the spawn
cascade — it was **the guard itself**, not only the stalled ego:

- `_carla_actor_alive` confirmed actors via `world.get_actor`, which is meaningless right after
  a `tick=False` spawn because the client's world snapshot predates the actor. Successful
  spawns were therefore judged stale.
- The old guard then popped the "stale" actor from `_carla_actor_pool` but **left the vehicle
  physically parked on the spawn point**, so every later attempt at that transform failed —
  self-inflicted, and permanent. `_destroy_leaked_actor` now removes it from the world too.
- `try_spawn_actor` returning `None` (occupied point) is no longer retried; retrying inside the
  same tick cannot help because nothing has moved, and `BackgroundBehavior` calls this every
  tick per traffic source.

**Measured effect on the same route, same config:**

| arm | job | steps | `Cannot spawn actor` | per env step |
|---|---|---|---|---|
| dev before | 10 | 3 520 | 59 326 | **16.9** |
| dev after | 21 | 410 | **0** | **0.00** |
| cast before | 15 | 20 000 | 299 554 | **15.0** |
| cast after | 26 | 430 | **0** | **0.00** |

The fix was applied to the `cast_relabel` worktree too (`git apply` of the same diff — the
pre-fix code there was byte-identical), so both arms get it.

`cot_temperature` dropped **1.0 → 0.5** on both arms via `--agent.steervla.cot_temperature=0.5`
(no config edit; verified in the run's own `flags.json`, not just the command line). The dev
arm also keeps `--pid-stuck-threshold 150`; the cast arm still cannot, having no creep.

Re-run as jobs **21** (dev) and **26** (cast) rather than reusing 10/15, so the pre-fix logs
survive for comparison. `cast 15` had already reached 20 000/20 000 before the swap and is kept.

### Still open: `was not alive after tick`

`Cannot spawn actor` is gone, but a second failure mode survives — and it thins out the traffic:

```
[carla spawn guard] spawned actor 6355 was not alive after tick (attempt 1/3); retrying
[carla spawn guard] spawned actor 6356 was not alive after tick (attempt 2/3); retrying
[carla spawn guard] spawned actor 6357 was not alive after tick (attempt 3/3); retrying
[carla spawn guard] could not spawn a live actor after 3 attempts; returning None
```

6 210 retries / 2 070 give-ups in 410 steps, with **zero** `Cannot spawn actor` — so
`try_spawn_actor` is succeeding every time (consecutive actor ids) and the liveness check is
rejecting the result. Probing the live world confirms the consequence: **11 vehicles in the
post-fix run vs 33-35 pre-fix**.

Likely cause: the fix routes `tick=False` spawns to the client-side `actor.is_alive` flag
instead of `world.get_actor` — but *both* are derived from the world snapshot, and with
`tick=False` no snapshot containing the actor exists yet. `background_activity.py:2130,2175`
passes `tick=False` as a keyword, and `_TICK_ARG_INDEX = 8` is correct, so `ticked` is being
read properly; the check itself is the problem. The consistent treatment would be to skip the
liveness check entirely when `tick=False` (trust a non-`None` return and re-validate after the
next tick) rather than substituting one snapshot-dependent test for another.

Not changed here — `carla_utils.py` is uncommitted work in progress and stomping it risked a
conflict.

### Revision 2 (14:18) — `world.get_actor` restored as authoritative

The next iteration of `carla_utils.py` went the other way: it drops the `rpc_check` /
`_TICK_ARG_INDEX` machinery and makes `world.get_actor` authoritative again, on the grounds
that `actor.is_alive` "reads False on a perfectly live actor" for a `tick=False` spawn.
`_destroy_leaked_actor` and the no-retry-on-`None` behaviour — the parts that actually killed
the cascade — are kept. Ported to the cast worktree (reset + re-apply, both trees now
byte-identical in the guard region) and relaunched as jobs **22** (dev) / **27** (cast), still
at `cot_temperature=0.5`.

**It made no measurable difference to the rejection rate:**

| version | job | steps | `Cannot spawn actor` | `not alive` retries | give-ups | retries/step | give-ups/step |
|---|---|---|---|---|---|---|---|
| rev 1 (`is_alive`) | dev 21 | 1 240 | 0 | 18 645 | 6 215 | 15.04 | 5.01 |
| rev 2 (`get_actor`) | dev 22 | 400 | 0 | 6 015 | 2 005 | **15.04** | **5.01** |
| rev 1 (`is_alive`) | cast 26 | 1 415 | 0 | 18 084 | 6 028 | 12.78 | 4.26 |
| rev 2 (`get_actor`) | cast 27 | 410 | 0 | 6 177 | 2 059 | 15.07 | 5.02 |

Identical to three significant figures on the dev arm. So **both** liveness tests reject a
`tick=False` spawn — swapping between them cannot help, because neither the client snapshot
nor the `get_actor` lookup reflects an actor created without a tick. Probing job 22's world
found **only the ego present**, no background vehicles.

The good news is unchanged and is the part that matters for stability: `Cannot spawn actor`
stays at **0** in both revisions, so the blocked-spawn-point cascade — and with it the >1 TB
CarlaUE4 leak — is genuinely fixed by `_destroy_leaked_actor` plus not retrying an occupied
point. What remains is that the guard rejects every background vehicle, so traffic never
populates.

Given both tests fail, the options are to skip the liveness check entirely when `tick=False`
(trust a non-`None` return, re-validate after the next tick) or to let `request_new_actor`
tick. There is no third choice that keeps a check at spawn time.

### Which uncommitted change is in which run

Two files were edited on 2026-08-03 and they have **different** reach, because only one of them
ports across branches:

| file | mtime | dev arm | cast arm |
|---|---|---|---|
| `ogbench/carla/carla_utils.py` (spawn guard, rev 2) | 14:18:56 | ✅ | ✅ ported via `git apply` |
| `impls/vlas/steervla.py` (CoT overflow resample) | 13:48:28 | ✅ | ❌ **patch does not apply** |

The guard region is byte-identical across the two trees (`md5` of
`_carla_actor_alive` → `_spawn_guard_installed = True` is `94112e510ef7` in both), and each job
imports its own tree (`PYTHONPATH` verified per process).

The `steervla.py` change adds `_reasoning_overflowed` / `_sample_cot_checked`: when the
reasoning segment burns its whole `max_reasoning_len` budget the chain has derailed (with
`cot_temperature > 0` the decoder samples the full vocabulary with no top-k/top-p, so one junk
token knocks it off-distribution and it never emits `END_OF_REASONING`), and it is resampled at
halved temperature, finally falling back to greedy. That is almost certainly the source of the
`"<loc1022>The vehicle remains stopped normally.;<loc1021>"` garbage found in the HL dataset.

It cannot be `git apply`-ed to the cast worktree — the two branches' `steervla.py` differ by
~915 lines. **So with `cot_temperature=0.5` on both arms, the dev arm repairs garbled CoT and
the cast arm does not.** A hand-port is feasible: the cast branch has `_sample_or_reuse_cot`,
`tokenized_reasoning_mask`, `max_reasoning_len`, the identical `self._sample_cot(...)` call
site (line 2936) and the `create_steervla_pi0_cot_sample_fn` config plumbing. Not done —
hand-editing a 915-line-divergent file is a bigger step than a clean patch application.

### Revision 3 (15:31) — skip the check when `tick=False`. This one works.

The next `carla_utils.py` revision does what the measurements pointed to: the liveness check
runs **only when the caller asked `request_new_actor` to tick**. Its docstring confirms the
diagnosis independently — checking a `tick=False` spawn "rejected ~100% of them", and since
`BackgroundBehavior._spawn_source_actor` is the only way road traffic is replenished as the ego
drives, that "silently disabled continuous background traffic: only the initial batch
population (`request_new_batch_actors`, which this guard does not wrap) ever survived, and it
thinned out to nothing as the ego moved away from it."

**Guard behaviour on `enter-actor-flow-004` across all three revisions:**

| revision | job | steps | `Cannot spawn actor` | `not alive` retries | give-ups |
|---|---|---|---|---|---|
| rev 1 (`is_alive`) | dev 21 | 1 240 | 0 | 18 645 | 6 215 |
| rev 2 (`get_actor`) | dev 23 | 1 107 | 0 | 16 650 | 5 550 |
| **rev 3 (skip when `tick=False`)** | dev 24 | 300 | **0** | **0** | **0** |
| **rev 3** | cast 29 | 305 | **0** | **0** | **0** |

And the traffic is back. Probing job 24's live world: **24 vehicles** — 10 moving (including
`scenario`-role actors at ~12 m/s), 4 held at red lights, 10 stationary further out, and
**0 frozen within 30 m of the ego**, i.e. no queue jammed against a stalled ego. Compare rev 2,
where the same probe found only the ego.

### Stale-code audit (what prompted the rev-3 relaunch)

Rev 3 landed at 15:31:31, *after* every then-running job had started, so all of them were
running superseded code. Audit and action:

| job | route | started | state | action |
|---|---|---|---|---|
| dev 20 | opposite-vehicle-…-004 | 11:18 | 6 921 steps, pre-fix (spawn storm ~5.9/step) | relaunched → **dev 30** |
| cast 25 | opposite-vehicle-…-004 | 11:22 | 7 820 steps, pre-fix (~6.1/step) | relaunched → **cast 35** |
| cast 28 | enter-actor-flow-004 | 15:00 | 1 280 steps, rev 2 | relaunched → **cast 29** |
| dev 23 | enter-actor-flow-004 | 15:00 | stopped externally ~15:36 (pidfile removed, no guard entry) | relaunched → **dev 24** |
| dev 13 | merger-…-005 | — | **20 000/20 000 complete** | left |
| cast 18 | merger-…-005 | — | **20 000/20 000 complete** | left |

The two `opposite-vehicle-…-004` runs gave up ~7-8k steps, but they carried the *original*
pre-fix guard and were accumulating the cascade at ~6 failures/step, so they were heading for
the 250 GB guard regardless.

All four relaunches verified: started after 15:31:31, `rl=False bc=False hl=True`,
`cot_temperature=0.5`, dev arm also `--pid_stuck_threshold=150`. Guard region `md5` is
`3b327f88dc33` in both trees.

**Still dev-only:** the `steervla.py` CoT-overflow resample (13:48). It does not apply to the
cast branch and has not been hand-ported, so at `cot_temperature=0.5` the dev arm repairs
garbled CoT and the cast arm does not.

### Monitoring

`spawnwatch.sh` snapshots every live run every 5 min into `spawnwatch.log` — steps,
`Cannot spawn actor` count, give-ups and the per-step rate — and flags any run that climbs back
above 0.5/step. First snapshot after the fix:

```
dev   job 13  steps=18652  cannot_spawn=115708  per_step=6.20  <<< ALERT   (pre-fix run)
dev   job 20  steps=4650   cannot_spawn=27472   per_step=5.91  <<< ALERT   (pre-fix run)
dev   job 21  steps=410    cannot_spawn=0       per_step=0.00             (post-fix)
cast  job 18  steps=18078  cannot_spawn=112572  per_step=6.23  <<< ALERT   (pre-fix run)
cast  job 25  steps=5160   cannot_spawn=31259   per_step=6.06  <<< ALERT   (pre-fix run)
cast  job 26  steps=430    cannot_spawn=0       per_step=0.00             (post-fix)
```

The four alerting runs started before the fix landed and carry the old code.

### The 250 GB guard earned its keep

`carla_guard.sh` stopped dev 11 and cast 16 when their servers hit 250 GB, and the cgroup
`oom_kill` counter stayed at **6** — no new kernel kills, and no collateral damage to the
healthy runs. Both had died of the spawn cascade (final log lines are `Cannot spawn actor` at a
single repeating transform), so both are candidates for a re-run on the fixed code.

---

## Results

*(filled in as runs complete)*

| set | route | steps done | episodes | RouteCompletion | driving score | collisions | HL updates | status |
|---|---|---|---|---|---|---|---|---|
| dev | enter-actor-flow-004 | | | | | | | running |
| cast | enter-actor-flow-004 | | | | | | | running |
| dev | signalized-junction-left-turn-001 | | | | | | | running |
| cast | signalized-junction-left-turn-001 | | | | | | | running |
| dev | opposite-vehicle-running-red-light-**004** | | | | | | | queued (job 20) |
| cast | opposite-vehicle-running-red-light-**004** | | | | | | | queued (job 25) |
| dev | opposite-vehicle-running-red-light-001 *(superseded, kept)* | 20000 | 34 | | | | | **complete** |
| cast | opposite-vehicle-running-red-light-001 *(superseded, kept)* | 20000 | 8 | | | | | **complete** |
| dev | merger-into-slow-traffic-v2-005 | | | | | | | queued |
| cast | merger-into-slow-traffic-v2-005 | | | | | | | queued |
| dev | generalization-wall-1095 | 20000 | 19 | | | | | **complete** |
| cast | generalization-wall-1095 | 20000 | 14 | | | | | **complete** |

---

## Reproducing / inspecting

```bash
# live status of all jobs
/home/cglossop/ogbench-carla/carla_job.sh list                      # dev set (jobs 10-14)
/home/cglossop/ogbench-carla-castrelabel/carla_job.sh list          # cast set (jobs 15-19)

# follow one run
/home/cglossop/ogbench-carla/carla_job.sh logs 10

# stop one run without disturbing the others (scoped kill by rpc port + display)
/home/cglossop/ogbench-carla/carla_job.sh stop 10
```

W&B project `OGBench-CARLA`, group `DevVsCastRelabel`.

Artifacts per run under `<root>/exp/OGBench-CARLA/DevVsCastRelabel/<run-name>/`:
`cast_relabel/` (per-window VLM review + annotated debug video), `cast_relabel_hl_dataset/`
(the high-level training samples actually consumed by `update_hl`), `videos/`.

The `cast_relabel` worktree is at `/home/cglossop/ogbench-carla-castrelabel`
(`git worktree add … cast_relabel`); remove with `git worktree remove` when done.
