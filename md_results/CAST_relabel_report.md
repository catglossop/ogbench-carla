# CAST-relabel HL-only runs — dev branch, 6 routes

**Status: running.** Launched 2026-08-03 15:55-15:57 PT. Six runs × 20 000 env steps, one per
GPU, all concurrent. Single arm — the `cast_relabel` branch comparison was dropped at this
point, so everything below is the **dev** branch.

## What is being run

| | |
|---|---|
| agent config | `impls/configs/steervla_cast_relabel_train_config.py` |
| checkpoint | `gs://cat-logs/…/pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000` |
| actor config | `pi05_steervla_cot_simplified_reasoning_no_ego_history` |
| env steps | 20 000 |
| **CoT temperature** | **0.5** (config ships 1.0; overridden per-run) |
| updates | **HL only** — `rl=False bc=False hl=True` |
| CAST relabel | window 150 env steps, `gemini-3.5-flash`, `store_hl_dataset=True`, `store_good_chunks=True` |
| HL update | `hl_update_every=8`, `batch=2`, `num_steps=1`, freeze `[".*img.*", ".*embedder.*"]` |
| HL replay mix | online 0.7 / simlingo 0.2 / simplified_reasoning 0.1 |
| critic feedback | `none` |
| stall recovery | `--pid-stuck-threshold 150` (default 800 never fired; see note) |
| seed | 0 |
| W&B | project `OGBench-CARLA`, group `DevVsCastRelabel`, run names prefixed `dev_` |

`enable_updates_rl=False`, `enable_updates_bc=False`, `enable_updates_bc_hl=True` are set in
the config **and** passed again on the command line, and confirmed per run in the log:

```
[main_carla] updates enabled -> rl=False bc=False hl=True
```

## Runs

| job | route | source | town | GPU | rpc / tm / display |
|---|---|---|---|---|---|
| 46 | `enter-actor-flow-004` | bench2drive | Town05 | 1 | 16600 / 22600 / :76 |
| 41 | `signalized-junction-left-turn-001` | bench2drive | Town12 | 2 | 16100 / 22100 / :71 |
| 42 | `opposite-vehicle-running-red-light-004` | bench2drive | Town12 | 3 | 16200 / 22200 / :72 |
| 43 | `merger-into-slow-traffic-v2-005` | bench2drive | Town06 | 4 | 16300 / 22300 / :73 |
| 44 | `generalization-wall-1095` | fail2drive | Town13 | 5 | 16400 / 22400 / :74 |
| 45 | `generalization-animals-1081` | fail2drive | Town13 | 7 | 16500 / 22500 / :75 |

`enter-actor-flow-004` runs as job **46**, not 40: rpc port 16000 was already held by another
tenant on this box, and `carla_job.sh` correctly refused to launch onto it.

Both fail2drive routes were asset-checked before launch — Town12/Town13 installed,
`generalization-wall-1095` needs `static.prop.brickwall` (present under
`Content/WallAssets/…/Brickwall`) and `generalization-animals-1081` is a `DynamicObjectCrossing`
needing the animal packs (`AnimalVarietyPack`, `AfricanAnimalsPack`, `FarmAnimalsPack`, all
present).

GPU pinning is one card per run — renderer, JAX learner, HL update and SigLIP all on the same
device. Three different index conventions had to be reconciled because GPUs 0 and 6 are
`compute_mode=Prohibited`: `--train-gpu`/`--hl-gpu` index `jax.devices("gpu")` = `[1,2,3,4,5,7]`,
while `--render-adapter` and `siglip_device` are physical `nvidia-smi` indices. `run_carla.sh`
hardcodes `siglip_device = cuda:<jax-rank>`, which would land SigLIP on the prohibited GPU 0, so
every run carries an explicit `--agent.siglip_device=cuda:<physical>` override.

## `Cannot spawn actor` tracking

This is the headline number to watch. Earlier runs on this same config were drowning in it —
**15-17 failures per env step**, 299 554 in a single 20 000-step run — which blocked the spawn
points, kept `BackgroundBehavior` retrying forever, and leaked the CarlaUE4 server to **1.43 TB**
RSS until the kernel OOM-killed it and stranded the run on a 2-hour RPC timeout.

`spawnwatch.sh` snapshots every live run every 5 min into `spawnwatch.log` — steps,
`Cannot spawn actor`, `was not alive after tick`, give-ups and the per-step rate — and flags any
run that climbs above 0.5/step.

**First reading (~200-300 steps in):**

| job | route | steps | `Cannot spawn actor` | `not alive` | give-ups |
|---|---|---|---|---|---|
| 46 | enter-actor-flow-004 | 197 | **0** | 0 | 0 |
| 41 | signalized-junction-left-turn-001 | 290 | **1** | 0 | 0 |
| 42 | opposite-vehicle-running-red-light-004 | 299 | **0** | 0 | 0 |
| 43 | merger-into-slow-traffic-v2-005 | 299 | **1** | 0 | 0 |
| 44 | generalization-wall-1095 | 273 | **0** | 0 | 0 |
| 45 | generalization-animals-1081 | 244 | **0** | 0 | 0 |

Two isolated failures total across ~1 600 env steps, versus ~24 000 expected at the old rate.
An occasional single failure is normal — it means a spawn point was genuinely occupied at that
instant, which is what the message is supposed to mean.

### Why it is fixed

Three revisions of `ogbench/carla/carla_utils.py`'s spawn guard, measured on
`enter-actor-flow-004`:

| revision | `Cannot spawn actor` | `not alive` retries | give-ups | background traffic |
|---|---|---|---|---|
| original | 16.9 / step | — | — | jammed; server leaks to >1 TB |
| rev 1 — `_destroy_leaked_actor`, no retry on occupied, check via `is_alive` | **0** | 15.0 / step | 5.0 / step | 11 vehicles |
| rev 2 — check via `world.get_actor` | **0** | 15.0 / step | 5.0 / step | ego only |
| **rev 3 — skip the check when `tick=False`** | **0** | **0** | **0** | **24 vehicles** |

- The original guard popped a rejected actor from `_carla_actor_pool` but left the vehicle
  physically parked on the spawn point, so that transform was blocked permanently — a
  self-inflicted cascade. `_destroy_leaked_actor` removes it from the world too, and an occupied
  point is no longer retried inside the same tick (nothing has moved, so it cannot help).
- Rev 1 and rev 2 then rejected ~100 % of `tick=False` spawns instead. Neither `actor.is_alive`
  nor `world.get_actor` is valid before a tick, which is why swapping between them changed
  nothing — the two revisions' rejection rates were identical to three significant figures.
- Rev 3 skips the liveness check entirely when `tick=False`. Since
  `BackgroundBehavior._spawn_source_actor` is the only path that replenishes road traffic as the
  ego drives, this is also what restores continuous background traffic. Probing a live world
  post-rev-3 found 24 vehicles — 10 moving (including `scenario`-role actors at ~12 m/s), 4 held
  at red lights, and **0 frozen within 30 m of the ego**.

## Notes on settings

**`--pid-stuck-threshold 150`.** `SimlingoStyleWaypointDecoder` forces a throttle creep after N
consecutive near-stopped PID calls, to break a standstill. The default 800 never fired in
practice (a full 20 000-step run fired it 0 times) because the counter resets on any twitch above
0.1 m/s. Sizing it from the measured distribution of stopped streaks — median 3, p90 15-16,
p99 61-97, pathological tails at 333/565/809 — 150 clears the p99 of legitimate stops on every
route while still catching deadlocks. A lower value (~40) would fire inside 2-5 % of legitimate
stops and creep the ego through red lights, corrupting the traffic-violation metric.

**CoT temperature 0.5.** Applied via `--agent.steervla.cot_temperature=0.5`; verified in each
run's own `flags.json`, not just the command line. Relevant companion change already on dev:
`steervla.py` gained `_reasoning_overflowed` / `_sample_cot_checked`, which detects a reasoning
chain that burned its whole `max_reasoning_len` budget (the signature of a derailed sample when
`cot_temperature > 0`, since the decoder samples the full vocabulary with no top-k/top-p) and
resamples at halved temperature, falling back to greedy. That is the likely source of garbled
subtasks like `"<loc1022>The vehicle remains stopped normally.;<loc1021>"` seen in earlier HL
datasets.

## Guards in place

- `spawnwatch.sh` — 5-min `Cannot spawn actor` snapshots, alerts above 0.5/step.
- `carla_guard.sh` — scoped `carla_job.sh stop` on any job whose CarlaUE4 server passes **250 GB**
  RSS (healthy is 4-6 GB), so a runaway cannot OOM the box and take healthy neighbours with it.
  It has already fired twice and kept the cgroup `oom_kill` counter from advancing.
- `ramwatch.sh` — host RAM, cgroup `oom_kill` counter and per-process RSS every 30 s.

## Results

*(filled in as runs complete)*

| job | route | steps | episodes | RouteCompletion | driving score | collisions | `Cannot spawn actor` | status |
|---|---|---|---|---|---|---|---|---|
| 46 | enter-actor-flow-004 | | | | | | | running |
| 41 | signalized-junction-left-turn-001 | | | | | | | running |
| 42 | opposite-vehicle-running-red-light-004 | | | | | | | running |
| 43 | merger-into-slow-traffic-v2-005 | | | | | | | running |
| 44 | generalization-wall-1095 | | | | | | | running |
| 45 | generalization-animals-1081 | | | | | | | running |

## Inspecting

```bash
cd /home/cglossop/ogbench-carla
./carla_job.sh list          # all jobs
./carla_job.sh logs 46       # follow one
./carla_job.sh stop 46       # scoped stop, leaves siblings alone
```

Per-run artifacts under `exp/OGBench-CARLA/DevVsCastRelabel/<run-name>/`: `cast_relabel/`
(per-window VLM review + annotated debug video), `cast_relabel_hl_dataset/` (the HL samples
`update_hl` consumes, plus `hl_update_batches/` showing the exact decoded tokens per update),
`videos/`.

Prior investigation — the ×7 decode bug, the OOM incidents, the ego-stall/traffic-jam analysis
and the dev vs `cast_relabel` branch comparison — is in
[`dev_vs_cast_relabel_hl.md`](dev_vs_cast_relabel_hl.md).
