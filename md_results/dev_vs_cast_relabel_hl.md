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

`opposite-vehicle-running-red-light` is not a route id — the registry has
`-001` … `-005`. Used **`opposite-vehicle-running-red-light-001`**, matching the `-001`
suffix used for `signalized-junction-left-turn-001`.

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

## Results

*(filled in as runs complete)*

| set | route | steps done | episodes | RouteCompletion | driving score | collisions | HL updates | status |
|---|---|---|---|---|---|---|---|---|
| dev | enter-actor-flow-004 | | | | | | | running |
| cast | enter-actor-flow-004 | | | | | | | running |
| dev | signalized-junction-left-turn-001 | | | | | | | running |
| cast | signalized-junction-left-turn-001 | | | | | | | running |
| dev | opposite-vehicle-running-red-light-001 | | | | | | | queued |
| cast | opposite-vehicle-running-red-light-001 | | | | | | | queued |
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
