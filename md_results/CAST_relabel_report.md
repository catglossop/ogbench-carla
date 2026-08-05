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

## Second config: Best-of-N + CAST relabel (2026-08-04)

`impls/configs/steervla_bon_cast_relabel_config.py` (commit `b088788`) merges critic-guided
Best-of-N selection with the CAST-relabel HL training path. Run on two routes, replacing the
cast-only runs there:

| job | route | GPU | replaces |
|---|---|---|---|
| 47 | `merger-into-slow-traffic-v2-005` | 4 | job 43 (stopped at 12 670/20 000) |
| 48 | `generalization-wall-1095` | 5 | job 48 (stopped at 12 810/20 000) |

Differences from the cast-only runs above:

- `agent_name=best_of_n`, `best_of_n=10` candidates per env step, `vla_cot_temperature=0.5`.
  The config pins **both** `cot_temperature` and `vla_cot_temperature` to 0.5 itself, so no CLI
  override is passed — moving only one of the pair would desync them.
- `critic_pretrained_weights=/raid/users/cglossop/critic_ckpts/step_0012000.pkl`, used purely to
  rank candidates. Shape-checked before launch: `value_net/Dense_0/kernel` is `(2, 3496, 256)`
  — ensemble dim 2 and input width **3496 = 3×1152 + 40**, exactly what the config documents.
- `hl_update_batch_size=16` (not 2), `hl_update_every=5`, `hl_update_num_steps=2`.
- **`enable_updates_rl=true`** — see below.

### Why RL updates must be on for Best-of-N

The config ships `enable_updates_rl=False` (critic frozen, only HL gradients flow), but with
that setting **the Best-of-N path never fires**. `main_carla.py:2882` gates the agent's own
action-selection path on `rl_updates_on`:

```python
if rl_updates_on and hasattr(agent, "sample_actions_with_vla"):
    return agent.sample_actions_with_vla(obs[None], seed=subkey), None
# else: fixed tanh-squashed latent, single sample
```

and `BestOfNAgent.sample_actions_with_vla` *is* the candidate sampling + critic argmax. So with
RL updates off the run silently degrades to a single-sample rollout with no selection at all.
Launched with `--enable_updates_rl=true`, confirmed in both logs:

```
[main_carla] updates enabled -> rl=True bc=False hl=True
[best_of_n] loaded pretrained critic from /raid/users/cglossop/critic_ckpts/step_0012000.pkl
```

and the selection is demonstrably live — 10 distinct candidates per step at temp 0.50:

```
[best_of_n][cand 0] temp=0.50 subtask='The vehicle accelerates normally while making a slight adjustment to the left.'
[best_of_n][cand 1] temp=0.50 subtask='The vehicle accelerates normally, following the route with steady lane keeping and a slight rightward adjustment.'
[best_of_n][cand 2] temp=0.50 subtask='The vehicle accelerates normally and makes a sharp left adjustment.'
```

Side effect worth knowing: with RL on, the critic no longer stays frozen at its pretrained
weights — it keeps training during the run, which is not what the config's docstring intends.
The cleaner fix is to make the gate agent-aware (Best-of-N's selection does not depend on RL
updates at all, since its critic is pretrained), rather than coupling selection to the update
switch. Not changed here.

## Third config: static-critic Best-of-N + residual RL + CAST relabel (2026-08-04)

`impls/configs/steervla_bon_cast_residual_config.py` — three mechanisms on one rollout:
frozen-critic Best-of-N picks *which* pi0 proposal to follow, a SAC residual *corrects* the
control it decodes to, and CAST relabel supervises the VLM backbone.

Queued on all five routes as jobs **50-54** (20 000 steps each), starting as GPUs free:

| job | route | status |
|---|---|---|
| 50 | `signalized-junction-left-turn-001` | running (GPU 7, smoke test) |
| 51 | `opposite-vehicle-running-red-light-004` | queued |
| 52 | `enter-actor-flow-004` | queued |
| 53 | `merger-into-slow-traffic-v2-005` | queued |
| 54 | `generalization-wall-1095` | queued |

### This combination did not previously exist

Neither loop supported all three, and no config combined cast with residual:

| loop | entry | cast_relabel | BoN | residual |
|---|---|---|---|---|
| `run_online_carla` | everything else | 22 refs | 13 refs | 23 refs |
| `run_online_residual` | `agent_name="sac_residual"` | **0 refs** | 1 | 38 |

`run_online_carla` has all three but they did not compose: every Best-of-N branch in
`_sample_agent_action` did `return best_chunk, None` **before** the residual branch, so the
residual actor never ran. Three explicit guards encoded this, raising
`"--bon_critic_ckpt … is incompatible with --train_mode='sac_residual'; use --train_mode=rl"`.

**Change made.** A new `_bon_selected_to_action(best_chunk, subkey)` helper hands the selected
chunk to the residual instead of executing it: PID-decode to `[accel, steer]`, then
`sample_actions_sac_residual(..., base_action=base2d)`, returning the usual
`(action, base_action)` pair so the buffer stores the base alongside the executed action.
During `residual_warmup_steps` it executes the selected candidate unmodified, so the residual's
critic sees on-distribution base actions first. All six Best-of-N return sites (critic-ckpt,
Gemini, online-critic; each with a cached-chunk and a fresh-score path) route through it, and
it is a no-op outside `accel_steer` residual mode. The three guards were narrowed from "any
residual mode" to "any residual mode that is not `accel_steer`" — the chunk-space residual
genuinely cannot consume a selected chunk this way. The inline `residual_action_space` check is
used rather than `_residual_2d`, which is not defined until ~270 lines later.

### Configuration

Inherits `steervla_cast_relabel_train_config` (so the whole CAST/HL block is unchanged), then:

| knob | value | note |
|---|---|---|
| `agent_name` | `dsrl` | the residual actor lives on `DSRLAgent`, not `BestOfNAgent` |
| `online_training_mode` | `sac_residual` | must also be passed as `--train-mode`, or `run_carla.sh` overwrites it with `rl` |
| selection critic | `--bon-critic-ckpt …/step_0012000.pkl`, `--bon-num-candidates 10` | **static** — no gradient ever touches it |
| `residual_action_space` | `accel_steer` | residual perturbs 2-D `[accel, steer]` |
| `residual_action_scale` | `(0.6, 0.6)` | |
| `residual_warmup_steps` | 500 | |
| **`batch_size`** | **64** | as requested, for the RL update |
| `enable_updates_rl` | **True** | drives the residual SAC update *and* gates selection |
| `enable_updates_bc_hl` | True | CAST HL update |
| `vla_cot_temperature` / `steervla.cot_temperature` | 0.5 / 0.5 | set in the config, so no CLI override |

Two critics are in play and they are not the same object: the **selection** critic is frozen at
its pretrained weights, while `update_sac_residual` trains its own DSRL TD critic alongside the
residual MLP. "Static critic" therefore holds for selection even though RL updates are on.

Smoke test (job 50) confirms all three active:

```
[run_carla.sh] train_mode=sac_residual
[main_carla] updates enabled -> rl=True bc=False hl=True
[main_carla] Best-of-N action selection enabled: ckpt=…/step_0012000.pkl N=10 obs_enc_dim=3456
[main_carla] CAST relabel enabled (provider=gemini, window=150 env steps, debug=True)
```

`obs_enc_dim=3456` = 3×1152, matching the checkpoint's `(2, 3496, 256)` input kernel
(3456 + 40 for the action chunk).

**Route note:** the request said `generalization-wall-005`, which is not in the registry — the
fail2drive wall routes are 1095-1099. Used **`generalization-wall-1095`**, the same route as the
earlier rounds.

## Runs and W&B ids

All in W&B project **`OGBench-CARLA`**, group **`DevVsCastRelabel`**
(`https://wandb.ai/catglossop/OGBench-CARLA/runs/<id>`). Snapshot 2026-08-04 11:37.

### A — cast-only (`steervla_cast_relabel_train_config`, HL-only, cot 0.5)

| job | run id | route | steps | `Cannot spawn actor` | status |
|---|---|---|---|---|---|
| 46 | `92wbycqg` | enter-actor-flow-004 | 11 588 | 33 | running |
| 41 | `vugzu8bc` | signalized-junction-left-turn-001 | 10 574 | 16 | running |
| 42 | `8hst74l6` | opposite-vehicle-running-red-light-004 | **20 000** | 39 | **complete** |
| 43 | `ra5p742b` | merger-into-slow-traffic-v2-005 | 12 670 | 146 | stopped (replaced by 47) |
| 44 | `97eev74o` | generalization-wall-1095 | 12 810 | 0 | stopped (replaced by 48) |
| 45 | `0wvli7wp` | generalization-animals-1081 | 12 120 | **0** | stopped on request |

### B — Best-of-N + cast (`steervla_bon_cast_relabel_config`)

⚠️ Launched with `enable_updates_rl=true`, so the pretrained ranking critic **trained during the
run** instead of staying frozen — see "The critic was not static in runs 47/48" below. Superseded
by jobs 55/56.

| job | run id | route | steps | status |
|---|---|---|---|---|
| 47 | `yse2tuzl` | merger-into-slow-traffic-v2-005 | **20 000** | complete (drifting critic) |
| 48 | `w3lakhnw` | generalization-wall-1095 | **20 000** | complete (drifting critic) |

### C — Best-of-N + residual RL + cast (`steervla_bon_cast_residual_config`)

Static selection critic (`_bon_q_fn`, a jitted closure over fixed params); the RL updates train
the residual actor and its own separate TD critic.

| job | run id | route | steps | status |
|---|---|---|---|---|
| 50 | `sqrj9v26` | signalized-junction-left-turn-001 | 2 237 | running |
| 51 | `m3mlx4nj` | opposite-vehicle-running-red-light-004 | 1 686 | running |
| 52 | `vsjgxk2p` | enter-actor-flow-004 | 160 | running |
| 53 | `akdg92ko` | merger-into-slow-traffic-v2-005 | 6 | running |
| 54 | *(pending)* | generalization-wall-1095 | — | queued |

### D — Best-of-N + cast, RL **off** — corrected re-runs of 47/48

| job | run id | route | steps | status |
|---|---|---|---|---|
| 55 | *(pending)* | merger-into-slow-traffic-v2-005 | — | queued |
| 56 | *(pending)* | generalization-wall-1095 | — | queued |

## The critic was not static in runs 47/48

`BestOfNAgent.update_with_vla(run_rl=True)` runs `apply_loss_fn(total_loss_vla)` +
`target_update`, and `total_loss_vla` includes `_critic_loss_vla_pure_math` — so with RL updates
on, the critic loaded from `critic_pretrained_weights` keeps training. Job 47's `train.csv`
confirms it fired: **1 996 gradient updates**, first at step 50 (right after `warmup_steps=50`),
last at step 20 000, mean 1.85 s each.

Root cause was a single flag controlling two unrelated things. `main_carla.py:2930` gated the
agent's action-selection path on `rl_updates_on`, and `BestOfNAgent.sample_actions_with_vla`
*is* the candidate sampling + critic argmax — so `enable_updates_rl=false` silently degraded the
run to a single sample with no selection, while `true` fired selection *and* trained the critic.

**Fix:** `_selection_needs_no_rl = (agent_name == "best_of_n")`, and the gate became
`if (rl_updates_on or _selection_needs_no_rl) and hasattr(agent, "sample_actions_with_vla")`.
Best-of-N has no noise actor and ranks with a frozen pretrained critic, so its selection never
depended on RL updates. Jobs 55/56 re-run the two routes with `enable_updates_rl=false`: BoN
selection fires, the critic stays pinned at `step_0012000.pkl`, and only the HL update trains.

**No residual is involved in group B or D.** `BestOfNAgent` contains zero residual code
(`grep -c residual impls/jax_agents/best_of_n.py` = 0), `online_training_mode` is `rl` in both
the config and the `--train-mode` flag `run_carla.sh` writes, the residual branch at
`main_carla.py:2902` needs both a residual training mode *and*
`hasattr(agent, "sample_actions_sac_residual")`, and `_bon_selected_to_action` short-circuits to
`return best_chunk, None` whenever `_residual_2d` is false. Jobs 47/48 logs confirm
`train_mode=rl` with zero residual mentions.

## Results

*(route metrics filled in as runs complete)*

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
