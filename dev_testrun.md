# `dev` branch merge test runs — 2026-07-31

Smoke-test of the `dev` branch after merging `cast_relabel`, `surya/rl-token`, and
`rc-integrate` (routing-commands + master) — commit `62ecc90`.

Three online CARLA runs were launched, all sharing one base SteerVLA checkpoint:

```
gs://cat-logs/pi05_steervla_cot_simplified_reasoning_no_ego_history/
  pi05_steervla_simplified_reasoning_no_ego_history_v1/
  pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000
```

| # | Run | Agent config | Train GPU | Render adapter | W&B |
|---|-----|--------------|-----------|----------------|-----|
| 1 | cast_relabel | `steervla_cast_relabel_train_config.py` | 1 | 5 | [3988noib](https://wandb.ai/catglossop/OGBench-CARLA/runs/3988noib) |
| 2 | sac_residual | `steervla_residual_config.py` | 2 | 4 | [OGBench-CARLA-Residual](https://wandb.ai/catglossop/OGBench-CARLA-Residual) |
| 3 | dsrl | `steervla_dsrl_config.py` | 3 | 7 | [ens1ps1q](https://wandb.ai/catglossop/OGBench-CARLA/runs/ens1ps1q) |

Route `parking-cut-in-001`, seed 0, 10 000 env steps each, run group `DevMergeTest`.

**Outcome: all three run.** Four defects were found and fixed along the way — three of
them genuine merge artifacts that made the affected entrypoints fail immediately.

### Verified working

| Run | Status | Evidence |
|---|---|---|
| cast_relabel | stepping, past all crash points | 4 VLM relabel windows written under `cast_relabel/ep0001_win000{1..4}`; HL dataset samples (`sample_*.npz` + `hl_samples.json`) per window; **20+ real HL gradient steps** (`[steervla.update_hl] device=cuda:1 live=54.54GB peak=65.21GB limit=112.59GB (bs=2, steps=1)`); 31 update-batch JSON/PNG debug panels |
| sac_residual | stepping | `_run_residual_entry` → `pi_prefix` state encoder → `SACResidualAgent`; fastest of the three (~2.9k steps) |
| dsrl | stepping | `[main_carla] updates enabled -> rl=True bc=False hl=False`, RL updates every 10 env steps after 200-step warmup |

**All three completed their full 10 000/10 000 steps** and exited 0.

### Route metrics at 10 000 steps

| Run | Episodes | RouteCompletion per episode | MinSpeedTest | Collisions |
|---|---|---|---|---|
| cast_relabel | 3 | 9.11 %, 17.37 %, 12.11 % | 90 %, 58 % | 0 |
| sac_residual | 14 | **100 % ×13**, then 21.88 % | 213 %, 165 % | 2, then 0 |
| dsrl | 3 | 3.10 %, 15.87 %, 12.86 % | 90 %, 57 % | 0 |

This gap is not a training-quality difference — it is a decode bug in the DSRL/cast_relabel
action path. See "Follow-up" below.

---

## How the jobs were launched

`carla_job.sh` derives every port/display from the `--job` index, so the three runs cannot
collide (`rpc = 12000 + 100k`, `tm = 18000 + 100k`, `display = 30 + k`).

Common environment for all three:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # see "GPU notes" — 3 JAX procs on one box
export WANDB_MODE=online
CKPT="gs://cat-logs/pi05_steervla_cot_simplified_reasoning_no_ego_history/pi05_steervla_simplified_reasoning_no_ego_history_v1/pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000"
```

### 1) cast_relabel — GPU 1

```bash
./carla_job.sh start --job 0 --train-gpu 0 --render-adapter 5 --route parking-cut-in-001 -- \
  --agent-config impls/configs/steervla_cast_relabel_train_config.py \
  --online-steps 10000 --train-mode rl --critic-mode none --hl-gpu 0 \
  --max-retries 3 --run-group DevMergeTest --seed 0 \
  -- --save_dir=/home/cglossop/ogbench-carla/exp --agent.siglip_device=cuda:1 \
     --agent.steervla.checkpoint="$CKPT"
```

### 2) sac_residual — GPU 2

```bash
./carla_job.sh start --job 1 --train-gpu 1 --render-adapter 4 --route parking-cut-in-001 -- \
  --agent-config impls/configs/steervla_residual_config.py \
  --online-steps 10000 --train-mode sac_residual --critic-mode none \
  --max-retries 3 --run-group DevMergeTest --seed 0 \
  -- --save_dir=/home/cglossop/ogbench-carla/exp --agent.siglip_device=cuda:2 \
     --agent.steervla.actor_config=pi05_steervla_cot_simplified_reasoning_no_ego_history \
     --agent.steervla.checkpoint="$CKPT"
```

This is the standalone `agent_name="sac_residual"` stack from `surya/rl-token`
(`_run_residual_entry` → `SACResidualAgent`, `state_encoder="pi_prefix"`), *not*
`pi0_residual_sac_config.py` (which is the DSRL residual sub-agent from routing-commands —
that one is still untested).

The config shipped pointing at an older checkpoint + `actor_config`
(`pi05_steervla_cot_simplified_reasoning` @ 8000), so both were overridden on the command
line to use the shared base checkpoint. `run_carla.sh` has no checkpoint flag; ml_collections
`--agent.<path>=<value>` overrides are applied *after* the config's `get_config()`, so they
win over the temp-config wrapper.

### 3) dsrl — GPU 3

```bash
./carla_job.sh start --job 2 --train-gpu 2 --render-adapter 7 --route parking-cut-in-001 -- \
  --agent-config impls/configs/steervla_dsrl_config.py \
  --online-steps 10000 --train-mode rl --critic-mode none \
  --max-retries 3 --run-group DevMergeTest --seed 0 \
  -- --save_dir=/home/cglossop/ogbench-carla/exp --agent.siglip_device=cuda:3 \
     --agent.steervla.checkpoint="$CKPT"
```

Monitor / stop:

```bash
./carla_job.sh list
./carla_job.sh logs 0
./carla_job.sh stop 0        # scoped — never touches sibling jobs
```

---

## GPU notes (machine-specific, not a code bug)

**GPUs 0 and 6 are `compute_mode=Prohibited` on this box.** Two consequences that cost the
first two launch attempts:

1. JAX cannot open them, so `jax.devices("gpu")` is `[cuda:1, cuda:2, cuda:3, cuda:4, cuda:5, cuda:7]`.
   `training_gpu_rank` indexes *that* list, so **rank N → physical GPU N+1**. To train on
   physical GPUs 1/2/3 you must pass `--train-gpu 0/1/2`. The log line to check is
   `[main_carla] JAX default device -> cuda:N (training_gpu_rank=M)`.
2. CARLA `-graphicsadapter=6` dies instantly with an empty `/tmp/carla_rpc*.log`. Render
   adapters 4/5/7 work. On this box CARLA's adapter index matches `nvidia-smi` ordering
   (no swap), contrary to the general warning in the README.

`siglip_device` is a **torch** index and does *not* skip prohibited GPUs, but
`run_carla.sh` sets it to `cuda:${TRAIN_GPU_RANK}` — i.e. the JAX rank. With the offset above
that would have put SigLIP on the prohibited GPU 0. Hence the explicit
`--agent.siglip_device=cuda:<physical>` override on every run. This coupling in
`run_carla.sh` is only correct when the JAX rank and the torch index coincide; worth fixing
properly if prohibited/masked GPUs are a recurring situation here.

`XLA_PYTHON_CLIENT_PREALLOCATE=false` is set so the three JAX processes don't each try to
preallocate 75 % of a card.

---

## Bugs found and fixed

### 1. `main_carla.py` — duplicate `enable_updates` flag (fatal at import)

`flags.DEFINE_bool("enable_updates", ...)` was defined twice, so **every** entrypoint died
before doing anything:

```
absl.flags._exceptions.DuplicateFlagError: The flag 'enable_updates' is defined twice.
```

The merge brought in a newer "master switch" definition that sits with its
`enable_updates_rl` / `_bc` / `_bc_hl` siblings, while the older standalone copy survived
further up the file. Removed the older orphan; kept the grouped one.

### 2. `run_carla.sh` — `PYTHONPATH` orphaned onto an `echo`

The merge inserted new `echo` lines between the `PYTHONPATH=... \` line-continuation and the
`uv run python impls/main_carla.py` invocation it was meant to prefix, so `PYTHONPATH` was
being set for an `echo` and never reached the run. Since the invocation now lives inside the
crash-supervisor `while` loop, a one-shot `VAR=... cmd` prefix is the wrong shape anyway —
changed to a plain `export` before the loop.

Low impact in practice (`CARLA_ROOT` is exported in the environment and `carla.py` prepends
it to `sys.path` itself), but it silently dropped `simlingo-rebuttal` from the path.

### 3. `vlas/steervla.py` — `setup()`'s tail stranded inside `update_hl()` (fatal on first action)

The block that builds the jitted prefix kernels had been spliced into the **end of
`update_hl()`, after its `return out`** — dead code. `setup()` therefore never created them,
and the first env step of every SteerVLA run died with:

```
AttributeError: 'SteerVLAActor' object has no attribute '_prefix_embed_fn'
```

(preceded, once that was patched, by `'SteerVLAActor' object has no attribute '_cached_policy_embed'`).

Fixes:
- Moved `_prefix_embed_fn`, `_prefix_cache_fn`, `_denoise_step_fn` into
  `_build_sample_wrappers()` — the right home, since they are `nnx_utils.module_jit`
  kernels bound to `self.model` and must be rebuilt after any in-place weight update.
- Deleted the dead copy from `update_hl()`.
- Hoisted `import types as _types` (it lived inside the dead block) to module scope.
- Initialised `_cached_policy_embed` / `_cached_policy_embed_obs_id` in `__init__`
  alongside the other caches; both were read before their first assignment.

### 4. `vlas/steervla.py` — HL train-state donation deleted the inference model's buffers

With `hl_training_gpu_rank == training_gpu_rank` (HL update sharing the inference GPU), the
cast_relabel run reached step ~300, ran its first HL gradient steps, and then died on the
next action sample:

```
jaxlib.xla_extension.XlaRuntimeError: INVALID_ARGUMENT: Buffer has been deleted or donated.
```

`_hl_train_step` is jitted with `donate_argnums=(1,)`, so each HL update deletes the buffers
of the `TrainState` passed in. The inference model was built with
`nnx.merge(train_state.model_def, train_state.params)` — **aliasing those exact buffers**.
The split-GPU path happened to be safe only because its `jax.device_put(..., infer_device)`
is a real cross-device copy; on a shared device `device_put` is a no-op that hands back the
same buffer.

Added `SteerVLAActor._params_for_inference()`, which copies explicitly when the HL and
inference devices coincide, and routed both the `setup()` construction sites and
`_refresh_inference_weights()` through it.

This one is not merge debris — it is a latent bug in the shared-GPU HL configuration,
reachable from any `load_trainable_params=True` run whose `hl_training_gpu_rank` is unset,
`-1`, or equal to `training_gpu_rank`. Verified fixed: the run now completes HL updates
(`[steervla.update_hl] device=cuda:1 live=54.54GB peak=65.21GB limit=112.59GB (bs=2, steps=1)`)
and keeps stepping.

---

## Follow-up — DSRL / cast_relabel drove at a crawl (2026-08-01)

### The A/B

Three **rollout-only** runs (`--enable-updates false`, 1000 steps, same checkpoint, same
route, jobs 4/5/6, run group `RolloutOnlyAB`) to separate "the training made it slow" from
"the rollout path is wrong":

```bash
./carla_job.sh start --job 4 --train-gpu 0 --render-adapter 5 --route parking-cut-in-001 -- \
  --agent-config impls/configs/steervla_dsrl_config.py --enable-updates false \
  --online-steps 1000 --max-retries 0 --save-buffer false --critic-mode none \
  --run-group RolloutOnlyAB -- --save_dir=exp --agent.siglip_device=cuda:1 \
     --agent.steervla.checkpoint="$CKPT"
# job 5 = steervla_cast_relabel_train_config.py, --train-gpu 1 --render-adapter 4, siglip cuda:2
# job 6 = steervla_residual_config.py,           --train-gpu 2 --render-adapter 7, siglip cuda:3
```

| arm | ego speed mean / p90 / max (m/s) | PID desired speed mean / max | RouteCompletion |
|---|---|---|---|
| sac_residual | 4.74 / 9.49 / **11.60** | 4.26 / 9.59 | 100 %, 76.71 % |
| dsrl | 0.05 / 0.23 / **0.41** | 0.23 / 0.36 | 1.6 % |
| cast_relabel | 0.01 / 0.07 / **0.37** | 0.11 / 0.32 | 0.0 % |

With **no gradient updates in any arm**, DSRL and cast_relabel still crawl. So the cause is
in the rollout/action path, not in training — and `debug_task`'s `reward = -ego_speed` was
not responsible.

### Root cause — the ×7 denormalization was dropped

`openpi`'s `denormalize_actions` reverses the RLDS scaling; for
`DELTA_XY_T_DELTA_XY_SPACE` it multiplies the **speed-waypoint deltas (columns 0:2) by 7.0**.
Exactly one party must apply it, chosen by `exec_cfg["action_input_space"]`:

* `normalized` → `SimlingoStyleWaypointDecoder` applies it.
* `policy_output` → decoder assumes the chunk is already physical and applies nothing.

`_steervla_action_execution_cfg` picked with the wrong condition:

```python
if residual and not remote:      # <-- residual is not the discriminator
    action_input_space = "normalized"
else:
    action_input_space = "policy_output"
```

Every local entrypoint gets its chunk from the **same** `vla_sample_fn`, and
`impls/vlas/steervla.py` deliberately disables OpenPI Normalize/Unnormalize for these
checkpoints (`STEERVLA_ENABLE_OPENPI_NORM`), so the local actor always returns raw model
output. Only the remote server (`steervla_server.py`, `steervla_physical_denormalize_actions`)
hands back physical units. So DSRL, cast_relabel, best_of_n and the routing-commands residual
sub-agent all landed on `policy_output` and silently skipped the ×7; the standalone residual
stack was the one path that happened to be right.

Replaying the 338 chunks DSRL actually logged through both decodes:

```
executed (policy_output): mean=0.2094 max=0.3814  frac below brake_speed(0.1) = 0.127
correct  (normalized)   : mean=1.4660 max=2.6698  frac below brake_speed(0.1) = 0.000
ratio (unique, rounded) : [7.]
```

The ratio is **exactly 7.0 on every chunk** — `desired_speed` is linear in those two columns
(`control_pid`: `norm(speed_wp[hi] - speed_wp[lo]) * 2`). The computed 0.209 matches the
logged `[RC-PID] Desired speed` mean of 0.229, so this is the real executed path.

Second-order: 12.7 % of DSRL's steps fall below `brake_speed = 0.1` and get commanded
**brake**, vs 0 % under the correct decode. And the model is speed-conditioned, so a car that
never gets moving keeps predicting small deltas — which is why the observed gap (~19×)
exceeds the raw 7×. The chunks themselves are fine; both stacks emit the same distribution.

### Verification

Same three arms re-run on the fixed code (jobs 7/8/9, run group `RolloutOnlyFixed`), 1000
rollout-only steps each:

| arm | ego speed mean / p50 / max — before → after | RouteCompletion before → after |
|---|---|---|
| dsrl | 0.05 / 0.00 / 0.41 → **5.39 / 5.83 / 11.60** | 1.6 % → **100 %, 100 %**, 0.85 % |
| cast_relabel | 0.01 / 0.00 / 0.37 → **4.46 / 4.74 / 11.70** | 0.0 % → **100 %, 66.95 %** |
| sac_residual (control) | 4.74 / 4.63 / 11.60 → 6.00 / 6.00 / 11.01 | 100 %, 76.71 % → 100 %, 100 %, 21.88 % |

DSRL went from 1.6 % of the route in 1000 steps to **two full 100 % completions** in the same
budget; cast_relabel from 0 % to 100 % + 67 %. Trailing sub-1 % entries are episodes still in
progress when the step budget ran out.

The control arm is unchanged within run-to-run variance, as expected: its
`action_input_space` resolves to `normalized` under both the old and the new condition.

For reference, the pre-fix 10 000-step DSRL run never exceeded 15.9 % route completion in
three episodes — the fixed policy clears the whole route in a few hundred steps.

### Fixes

**5. `main_carla.py` — `action_input_space` keyed on the wrong condition.** Now
`"policy_output" if remote else "normalized"`, with the ownership rule written down. Dropped
the dead `residual` kwarg and its call site, and fixed a stale
`.get(..., "policy_output")` default that disagreed with its twin at the residual site. This
also repairs `pi0_residual_sac_config.py` (the routing-commands residual sub-agent), which
ran through the same broken branch.

**6. `main_carla.py` — DSRL rollout bypassed the trained noise actor (merge regression).**
The merge replaced `agent.sample_actions_with_vla(obs, seed=subkey)` with a hardcoded
`tanh(N(0,1)) * vla_noise_scale` fed to `vla_sample_fn`, leaving `sample_actions_with_vla`
called *nowhere* in the file (original at `c22d346` / `63e19f7`). Consequence:
`enable_updates_rl=True` trained a critic and noise actor that never touched behaviour — also
why the dsrl and cast_relabel arms were near-identical despite different configs. Restored
the learned-actor path when `rl_updates_on`, keeping the fixed-latent fallback for
rollout-only/eval, which is the case the original comment was actually protecting (the noise
actor is uncalibrated at random init).

**7. `steervla_dsrl_config.py` — `debug_task` back to `False`.** Switched on in one-off commit
`aaa4c7a`; it makes RL updates optimise `reward = -ego_speed`, i.e. trains the car to stop.
It was inert while fix 6's bug bypassed the noise actor, but reconnecting the actor makes it
live. cast_relabel inherits it.

---

## Follow-up 2 — the violation banners on the rollout video (2026-08-02)

The merge silently gutted the collision banner. In `run_online_carla`'s nested
`_annotate_collision_frame`, everything between `import cv2` and `return annotated` was
deleted by `62ecc90`, leaving:

```python
        annotated = np.array(frame, copy=True)
        try:
            import cv2  # type: ignore

            return annotated      # <- the whole COLL badge draw was here
        except Exception:
            return annotated
```

so the function still ran on every collision step and still returned a frame — just an
unannotated one. `_annotate_traffic_violation_frame` was untouched, and every call site
(`main_carla.py:3330-3341`) survived, which is why only the red banner went missing.
Verified by `git log -L 2043,2056:impls/main_carla.py`: the drawing block is present in
`22bac1a` and absent in the merge. Nothing else regressed — `_annotate_text_panel`,
`_annotate_waypoints`, `_annotate_reward_corner`, `_annotate_candidates_panel`,
`_plot_bon_candidates` and `coaches/cast_relabel.py::annotate_cast_relabel_frames` all
compare byte-identical to `cast_relabel` / `rc-integrate`.

**8. Restored the collision banner, deduplicated against `_draw_corner_badge`.** Both nested
helpers now delegate to the module-level badge drawer, which grew a `row` argument so badges
stack in the same corner (row 0 = collision at `y0=6`, row 1 = traffic violation at `y0=22`,
exactly the old hardcoded offsets). Checked pixel-exact against the pre-merge
routing-commands implementations — a rendered frame is byte-identical for both badges.

**9. The residual loop now annotates violations too.** `run_online_residual` goes through the
module-level `_annotate_full_frame` / `_maybe_capture_frame` pipeline, which knew about
collisions but not traffic violations, so videos from the three stacks did not read the same
way. Added:

- `traffic_violation=(count, episode_events)` to `_annotate_full_frame` / `_maybe_capture_frame`;
- traffic-violation bookkeeping + the `[main_carla] TRAFFIC VIOLATION at step N` print in
  `run_online_residual`, matching `run_online_carla`;
- `rollout/traffic_violation_events` + `rollout/traffic_violation_count` per-step, and
  `rollout/episode_traffic_violations` per episode (via `_episode_summary_log`, so both loops
  get it);
- **force-capture on event steps.** `_maybe_capture_frame` only fired on
  `episode_steps % video_every == 0 or done`, so a collision badge appeared only if the crash
  happened to land on a capture step. `run_online_carla` already force-captured collision and
  violation steps; the module-level path now does too.

Not annotated, in either loop, before or after: `outside_route_delta` — the env computes and
penalises it (`carla_utils.py:2645`) but no branch ever drew it. Left alone.

Verified offline (no CARLA needed): badge rendering is pixel-identical to the pre-merge code;
`_maybe_capture_frame` captures an event step that is not a multiple of `video_every`, draws
both banners on it, and still skips a quiet non-capture step. Ruff on `main_carla.py` goes
54 -> 51 errors (the gutted function's now-unused `import cv2` was one of them); no new ones.

### Also hit

`main_carla.py:147` defaults `save_dir` to `/home/celinet/carla_exps` and `run_carla.sh`
never overrides it, so a launch without an explicit `--save_dir` dies instantly with
`PermissionError: [Errno 13] Permission denied: '/home/celinet'`. Pass `--save_dir=exp`.
Worth fixing properly.

---

## Still-latent issues (found, not fixed)

These pre-date the merge fixes above and are **not** reachable by the three runs tested, so
they were left alone rather than guessed at:

- `vlas/steervla.py` — `postprocess_sampled_trajectory()` has an unreachable block after its
  `return` (lines ~2879-2909) referencing undefined names (`batch_size`, `input_noise`,
  `rng`, `obs_full`, `openpi_observation`). Same splice-after-return signature as bug 3, so
  it is probably another stranded method tail; harmless while unreachable, but it means some
  method somewhere may be missing *its* tail.
- `vlas/steervla.py` — `sample_candidates` is defined twice (~line 2928 and ~4323); Python
  keeps the later one. Best-of-N path. Worth confirming which is authoritative.
- `vlas/steervla.py:3107` — undefined name `_openpi_gemma`.
- `impls/configs/steervla_residual_config.py` still points at the older
  `pi05_steervla_cot_simplified_reasoning` @ 8000 checkpoint, which does not match the
  `_no_ego_history` actor config used elsewhere.
- `pi0_residual_sac_config.py` (routing-commands DSRL residual sub-agent,
  `online_training_mode="sac_residual"` with `agent_name="dsrl"`) was **not** exercised —
  it is a fourth distinct code path. It shared the `action_input_space` bug and is repaired
  by fix 5, but that has not been confirmed on a run.
- The SimLingo residual-SAC stack (`main_carla_simlingo.py`) cannot run here:
  `simlingo-rebuttal/` is empty and the stack needs the py3.8 conda env.

## Caveats on the numbers

- The three 10 000-step runs were made **before** fixes 5-7, so their reward curves and route
  metrics reflect the broken decode. They test code paths, not driving quality.
- `debug_task=True` was the shipped default in `steervla_dsrl_config.py` (and inherited by
  `steervla_cast_relabel_config.py`) at the time of those runs, so **their RL updates used
  `reward = -ego_speed`, not the env reward**. Fix 7 turns it off; re-check it before reading
  any reward curve from an older run.
- `--critic-mode none` was used throughout (`run_carla.sh`'s default), so the critic
  language-label paths (`action_delta`, `delta_commentary_bow`, `commentary_bow`,
  `vlm_chunk_bow`) are untested.
- `--max-retries 3` (default is 50) so genuine crashes surface instead of being retried away.

## Files changed

```
impls/configs/steervla_dsrl_config.py |   6 ++--
impls/main_carla.py                   | 182 ++++++++++++++++++++++++----------
impls/vlas/steervla.py                |  66 +++++++++----
run_carla.sh                          |   5 +-
```

Working tree only — nothing committed.
