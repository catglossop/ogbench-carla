# Handover — SimLingo SteerVLA CAST sweep on the Fail2Drive 16-route subset

Sweep name: **`f2dsteervla_simlingo_fixedcarla_kl005_seed0`**. Ran 2026-09-15 15:48 → 2026-09-18 15:05 on
`dgx5` (GPUs 3, 5, 6), 16/16 routes complete. Written for an agent picking this up on another machine.

Everything below is what the runs *actually* used, read back from a route's `flags.json`, not from memory.

---

## 1. What the sweep is

CAST-relabel **high-level-only** online training on top of a frozen SimLingo SteerVLA driving stack:

- Base VLA: `impls/vlas/simlingo_steervla.py` (`steervla.vla="simlingo_steervla"`) — torch SimLingo
  HL planner (InternVL2-1B + LoRA) → meta-action-conditioned LL waypoint policy, both in-process.
- A Gemini coach reviews each rollout window and relabels chunk subtasks
  (`impls/coaches/cast_relabel.py`); those samples train the HL via `update_hl`. **RL and BC are off**
  (`enable_updates_rl=False`, `enable_updates_bc=False`, `enable_updates_bc_hl=True`); the DSRL agent is
  only the rollout shell (`agent_name="dsrl"`, `online_training_mode="rl"`).
- Each route trains **independently** (per-route policy), stops at the first of: 10 000 training env
  steps, 500 HL gradient steps, or 3 consecutive training episodes with driving score ≥ 95 — then runs
  one frozen 3-seed eval of the checkpoint it stopped at.

**Result: eval split 47.53** (avg per-route SD 18.66) over the 16 routes. Per-route table in §9.

---

## 2. The exact launch

Run from the checkout (the worktree used here was `.claude/worktrees/simlingo-steervla-vla`, branch
`simlingo-steervla-vla`, at commit `b664c89`). `EVAL_EVERY=` empty is what turns periodic evals off.

```bash
cd <checkout>
eval "$(grep -m1 '^export GEMINI_API_KEY=' ~/.bashrc)"     # CAST relabel is a Gemini client
export UV_PROJECT_ENVIRONMENT=/home/cglossop/ogbench-carla/.venv
export PYTHONPATH=<checkout>:/raid/users/cglossop/ogbench-simlingo-deps

export SWEEP_NAME=f2dsteervla_simlingo_fixedcarla_kl005_seed0 \
       ROUTES_FILE=f2d_steervla_subset.txt \
       AGENT_CFG=impls/configs/simlingo_steervla_cast_relabel_train_config.py \
       SWEEP_GPUS="6" SEED=0 \
       ONLINE_STEPS=10000 MAX_HL_UPDATES=500 \
       STOP_ON_SCORE=95 STOP_SCORE_STREAK=3 UPDATES_AFTER_SCORE=0 \
       FIXED_CARLA_SEED=true EVAL_EVERY= POST_STOP_EVAL_EPISODES=3 \
       RESULT_TRAIN_STEP=10000 \
       CARLA_PORT_BASE=17400 TM_PORT_BASE=17500 DISPLAY_BASE=994 \
       F2D_CARLA_0915_ROOT=/home/cglossop/f2d_carla \
       F2D_CARLA_0915_PYTHON=/home/cglossop/ogbench-carla/.venv-f2d-eval/bin/python \
       EXTRA_ARGS="--cot-temperature 0.1 --hl-kl-coef 0.05 --hl-ckpt-keep-last 5"

./.run_carla/b2d_subset_sweep.sh              # dry run: prints the plan, writes queue.txt
setsid nohup ./.run_carla/b2d_subset_sweep.sh --arm >> .run_carla/jobs/$SWEEP_NAME/sweep.log 2>&1 &
```

Add capacity on another GPU at any time (pops from the same `flock`'d queue, safe with the driver):

```bash
GPU=3 CARLA_PORT=18000 TM_PORT=18100 DISPLAY_NUM=997 \
  setsid nohup ./.run_carla/sweep_extra_worker.sh > /dev/null 2>&1 &
```

`--status` prints progress, `--stop` kills the driver's process group and its CARLA servers.

---

## 3. Config (read this first)

**[`impls/configs/simlingo_steervla_cast_relabel_train_config.py`](impls/configs/simlingo_steervla_cast_relabel_train_config.py)**
— inherits `steervla_cast_relabel_train_config.py` → `steervla_cast_relabel_config.py` and swaps in the
SimLingo base via `apply_simlingo_steervla()`. Values it set for this sweep:

| `steervla.` key | Value |
| --- | --- |
| `vla` | `simlingo_steervla` |
| `hl_checkpoint` | `/raid/users/celine/steervla-ckpts/2026_05_24_06_52_33_simlingo_seed1_bellman/checkpoints/epoch=019.ckpt` |
| `ll_checkpoint` | `/raid/users/celine/steervla-ckpts/2026_05_23_21_39_41_simlingo_ll_vla_meta_conditioned/checkpoints/epoch=029.ckpt` |
| `simlingo_source_root` | `/home/cglossop/simlingo-steervla` (simlingo-steervla checkout; `simlingo_training` imported from it) |
| `image_key` | `image_viz` (native 1024×512 front camera) |
| `actions_per_model_query` / `actions_per_cot` | 3 / 5 (LL every 3 env steps; HL re-plans every 6) |
| `cot_temperature` | 0.1 (from `EXTRA_ARGS`; config default is 0.0) |
| `hl_kl_coef` | 0.05 (from `EXTRA_ARGS`) |
| `hl_update_every` / `hl_update_num_steps` | 100 update calls / 10 grad steps ⇒ 10 grad steps per 200 env steps |
| `hl_update_batch_size` / `hl_micro_batch_size` | 64 / 8 |
| `hl_online_weight` / `hl_online_bad_fraction` | 0.5 / 0.9 (precursor 2/3 of the bad share) |
| `hl_replay_root` / `hl_replay_pools` | `/raid/users/cglossop/simlingo_hl_pools` / `[{simlingo_hl_simplified, 0.5}]` |
| `use_adaptive_sampling` | **False** |
| `hl_checkpoint_every_steps` / `hl_checkpoint_keep_last` | 2000 / 5 (`--hl-ckpt-keep-last 5`) |

Per-run flags that reached `main_carla` (from `flags.json`):

```
--online_steps=10000 --max_hl_updates=500 --stop_on_driving_score=95 --stop_on_driving_score_streak=3
--post_stop_eval_episodes=3 --eval_crash_stuck_steps=1000000000 --fixed_train_carla_seed=true --eval_mode=true
--carla_seed=0 --train_seed=0 --eval_seeds=1001,1002,1003        (eval_every_env_steps = 0, i.e. off)
```

`--eval_crash_stuck_steps=1000000000` makes eval episodes use Bench2Drive's `AgentBlockedTest` (60 s)
instead of the training wrapper's 1 s post-collision cutoff. Training episodes keep the 1 s cutoff.

---

## 4. The runners

| File | Role |
| --- | --- |
| [`.run_carla/b2d_subset_sweep.sh`](.run_carla/b2d_subset_sweep.sh) | Sweep driver (used for f2d too, despite the name): builds `queue.txt` from `ROUTES_FILE` minus routes that already have a `run_summary.json`, one worker per `SWEEP_GPUS` entry, CARLA-gone and stall watchdogs, `--status` / `--stop`, calls `sweep_results.sh` after every route. |
| [`.run_carla/eval_subset_run.sh`](.run_carla/eval_subset_run.sh) | One route. Owns the seed contract (seed N ⇒ `carla_seed=N`, `train_seed=N`, `eval_seeds=N+1001..N+1003`), exports the W&B school key, pins ports/displays, and switches **only `generalization-animals-*` routes** to CARLA 0.9.15 when `F2D_CARLA_0915_ROOT` is set. |
| [`.run_carla/sweep_extra_worker.sh`](.run_carla/sweep_extra_worker.sh) | Attach one more GPU to a live sweep; pops from the same queue under `flock`. **Use this one, not `f2d_extra_worker.sh`** — see §8. |
| [`.run_carla/gpu_chain.sh`](.run_carla/gpu_chain.sh) | Hands one GPU a list of sweep queues to serve in order once its current job ends. |
| [`.run_carla/sweep_results.sh`](.run_carla/sweep_results.sh) | Writes `RESULTS.md`, `results.csv`, `strategies.md`, `summaries/`. `RESULT_TRAIN_STEP=10000` scores a run that trained past 10 k by its 10 000-checkpoint eval. |
| [`f2d_steervla_subset.txt`](f2d_steervla_subset.txt) | The 16 routes (§7). |
| [`impls/vlas/extract_simlingo_hl_replay.py`](impls/vlas/extract_simlingo_hl_replay.py) | Builds the SimLingo HL replay pool (§5). |

Route logs: `.run_carla/jobs/<SWEEP_NAME>/<route>.log`; driver log `sweep.log`; worker logs
`extra_worker_gpu<N>.log`; queue `queue.txt` (+ `queue.lock`, `carla_died.txt`).

---

## 5. What the new machine needs

1. **SimLingo checkpoints** — the two `.ckpt` dirs in §3. DeepSpeed dirs work as-is (`converted/pytorch_model.bin`
   is used if present); a `.hydra/config.yaml` must sit above the weights.
2. **simlingo-steervla checkout** at `simlingo_source_root`. Source only — its py3.8 env is not needed.
3. **Python deps**: `uv sync --extra all-gpu --extra simlingo`, or reuse a prebuilt dir on `PYTHONPATH`
   (here `/raid/users/cglossop/ogbench-simlingo-deps`), which is what these runs did.
4. **HL replay pool** `simlingo_hl_simplified` (4000 samples) under `hl_replay_root`. Rebuild with:
   ```bash
   .venv/bin/python impls/vlas/extract_simlingo_hl_replay.py \
     --data-path /raid/datasets/simlingo/database/simlingo \
     --simlingo-source-root /home/cglossop/simlingo-steervla \
     --out-root /raid/users/cglossop/simlingo_hl_pools --name simlingo_hl_simplified --n 4000
   ```
   A missing pool only warns and falls back to online-only samples, which silently changes the recipe.
5. **CARLA 0.9.16 + the Fail2Drive content pack** (`CARLA_ROOT`, default `/home/cglossop/carla`) for 15 routes,
   **plus CARLA 0.9.15** (`F2D_CARLA_0915_ROOT=/home/cglossop/f2d_carla`) and its py3.10 env
   (`F2D_CARLA_0915_PYTHON=.../.venv-f2d-eval/bin/python`) for the animals route. See `ogbench/carla/README.md` §"Fail2Drive routes".
6. **Keys**: `GEMINI_API_KEY` (CAST coach) and the W&B school key at `/home/cglossop/.wandb_school_key`
   with `WANDB_ENTITY=catherineglossop` — `eval_subset_run.sh` exports both. **Runs must log as the
   `catherine_glossop` W&B user**; check `wandb.Api().viewer.username` before launching.
7. **Disk**: ~10–20 GB of checkpoints per route under `OGBENCH_SAVE_DIR` (defaults to
   `/raid/users/cglossop/sweeps/<SWEEP_NAME>`), so keep it off `/home`.

---

## 6. W&B and outputs

- Project `OGBench-CARLA`, run group = `SWEEP_NAME`, one run per route.
- Runs: `/raid/users/cglossop/sweeps/f2dsteervla_simlingo_fixedcarla_kl005_seed0/OGBench-CARLA/<SWEEP_NAME>/<run>/`
  with `checkpoints/<train_step>/` (HL only, last 5), `run_summary.json`, `cast_relabel/` (window artifacts,
  `strategy_memory.json`), `videos/`, `flags.json`.
- Report: `/raid/users/cglossop/sweep_results/f2dsteervla_simlingo_fixedcarla_kl005_seed0/`
  (`RESULTS.md`, `results.csv`, `strategies.md`, `summaries/`). Regenerate any time:
  ```bash
  OGBENCH_SAVE_DIR=/raid/users/cglossop/sweeps/$SWEEP_NAME \
  RESULTS_DIR=/raid/users/cglossop/sweep_results/$SWEEP_NAME \
  SWEEP=$SWEEP_NAME ROUTES_FILE=f2d_steervla_subset.txt RESULT_TRAIN_STEP=10000 \
    ./.run_carla/sweep_results.sh
  ```

---

## 7. Routes (`f2d_steervla_subset.txt`)

```
generalization-construction-permutations-1019   generalization-custom-obstacles-1020
generalization-pedestrians-on-road-1085         generalization-construction-pedestrian-1011
generalization-pedestrian-crowd-1069            generalization-custom-obstacles-1024
generalization-fully-blocked-1032               generalization-hard-brake-1036
generalization-pedestrian-other-blocker-1072    generalization-right-construction-1094
generalization-right-of-way-1056                generalization-image-on-object-1041
generalization-obscured-stop-1046               generalization-bad-parking-1004
generalization-animals-1076                     generalization-wall-1097
```

`generalization-animals-1076` **must** run on CARLA 0.9.15. On 0.9.16 the walker `walker.animal.1007`
does not exist, CARLA silently substitutes `vehicle.tesla.model3`, and the scenario is skipped with a setup
error — the route trains against a route with no animal. It cost one wasted run here; check the log for
`[fail2drive animal] requested=... spawned=[...] matched=True`.

Throughput: 2.7–4.9 h per route per GPU (~3.5 h typical). Whole sweep ≈ 60 GPU-hours.

---

## 8. Gotchas that cost time here

1. **Orphaned `main_carla` (the big one).** `run_carla.sh` runs `main_carla` in its own process group, so a
   watchdog abort that kills the launcher's group leaves `main_carla` alive holding ~110 GB of GPU. It happened
   three times (GPU 5 blocked ~4 h). Worse, the watchdog's own cleanup never runs: its TERM makes the worker's
   `wait` return, and the worker then kills the watchdog mid-`sleep 15`. **`sweep_extra_worker.sh` fixes this**
   by cleaning up in the main loop after every route. `b2d_subset_sweep.sh` and the older `f2d_extra_worker.sh`
   still have the bug — if a route is aborted there, check `nvidia-smi` and kill leftovers by hand.
2. **A retry restarts from step 0.** `run_carla.sh` resumes nothing on the DSRL/CAST path, so a mid-run CARLA
   crash throws the route's progress away. Checkpoints on disk are still usable: evaluate one frozen with
   `--frozen-eval true --steervla-checkpoint <ckpt> --frozen-eval-out <run>/ckpt_evals/<step>` rather than
   retraining (`sweep_results.sh` reports such a run from its checkpoint eval).
3. **A route that dies fast is requeued, one that dies slowly is not.** `MIN_HEALTHY_SECS=300` in the workers:
   a failure after that is treated as a finished route. A CARLA world-setup segfault at ~350 s therefore
   consumed the route silently once.
4. **Ports/displays are per worker.** `CARLA_PORT_BASE`/`TM_PORT_BASE`/`DISPLAY_BASE` must not collide with
   another sweep or another user; `eval_subset_run.sh` aborts if the port is listening, and it reclaims a stale
   `/tmp/.X<N>-lock` only if no live Xvfb/CARLA owns it. Check `/tmp/carla_rpc<port>.log` ownership first.
5. **Never `pkill -f` a pattern that appears in your own command line** — it matches the shell running it.
   Use a bracketed pattern (`[m]ain_carla`) or kill by PID.
6. **GPU etiquette on dgx5**: GPUs 0 and 7 are other users'. These runs used 3, 5 and 6 only.

---

## 9. Results (2026-09-18, 16/16)

Eval of each route's stop checkpoint, seeds 1001–1003. `stop`: `budget` = 10 k training steps,
`cap` = 500 HL updates, `@10000` = scored at the 10 000 checkpoint.

| Route | DS | stop | HL updates |
| --- | ---: | --- | ---: |
| `generalization-image-on-object-1041` | 100.00 | cap | 80 |
| `generalization-obscured-stop-1046` | 100.00 | cap | 110 |
| `generalization-pedestrian-other-blocker-1072` | 83.33 | budget | 490 |
| `generalization-fully-blocked-1032` | 60.00 | budget | 490 |
| `generalization-hard-brake-1036` | 52.00 | budget | 490 |
| `generalization-right-of-way-1056` | 51.92 | budget | 490 |
| `generalization-bad-parking-1004` | 44.00 | budget | 490 |
| `generalization-animals-1076` | 41.80 | budget | 490 |
| `generalization-pedestrians-on-road-1085` | 36.17 | budget | 490 |
| `generalization-construction-pedestrian-1011` | 34.42 | budget | 490 |
| `generalization-right-construction-1094` | 32.75 | cap | 450 |
| `generalization-custom-obstacles-1024` | 30.68 | budget | 490 |
| `generalization-wall-1097` | 28.31 | budget | 490 |
| `generalization-pedestrian-crowd-1069` | 23.98 | budget | 0 |
| `generalization-construction-permutations-1019` | 21.05 | budget | 490 |
| `generalization-custom-obstacles-1020` | 20.03 | budget | 490 |

**Split mean 47.53**, avg per-route SD 18.66. `pedestrian-crowd-1069` reached the step budget with 0 HL
updates — worth a look before trusting that row.

The comparable π0.5 ll-heavy f2d sweep is `f2dsubset_fixedcarla_kl005_seed0` (its own 16-route list, only 12
routes overlap with this one) plus `f2dllheavy_fixedcarla_kl005_newroutes_seed0` for the 4 routes it missed.

---

## 10. ⚠️ What is NOT on `dev` yet

As of `dev` = `90949a3`, a fresh clone **cannot** reproduce this sweep. Missing or stale there:

| Path | State |
| --- | --- |
| `f2d_steervla_subset.txt` | untracked — the route list |
| `.run_carla/sweep_extra_worker.sh`, `.run_carla/gpu_chain.sh`, `.run_carla/f2d_extra_worker.sh` | untracked (`.run_carla/` is gitignored; tracked launchers were force-added) |
| `.run_carla/eval_subset_run.sh` | dev's copy has **no** `F2D_CARLA_0915_ROOT` per-route switch, so the animals route would run on 0.9.16 and be invalid |
| `.run_carla/sweep_results.sh` | dev has `RESULT_TRAIN_STEP`, but not the fix that applies `ROUTES_FILE` to the periodic-eval table/CSV |

Committed and fine on dev: the config, `impls/vlas/simlingo_steervla.py`, `impls/vlas/extract_simlingo_hl_replay.py`,
`b2d_subset_sweep.sh`, and the budget-stop / eval-cadence changes in `impls/main_carla.py` (`b664c89`).

Before handing over, commit the four rows above from the `simlingo-steervla-vla` worktree
(`git add -f` for the `.run_carla/` files) and push, or copy them across by hand.
