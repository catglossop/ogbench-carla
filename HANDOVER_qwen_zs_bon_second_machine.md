# Handover: second machine for the f2d mixed zero-shot Qwen BoN sweep

Split the running Fail2Drive sweep `f2dsteervla_kl005_seed0_qwenzs_bon_mixed` across two machines.
**Machine A** (`bellman`, GPUs 5 and 6) keeps its current queue, walking routes **first → last** and
CARLA seeds **0 → 1 → 2**. **Machine B** walks the same 48 cells in **reverse**: routes
**last → first**, seeds **2 → 1 → 0**. Both machines log to the same W&B group, and W&B is the only
state they share, so it is also how they stay off each other's cells:

- B skips any cell another host has `running`, `finished` or `crashed` on W&B
  (`SKIP_WANDB_CLAIMED=1`, checked right before each cell starts).
- A's driver was armed before this existed and pops a fixed `queue.txt`, so A runs a small
  pruner that drops from that queue every cell B has claimed.

Reversing the order only moves the clash to where the queues meet. The two mechanisms above are
what prevent it. Branch: **`bon-candidate-log`** (`catglossop/ogbench-carla`), which is the sweep
commit `856440b` plus the tooling below.

## State at handover (2026-09-21, ~15:50 on bellman)

16 of 48 cells done, 2 running, 30 queued on A. A cell takes about 1.5–2.5 h per worker.

| # | route (A's order; B starts at the bottom) | cs0 | cs1 | cs2 |
|---|---|---|---|---|
| 1 | generalization-construction-permutations-1019 | done | done | done |
| 2 | generalization-custom-obstacles-1020 | done | done | done |
| 3 | generalization-pedestrians-on-road-1085 | done | done | done |
| 4 | generalization-construction-pedestrian-1011 | done | done | done |
| 5 | generalization-pedestrian-crowd-1069 | done\* | done\* | done |
| 6 | generalization-custom-obstacles-1024 | running (A) | | |
| 7 | generalization-fully-blocked-1032 | | | done |
| 8 | generalization-hard-brake-1036 | running (A) | | |
| 9 | generalization-pedestrian-other-blocker-1072 | | | |
| 10 | generalization-right-construction-1094 (ckpt 9141) | | | |
| 11 | generalization-right-of-way-1056 | | | |
| 12 | generalization-image-on-object-1041 (ckpt 1669) | | | |
| 13 | generalization-obscured-stop-1046 (ckpt 2276) | | | |
| 14 | generalization-bad-parking-1004 | | | |
| 15 | generalization-animals-1076 | | | |
| 16 | generalization-wall-1097 | | | |

\* Summary written, but the run shows **`crashed`** on W&B: its heartbeat dropped. That is why
`crashed` counts as claimed; see Gotchas.

B's first cell is `wall-1097` cs2, then `wall-1097` cs1, and so on. Live status on A:
`cd ~/ogbench-carla/.claude/worktrees/qwen-zs-bon-sweep && tail .run_carla/jobs/f2dsteervla_kl005_seed0_qwenzs_bon_mixed/sweep.log`.

## Protocol (B must match A exactly; the launcher sets all of it)

- **Actor.** "Mixed" SteerVLA: the per-route InternVL2 HL, which is the route's end-of-training
  CAST-relabel export from `f2dsteervla_simlingo_fixedcarla_kl005_seed0`, sampled at CoT
  temperature 1.0. It drives the frozen pi05 LL `pi05_steervla_simplified_reasoning_no_ego_history_v1`
  @ 6000 (`impls/configs/steervla_mixed_eval_config.py`).
- **Best-of-N.** 4 candidates sampled sequentially, re-scored every 3 env steps, no brake candidate,
  labels = subtask.
- **Critic.** Zero-shot `Qwen/Qwen3.8-27B` with the qwen-critic README settings
  (`.run_carla/qwen_zs_critic_server.sh` enforces them via `/health`), one critic per worker.
- **Episodes.** One frozen-eval episode per cell, with model seed = CARLA seed. Episodes end only on
  leaderboard criteria (no step cap, crash-stuck cutoff disabled).
- **Simulator.** Fail2Drive CARLA 0.9.15, run through the Python 3.10 subprocess env.

## Machine B setup

Keeping the same absolute paths as bellman (`/raid/users/cglossop/...`, `/home/cglossop/...`) means
nothing needs editing. If B's paths differ, every one of them is overridable by env var;
see "If paths differ".

1. **Code.**
   `git clone git@github.com:catglossop/ogbench-carla.git && cd ogbench-carla && git checkout bon-candidate-log`,
   then build the main venv: `GIT_LFS_SKIP_SMUDGE=1 uv sync --extra all-gpu`, followed by
   `uv pip install git+https://github.com/catglossop/fail2drive.git`. Bellman's sweep venv has
   `carla 0.9.16` + `bench2drive` + `fail2drive` + JAX/openpi.
2. **CARLA 0.9.15 subprocess env** `.venv-carla-0915` (Python 3.10, `carla 0.9.15`, `fail2drive`):
   follow `ogbench/carla/README.md` → "Fail2Drive routes" §2a–2b. Check it with
   `.venv-carla-0915/bin/python -c "import carla, srunner, fail2drive"`.
3. **Fail2Drive CARLA 0.9.15 install** (~30 GB) at `/home/cglossop/f2d_carla`, with the loose-asset
   pack: `./install_f2d_content.sh /home/cglossop/f2d_carla <f2d_content_pack.zip>`. The animals
   route needs `walker.animal.*`.
4. **Qwen critic.** `git clone git@github.com:celineltan/qwen-critic.git ~/qwen-critic`, check out
   `905c918` (the commit bellman's critics run), then `uv sync`. Weights are offline from an HF
   cache holding `Qwen/Qwen3.8-27B` (~57 GB; bellman reads Celine's at
   `/raid/users/celine/qwen-critic/huggingface`): copy it and set `QWEN_HF_HOME`.
5. **SimLingo source** (~9 GB) at `/home/cglossop/simlingo-steervla`, for the InternVL2 HL worker.
6. **Checkpoints and HL deps** (~90 GB), mirrored at identical paths:
   ```bash
   # from B (or push from bellman); -R keeps the absolute layout the sweep's discovery expects.
   # -r is explicit on purpose: with --files-from, -a does NOT imply -r (directories would copy empty).
   rsync -aRr --info=progress2 --files-from=<(grep -v '^#' .run_carla/handover/f2d_assets.txt) \
     bellman:/ /
   ```
   That is 16 per-route HL exports (2.6 GB each) plus the `run_summary.json` naming each one final,
   the 46 GB pi05 LL, and `ogbench-simlingo-deps` + `hl_python.sh`.
7. **`hl_python.sh`** runs the HL worker as the *main* ogbench venv's python with
   `PYTHONPATH=/raid/users/cglossop/ogbench-simlingo-deps`. On B, edit its two paths or point
   `MIXED_HL_PYTHON` at your own copy.
8. **W&B.** Put the catherine_glossop school key at `~/.wandb_school_key`, or set `WANDB_KEY_FILE`.
   The launcher refuses to arm unless the key resolves to `catherine_glossop`. Never let it fall
   through to `~/.netrc` (`catglossop`).

## GPU layout on B

Each worker needs its own critic (~70 GB; the start script waits for **80 GB free**) plus
policy + HL worker + CARLA (~35 GB).

- **≥ 141 GB GPUs (bellman's layout):** co-locate each worker with its critic,
  `SWEEP_GPUS="0 1" QWEN_GPUS="0 1" QWEN_PORTS="18850 18851"`.
- **80 GB GPUs:** give critics their own cards,
  `SWEEP_GPUS="0 1" QWEN_GPUS="2 3" QWEN_PORTS="18850 18851"`.

More workers finish sooner. Each additional worker needs its own entry in all three lists.

## Run it

**On B**, from the repo root:

```bash
./.run_carla/handover/launch_f2d_mixed_reverse.sh --dry-run    # first job must be wall-1097 cs2
SWEEP_GPUS="0 1" QWEN_GPUS="0 1" QWEN_PORTS="18850 18851" \
  setsid nohup ./.run_carla/handover/launch_f2d_mixed_reverse.sh > sweep_b.log 2>&1 &
./.run_carla/handover/launch_f2d_mixed_reverse.sh --status
tail -f .run_carla/jobs/f2dsteervla_kl005_seed0_qwenzs_bon_mixed/sweep.log   # START / SKIP / DONE
```

**On A** (bellman), once, as soon as B is armed. The pruner lives on the `bon-candidate-log`
worktree and edits A's live queue under the sweep's own lock:

```bash
cd ~/ogbench-carla/.claude/worktrees/bon-candidate-log
INTERVAL=600 setsid nohup ./.run_carla/prune_queue_from_wandb.sh \
  ~/ogbench-carla/.claude/worktrees/qwen-zs-bon-sweep/.run_carla/jobs/f2dsteervla_kl005_seed0_qwenzs_bon_mixed/queue.txt \
  f2dsteervla_kl005_seed0_qwenzs_bon_mixed > /raid/users/cglossop/sweep_results/qwenzs_mixed_b2d_sources/prune_f2d.log 2>&1 &
```

It logs `dropped <route> csN (claimed on <B's hostname>)` for each cell B takes, and exits when A's
queue is empty.

## How it ends

Each machine only counts the *other* host's claims, so its own crashed cells stay its own retries.
B skips A's cells at start time; A's pruner drops B's cells within `INTERVAL`. The one residual
race is the two machines starting the *same* cell inside one pruner interval right at the meeting
point. That only wastes one cell and is visible on W&B as two runs of the same cell. Both drivers
exit when their queues run dry.

Then merge B's results into A's results dir and tabulate:

```bash
rsync -a B:/raid/users/cglossop/sweep_results/f2dsteervla_kl005_seed0_qwenzs_bon_mixed/ \
  /raid/users/cglossop/sweep_results/f2dsteervla_kl005_seed0_qwenzs_bon_mixed/
cd ~/ogbench-carla/.claude/worktrees/qwen-zs-bon-sweep && BENCH=f2d \
  SOURCE_SWEEP=f2dsteervla_simlingo_fixedcarla_kl005_seed0 RUN_GROUP=f2dsteervla_kl005_seed0_qwenzs_bon_mixed \
  ROUTES_FILE=/raid/users/cglossop/sweep_results/qwenzs_mixed_b2d_sources/routes_f2d.txt CARLA_SEEDS="0 1 2" \
  ./.run_carla/qwen_zs_bon_sweep.sh --results
```

The record of what finished is `run_summary_frozen_eval.json` per cell, **not** the W&B state.
Any cell still showing `-` after the merge had no summary on either machine: re-arm one machine
and it runs exactly those cells, because finished cells are skipped.

## Gotchas

- **`crashed` on W&B ≠ failed.** pedestrian-crowd-1069 cs0/cs1 are `crashed` on W&B with valid
  summaries. That is why `wandb_claimed_cells.py` treats `crashed` as claimed, so neither machine
  re-runs it; only `failed`/`killed` free a cell.
- **Early-stop checkpoints.** right-construction-1094 → 9141, image-on-object-1041 → 1669,
  obscured-stop-1046 → 2276. These are the "final" ones their `run_summary.json` names; A used them
  too.
- **Stale plan label.** The dry-run prints "pi05 LL ll_heavy_unnormed_matchcrop/6000"; the command
  it actually runs uses the no_ego_history v1 @ 6000 LL, same as A.
- **Always pass `QWEN_PORT` to `qwen_zs_critic_server.sh`.** Its `start`/`stop`/`status` default
  to port 18850, and the pid files are shared across worktrees under
  `sweep_results/qwen_zs_critic/`. A bare `stop` kills whatever critic owns 18850: on
  2026-09-21 that was A's live GPU-5 critic, and its in-flight cell died.
- **Not every GPU can host CARLA.** On bellman UE4 cannot run on GPU 7, and it died with
  `VK_ERROR_DEVICE_LOST` on GPU 0. The sweep renders CARLA on the worker's own GPU, so leave any
  such card out of `SWEEP_GPUS`; it can still hold a critic via `QWEN_GPUS`. Vulkan's adapter
  order matched `nvidia-smi` on bellman (check B with `vulkaninfo --summary`).
- **Cleanup.** Use `launch_f2d_mixed_reverse.sh --stop` (only this worktree's workers and their
  CARLA; the critics keep running) or `qwen_zs_critic_server.sh stop`. **Not** `reset_carla.sh`,
  which kills every CARLA on the box. On bellman, `qwen_zs_bon_sweep.sh --stop` also pkills slot
  ports 17400–17540, which includes the separate candidate-logging rerun on slot 2 (port 17440).

## Files (branch `bon-candidate-log`)

| file | what |
|---|---|
| `.run_carla/handover/launch_f2d_mixed_reverse.sh` | B's launcher: reversed routes, seeds `2 1 0`, W&B claim skip, identity guard |
| `.run_carla/handover/routes_f2d_reversed.txt` | `routes_f2d.txt`, last → first |
| `.run_carla/handover/f2d_assets.txt` | absolute paths to mirror onto B |
| `.run_carla/wandb_claimed_cells.py` | `GROUP --list` / `GROUP ROUTE SEED` (exit 0 claimed, 1 free, 2 W&B error) |
| `.run_carla/prune_queue_from_wandb.sh` | A-side pruner for an already-armed driver's `queue.txt` |
| `.run_carla/qwen_zs_bon_sweep.sh` | + opt-in `SKIP_WANDB_CLAIMED`, `WANDB_KEY_FILE` (default behaviour unchanged) |
| `.run_carla/qwen_zs_bon_run.sh` | + `WANDB_KEY_FILE`, `EXTRA_MAIN_FLAGS` (both default to the old behaviour) |
