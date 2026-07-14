# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [OGBench](https://github.com/seohongpark/ogbench) extended with a **CARLA Bench2Drive** Gymnasium env and an **online DSRL + SteerVLA (OpenPI Pi0-CoT)** training stack. The upstream MuJoCo benchmarks and reference agents (GCBC/GCIVL/GCIQL/QRL/CRL/HIQL) still live here but the active development surface is the CARLA driving stack.

There are two entrypoints with very different semantics:

- `impls/main.py` — original OGBench offline goal-conditioned RL on MuJoCo (unchanged from upstream).
- `impls/main_carla.py` — online RL loop against `CarlaBench2DriveWrapper`, optionally driven by a frozen SteerVLA actor. **This is what `run_carla.sh` calls.**

Don't run `pip install` — this project uses `uv`. Python is pinned to `>=3.11,<3.12` in `pyproject.toml`.

## Common commands

```bash
# Install (pick your platform)
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra all-gpu        # or --extra all-tpu
# Bench2Drive-only or Fail2Drive-only environments:
#   uv sync --extra all-b2d-gpu   (or all-b2d-tpu)
#   uv sync --extra all-f2d-gpu   (or all-f2d-tpu)
# To keep the two side-by-side in separate venv subdirectories, point uv at
# a different env path per stack (bench2drive and fail2drive ship conflicting
# srunner/scenario files, so a shared .venv will only ever hold one of them):
#   UV_PROJECT_ENVIRONMENT=ogbench-b2d uv sync --extra all-b2d-gpu
#   UV_PROJECT_ENVIRONMENT=ogbench-f2d uv sync --extra all-f2d-gpu
# Then run with the matching env, e.g. `ogbench-b2d/bin/python impls/main_carla.py ...`

# List available Bench2Drive routes
WANDB_MODE=disabled uv run python impls/main_carla.py --list_routes=true | head -20

# Online run on a single route (the canonical wrapper, handles tmp configs + GPU pinning)
./run_carla.sh --train-gpu 2 --render-adapter 4 --carla-port 12045 \
  --carla-streaming-port 12091 --tm-port 18019 --x-display-num 30

# Direct invocation (no wrapper)
uv run python impls/main_carla.py \
  --agent=impls/configs/steervla_dsrl_config.py \
  --carla_config=impls/configs/carla_config.yaml \
  --route=parking-cut-in-001 \
  --online_steps=5000 --save_buffer=true --seed=0

# Eval-only: load a SteerVLA OpenPI checkpoint without spinning CARLA
WANDB_MODE=disabled .venv/bin/python impls/main_carla.py \
  --eval_only=true \
  --steervla_checkpoint=gs://cat-logs/.../90000 \
  --steervla_actor_config=pi05_steervla_inference

# Run the SteerVLA HTTP inference server (used when steervla.actor_url is set)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  uv run python impls/vlas/steervla_server.py \
  --actor-config <openpi-config> --checkpoint <gcs-or-local-path>

# Launch SteerVLA server on a TPU VM
cd impls/vlas && ./launch_steervla.sh <tpu-vm-name> <config> <checkpoint>

# Lint (configured in pyproject.toml: ruff, line-length 120, single quotes)
uv run ruff check .
uv run ruff format .
```

There is no unit test suite (`pytest` is a dependency but no `tests/` exists). Don't claim a "single test" command — there isn't one. The one exception is `impls/coaches/test_action_chunk_feedback_integration.py`, a standalone integration script you run directly, not via a test runner.

### Recovering from a crashed run

CARLA runs leave orphaned processes (UE4 server, `main_carla`, `Xvfb`, wandb helpers) that hold GPU VRAM and ports. `./reset_carla.sh` SIGKILLs all of them and cleans stale `/tmp/.X*-lock` files. Run it before restarting after any crash. Note its caveat: zombie/`<defunct>` processes and primary/display GPUs can't be reset live — a reboot is the only fix; secondary (headless) GPUs may respond to `sudo nvidia-smi --gpu-reset -i <index>`.

## Required env vars / external state

- `CARLA_ROOT=/home/carla/carla-0-9-16` (or `CARLA_PYTHON_API_ROOT=<root>/PythonAPI/carla`) — `ogbench/carla/carla.py` prepends this to `sys.path` before importing `leaderboard`/`srunner`. Without it the env will not import.
- A packaged CARLA install at `CARLA_ROOT` (its `CarlaUE4.sh` must be runnable). You do **not** start the UE4 server yourself: `CarlaBench2DriveWrapper._setup_simulation` (`ogbench/carla/carla_utils.py`) launches its own Xvfb + `CarlaUE4.sh` per run, using `port`/`streaming_port`/`traffic_manager_port`/`x_display_num`/`gpu_rank` from `carla_config.yaml`. Any of those left as `0` is auto-assigned to a free port/display; at startup the wrapper kills only stale processes matching *its own* rpc-port/display (`carla_utils.py` `_kill_stale_carla_processes`), so distinct ports are fully isolated.
- Running several jobs at once: use `./carla_job.sh` (start/list/logs/stop/stop-all). It derives disjoint ports/display per integer `--job k` and its `stop` mirrors the wrapper's scoped kill, so it never disturbs sibling jobs — unlike `reset_carla.sh`, which SIGKILLs **all** CARLA processes on the box.
- For **Fail2Drive** routes: the loose-asset content pack must be dropped into your packaged CARLA install first via `./install_f2d_content.sh CARLA_ROOT ZIP`. It only adds `.uasset` files (animal walkers, occluder/wall props) and does not touch bench2drive features.
- `WANDB_MODE` (`online`/`offline`/`disabled`). `run_carla.sh` and `main_carla.py` both honor it.
- For OpenPI checkpoints loaded from GCS: standard `gcloud auth` / Application Default Credentials.

## Architecture: how a CARLA step actually flows

This is the part that is **not** discoverable by reading any single file. Trace through these layers in order:

1. **`ogbench/carla/carla.py` :: `CarlaBench2DriveWrapper`** — a Gymnasium env wrapping the CARLA leaderboard. The leaderboard's "autoagent" (`ogbench/carla/leaderboard_agents/observation_only.py`) only **registers sensors** and stashes the latest sensor dict; the gym wrapper drives the ego vehicle directly via `step(action)`. Observation is a `Dict({"state": float32, "image": uint8, "image_viz": uint8})`. Action space is normally `Box(-1, 1, shape=(2,))` as `[accel, steer]`, but flips to a flattened OpenPI chunk `(action_horizon * action_dim,)` when `carla_config["steervla_action_execution"]` is set (see `_steervla_action_execution_cfg` in `main_carla.py`). Chunk controls are converted to throttle/steer the same way as `simlingo/team_code/agent_steervla.py` (cumsums + PID, see `ogbench/carla/steervla_simlingo_control.py`).

2. **`impls/vlas/steervla.py` :: `SteerVLAActor`** — wraps OpenPI Pi0-CoT inference. Two modes:
   - Local checkpoint: restores OpenPI weights once via `init_train_state` and shares them across `sample_cot` / `sample_actions`.
   - Remote: HTTP client (`impls/vlas/utils.py :: RemoteActor`) talks to `impls/vlas/steervla_server.py` (FastAPI). Use this when `steervla.actor_url` is set.
   The actor produces a `(action_horizon, action_dim)` chunk. CoT reasoning can be **reused** across multiple env steps (`actions_per_cot`) and chunks can be **executed open-loop** for several rows before re-querying (`actions_per_model_query`) — these are the two speed knobs called out in `ogbench/carla/README.md`.

3. **`impls/jax_agents/dsrl.py` :: `DSRLAgent`** — Diffusion Steering RL (arXiv:2506.15799). A SAC-style stochastic *noise* actor seeds a BC flow that produces env actions. Two relevant entry points:
   - `sample_actions_with_vla(obs, noise, vla_sample_fn)` — used at rollout time; substitutes the SteerVLA chunk-producing function for the internal BC flow.
   - `update_with_vla(...)` — joint VLA + DSRL training path (requires OpenPI `vla_train_state` and an `openpi_train_config`). Action-expert-only fine-tuning otherwise belongs in OpenPI's `TrainConfig.freeze_filter` — **not** inside DSRL's Flax losses.
   The DSRL critic optionally consumes a **language label** concatenated onto the encoded observation; the label width is set by `coaches/critic_feedback.py :: critic_language_dim`.

4. **`impls/coaches/`** — produces language / delta labels for the critic. Modes (resolved in `coaches/critic_feedback.py :: resolve_critic_feedback_mode`):
   - `commentary_bow` — BOW of expert commentary from the SimLingo-style labeler (`coaches/expert_label.py`, `coaches/simlingo/`).
   - `delta_commentary_bow` — BOW of corrective language derived from expert-vs-agent action delta ("Adjust right. Decelerate more heavily.").
   - `action_delta` — numeric `(critic_action_dim,)` vector = expert first-step minus agent first-step.
   - `vlm_chunk_bow` — VLM (Gemini/Perceptron) feedback from `coaches/online_vlm_coach.py`.
   - `none` — disable; `language_label_dim` is forced to 0.
   `main_carla.py` auto-sets `agent_config.language_label_dim` from this resolution; don't hand-set it.

5. **`impls/main_carla.py`** — orchestrates: builds env → builds `SteerVLAActor` (if `steervla.enabled`) → wires `vla_sample_fn` into the DSRL agent → runs the online loop, computing critic labels per step and pushing to a `utils.datasets.ReplayBuffer`. Optionally serves a live MP4 viewer at `http://<host>:<port>/` (`utils/live_policy_viewer.py`).

### Two-GPU split (gotcha)

CARLA's UE4 renderer and JAX run on **different GPUs**, and the rank conventions are not the same:

- `gpu_rank` in `impls/configs/carla_config.yaml` → passed as CARLA `-graphicsadapter`. CARLA's adapter ordering is **swapped** relative to `nvtop`/`nvidia-smi` — see the inline comment in `carla_config.yaml`.
- `training_gpu_rank` in the agent config → pins JAX's default device via `jax.config.update("jax_default_device", devs[rank])`. Set to `-1` to leave alone.

`run_carla.sh` exposes both as `--render-adapter` (alias `--sim-gpu`) and `--train-gpu`, generates a temp `carla_config.yaml` + temp agent config under `.run_carla/` for the run, and cleans them up on exit. When debugging multi-run setups, look at the printed `[run_carla.sh]` summary lines — they show every resolved port and GPU rank.

## Config layering for `run_carla.sh`

`run_carla.sh` doesn't edit your configs in place. It writes two temp files under `.run_carla/run.XXXXXX/`:

- `agent_config.py` — `runpy`-loads `impls/configs/steervla_dsrl_config.py` (the default; other experiment variants live alongside it — `steervla_best_of_n_config.py`, `steervla_cast_relabel_config.py`, `steervla_dsrl_config_no_subtask_attention.py` — selected via `--agent=`), then overrides `training_gpu_rank`, `critic_feedback_mode`, and `online_training_mode` from CLI flags. So `--critic-mode {none|delta|delta-lang|expert-lang}` and `--train-mode {rl|dagger}` are the canonical user-facing knobs; **don't edit `steervla_dsrl_config.py` to change them per-run**.
- `carla_config.yaml` — loads the base yaml and overrides `host`/`port`/`streaming_port`/`traffic_manager_port`/`gpu_rank`/`x_display_num`. `use_cuda_visible_devices` is forced to `False` because the script is already managing GPU pinning.

## Custom Python packages

Two non-PyPI packages are pulled in via the `carla` extra (see `pyproject.toml`):

- `bench2drive @ git+https://github.com/catglossop/Bench2Drive.git` — the leaderboard / scenario runner. **Use the catglossop fork**, not upstream.
- `openpi @ git+https://github.com/catglossop/steervla-pi.git` — SteerVLA Pi0-CoT model code (`openpi.models.pi0_cot`, `openpi.policies.steervla_policy`, etc.). Again, the catglossop fork.

When changing things like routing-command prompt format or normalization, the source of truth often lives in those repos, not here.

## File-organization conventions worth knowing

- `impls/jax_agents/` is named `jax_agents/` (not `agents/`) deliberately, because CARLA's `PythonAPI/carla/agents` (navigation agents) takes the `agents` import name. Don't rename it.
- Coach artifacts (`metadata/`, `results/`, `videos/`) under `impls/coaches/` are run outputs — they are not source. Don't grep them for examples of "how things work."
- `data/` holds Bench2Drive route XMLs and weather definitions; it is not training data.
- `exp/` is the local experiment / wandb scratch dir (per `.gitignore`).
- The `simulation_results.json` and `debug_checkpoint` paths in `carla_config.yaml` are written by the leaderboard subprocess at the repo root.
