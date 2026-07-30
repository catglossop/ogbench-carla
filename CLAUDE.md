# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [OGBench](https://github.com/seohongpark/ogbench) extended with a **CARLA Bench2Drive** Gymnasium env and two online driving-RL stacks. The upstream MuJoCo benchmarks and reference agents (GCBC/GCIVL/GCIQL/QRL/CRL/HIQL) still live here but the active development surface is CARLA.

There are three entrypoints with very different semantics:

- `impls/main.py` — original OGBench offline goal-conditioned RL on MuJoCo (unchanged from upstream).
- `impls/main_carla.py` — online RL loop against `CarlaBench2DriveWrapper`, driven by a **SteerVLA (OpenPI Pi0-CoT)** actor. **This is what `run_carla.sh` calls** and is where nearly all active work happens.
- `impls/main_carla_simlingo.py` — a separate **SimLingo residual-SAC** stack that runs under a Python 3.8 conda env and talks to CARLA through a subprocess. See "The SimLingo residual-SAC stack" below. `run_simlingo.sh` calls this.

Don't run `pip install` — this project uses `uv`. Python is pinned to `>=3.11,<3.12` in `pyproject.toml` (the SimLingo stack is the sole exception; it deliberately runs on a foreign 3.8 interpreter).

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
- Running several jobs at once: use `./carla_job.sh` (start/list/logs/stop/stop-all). Every port and display is derived deterministically from the integer `--job k` (`rpc = 12000 + 100k`, `streaming = rpc + 1`, `tm = 18000 + 100k`, `display = 30 + k`), so indices can't collide. GPUs are *not* derived — always pass `--train-gpu`/`--render-adapter`. Its `stop` mirrors the wrapper's scoped kill, so it never disturbs sibling jobs — unlike `reset_carla.sh`, which SIGKILLs **all** CARLA processes on the box.
- For **Fail2Drive** routes: `pip install git+https://github.com/catglossop/fail2drive.git` (route XMLs + scenario classes) and drop the loose-asset content pack into your packaged CARLA install via `./install_f2d_content.sh CARLA_ROOT ZIP`. It only adds `.uasset` files (animal walkers, occluder/wall props) and does not touch bench2drive features. Fail2Drive routes accept three name forms: `f2d:<id>`, kebab (`base-pedestrians-on-road-0085`), or original filename (`Base_PedestriansOnRoad_0085`) — see `ogbench/carla/route_registry.py` and `fail2drive_compat.py`.
- `WANDB_MODE` (`online`/`offline`/`disabled`). `run_carla.sh` and `main_carla.py` both honor it.
- For OpenPI checkpoints loaded from GCS: standard `gcloud auth` / Application Default Credentials.

## Architecture: how a CARLA step actually flows

This is the part that is **not** discoverable by reading any single file. Trace through these layers in order:

1. **`ogbench/carla/carla.py` :: `CarlaBench2DriveWrapper`** — a Gymnasium env wrapping the CARLA leaderboard. The leaderboard's "autoagent" (`ogbench/carla/leaderboard_agents/observation_only.py`) only **registers sensors** and stashes the latest sensor dict; the gym wrapper drives the ego vehicle directly via `step(action)`. Observation is a `Dict({"state": float32, "image": uint8, "image_viz": uint8})`. Action space is normally `Box(-1, 1, shape=(2,))` as `[accel, steer]`, but flips to a flattened OpenPI chunk `(action_horizon * action_dim,)` when `carla_config["steervla_action_execution"]` is set (see `_steervla_action_execution_cfg` in `main_carla.py`). Chunk controls are converted to throttle/steer the same way as `simlingo/team_code/agent_steervla.py` (cumsums + PID, see `ogbench/carla/steervla_simlingo_control.py` and `steervla_pid_utils.py`).

2. **`impls/vlas/steervla.py` :: `SteerVLAActor`** — wraps OpenPI Pi0-CoT inference (the single largest file in the repo, ~3.4k lines). Three axes to keep straight:
   - **Local vs. remote**: local restores OpenPI weights once and shares them across `sample_cot` / `sample_actions`; remote is an HTTP client (`impls/vlas/utils.py :: RemoteActor`) talking to `impls/vlas/steervla_server.py` (FastAPI), used when `steervla.actor_url` is set.
   - **Inference-only vs. trainable**: `steervla.load_trainable_params=True` restores the full OpenPI `TrainState` (optimizer + opt_state + freeze/trainable filters, like OpenPI's `scripts/train.py`) instead of bare params. Required for the CAST-relabel HL update path; costs a lot more VRAM.
   - **Speed knobs**: CoT reasoning can be **reused** across env steps (`actions_per_cot`) and chunks executed **open-loop** for several rows before re-querying (`actions_per_model_query`) — see `ogbench/carla/README.md`.
   The actor produces a `(action_horizon, action_dim)` chunk; `sample_candidates` produces N of them for Best-of-N.

3. **`impls/jax_agents/`** — RL agents, selected by `config.agent_name` against the `agents` dict in `jax_agents/__init__.py`. Two matter for CARLA:
   - **`dsrl.py` :: `DSRLAgent`** — Diffusion Steering RL (arXiv:2506.15799). A SAC-style stochastic *noise* actor seeds a BC flow that produces env actions. Entry points: `sample_actions_with_vla(obs, noise, vla_sample_fn)` at rollout time (substitutes the SteerVLA chunk-producing function for the internal BC flow), `sample_actions_dagger` / `update_dagger` for the DAgger regime, and `update_with_vla(...)` for the joint VLA + DSRL training path (requires an OpenPI `vla_train_state` and `openpi_train_config`; `run_hl=True` additionally runs the high-level VLM-backbone step). Action-expert-only fine-tuning belongs in OpenPI's `TrainConfig.freeze_filter` — **not** inside DSRL's Flax losses.
   - **`best_of_n.py` :: `BestOfNAgent`** — no noise actor. Each env step samples `best_of_n` CoTs at `vla_cot_temperature` in one batched forward, scores each candidate chunk with the critic, and executes `argmax_i Q([obs_e; subtask_i], action_i)`. Candidate subtasks are embedded with SigLIP and stored as the critic language label (`critic_feedback_mode="subtask_siglip"`). `main_carla` injects `steervla_actor` and the shared `siglip_encoder` into `BestOfNAgent.create` automatically.
   The DSRL/BoN critic optionally consumes a **language label** concatenated onto the encoded observation; its width comes from `coaches/critic_feedback.py :: critic_language_dim`.

4. **`impls/coaches/`** — produces the critic's language / delta labels, and (for CAST) relabeled subtasks. See the two sections below.

5. **`impls/main_carla.py`** (~1.9k lines) — orchestrates everything: builds env → builds `SteerVLAActor` (if `steervla.enabled`) → wires `vla_sample_fn` into the agent → runs the online loop, computing critic labels per step and pushing to a `utils.datasets.ReplayBuffer`, optionally driving a `OnlineCastRelabelSession`. Optionally serves a live MP4 viewer at `http://<host>:<port>/` (`utils/live_policy_viewer.py`). It inserts `impls/` on `sys.path`, so intra-`impls` imports are top-level (`from coaches...`, `from vlas...`) — but note it *also* imports `from impls.coaches.cast_relabel import ...` in one place, so both spellings resolve; keep new imports in the top-level style.

### Critic feedback modes (read `coaches/critic_feedback.py` before touching)

The mode string is **resolved**, not read directly, by `resolve_critic_feedback_mode(agent_config)`, in this precedence order:

1. If `critic_pretrained_weights` is set → forced to `"none"` (a pretrained critic was trained without a language label; this must stay in sync with `BestOfNAgent.create`, which forces the same).
2. Else if `config.language_feedback` exists → `source="vlm"` gives `vlm_chunk_bow`; `source="expert"` uses `language_feedback.expert_mode`.
3. Else the legacy flat `config.critic_feedback_mode` field.

So **adding a `language_feedback` block silently disables a flat `critic_feedback_mode`** — that's why `steervla_best_of_n_config.py` deliberately has no `language_feedback` block.

Modes: `commentary_bow` (BOW of expert commentary from the SimLingo-style labeler, `coaches/expert_label.py` + `coaches/simlingo/`) · `delta_commentary_bow` (BOW of corrective language from the expert-vs-agent action delta, "Adjust right. Decelerate more heavily.") · `action_delta` (numeric `(critic_action_dim,)` = expert first step minus agent first step) · `vlm_chunk_bow` (Gemini/Perceptron feedback from `coaches/online_vlm_coach.py`) · `subtask_siglip` (SigLIP embedding of the executed Best-of-N candidate's subtask) · `none`.

`main_carla.py` auto-sets `agent_config.language_label_dim` from this resolution; **don't hand-set it.**

### CAST relabel (`impls/coaches/cast_relabel.py`)

The newest pipeline and the reason the `cast_relabel` branch exists. It reuses the chunking machinery of `coaches/action_chunk_feedback.py` but changes the output from corrective steering commentary to **suggested subtasks**, with an explicit credit-assignment step:

1. Roll out for a window of env steps (rounded to whole action chunks).
2. A VLM reviews the *whole window video* and returns timestamped GOOD/BAD events.
3. A second VLM call maps those events onto specific chunks. Credit is **causal, not temporal**: a chunk is BAD either because an event overlaps it (`credit_source="direct"`) or because it's part of the lead-up (`credit_source="precursor"`).
4. Per chunk, the VLM suggests subtasks (open-vocab, seeded by `SEED_SUBTASKS`) plus a fresh CoT reasoning trace.

Outputs go to `cast_relabel.json` + annotated W&B debug videos, and — when `cast_relabel.store_hl_dataset` is set — to a **SteerVLA high-level dataset** in OpenPI's `steervla_hl_dataset_format` schema (image + ego state + prompt + corrected subtask + new reasoning + the executed chunk with `action_loss_mask` all-`False`, so only the CoT/VLM backbone is supervised). GOOD/unlabeled chunks are stored too when `store_good_chunks` is set, but with the model's *original* subtask (reinforcing rather than correcting). Those samples are consumed online by `SteerVLAActor.update_hl`, driven from `DSRLAgent.update_with_vla(..., run_hl=True)` and gated by `enable_updates_bc_hl`.

Configs: `steervla_cast_relabel_config.py` (observer only — writes artifacts, doesn't train) vs. `steervla_cast_relabel_train_config.py` (inherits it, then flips on `load_trainable_params` + `store_hl_dataset` + the `hl_update_*` cadence knobs).

### Update gating (why "nothing is training")

`steervla_dsrl_config.py` has a master switch plus three per-kind switches, each ANDed with the master:

- `enable_updates` — master; `False` = rollout-only, no gradients at all.
- `enable_updates_rl` — DSRL critic/actor RL updates.
- `enable_updates_bc` — the full BC / DAgger imitation path (`update_dagger`).
- `enable_updates_bc_hl` — the high-level VLM-backbone update (`update_hl` on CAST data).

Also relevant: `warmup_steps` (collect with the rollout policy, no updates), `update_interval` (run updates every N env steps), `updates_per_step`, and `debug_task` (**RL updates use `reward = -ego_speed` instead of env reward** — this is on by default in the DSRL config, so check it before interpreting a reward curve).

### Two-GPU split (gotcha)

CARLA's UE4 renderer and JAX run on **different GPUs**, and the rank conventions are not the same:

- `gpu_rank` in `impls/configs/carla_config.yaml` → passed as CARLA `-graphicsadapter`. CARLA's adapter ordering is **swapped** relative to `nvtop`/`nvidia-smi` — see the inline comment in `carla_config.yaml`.
- `training_gpu_rank` in the agent config → pins JAX's default device via `jax.config.update("jax_default_device", devs[rank])`. Set to `-1` to leave alone.
- `steervla.hl_training_gpu_rank` → optionally isolates the HL (VLM-backbone) update on its own JAX GPU.

`run_carla.sh` exposes these as `--render-adapter` (alias `--sim-gpu`), `--train-gpu`, and `--hl-gpu`. When debugging multi-run setups, look at the printed `[run_carla.sh]` summary lines — they show every resolved port and GPU rank.

## Config layering for `run_carla.sh`

`run_carla.sh` doesn't edit your configs in place. It writes two temp files under `.run_carla/run.XXXXXX/`:

- `agent_config.py` — `runpy`-loads the base agent config (`--agent-config`, default `impls/configs/steervla_dsrl_config.py`; variants alongside it: `steervla_best_of_n_config.py`, `steervla_cast_relabel_config.py`, `steervla_cast_relabel_train_config.py`, `steervla_dsrl_config_no_subtask_attention.py`), then overrides `training_gpu_rank`, `hl_training_gpu_rank`, `critic_feedback_mode`, and `online_training_mode` from CLI flags. So `--critic-mode {none|delta|delta-lang|expert-lang}` and `--train-mode {rl|dagger}` are the canonical user-facing knobs; **don't edit `steervla_dsrl_config.py` to change them per-run.**
- `carla_config.yaml` — loads the base yaml and overrides `host`/`port`/`streaming_port`/`traffic_manager_port`/`gpu_rank`/`x_display_num`. `use_cuda_visible_devices` is forced to `False` because the script is already managing GPU pinning.

## The SimLingo residual-SAC stack (separate, cross-interpreter)

`impls/main_carla_simlingo.py` is **not** part of the SteerVLA/DSRL path and does not share its process. It runs under a Python 3.8 **conda** env (`simlingo`), because SimLingo's deps are incompatible with this repo's 3.11 pin. CARLA therefore lives in a *different* process:

- `impls/carla_env_server.py` runs in the **uv 3.11 env**, owns `CarlaBench2DriveWrapper`, and speaks newline-delimited JSON over its own stdin/stdout. It dups fd 1 to a saved wire fd and redirects fd 1 → stderr at the OS level, so any stray `print()` or C-extension write from CARLA/UE4 can't corrupt the protocol. Don't add anything that writes to real stdout there.
- `main_carla_simlingo.py` (3.8) launches that server as a subprocess, runs the frozen SimLingo VLM to get `base_action` (PID over predicted waypoints) plus `vlm_features` (896,), and trains a small PyTorch residual actor/critic (`impls/torch_agents/residual_sac.py`): `final_action = clip(base_action + res_scale * residual_action, -1, 1)`.
- Launchers: `run_simlingo.sh` (Bench2Drive) and `run_simlingo_fail2drive.sh` (Fail2Drive). Both take `--instance N` for parallel runs; `run_simlingo.sh`'s header carries the cross-policy route shortlist used to pick RL routes.
- `simlingo-rebuttal/` at the repo root is expected on `sys.path` by `carla_env_server.py` but is **not** a tracked submodule and is empty in a fresh clone — the SimLingo path won't run until it's populated.

## Custom Python packages

Two non-PyPI packages are pulled in via the `carla` extra (see `pyproject.toml`):

- `bench2drive @ git+https://github.com/catglossop/Bench2Drive.git` — the leaderboard / scenario runner. **Use the catglossop fork**, not upstream.
- `openpi @ git+https://github.com/catglossop/steervla-pi.git` — SteerVLA Pi0-CoT model code (`openpi.models.pi0_cot`, `openpi.policies.steervla_policy`, `openpi.training.steervla_rlds_dataset`, etc.). Again, the catglossop fork.

When changing things like routing-command prompt format, normalization, the HL dataset schema, or freeze filters, the source of truth often lives in those repos, not here.

## File-organization conventions worth knowing

- `impls/jax_agents/` is named `jax_agents/` (not `agents/`) deliberately, because CARLA's `PythonAPI/carla/agents` (navigation agents) takes the `agents` import name. Don't rename it.
- Coach artifacts (`metadata/`, `results/`, `videos/`) under `impls/coaches/` are run outputs — they are not source. Don't grep them for examples of "how things work."
- `data/` holds Bench2Drive route XMLs and weather definitions; it is not training data.
- `exp/` is the local experiment / wandb scratch dir (per `.gitignore`); `.run_carla/` holds generated temp configs and `carla_job.sh` bookkeeping.
- `simulation_results.json` and `debug_checkpoint` at the repo root are written by the leaderboard subprocess.
- `README_celine.md` and `scratch_check_freeze.py` are personal scratch notes/snippets, not documentation.
