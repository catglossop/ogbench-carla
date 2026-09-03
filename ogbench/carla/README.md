# CARLA OGBench Instructions

## Installation

First, you will need to clone the repo and install the environment

```
git clone https://github.com/catglossop/ogbench-carla.git
cd ogbench-carla
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra all-<gpu, tpu> # choose your platform
```

You must also install CARLA. We follow most of the same steps from the Bench2Drive: 

```
mkdir carla
cd carla
wget https://tiny.carla.org/carla-0-9-16-linux
tar -xvf carla-0-9-16-linux
cd Import && wget https://tiny.carla.org/additional-maps-0-9-16-linux
cd .. && bash ImportAssets.sh
```

For ease of use, you can directly add the `CARLA_ROOT` to your `.bashrc`

```
vim ~/.bashrc
export CARLA_ROOT=<your carla path>
soumainrce ~/.bashrc
```



## Quickstart

To start running eval, you can run: 

```
WANDB_MODE=disabled .venv/bin/python impls/main_carla.py \
  --eval_only=true \
  --steervla_checkpoint=gs://cat-logs/pi05_steervla_cot_ki/pi05_steervla_cot_ki/90000 \
  --steervla_actor_config=pi05_steervla_inference
```

### Frozen Qwen critic best-of-N

`steervla_dsrl_config.py` defaults to the 14000-step commentary policy checkpoint on
GCS. On machines with a RAID mirror, pass the local checkpoint explicitly when
launching CARLA to avoid another large download. Start the critic from the
`qwen-critic` repository on one GPU:

```
CUDA_VISIBLE_DEVICES=<qwen-gpu> \
  .venv/bin/python scripts/serve_bon.py \
  --adapter /raid/users/celine/qwen-critic/<run>/step-<best-step>/adapter \
  --port 18784 --traffic-weight 0 --traffic-threshold 1.01
```

Then run CARLA on a separate GPU:

```
./run_carla.sh \
  --agent-config impls/configs/steervla_dsrl_config.py \
  --steervla-checkpoint /raid/users/celine/openpi/checkpoints/commentary-steervla-14000 \
  --route generalization-wall-1097 \
  --train-mode rl --eval-only false --enable-updates false \
  --bon-online-critic false --bon-qwen-select true \
  --qwen-bon-url http://127.0.0.1:18784 \
  --bon-num-candidates 8 --bon-max-sample-attempts 1 \
  --bon-qwen-cadence 5 --max-episodes 1 \
  --bon-candidates-wandb false --save-video-local true \
  --train-gpu <carla-gpu> --render-adapter <carla-gpu> -- \
  --bon_include_brake_candidate=true
```

Each candidate is a complete 10-action chunk; `--bon-qwen-cadence` controls how much
of the selected chunk is rolled out before resampling. Candidate images remain local
when `--bon-candidates-wandb false`, while episode videos are still logged. Keep
`--enable-updates false`, `--bon-online-critic false`, and omit `--online-train` from
the Qwen server for frozen offline-critic evaluation.



## Configuring your env

This repo is built to work with [openpi](https://github.com/catglossop/steervla-pi.git) and [bench2drive](https://github.com/catglossop/Bench2Drive.git)

We provide the ability to treat each Bench2Drive route as a task in the CARLA environemtn or run the entire benchmark. 

To list the available routes or look for a specific kind of route:

```
WANDB_MODE=disabled uv run python impls/main_carla.py --list_routes=true | head -20
WANDB_MODE=disabled uv run python impls/main_carla.py --list_routes=true | grep parking
```



## Machine-specific settings (change these first)

Most defaults in this repo were written for one of two boxes and **will not work
unmodified anywhere else**. Work through this list before your first run — almost every
"it won't start" report traces back to something here.

### 1. Paths — the ones that will break a fresh clone


| What                    | Where                                                                                                                              | Current default                                                     | Notes                                                                                                                                                                                                                    |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `CARLA_ROOT`            | env var; fallback in `ogbench/carla/carla.py`, `impls/carla_env_server.py`                                                         | `/home/carla/carla-0-9-16`, or `/home/celinet/VLA_driving/software` | **Required.** `carla.py` prepends it to `sys.path` before importing `leaderboard`/`srunner`; without it the env will not import. `CARLA_PYTHON_API_ROOT` overrides just the PythonAPI path.                              |
| `--save_dir`            | `impls/main_carla.py`                                                                                                              | `/home/carla/exps`                                                  | Where checkpoints, videos and replay buffers land. Pass `--save_dir` or edit the flag.                                                                                                                                   |
| SteerVLA checkpoints    | `impls/configs/steervla_*.py`                                                                                                      | mixture of `gs://cat-logs/...` and `/home/carla/.cache/openpi/...`  | **Prefer the** `gs://` **form.** OpenPI caches GCS downloads under `~/.cache/openpi`, so the local mirrors are just a machine-specific copy of the same checkpoint.                                                      |
| `FAIL2DRIVE_CARLA_ROOT` | env var / `--fail2drive-carla-root`                                                                                                | unset                                                               | Only needed if Fail2Drive routes must use a *different* CARLA install **of the same version**. Unset = stay on `CARLA_ROOT`. See `install_f2d_content.sh`.                                                               |
| `CARLA_0915_ROOT`       | env var, read by `run_carla.sh`                                                                                                    | unset                                                               | Points the env at a CARLA **0.9.15** install and runs it in a Python 3.10 subprocess (`.venv-carla-0915`, overridable via `CARLA_0915_PYTHON`). Required for `generalization-animals-`* — see "Fail2Drive routes" below. |
| SimLingo paths          | `run_simlingo*.sh`, `impls/main_carla_simlingo.py`, `impls/vlas/simlingo_base.py`, `impls/configs/simlingo_residual_sac_config.py` | `/home/celinet/...`, `/scratch/current/celinet/...`                 | The whole SimLingo stack (checkpoints, source roots, and the py3.8 conda interpreter `/home/celinet/miniconda3/envs/simlingo/bin/python`) is hardcoded.                                                                  |
| Critic pretrain output  | `impls/pretrain_critic.py`, `impls/pretrain_critic_simlingo.py`                                                                    | `/scratch/current/celinet/critic_pretrain*`                         | `--checkpoint_dir`.                                                                                                                                                                                                      |
| TPU launch user         | `impls/vlas/launch_steervla.sh`                                                                                                    | `USER=carla`, `ZONE=us-central2-b`                                  | Change both for your GCP setup.                                                                                                                                                                                          |




### 2. GPUs — two different rank conventions

CARLA's UE4 renderer and JAX run on **different GPUs**, and the numbering is not the same:


| Knob                            | Where                                           | Meaning                                                                                                                                                                                               |
| ------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gpu_rank`                      | `impls/configs/carla_config.yaml` (default `1`) | Passed as CARLA `-graphicsadapter`. **CARLA's adapter ordering is swapped relative to** `nvtop`**/**`nvidia-smi` — see the inline comment in the yaml. `run_carla.sh --render-adapter` / `--sim-gpu`. |
| `training_gpu_rank`             | agent config                                    | Pins JAX's default device. `-1` leaves it alone. `run_carla.sh --train-gpu`.                                                                                                                          |
| `steervla.hl_training_gpu_rank` | agent config                                    | Optional separate GPU for the high-level (VLM backbone) update. `run_carla.sh --hl-gpu`.                                                                                                              |
| `siglip_device`                 | agent config (e.g. `"cuda:0"`)                  | Torch device for the SigLIP encoder — a *torch* index, unrelated to the two above.                                                                                                                    |


`run_carla.sh` prints every resolved GPU rank and port in its `[run_carla.sh]` summary
lines; check those first when a multi-run setup misbehaves.

### 3. Ports and displays

`carla_config.yaml` ships with `port`, `streaming_port`, `traffic_manager_port` and
`x_display_num` all set to `0`, which means **auto-assign a free one at startup**
(`carla_utils._setup_simulation`). That is the right default for concurrent runs — the
wrapper only kills stale processes matching *its own* rpc-port/display, so distinct ports
are fully isolated. Pin explicit values only when you deliberately want a fixed port.

You do **not** start the UE4 server yourself: the wrapper launches its own Xvfb +
`CarlaUE4.sh` per run.

- `run_carla.sh` defaults: RPC `2020`, TM `8020`, train GPU `2`, render adapter `3`.
- `carla_job.sh` derives everything from a single `--job k` index, so parallel jobs can't
collide: `rpc = 12000 + 100k`, `streaming = rpc + 1`, `tm = 18000 + 100k`,
`display = 30 + k`. Override the bases with `CARLA_RPC_BASE` / `CARLA_TM_BASE` if they
clash with other tenants. GPUs are **not** derived — always pass `--train-gpu` /
`--render-adapter`.
- `--live_policy_view` serves an MP4 viewer on `--live_policy_port`; pick a free port per run.



### 4. Credentials and other env vars


| Variable                      | Needed for                                                                                                                                                                                                           |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GEMINI_API_KEY`              | VLM coaching and CAST relabel — read by `coaches/vlm_feedback.py`, `coaches/cast_relabel.py`, `coaches/static_coach.py`, `main_carla_simlingo.py`                                                                    |
| `gcloud` ADC                  | any `gs://` OpenPI checkpoint                                                                                                                                                                                        |
| `WANDB_MODE`                  | `online` / `offline` / `disabled`                                                                                                                                                                                    |
| `VK_ICD_FILENAMES`            | NVIDIA Vulkan ICD. The default differs by distro — `/usr/share/vulkan/icd.d/nvidia_icd.json` vs `/etc/vulkan/icd.d/nvidia_icd.json`. If CARLA boots on the wrong GPU or fails to boot headless, set this explicitly. |
| `XDG_RUNTIME_DIR`             | CARLA 0.9.16+ requires it; the wrapper synthesizes `/run/user/$UID` (falling back to `/tmp`) when it's unset.                                                                                                        |
| `SCENARIO_RUNNER_ROOT`        | leaderboard/scenario_runner imports                                                                                                                                                                                  |
| `STEERVLA_ENABLE_OPENPI_NORM` | set to `1` only to A/B the OpenPI `Normalize`/`Unnormalize` transforms, which are **off by default on purpose** (see `impls/vlas/steervla.py`)                                                                       |




### 5. Not machine-specific, but check before a real run

- `debug_task` in the agent config replaces the env reward with `-ego_speed` (i.e. trains
the car to *stop*). Some configs ship with it `True`.
- `enable_updates` (and `enable_updates_rl` / `_bc` / `_bc_hl`) gate all gradient updates.
`False` = rollout only. This is the usual answer to "why is nothing training?"



## Configs

There are a couple configs to be aware of: 

### Agent configs

Under `impls/configs`, agent configs can be found. Here, you can specify any arguments related to your RL agent (including SteerVLA configs)

You can also select the kind of observation you want to use ("image" or "state" - note that image generation is slow so the sim runs at about 1/3 of real time)

To use a remote actor, first on your server workstation (or TPU, whatever)

```
XLA_PYTHON_CLIENT_MEM_FRACTION="0.95" python impls/vlas/steervla_server.py --actor-config <config_name_from_steervla-pi> --checkpoint <gcs_checkpoint_path>
```

To launch on a TPU, you can use:

```
cd ~/ogbench-carla/impls/vlas
./launch_steervla.sh <config_name> <checkpoint_path>
```

Change your user name in line 10 of `launch_steervla.sh`. 
 To get the IP of a TPU, find the external IP on the TPU

Set the `actor_url` in the steervla config (see `impls/configs/steervla_dsrl_config.py` for an example)

### CARLA config

The carla config is located in `impls/config/carla_config.yaml`. This can be used to set the port for the sim (if using a remote sim), the timeout for the sim etc. 

### Reward settings

The per-step reward is assembled in `CarlaBench2DriveWrapper._compute_reward_and_info`
(`ogbench/carla/carla_utils.py`). Every weight has a `DEFAULT_*` constant there and can be
overridden by the **same key, lowercased, in** `carla_config.yaml` — e.g.
`DEFAULT_PROGRESS_REWARD_WEIGHT` ← `progress_reward_weight`.

The dense term is:

```python
route_progress_delta = max(0.0, route_progress_pct - prev_route_progress_pct)   # 0..100
progress_reward      = progress_reward_weight * route_progress_delta / 100.0
```


| Key (yaml)                    | Default                            | Effect                                                                                                    |
| ----------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `progress_reward_weight`      | **5.0**                            | Dense route progress; the dominant positive term. Total over a fully completed route ≈ the weight itself. |
| `collision_event_penalty`     | −20.0                              | Applied while contact is active (`collision_contact_penalty` splits this if set).                         |
| `outside_route_event_penalty` | −20.0                              | Leaving the route corridor.                                                                               |
| `traffic_violation_penalty`   | −20.0                              | Per new RunningStop / RunningRedLight event.                                                              |
| `crash_stuck_penalty`         | −5.0 (yaml) / −20.0 (code default) | Fired after `crash_stuck_steps` ticks below `crash_stuck_speed_threshold`.                                |
| `speed_limit_penalty_weight`  | 0.1                                | `weight * max(0, speed/limit − 1)`.                                                                       |
| `steer_penalty_weight`        | 0.0                                | Smoothness penalty; off by default.                                                                       |
| `brake_penalty_weight`        | 0.0                                | Smoothness penalty; off by default.                                                                       |


**Read this before changing** `progress_reward_weight`**.** The branches disagreed on both the
weight *and* the formula, and the two changes live ~1700 lines apart so they merge without
a conflict:

- the older formula was `weight * route_completion_delta * centering_factor * heading_factor`
(no `/100`), and `routing-commands` raised the weight to `10.0` to suit it;
- `master` rewrote the formula to the `/100`-normalized version above but kept `5.0`.

Merging the two naively yields the new normalized formula running at the old branch's
doubled weight, silently. It is currently **pinned to** `5.0` with a comment at the
constant. If you change it, change it deliberately, and remember that any critic
checkpoint or reward curve produced at a different scale is no longer comparable.

Two more things that change what the reward means:

- `debug_task` (agent config) replaces the env reward entirely with `-ego_speed` for RL
updates. The env reward is still logged, so a reward curve can look normal while the
agent is being trained to stop.
- `max_episode_steps` (yaml, default `4000`) terminates the route via
`_apply_episode_max_steps` with `termination_reason="episode_max_steps"` and **no**
terminal bonus, so long episodes end neutrally rather than being penalized.

Reward components are logged per step under `reward/*` and, at episode end, under
`rollout/final_step_*`.

## Run an experiment

```
WANDB_MODE=online uv run python impls/main_carla.py \
  --agent=impls/configs/steervla_dsrl_config.py \
  --route=parking-cut-in-001 \
  --online_steps=5000 \
  --save_buffer=true \
  --seed=0
```

If desired, increase the allowed mem allocation for JAX

```
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE="true" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 \
WANDB_MODE=online \
uv run python impls/main_carla.py \
  --agent=impls/configs/steervla_dsrl_config.py \
  --route=enter-actor-flow-004 \
  --online_steps=50000 \
  --save_buffer=true \
  --seed=0 \
  --save_dir=\home\cglossop\exps
```

There are a couple levers to pull to optimize the speed a bit: 

- `actions_per_model_query`: int - how many **env steps** (20 Hz CARLA ticks) to serve from one
  model query before querying again (speed)
- `actions_per_cot`: int - how many actions to execute before getting new CoT (speed)
- `reanchor_cached_chunk`: bool, default `True` - re-express the replayed chunk's route waypoints
  in the ego's current body frame on every held tick

### What `actions_per_model_query` actually holds

Env steps and chunk rows are not the same rate. CARLA ticks at 20 Hz and one env step is one
tick, while a chunk row is a waypoint at the 4 Hz policy rate the model was trained at
(`env_steps_per_chunk_row = carla_fps // policy_fps = 5`). So `actions_per_model_query=5` holds
**one** chunk row's worth of time (0.25 s) — it does not execute five rows of the plan. Executing
five rows would be `actions_per_model_query=25`.

During the hold the env re-runs the PID every tick, but hands the decoder only a fresh ego
*speed* (`SimlingoStyleWaypointDecoder._flat_action_to_pid` reads `state_vec` solely for
`EGO_STATE_IDX_SPEED`). The longitudinal loop therefore stays closed, but the lateral loop would
not: `interpolate_waypoints` re-zeroes arc length at the plan's first point, so the decoder
assumes the ego is sitting exactly at the pose the chunk was sampled from and re-issues the
steering appropriate to *that* pose for the whole hold. Cross-track and heading error accumulated
in between are invisible until the next query.

`reanchor_cached_chunk` fixes this: the cached waypoints get a rigid SE(2) transform into the
current body frame (the pose comes from `obs["state"]`, x/y at 0/1 and yaw at 5), the points now
behind the ego are dropped, and the deltas are re-derived. The speed columns are left alone — the
PID reads them as `|wp[2] - wp[0]| * 2`, a displacement magnitude that is invariant to rotation and
translation, and that channel is time-indexed so only the whole-row shift can advance it. Run
`impls/vlas/test_reanchor_cached_chunk.py` for the correctness check and an off/on A/B.

Two things to watch when tuning this knob, both logged to W&B by `main_carla.py`:

- `pid/heading_error` — pinned to a constant between model queries means the lateral loop is open.
- `vla/action_cached` — fraction of executed actions replayed from the cache. A replayed chunk
  ignores the sampled noise, so under DSRL the noise actor only shapes 1 in
  `actions_per_model_query` executed actions. Use `actions_per_model_query=1` for a fully
  on-policy RL run; the action expert cannot be re-run cheaply on held ticks, because a new
  observation needs a fresh (expensive) prefix forward.



## Fail2Drive routes:

The route XMLs and scenario classes come from the `fail2drive` package, which is already
part of the `carla` extra — a plain `uv sync --extra all-<gpu,tpu>` pulls it in, no extra
install step. Route names accept three forms: `f2d:<id>`, kebab
(`base-pedestrians-on-road-0085`), or the original filename
(`Base_PedestriansOnRoad_0085`). See `ogbench/carla/route_registry.py`.

Everything below is about **assets**. Most Fail2Drive routes run on the stock 0.9.16
install once you drop in the content pack. The `generalization-animals-*` routes do not,
and need a second, 0.9.15 simulator — see part 2.

### 1. Props / occluders on 0.9.16 (everything except animals)

First pull down the f2d_content_pack.zip (see our slack channel)

Then run `install_f2d_content.sh`: 

```
./install_f2d_content.sh <CARLA_ROOT> <ZIP_PATH>
```

This copies loose `.uasset` files (occluder/wall props, animal meshes) into the packaged
install and registers the *static props* via `Content/*/Config/*.Package.json`. It does
not touch any bench2drive feature.

### 2. Animals need a separate CARLA 0.9.15 install

#### 2a. Download the Fail2Drive simulator

Anywhere outside the repo (`$HOME` is fine):

```
mkdir f2d_carla
curl -L \
  https://huggingface.co/datasets/SimonGer/fail2drive/resolve/main/fail2drive_simulator.tar.gz \
  | tar -xz -C f2d_carla
```

Sanity check — both of these must exist:

```
ls f2d_carla/CarlaUE4.sh
ls f2d_carla/CarlaUE4/Content/AnimalVarietyPack
```



#### 2b. Build the Python 3.10 env

From the repo root. This env only runs the env server — no JAX, no openpi, no torch:

```
uv venv --python 3.10 .venv-carla-0915
uv pip install --python .venv-carla-0915/bin/python \
  "carla==0.9.15" "numpy<2" "py-trees==0.8.3" \
  absl-py pyyaml gymnasium networkx shapely tabulate xmlschema \
  opencv-python-headless matplotlib imageio scipy pillow tqdm \
  "bench2drive @ git+https://github.com/catglossop/Bench2Drive.git" \
  "fail2drive @ git+https://github.com/catglossop/fail2drive.git"
```

Check:

```
.venv-carla-0915/bin/python -c \
  "import carla, srunner, fail2drive; print(carla.__file__, fail2drive.__file__)"
```

#### 2c. Run

Set `CARLA_0915_ROOT`; everything else is automatic. `run_carla.sh` then points
`CARLA_ROOT` at the 0.9.15 install and sets `CARLA_ENV_SUBPROCESS_PYTHON` to
`.venv-carla-0915/bin/python` (override the interpreter with `CARLA_0915_PYTHON`). It
prints `[run_carla.sh] CARLA 0.9.15 mode: CARLA_ROOT=...` when the switch is active.

```
CARLA_0915_ROOT=$HOME/f2d_carla ./carla_job.sh start --job 45 \
  --train-gpu 5 --render-adapter 6 \
  --route generalization-animals-1081 -- \
  --agent-config impls/configs/steervla_cast_collect_config.py \
  --train-mode dagger --online-steps 20000 --run-group Animals -- \
  --save_dir=/raid/users/<you>/exps
```

The same variable works with `./run_carla.sh` directly. Unset it and the repo behaves  
exactly as before, on 0.9.16.
