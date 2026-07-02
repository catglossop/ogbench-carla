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


## Configuring your env

This repo is built to work with [openpi](https://github.com/catglossop/steervla-pi.git) and [bench2drive](https://github.com/catglossop/Bench2Drive.git)

We provide the ability to treat each Bench2Drive route as a task in the CARLA environemtn or run the entire benchmark. 

To list the available routes or look for a specific kind of route:

```
WANDB_MODE=disabled uv run python impls/main_carla.py --list_routes=true | head -20
WANDB_MODE=disabled uv run python impls/main_carla.py --list_routes=true | grep parking
```
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

 ```
 gcloud compute tpus tpu-vm describe <tpu-name> \
    --zone=<zone> \
    --format="value(networkEndpoints[0].ipAddress)"
```

Set the `actor_url` in the steervla config (see `impls/configs/steervla_dsrl_config.py` for an example)

### CARLA config

The carla config is located in `impls/config/carla_config.yaml`. This can be used to set the port for the sim (if using a remote sim), the timeout for the sim etc. 

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
XLA_PYTHON_CLIENT_PREALLOCATE="true" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 \
WANDB_MODE=online \
uv run python impls/main_carla.py \
  --agent=impls/configs/steervla_dsrl_config.py \
  --route=signalized-junction-left-turn-001 \
  --online_steps=50000 \
  --save_buffer=true \
  --seed=0
```

There are a couple levers to pull to optimize the speed a bit: 

- `actions_per_model_query`: int - how many actions in the action chunk to execute open loop (speed)
- `actions_per_cot`: int - how many actions to execute before getting new CoT (speed)


## Fail2Drive routes: 

To add fail2drive routes, first pull down the f2d_content_pack.zip (see our slack channel)

Then run `install_f2d_content.sh`: 
```
./install_f2d_content.sh <CARLA_ROOT> <ZIP_PATH>
```


