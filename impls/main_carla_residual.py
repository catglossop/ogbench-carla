"""Online residual RL on CARLA Bench2Drive with a frozen SteerVLA base policy.

    SteerVLA proposes a base action chunk -> RL state x = proprio slice -> the
    residual SAC agent adds a small correction -> the env executes one tick ->
    transitions feed a replay buffer.

Calling patterns::

    uv run python impls/main_carla_residual.py \
        --agent=impls/configs/steervla_residual_config.py \
        --route=parking-cut-in-001 --online_steps=5000 --save_buffer=true

    uv run python impls/main_carla_residual.py --list_routes=true

JAX RL agents live under ``jax_agents/`` so the top-level ``agents`` name stays
free for CARLA's ``PythonAPI/carla/agents``.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

_IMPLS_ROOT = Path(__file__).resolve().parent
if str(_IMPLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPLS_ROOT))

import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from utils.datasets import ReplayBuffer
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "route",
    None,
    "Bench2Drive route: scenario-name (parking-cut-in-001), file basename "
    "(bench2drive_007), or route id (1711). See --list_routes=true.",
)
flags.DEFINE_bool("list_routes", False, "Print all known routes and exit.")

flags.DEFINE_string("save_dir", "/home/carla/exps", "Save directory.")
flags.DEFINE_string("restore_path", None, "Restore path for the JAX agent.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")

flags.DEFINE_integer("online_steps", 1000, "Number of online environment steps to run.")
flags.DEFINE_integer("log_interval", 10, "Logging interval (env steps).")
flags.DEFINE_integer("save_interval", 5_000, "Agent-checkpoint interval (env steps).")
flags.DEFINE_bool("save_buffer", False, "Dump the replay buffer to <save_dir>/buffer.npz at the end.")
flags.DEFINE_string("buffer_path", None, "Optional explicit path for the saved buffer.")
flags.DEFINE_string("carla_config", None, "Path to carla_config.yaml (default: impls/configs/carla_config.yaml).")
flags.DEFINE_string("wandb_mode", None, "W&B mode (online/offline/disabled). Default: env WANDB_MODE or online.")
flags.DEFINE_bool("enable_updates", None, "Override config.enable_updates. If false, rollout/buffer only.")

config_flags.DEFINE_config_file("agent", "configs/steervla_residual_config.py", lock_config=False)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _configure_jax_training_device(training_gpu_rank: int) -> None:
    """Pin JAX's default device. CARLA's render GPU is ``gpu_rank`` in carla_config.yaml."""
    if training_gpu_rank < 0:
        return
    try:
        devs = jax.devices("gpu")
    except RuntimeError:
        devs = []
    if not devs:
        print("[main_carla_residual] training_gpu_rank set but JAX has no GPU; using default backend.", flush=True)
        return
    if training_gpu_rank >= len(devs):
        raise ValueError(f"training_gpu_rank={training_gpu_rank} invalid: only {len(devs)} JAX GPU(s): {devs}")
    dev = devs[training_gpu_rank]
    jax.config.update("jax_default_device", dev)
    print(f"[main_carla_residual] JAX default device -> {dev} (training_gpu_rank={training_gpu_rank})", flush=True)


def _list_routes_and_exit() -> None:
    from ogbench.carla.route_registry import list_routes

    entries = list_routes()
    print(f"# {len(entries)} bench2drive routes")
    print(f"{'scenario_name':<48} {'file_name':<20} {'route_id':<10} {'town':<10} {'scenario_type'}")
    for e in entries:
        print(f"{e.scenario_name:<48} {e.file_name:<20} {e.route_id:<10} {e.town:<10} {e.scenario_type}")


def _steervla_action_execution_cfg(steervla_cfg) -> Optional[dict[str, Any]]:
    """Env + replay-buffer layout for OpenPI SteerVLA chunks (simlingo-style control)."""
    if steervla_cfg is None or not steervla_cfg.get("enabled"):
        return None
    if not steervla_cfg.get("use_pi_action_chunk_for_env", True):
        return None
    return {
        "output_action_format": steervla_cfg.get("output_action_format") or "DELTA_XY_T_DELTA_XY_SPACE",
        "action_horizon": int(steervla_cfg.get("action_horizon", 10)),
        "action_dim": int(steervla_cfg.get("action_dim", 4)),
        "action_input_space": "policy_output",
    }


def _make_carla_env(carla_config_path, route, *, extra_carla_config=None):
    from ogbench.carla.carla_utils import CarlaBench2DriveWrapper, load_carla_config

    cfg = load_carla_config(carla_config_path)
    if extra_carla_config:
        cfg = {**cfg, **extra_carla_config}
    return CarlaBench2DriveWrapper(cfg, route=route)


def _proprio(obs: dict, slice_lo_hi) -> np.ndarray:
    """RL state = proprio slice of obs['state'] -> float32 [proprio_dim]."""
    state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
    return state[int(slice_lo_hi[0]):int(slice_lo_hi[1])]


def _base_chunk(vla_sample_fn, raw_holder: dict, obs: dict, action_dim: int) -> np.ndarray:
    """Frozen SteerVLA base action chunk (noise=0 -> deterministic) -> float32 [action_dim]."""
    raw_holder["obs"] = obs
    out = vla_sample_fn(jnp.zeros((1, 1), dtype=jnp.float32), jnp.zeros((1, action_dim), dtype=jnp.float32))
    return np.asarray(jax.device_get(out), dtype=np.float32).reshape(-1)


# --------------------------------------------------------------------------- #
# Online loop                                                                  #
# --------------------------------------------------------------------------- #


def run_online(env, agent, config, obs, *, vla_sample_fn, steervla_actor, raw_holder):
    """Residual-SAC online loop. One env.step == one CARLA tick == one transition."""
    proprio_slice = tuple(config["ego_state_slice"])
    action_dim = int(config["steervla"]["action_horizon"]) * int(config["steervla"]["action_dim"])
    warmup = int(config["residual_warmup_steps"])
    batch_size = int(config["batch_size"])
    updates_per_step = int(config["updates_per_step"])
    capacity = int(config["buffer_capacity"])
    enable_updates = bool(FLAGS.enable_updates) if FLAGS.enable_updates is not None else bool(config["enable_updates"])
    if not enable_updates:
        print("[main_carla_residual] enable_updates=False: rollout-only (no RL gradient updates).", flush=True)

    rng = jax.random.PRNGKey(FLAGS.seed)

    x = _proprio(obs, proprio_slice)
    base = _base_chunk(vla_sample_fn, raw_holder, obs, action_dim)

    buffer: Optional[ReplayBuffer] = None
    episode_return = 0.0
    episode_steps = 0
    episode_count = 0
    start_time = time.time()

    def _flush_checkpoint(step_tag: int) -> None:
        """Persist the latest agent (+ optional buffer). Runs on normal exit, a Python
        exception, or Ctrl-C. Note: a C++ SIGABRT (CARLA teardown) bypasses ``finally`` --
        only the periodic ``save_interval`` checkpoints survive that.
        """
        save_agent(agent, FLAGS.save_dir, step_tag)
        if FLAGS.save_buffer and buffer is not None:
            path = FLAGS.buffer_path or os.path.join(FLAGS.save_dir, "buffer.npz")
            saved = buffer.save(path)
            print(f"[main_carla_residual] Saved replay buffer ({buffer.size} transitions) -> {saved}", flush=True)

    last_step = 0
    try:
        for step in tqdm.tqdm(range(1, FLAGS.online_steps + 1), dynamic_ncols=True):
            last_step = step
            residual_active = step > warmup and buffer is not None and buffer.size >= batch_size
            if residual_active:
                rng, sample_key = jax.random.split(rng)
                final_b, _ = agent.sample_actions(x[None], base[None], seed=sample_key)
                final = np.asarray(jax.device_get(final_b), dtype=np.float32).reshape(-1)
            else:
                final = base

            next_obs, reward, terminated, truncated, info = env.step(final)
            done = bool(terminated or truncated)

            next_x = _proprio(next_obs, proprio_slice)
            next_base = _base_chunk(vla_sample_fn, raw_holder, next_obs, action_dim)

            transition = dict(
                observations=x,
                actions=final,
                base_actions=base,
                rewards=np.float32(reward),
                next_observations=next_x,
                next_base_actions=next_base,
                masks=np.float32(1.0 - float(terminated)),
            )
            if buffer is None:
                buffer = ReplayBuffer.create(transition, size=capacity)
            buffer.add_transition(transition)

            episode_return += float(reward)
            episode_steps += 1

            train_info: dict[str, Any] = {}
            if enable_updates and buffer.size >= batch_size:
                for _ in range(updates_per_step):
                    agent, train_info = agent.update(buffer.sample(batch_size))

            if step % FLAGS.log_interval == 0:
                log = {
                    "env/reward": float(reward),
                    "env/episode_count": episode_count,
                    "env/buffer_size": int(buffer.size),
                    "env/residual_active": int(residual_active),
                    "env/sps": step / max(time.time() - start_time, 1e-6),
                }
                if "reward_total" in info:
                    log["reward/total"] = float(info["reward_total"])
                if train_info:
                    log.update({k: float(jax.device_get(v)) for k, v in train_info.items()})
                wandb.log(log, step=step)

            if FLAGS.save_interval > 0 and step % FLAGS.save_interval == 0:
                save_agent(agent, FLAGS.save_dir, step)

            if done:
                wandb.log({"rollout/episode_return": episode_return, "rollout/episode_length": episode_steps}, step=step)
                episode_count += 1
                episode_return = 0.0
                episode_steps = 0
                obs, _info = env.reset(seed=FLAGS.seed + episode_count)
                steervla_actor.reset_action_cache()
                x = _proprio(obs, proprio_slice)
                base = _base_chunk(vla_sample_fn, raw_holder, obs, action_dim)
            else:
                obs, x, base = next_obs, next_x, next_base
    finally:
        _flush_checkpoint(last_step or FLAGS.online_steps)


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def main(_):
    if FLAGS.list_routes:
        _list_routes_and_exit()
        return

    config = FLAGS.agent

    if FLAGS.route is None:
        raise ValueError("--route is required (see --list_routes=true).")

    wandb_mode = FLAGS.wandb_mode or os.environ.get("WANDB_MODE", "online")
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project="OGBench-CARLA-Residual", group=FLAGS.run_group, name=exp_name, mode=wandb_mode)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(get_flag_dict(), f)

    carla_yaml = FLAGS.carla_config or str(_IMPLS_ROOT / "configs" / "carla_config.yaml")

    steervla_cfg = config.get("steervla", None)
    if steervla_cfg is None or not steervla_cfg.get("enabled"):
        raise ValueError("config.steervla.enabled must be true: residual RL needs a base policy.")

    extra_carla: dict[str, Any] = {}
    exec_cfg = _steervla_action_execution_cfg(steervla_cfg)
    if exec_cfg is not None:
        extra_carla["steervla_action_execution"] = exec_cfg

    # Bring CARLA up before JAX initializes its thread pool (forking afterwards can
    # deadlock the UE4 RenderThread). The reset below starts the simulator.
    env = _make_carla_env(carla_yaml, FLAGS.route, extra_carla_config=extra_carla or None)
    try:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

        obs, _info = env.reset(seed=FLAGS.seed)
        if not isinstance(obs, dict) or "state" not in obs or "image" not in obs:
            raise ValueError("CARLA env must return a Dict obs with 'state' and 'image'.")

        raw_holder: dict = {"obs": obs}
        training_gpu_rank = int(config.get("training_gpu_rank", -1))

        from vlas.steervla import create_steervla_pi0_cot_sample_fn

        vla_sample_fn, steervla_actor = create_steervla_pi0_cot_sample_fn(
            steervla_cfg, raw_holder, training_gpu_rank=training_gpu_rank
        )

        _configure_jax_training_device(training_gpu_rank)

        proprio_slice = tuple(config["ego_state_slice"])
        x_dim = int(proprio_slice[1]) - int(proprio_slice[0])
        action_dim = int(steervla_cfg["action_horizon"]) * int(steervla_cfg["action_dim"])
        ex_obs = np.zeros((1, x_dim), dtype=np.float32)
        ex_base = np.zeros((1, action_dim), dtype=np.float32)
        print(f"[main_carla_residual] proprio x_dim={x_dim}; action_dim={action_dim}", flush=True)

        from jax_agents.sac_residual import SACResidualAgent

        agent = SACResidualAgent.create(FLAGS.seed, ex_obs, ex_base, config)
        if FLAGS.restore_path is not None:
            agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

        run_online(
            env, agent, config, obs,
            vla_sample_fn=vla_sample_fn,
            steervla_actor=steervla_actor,
            raw_holder=raw_holder,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
