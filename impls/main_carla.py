"""Online RL on CARLA Bench2Drive (and SteerVLA checkpoint smoke tests).

Three calling patterns:

1) **Online RL on a single Bench2Drive route**::

     uv run python impls/main_carla.py \\
       --agent=impls/configs/steervla_dsrl_config.py \\
       --route=parking-cut-in-001 \\
       --online_steps=5000 \\
       --save_buffer=true

   ``--route`` accepts any of three name styles:
     * scenario-name kebab    (e.g. ``parking-cut-in-001``)
     * file basename           (e.g. ``bench2drive_007``)
     * numeric route id        (e.g. ``1711``)
   See :func:`ogbench.carla.route_registry.list_routes`.

2) **List routes** (no env spin-up)::

     uv run python impls/main_carla.py --list_routes=true

3) **SteerVLA checkpoint smoke test** (no CARLA needed)::

     uv run python impls/main_carla.py \\
       --eval_only=true \\
       --steervla_checkpoint=gs://cat-logs/.../90000 \\
       --steervla_actor_config=pi05_steervla_inference

JAX RL algorithms live under ``jax_agents/`` so the top-level ``agents`` name
remains free for CARLA's ``PythonAPI/carla/agents`` (navigation, etc.).
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

_IMPLS_ROOT = Path(__file__).resolve().parent
if str(_IMPLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPLS_ROOT))

import wandb
from absl import app, flags
from ml_collections import config_flags

from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "env_name",
    "carla-bench2drive",
    "Environment family. Use --route to pick a specific Bench2Drive scenario.",
)
flags.DEFINE_string(
    "route",
    None,
    "Bench2Drive route: scenario-name (parking-cut-in-001), file basename "
    "(bench2drive_007), or route id (1711). See --list_routes=true.",
)
flags.DEFINE_bool("list_routes", False, "Print all known routes and exit.")

flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("restore_path", None, "Restore path for JAX agents.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")

flags.DEFINE_integer("online_steps", 1000, "Number of online environment steps to run.")
flags.DEFINE_integer("log_interval", 100, "Logging interval (env steps).")
flags.DEFINE_integer("save_interval", 100_000, "Agent-checkpoint interval (env steps).")
flags.DEFINE_bool("save_buffer", False, "Dump the replay buffer to <save_dir>/buffer.npz at the end.")
flags.DEFINE_string("buffer_path", None, "Optional explicit path for the saved buffer.")

flags.DEFINE_bool(
    "eval_only",
    False,
    "If true, skip training. With --steervla_checkpoint, only load OpenPI SteerVLA weights.",
)
flags.DEFINE_string(
    "steervla_actor_config",
    "pi05_steervla_inference",
    "OpenPI training config name (must match the architecture used when saving).",
)
flags.DEFINE_string(
    "steervla_checkpoint",
    None,
    "gs:// or local path passed to openpi.shared.download.maybe_download.",
)
flags.DEFINE_string(
    "carla_config",
    None,
    "Optional path to carla_config.yaml; default is impls/configs/carla_config.yaml.",
)
flags.DEFINE_string(
    "wandb_mode",
    None,
    "W&B mode (online/offline/disabled). Default: env WANDB_MODE or online.",
)

config_flags.DEFINE_config_file("agent", "jax_agents/dsrl.py", lock_config=False)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _extract_agent_obs(env, env_obs: dict, mode: str) -> np.ndarray:
    """Pick the tensor the RL agent trains on (env always exposes both keys)."""
    if mode == "state":
        return np.asarray(env_obs["state"], dtype=np.float32)
    if mode == "image":
        return np.asarray(env_obs["image"], dtype=np.uint8)
    raise ValueError(f"Unknown observation_mode {mode!r}; expected 'state' or 'image'.")


def _carla_env_p(name: Optional[str]) -> bool:
    if not name:
        return False
    n = name.lower()
    return n.startswith("carla") or "bench2drive" in n


def _resolve_wandb_mode() -> str:
    if FLAGS.wandb_mode is not None:
        return FLAGS.wandb_mode
    return os.environ.get("WANDB_MODE", "online")


def _list_routes_and_exit() -> None:
    from ogbench.carla.route_registry import list_routes

    entries = list_routes()
    print(f"# {len(entries)} Bench2Drive routes")
    print(f"{'scenario_name':<48} {'file_name':<24} {'route_id':<8} {'town':<10} {'scenario_type'}")
    for e in entries:
        print(f"{e.scenario_name:<48} {e.file_name:<24} {e.route_id:<8} {e.town:<10} {e.scenario_type}")


def run_steervla_checkpoint_smoke() -> None:
    from vlas.steervla import SteerVLALocalActor

    assert FLAGS.steervla_checkpoint, "steervla_checkpoint must be set"
    actor = SteerVLALocalActor(FLAGS.steervla_actor_config, FLAGS.steervla_checkpoint)
    actor.setup()
    print("[SteerVLA] Checkpoint ready at:", actor.checkpoint_dir, flush=True)


def _make_carla_env(carla_config_path: Optional[str], route: Optional[str]):
    from ogbench.carla.carla_utils import make_env_and_datasets

    env, _, _ = make_env_and_datasets(
        env_name=FLAGS.env_name,
        env_only=True,
        carla_config_path=carla_config_path,
        route=route,
    )
    return env


def _build_vla_sample_fn(steervla_cfg):
    """Optionally construct a callable ``(obs, noise) -> action`` from a SteerVLA checkpoint."""
    if not steervla_cfg.get("enabled", False):
        return None
    if not steervla_cfg.get("checkpoint"):
        print("[SteerVLA] enabled=True but no checkpoint provided; skipping VLA hookup.", flush=True)
        return None
    from vlas.steervla import SteerVLALocalActor  # noqa: F401

    print("[SteerVLA] vla_sample_fn wiring is a placeholder; supply a real obs->action mapping.")

    def _identity(observations, noise):
        return noise

    return _identity


# --------------------------------------------------------------------------- #
# Online RL loop                                                              #
# --------------------------------------------------------------------------- #


def run_online_carla(env, agent, agent_config, exp_name: str) -> None:
    import jax
    import tqdm

    from utils.datasets import ReplayBuffer
    from utils.flax_utils import save_agent

    obs_mode = str(agent_config.get("observation_mode", "state"))

    capacity = int(agent_config.get("buffer_capacity", 100_000))
    warmup = int(agent_config.get("warmup_steps", 1000))
    updates_per_step = int(agent_config.get("updates_per_step", 1))
    batch_size = int(agent_config.get("batch_size", 256))

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))

    obs, info = env.reset(seed=FLAGS.seed)
    obs = _extract_agent_obs(env, obs, obs_mode)
    action_dim = int(env.action_space.shape[0])
    example_transition = dict(
        observations=np.array(obs),
        actions=np.zeros((action_dim,), dtype=np.float32),
        rewards=np.float32(0.0),
        next_observations=np.array(obs),
        masks=np.float32(1.0),
        terminals=np.float32(0.0),
    )
    buffer = ReplayBuffer.create(example_transition, size=capacity)

    rng = jax.random.PRNGKey(FLAGS.seed + 1)
    episode_return, episode_steps, episode_count = 0.0, 0, 0
    last_log_time = time.time()

    for step in tqdm.tqdm(range(1, FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, sub = jax.random.split(rng)
        if step <= warmup:
            action = env.action_space.sample()
        else:
            action_jax = agent.sample_actions(obs[None], seed=sub)
            action = np.asarray(action_jax[0])
            
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = _extract_agent_obs(env, next_obs, obs_mode)
        done = bool(terminated or truncated)
        buffer.add_transition(
            dict(
                observations=np.asarray(obs),
                actions=action.astype(np.float32),
                rewards=np.float32(reward),
                next_observations=np.asarray(next_obs),
                masks=np.float32(0.0 if terminated else 1.0),
                terminals=np.float32(1.0 if done else 0.0),
            )
        )
        obs = next_obs
        episode_return += float(reward)
        episode_steps += 1

        if done:
            episode_count += 1
            wandb.log(
                {
                    "rollout/episode_return": episode_return,
                    "rollout/episode_steps": episode_steps,
                    "rollout/episodes": episode_count,
                    "rollout/route": info.get("route", "?"),
                },
                step=step,
            )
            obs, info = env.reset(seed=FLAGS.seed + episode_count)
            obs = _extract_agent_obs(env, obs, obs_mode)
            episode_return, episode_steps = 0.0, 0

        if step > warmup and buffer.size >= batch_size:
            for _ in range(updates_per_step):
                batch = buffer.sample(batch_size)
                agent, update_info = agent.update(batch)

            if step % FLAGS.log_interval == 0:
                metrics = {f"training/{k}": float(v) for k, v in update_info.items()}
                metrics["training/buffer_size"] = int(buffer.size)
                metrics["time/steps_per_sec"] = FLAGS.log_interval / max(time.time() - last_log_time, 1e-6)
                last_log_time = time.time()
                wandb.log(metrics, step=step)
                train_logger.log(metrics, step=step)

        if step % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, step)

    train_logger.close()

    if FLAGS.save_buffer:
        buffer_path = FLAGS.buffer_path or os.path.join(FLAGS.save_dir, "buffer.npz")
        path = buffer.save(buffer_path)
        print(f"[buffer] saved {buffer.size} transitions -> {path}", flush=True)


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #


def main(_):
    if FLAGS.list_routes:
        _list_routes_and_exit()
        return

    wandb_mode = _resolve_wandb_mode()

    if FLAGS.eval_only and FLAGS.steervla_checkpoint:
        exp_name = get_exp_name(FLAGS.seed)
        setup_wandb(project="OGBench-CARLA", group=FLAGS.run_group, name=exp_name, mode=wandb_mode)
        try:
            run_steervla_checkpoint_smoke()
        except Exception:
            print("[SteerVLA] Checkpoint smoke test failed:", flush=True)
            traceback.print_exc()
            sys.exit(1)
        wandb.finish()
        print("[SteerVLA] Smoke test finished OK.", flush=True)
        return

    import jax
    import numpy as np

    from jax_agents import agents
    from utils.flax_utils import restore_agent

    config = FLAGS.agent

    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project="OGBench-CARLA", group=FLAGS.run_group, name=exp_name, mode=wandb_mode)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(get_flag_dict(), f)

    if not _carla_env_p(FLAGS.env_name):
        raise ValueError(
            f"main_carla.py only supports CARLA Bench2Drive envs; got --env_name={FLAGS.env_name!r}."
            " Use main.py for the OGBench MuJoCo tasks."
        )

    carla_yaml = FLAGS.carla_config
    if carla_yaml is None:
        default_yaml = _IMPLS_ROOT / "configs" / "carla_config.yaml"
        if default_yaml.is_file():
            carla_yaml = str(default_yaml)

    env = _make_carla_env(carla_yaml, FLAGS.route)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    obs_mode = str(config.get("observation_mode", "state"))
    obs, _info = env.reset(seed=FLAGS.seed)
    if not isinstance(obs, dict) or "state" not in obs or "image" not in obs:
        raise ValueError(
            "CARLA env must return a Dict observation with 'state' and 'image'; "
            f"got {type(obs).__name__}."
        )
    agent_obs = _extract_agent_obs(env, obs, obs_mode)
    ex_obs = np.expand_dims(agent_obs, 0)
    ex_actions = np.zeros((1,) + tuple(env.action_space.shape), dtype=np.float32)

    agent_class = agents[config["agent_name"]]
    create_kwargs = {}
    if config["agent_name"] == "dsrl":
        steervla_cfg = config.get("steervla", None)
        vla_sample_fn = _build_vla_sample_fn(steervla_cfg) if steervla_cfg is not None else None
        if vla_sample_fn is not None:
            create_kwargs["vla_sample_fn"] = vla_sample_fn

    agent = agent_class.create(FLAGS.seed, ex_obs, ex_actions, config, **create_kwargs)
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    if FLAGS.eval_only:
        # No offline-eval pipeline yet for CARLA; do a single rollout.
        FLAGS.online_steps = max(FLAGS.online_steps, 200)
        FLAGS.save_buffer = FLAGS.save_buffer or False

    try:
        run_online_carla(env, agent, config, exp_name)
    finally:
        try:
            env.close()
        except Exception:
            pass
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
