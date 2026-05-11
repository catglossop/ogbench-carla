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
from typing import Any, Optional

import numpy as np
import jax

from jax_agents import agents
from utils.flax_utils import restore_agent

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


def _configure_jax_training_device(training_gpu_rank: int) -> None:
    """Pin JAX default device for RL. CARLA uses ``gpu_rank`` in ``carla_config.yaml`` separately."""
    if training_gpu_rank < 0:
        return
    import jax

    try:
        devs = jax.devices("gpu")
    except RuntimeError:
        devs = []
    if not devs:
        print(
            "[main_carla] training_gpu_rank is set but JAX has no GPU; using default backend.",
            flush=True,
        )
        return
    if training_gpu_rank >= len(devs):
        raise ValueError(
            f"training_gpu_rank={training_gpu_rank} invalid: only {len(devs)} JAX GPU(s) visible: {devs}"
        )
    dev = devs[training_gpu_rank]
    jax.config.update("jax_default_device", dev)
    print(
        f"[main_carla] JAX default device -> {dev} (training_gpu_rank={training_gpu_rank})",
        flush=True,
    )


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


def _steervla_action_execution_cfg(steervla_cfg) -> dict[str, Any] | None:
    """Env + replay-buffer layout for OpenPI SteerVLA chunks (simlingo-style control)."""
    if steervla_cfg is None or not steervla_cfg.get("enabled"):
        return None
    if not steervla_cfg.get("use_pi_action_chunk_for_env", True):
        return None
    fmt = steervla_cfg.get("output_action_format") or "DELTA_XY_T_DELTA_XY_SPACE"
    ah = int(steervla_cfg.get("action_horizon", 10))
    ad = int(steervla_cfg.get("action_dim", 4))
    url = steervla_cfg.get("actor_url")
    remote = bool(url and str(url).strip())
    return {
        "output_action_format": fmt,
        "action_horizon": ah,
        "action_dim": ad,
        # Remote HTTP policy applies ``Unnormalize`` (dataset units); local JAX returns raw flow outputs.
        "action_input_space": "policy_output" if remote else "normalized",
    }


def _make_carla_env(
    carla_config_path: Optional[str],
    route: Optional[str],
    *,
    extra_carla_config: Optional[dict[str, Any]] = None,
):
    from ogbench.carla.carla_utils import CarlaBench2DriveWrapper, load_carla_config

    cfg = load_carla_config(carla_config_path)
    if extra_carla_config:
        cfg = {**cfg, **extra_carla_config}
    return CarlaBench2DriveWrapper(cfg, route=route)


def _build_vla_sample_fn(
    steervla_cfg,
    raw_carla_obs_holder: dict | None,
    *,
    training_gpu_rank: int = -1,
):
    """Construct ``(obs, noise) -> action`` using OpenPI Pi0-CoT SteerVLA (:mod:`vlas.steervla`)."""
    if not steervla_cfg.get("enabled", False):
        return None
    if raw_carla_obs_holder is None:
        raise ValueError("SteerVLA requires raw_carla_obs_holder for full gym obs (image + state + prompt fields).")

    actor_url = steervla_cfg.get("actor_url")
    if actor_url and str(actor_url).strip():
        from vlas.steervla import create_steervla_pi0_cot_sample_fn

        print(
            "[SteerVLA] Remote inference at",
            str(actor_url).strip(),
            "(no local OpenPI checkpoint restore).",
            flush=True,
        )
        return create_steervla_pi0_cot_sample_fn(
            steervla_cfg,
            raw_carla_obs_holder,
            training_gpu_rank=training_gpu_rank,
        )

    if not steervla_cfg.get("checkpoint"):
        print("[SteerVLA] enabled=True but no checkpoint provided; skipping VLA hookup.", flush=True)
        return None
    if not steervla_cfg.get("actor_config"):
        raise ValueError("steervla.actor_config must name an OpenPI TrainConfig (e.g. pi05_steervla_cot_ki).")

    from vlas.steervla import create_steervla_pi0_cot_sample_fn, openpi_action_expert_trainable_hint

    print(openpi_action_expert_trainable_hint(str(steervla_cfg["actor_config"])), flush=True)
    return create_steervla_pi0_cot_sample_fn(
        steervla_cfg,
        raw_carla_obs_holder,
        training_gpu_rank=training_gpu_rank,
    )


# --------------------------------------------------------------------------- #
# Online RL loop                                                              #
# --------------------------------------------------------------------------- #


def run_online_carla(
    env,
    agent,
    agent_config,
    exp_name: str,
    raw_carla_obs_holder: dict | None = None,
    steervla_actor=None,
) -> None:
    import jax
    import tqdm

    from ogbench.carla.carla_utils import ego_drive_metrics_from_state_vec

    from utils.datasets import ReplayBuffer
    from utils.flax_utils import save_agent

    obs_mode = str(agent_config.get("observation_mode", "state"))

    capacity = int(agent_config.get("buffer_capacity", 5_000))
    warmup = int(agent_config.get("warmup_steps", 1000))
    updates_per_step = int(agent_config.get("updates_per_step", 1))
    batch_size = int(agent_config.get("batch_size", 256))
    image_curr_interval = int(agent_config.get("image_log_curr_interval", 10))

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))

    def _cot_fields_from_raw(raw: dict | None) -> dict[str, np.ndarray]:
        if steervla_actor is None or getattr(steervla_actor, "model_cfg", None) is None:
            return {}
        rlen = int(steervla_actor.model_cfg.max_reasoning_len)
        slen = int(steervla_actor.model_cfg.max_subtask_len)
        out = {
            "reasoning": np.zeros((rlen,), dtype=np.int32),
            "reasoning_mask": np.zeros((rlen,), dtype=bool),
            "subtask": np.zeros((slen,), dtype=np.int32),
            "subtask_mask": np.zeros((slen,), dtype=bool),
        }
        if not isinstance(raw, dict):
            return out
        for k, dt in (
            ("reasoning", np.int32),
            ("reasoning_mask", bool),
            ("subtask", np.int32),
            ("subtask_mask", bool),
        ):
            if raw.get(k) is None:
                continue
            arr = np.asarray(raw[k], dtype=dt).reshape(-1)
            n = min(arr.size, out[k].size)
            out[k][:n] = arr[:n]
        return out
    
    raw_obs_holder = raw_carla_obs_holder

    if raw_obs_holder is not None and raw_obs_holder.get("obs") is not None:
        obs_raw = raw_obs_holder["obs"]
    else:
        obs_raw, _info = env.reset(seed=FLAGS.seed)
    if raw_obs_holder is not None:
        raw_obs_holder["obs"] = obs_raw
        raw_obs_holder["next_obs"] = obs_raw
    obs = _extract_agent_obs(env, obs_raw, obs_mode)
    log_images = obs_mode == "image"
    if log_images:
        wandb.log(
            {
                "rollout/start_obs": wandb.Image(
                    np.asarray(obs), caption="episode 1 start",
                ),
            },
            step=0,
        )

    action_dim = int(env.action_space.shape[0])
    example_transition = dict(
        observations=np.array(obs),
        actions=np.zeros((action_dim,), dtype=np.float32),
        action_chunked=np.zeros((action_dim,), dtype=np.float32),
        rewards=np.float32(0.0),
        next_observations=np.array(obs),
        masks=np.float32(1.0),
        terminals=np.float32(0.0),
    )
    if steervla_actor is not None:
        cot0 = _cot_fields_from_raw(obs_raw)
        example_transition.update(cot0)
        example_transition.update({f"next_{k}": np.array(v) for k, v in cot0.items()})
    buffer = ReplayBuffer.create(example_transition, size=capacity)

    rng = jax.random.PRNGKey(FLAGS.seed + 1)
    episode_return, episode_steps, episode_count = 0.0, 0, 0
    last_log_time = time.time()

    for step in tqdm.tqdm(range(1, FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if raw_obs_holder is not None:
            raw_obs_holder["obs"] = obs_raw
        rng, sub = jax.random.split(rng)
        if step <= warmup:
            action = env.action_space.sample()
        else:
            if getattr(agent, "vla_sample_fn", None) is not None:
                action_jax = agent.sample_actions_with_vla(obs[None], seed=sub)
            else:
                action_jax = agent.sample_actions(obs[None], seed=sub)
            action = np.asarray(action_jax[0])
        
        action_chunked = np.asarray(action, dtype=np.float32).reshape(-1)

        next_obs_raw, reward, terminated, truncated, info = env.step(action)
        if raw_obs_holder is not None:
            raw_obs_holder["next_obs"] = next_obs_raw
        drive_metrics = ego_drive_metrics_from_state_vec(next_obs_raw["state"])
        next_obs = _extract_agent_obs(env, next_obs_raw, obs_mode)
        done = bool(terminated or truncated)
        end_img = np.copy(next_obs) if done and log_images else None
        buffer.add_transition(
            {
                "observations": np.asarray(obs),
                "actions": action.astype(np.float32),
                "action_chunked": action_chunked.astype(np.float32),
                "rewards": np.float32(reward),
                "next_observations": np.asarray(next_obs),
                "masks": np.float32(0.0 if terminated else 1.0),
                "terminals": np.float32(1.0 if done else 0.0),
                **(_cot_fields_from_raw(obs_raw) if steervla_actor is not None else {}),
                **({f"next_{k}": np.array(v) for k, v in _cot_fields_from_raw(next_obs_raw).items()} if steervla_actor is not None else {}),
            }
        )
        obs = next_obs
        obs_raw = next_obs_raw
        episode_return += float(reward)
        episode_steps += 1

        step_wb = {f"rollout/{k}": v for k, v in drive_metrics.items()}
        if log_images and image_curr_interval > 0 and step % image_curr_interval == 0:
            step_wb["rollout/curr_obs"] = wandb.Image(
                np.asarray(obs),
                caption=f"env step {step}",
            )
        wandb.log(step_wb, step=step)

        if done:
            episode_count += 1
            rollout_log = {
                "rollout/episode_return": episode_return,
                "rollout/episode_steps": episode_steps,
                "rollout/episodes": episode_count,
                "rollout/route": info.get("route", "?"),
            }
            if log_images and end_img is not None:
                rollout_log["rollout/final_obs"] = wandb.Image(
                    end_img,
                    caption=f"episode {episode_count} final",
                )
            wandb.log(rollout_log, step=step)
            obs_raw, _info = env.reset(seed=FLAGS.seed + episode_count)
            if raw_obs_holder is not None:
                raw_obs_holder["obs"] = obs_raw
                raw_obs_holder["next_obs"] = obs_raw
                
            reset_vla_cache = getattr(getattr(agent, "vla_sample_fn", None), "reset_action_cache", None)
            if reset_vla_cache is not None:
                reset_vla_cache()
            obs = _extract_agent_obs(env, obs_raw, obs_mode)
            if log_images:
                wandb.log(
                    {
                        "rollout/start_obs": wandb.Image(
                            np.asarray(obs),
                            caption=f"episode {episode_count + 1} start",
                        ),
                    },
                    step=step,
                )
            episode_return, episode_steps = 0.0, 0

        if step > warmup and buffer.size >= batch_size:
            for _ in range(updates_per_step):
                batch = buffer.sample(batch_size)
                if getattr(agent, "vla_sample_fn", None) is not None:
                    agent, update_info = agent.update_with_vla(batch)
                else:
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

    steervla_cfg = config.get("steervla", None)
    extra_carla: dict[str, Any] = {}
    exec_cfg = _steervla_action_execution_cfg(steervla_cfg)
    if exec_cfg is not None:
        extra_carla["steervla_action_execution"] = exec_cfg

    # Leaderboard starts CARLA with subprocess (fork + exec). JAX initializes a native
    # thread pool; forking afterward triggers the stdlib warning and can deadlock the child,
    # which often surfaces as UE4 "RenderThread" timeouts. Bring the simulator up first.
    env = _make_carla_env(carla_yaml, FLAGS.route, extra_carla_config=extra_carla or None)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    obs_mode = str(config.get("observation_mode", "state"))
    obs_dict, _info = env.reset(seed=FLAGS.seed)
    if not isinstance(obs_dict, dict) or "state" not in obs_dict or "image" not in obs_dict:
        raise ValueError(
            "CARLA env must return a Dict observation with 'state' and 'image'; "
            f"got {type(obs_dict).__name__}."
        )

    raw_carla_holder: dict | None = None
    if steervla_cfg is not None and steervla_cfg.get("enabled"):
        raw_carla_holder = {"obs": obs_dict, "next_obs": obs_dict}

    agent_obs = _extract_agent_obs(env, obs_dict, obs_mode)
    ex_obs = np.expand_dims(agent_obs, 0)
    ex_actions = np.zeros((1,) + tuple(env.action_space.shape), dtype=np.float32)

    _configure_jax_training_device(int(config.get("training_gpu_rank", -1)))

    agent_class = agents[config["agent_name"]]
    create_kwargs = {}
    steervla_actor = None
    if config["agent_name"] == "dsrl":
        tr_rank = int(config.get("training_gpu_rank", -1))
        vla_bundle = (
            _build_vla_sample_fn(steervla_cfg, raw_carla_holder, training_gpu_rank=tr_rank)
            if steervla_cfg is not None
            else None
        )
        if vla_bundle is not None:
            vla_sample_fn, steervla_actor = vla_bundle
            create_kwargs["vla_sample_fn"] = vla_sample_fn
            url = steervla_cfg.get("actor_url") if steervla_cfg else None
            if not (url and str(url).strip()):
                create_kwargs["vla_train_state"] = steervla_actor.train_state
                create_kwargs["openpi_train_config"] = steervla_actor.train_cfg
                create_kwargs["steervla_actor"] = steervla_actor
    

    agent = agent_class.create(FLAGS.seed, ex_obs, ex_actions, config, **create_kwargs)

    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    if FLAGS.eval_only:
        # No offline-eval pipeline yet for CARLA; do a single rollout.
        FLAGS.online_steps = max(FLAGS.online_steps, 200)
        FLAGS.save_buffer = FLAGS.save_buffer or False

    try:
        run_online_carla(
            env,
            agent,
            config,
            exp_name,
            raw_carla_obs_holder=raw_carla_holder,
            steervla_actor=steervla_actor,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
