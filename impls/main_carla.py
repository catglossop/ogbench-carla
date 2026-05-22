"""Online RL on CARLA Bench2Drive (and SteerVLA checkpoint smoke tests).

Three calling patterns:

1) **Online RL on a single Bench2Drive route**::

     uv run python impls/main_carla.py \\
       --agent=impls/configs/steervla_dsrl_config.py \\
       --route=parking-cut-in-001 \\
       --online_steps=5000 \\
       --save_buffer=true

   Set ``warmup_steps`` in the agent config to run the policy without RL updates
   while prefilling the replay buffer (default 500 in ``steervla_dsrl_config.py``).

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
from pathlib import Path
from typing import Any, Optional

import numpy as np
import jax
import jax.numpy as jnp

from jax_agents import agents
from jax_agents.sac_residual import SACResidualAgent
from utils.flax_utils import restore_agent
from coaches.expert_label import NUM_COMMENTARY_WORDS, NUM_DELTA_COMMENTARY_WORDS
from coaches.critic_feedback import (
    compute_action_delta,
    compute_action_delta_commentary,
    critic_language_dim,
    denormalize_action_chunk,
)

_IMPLS_ROOT = Path(__file__).resolve().parent
if str(_IMPLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPLS_ROOT))

import wandb
from absl import app, flags
from ml_collections import config_flags

import tqdm

from ogbench.carla.carla_utils import ego_drive_metrics_from_state_vec

from utils.datasets import ReplayBuffer
from utils.flax_utils import save_agent

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
flags.DEFINE_bool(
    "expert_debug",
    False,
    "Debug mode: drive with the PDM-Lite expert action instead of the RL agent. "
    "Useful to verify that expert_action values are sensible.",
)
flags.DEFINE_bool(
    "expert_recover_debug",
    False,
    "Debug mode: roll out the SteerVLA agent for a random [70, 200] steps per episode, "
    "then switch to the PDM-Lite expert for the remainder of the episode.",
)

flags.DEFINE_string("save_dir", "/raid/users/celine/carla_exps", "Save directory.")
#flags.DEFINE_string("save_dir", "/home/carla/exps", "Save directory.")
flags.DEFINE_string("restore_path", None, "Restore path for JAX agents.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")

flags.DEFINE_integer("online_steps", 1000, "Number of online environment steps to run.")
flags.DEFINE_integer("log_interval", 1, "Logging interval (env steps).")
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

    try:
        devs = jax.devices("gpu")
    except RuntimeError:
        devs = []
    if not devs:
        print(
            "[WARNING - main_carla] training_gpu_rank is set but JAX has no GPU; using default backend.",
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
    """Pick the tensor the RL agent trains on (env always exposes both keys).

    The language label (BOW or delta) is stored separately in the replay buffer
    and concatenated to the encoded observation ONLY inside the critic (dsrl.py).
    """
    if mode == "state":
        return np.asarray(env_obs["state"], dtype=np.float32)
    if mode == "image":
        return np.asarray(env_obs["image"], dtype=np.uint8)
    raise ValueError(f"Unknown observation_mode {mode!r}; expected 'state' or 'image'.")


# Check if valid task environment
def _carla_env_p(name: Optional[str]) -> bool:
    if not name:
        return False
    n = name.lower()
    return n.startswith("carla") or "bench2drive" in n

# Check wandb mode
def _resolve_wandb_mode() -> str:
    if FLAGS.wandb_mode is not None:
        return FLAGS.wandb_mode
    return os.environ.get("WANDB_MODE", "online")

# List routes and exit  
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

    from vlas.steervla import create_steervla_pi0_cot_sample_fn 

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

    obs_mode = str(agent_config.get("observation_mode", "state"))

    capacity = int(agent_config.get("buffer_capacity", 5_000))
    warmup = int(agent_config.get("warmup_steps", 1000))
    updates_per_step = int(agent_config.get("updates_per_step", 1))
    batch_size = int(agent_config.get("batch_size", 256))

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    if warmup > 0 and agent is not None and not FLAGS.expert_debug:
        policy_src = "rollout policy (no RL updates)"
        print(f"[main_carla] warmup: {warmup} steps using {policy_src}", flush=True)
    
    # Get openpi fields from raw observation
    def _openpi_fields_from_raw(raw: dict | None) -> dict[str, np.ndarray]:
        if (
            steervla_actor is None
            or getattr(steervla_actor, "model_cfg", None) is None
            or raw is None
            or not isinstance(raw, dict)
        ):
            return {}

        from openpi.shared import image_tools as openpi_image_tools

        obs_struct = steervla_actor.build_observation_batch_numpy(batch_size=1, raw=raw)
        from vlas.steervla import openpi_replay_fields_from_observation

        out = openpi_replay_fields_from_observation(obs_struct)
        # ``build_observation_batch_numpy`` leaves CoT/FAST empty; overlay tokens stashed by VLA.
        for src_key, dst_key in (
            ("reasoning", "openpi_tokenized_reasoning"),
            ("reasoning_mask", "openpi_tokenized_reasoning_mask"),
            ("subtask", "openpi_tokenized_subtask"),
            ("subtask_mask", "openpi_tokenized_subtask_mask"),
            ("openpi_tokenized_fast", "openpi_tokenized_fast"),
            ("openpi_tokenized_fast_mask", "openpi_tokenized_fast_mask"),
            ("fast", "openpi_tokenized_fast"),
            ("fast_mask", "openpi_tokenized_fast_mask"),
        ):
            if src_key in raw:
                out[dst_key] = np.asarray(raw[src_key])
        if "reasoning" in raw:
            out["reasoning"] = np.asarray(raw["reasoning"], dtype=np.int32)
            out["reasoning_mask"] = np.asarray(raw.get("reasoning_mask", raw["reasoning"] != 0), dtype=bool)
        if "subtask" in raw:
            out["subtask"] = np.asarray(raw["subtask"], dtype=np.int32)
            out["subtask_mask"] = np.asarray(raw.get("subtask_mask", raw["subtask"] != 0), dtype=bool)
        if "fast" in raw or "openpi_tokenized_fast" in raw:
            fk = raw.get("openpi_tokenized_fast", raw.get("fast"))
            fmk = raw.get("openpi_tokenized_fast_mask", raw.get("fast_mask"))
            if fk is not None:
                out["openpi_tokenized_fast"] = np.asarray(fk, dtype=np.int32)
                out["fast"] = out["openpi_tokenized_fast"]
            if fmk is not None:
                out["openpi_tokenized_fast_mask"] = np.asarray(fmk, dtype=bool)
                out["fast_mask"] = out["openpi_tokenized_fast_mask"]
        return out

    def _maybe_log_vla_action_support(step: int, raw: dict | None) -> dict[str, Any]:
        if steervla_actor is None or raw is None or not isinstance(raw, dict):
            return {}
        interval = int(agent_config.get("steervla_debug_action_dist_interval", 0))
        num_samples = int(agent_config.get("steervla_debug_action_dist_num_samples", 32))
        if interval <= 0 or step % interval != 0 or num_samples <= 1:
            return {}
        expert_action = raw.get("expert_action")
        if expert_action is None:
            return {}
        try:
            sampled = steervla_actor.sample_action_distribution(raw, num_samples=num_samples, seed=step)
            if not sampled or "policy_output" not in sampled:
                return {}
            expert_flat = np.asarray(expert_action, dtype=np.float32).reshape(-1)
            samples = np.asarray(sampled["policy_output"], dtype=np.float32)
            zero_sample = np.asarray(sampled["zero_policy_output"], dtype=np.float32).reshape(-1)
            if samples.ndim != 2 or samples.shape[1] != expert_flat.size:
                return {}

            diff = samples - expert_flat[None, :]
            l2 = np.linalg.norm(diff, axis=1)
            nearest = int(np.argmin(l2))
            first = min(4, expert_flat.size)
            metrics: dict[str, Any] = {
                "debug_action_support/expert_min_l2": float(np.min(l2)),
                "debug_action_support/expert_mean_l2": float(np.mean(l2)),
                "debug_action_support/expert_zero_l2": float(np.linalg.norm(zero_sample - expert_flat)),
            }
            for i in range(first):
                metrics[f"debug_action_support/expert_first_{i}"] = float(expert_flat[i])
                metrics[f"debug_action_support/sample_mean_first_{i}"] = float(np.mean(samples[:, i]))
                metrics[f"debug_action_support/sample_std_first_{i}"] = float(np.std(samples[:, i]))
                metrics[f"debug_action_support/sample_nearest_first_{i}"] = float(samples[nearest, i])
                metrics[f"debug_action_support/zero_first_{i}"] = float(zero_sample[i])

            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 2, figsize=(10, 6))
                axes = np.asarray(axes).reshape(-1)
                for i in range(4):
                    ax = axes[i]
                    if i < expert_flat.size:
                        ax.hist(samples[:, i], bins=20, color="steelblue", alpha=0.85)
                        ax.axvline(float(expert_flat[i]), color="crimson", linewidth=2)
                        ax.axvline(float(zero_sample[i]), color="darkgreen", linewidth=2, linestyle="--")
                        ax.set_title(f"first-step dim {i}")
                    else:
                        ax.axis("off")
                fig.tight_layout()
                metrics["debug_action_support/first_step_hist"] = wandb.Image(fig)
                plt.close(fig)
            except Exception:
                pass
            return metrics
        except Exception as exc:
            print(f"[vla_action_support] failed: {exc}", flush=True)
            return {}
    
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

    _critic_feedback_mode = str(agent_config.get("critic_feedback_mode", "commentary_bow"))
    _online_training_mode = str(agent_config.get("online_training_mode", "rl")).strip().lower()
    _lang_dim = critic_language_dim(agent_config)
    _residual_warmup = int(agent_config.get("residual_warmup_steps", 0))

    steervla_cfg = agent_config.get("steervla") or {}
    env_ah = int(steervla_cfg.get("action_horizon", agent_config.get("vla_action_horizon", 10)))
    env_ad = int(steervla_cfg.get("action_dim", agent_config.get("vla_action_dim", 4)))
    action_dim = env_ah * env_ad

    # If the agent's executed action is in Pi0 normalized model space, denormalize
    # it to env units before any side-by-side comparison with the expert (which is
    # always stored in env units). Otherwise the delta / MSE are off by ~7x for
    # DELTA_XY formats and bias the critic-feedback labels.
    _exec_cfg = _steervla_action_execution_cfg(steervla_cfg)
    if _exec_cfg is not None and _exec_cfg.get("action_input_space") == "normalized":
        _agent_action_denorm_kwargs: dict[str, Any] | None = {
            "action_horizon": int(_exec_cfg["action_horizon"]),
            "action_dim": int(_exec_cfg["action_dim"]),
            "output_action_format": str(_exec_cfg["output_action_format"]),
        }
    else:
        _agent_action_denorm_kwargs = None
    example_transition = dict(
        observations=np.array(obs),
        actions=np.zeros((action_dim,), dtype=np.float32),
        rewards=np.float32(0.0),
        next_observations=np.array(obs),
        masks=np.float32(1.0),
        terminals=np.float32(0.0),
        language_label=np.zeros(_lang_dim, dtype=np.float32),
        next_language_label=np.zeros(_lang_dim, dtype=np.float32),
    )
    if _online_training_mode in {"sac_residual", "dagger_residual"}:
        # Base Pi0 action used at rollout time; residual = stored action - base.
        # (In dagger_residual the residual is supervised toward expert - base.)
        example_transition["base_actions"] = np.zeros((action_dim,), dtype=np.float32)
    if _online_training_mode == "sac_residual":
        # Pi0(next_obs) computed at rollout time so the SAC critic update does not
        # need to re-run Pi0 inference for every training batch.
        example_transition["base_next_actions"] = np.zeros((action_dim,), dtype=np.float32)
    if steervla_actor is not None:
        openpi0 = _openpi_fields_from_raw(obs_raw)
        example_transition.update(openpi0)
        example_transition.update({f"next_{k}": np.array(v) for k, v in openpi0.items()})
        
    # Create replay buffer
    buffer = ReplayBuffer.create(example_transition, size=capacity)
    
    rng = jax.random.PRNGKey(FLAGS.seed + 1)

    def _sample_base_action_at(obs_np: np.ndarray, obs_raw_: dict) -> np.ndarray | None:
        """Run Pi0 on an arbitrary obs to get its base action (for base_next_actions).

        Temporarily swaps raw_obs_holder so the VLA sample fn sees the right image/state.
        Bypasses the action chunk cache so it never corrupts the rollout cache.
        """
        if (
            agent is None
            or _online_training_mode != "sac_residual"
            or raw_obs_holder is None
            or not hasattr(agent, "vla_sample_fn")
            or agent.vla_sample_fn is None
        ):
            return None
        saved = raw_obs_holder.get("obs")
        raw_obs_holder["obs"] = obs_raw_
        try:
            rng_bn = jax.random.PRNGKey(int(np.random.randint(0, 2**30)))
            noise = jax.random.normal(rng_bn, (1, agent._flat_noise_dim()))
            # Call the underlying _forward_pi0 path directly when possible to skip
            # the chunk cache; otherwise fall back to the public sample fn.
            vla_actor = getattr(agent.vla_sample_fn, "__self__", None)
            if vla_actor is not None and hasattr(vla_actor, "_forward_pi0"):
                base_jax = vla_actor._forward_pi0(1, noise, raw=None)
            else:
                base_jax = jnp.asarray(agent.vla_sample_fn(obs_np[None], noise))
            base_jax = agent._clip_actions_to_env(base_jax)
            return np.asarray(base_jax[0], dtype=np.float32)
        except Exception as _e:
            print(f"[base_next] failed: {_e}", flush=True)
            return None
        finally:
            raw_obs_holder["obs"] = saved

    episode_return, episode_steps, episode_count = 0.0, 0, 0
    episode_collision_count = 0
    episode_collision_events = 0
    prev_collision_count = 0
    last_log_time = time.time()
    episode_video_every = 2
    episode_video_frames: list[np.ndarray] = []
    last_video_reward: float = 0.0
    last_video_critic_text: str = ""

    def _as_video_frame(image: np.ndarray) -> np.ndarray:
        frame = np.asarray(image)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def _viz_image_from_raw(raw: dict[str, Any] | np.ndarray) -> np.ndarray:
        """High-res camera frame for W&B video; falls back to policy ``image``."""
        if isinstance(raw, dict):
            if raw.get("image_viz") is not None:
                return np.asarray(raw["image_viz"], dtype=np.uint8)
            if raw.get("image") is not None:
                return np.asarray(raw["image"], dtype=np.uint8)
        return np.asarray(raw, dtype=np.uint8)

    def _annotate_collision_frame(
        frame: np.ndarray,
        *,
        collision_count: int,
        collision_events: int,
    ) -> np.ndarray:
        annotated = np.array(frame, copy=True)
        try:
            import cv2  # type: ignore

            _, w = annotated.shape[:2]
            label = f"COLL c={collision_count} e={collision_events}"
            font_scale = 0.38
            thickness = 1
            pad = 4
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x1 = w - 6
            x0 = max(6, x1 - tw - 2 * pad)
            y0 = 6
            y1 = y0 + th + baseline + 2 * pad
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 0, 0), thickness=-1)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 255, 255), thickness=1)
            cv2.putText(
                annotated,
                label,
                (x0 + pad, y1 - baseline - pad),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            return annotated
        except Exception:
            return annotated

    def _annotate_reward_corner(frame: np.ndarray, reward_value: float) -> np.ndarray:
        annotated = np.array(frame, copy=True)
        try:
            import cv2  # type: ignore

            label = f"r={reward_value:+.3f}"
            font_scale = 0.38
            thickness = 1
            pad = 4
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x0, y0 = 6, 6
            x1 = x0 + tw + 2 * pad
            y1 = y0 + th + baseline + 2 * pad
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 255, 255), thickness=1)
            cv2.putText(
                annotated,
                label,
                (x0 + pad, y1 - baseline - pad),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            return annotated
        except Exception:
            return annotated

    def _format_text_field(raw: dict[str, Any] | None, key: str) -> str:
        
        if not isinstance(raw, dict) or key not in raw or raw.get(key) is None:
            return ""
        value = raw.get(key)
        
        if isinstance(value, str):
            return value
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return ""
        if arr.dtype == bool:
            return " ".join(map(str, arr.astype(np.int32)[:16].tolist()))
        # Token ids or numeric payload fallback.
        return " ".join(map(str, arr.astype(np.int32)[:24].tolist()))

    def _critic_input_text(
        critic_mode: str,
        critic_label: np.ndarray,
        critic_text: str,
        raw: dict[str, Any] | None,
    ) -> str:
        if critic_mode == "none":
            return "none"
        if critic_mode == "action_delta":
            arr = np.asarray(critic_label, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return "[]"
            show = " ".join(f"{v:+.3f}" for v in arr[:8])
            return show if arr.size <= 8 else f"{show} ..."
        if critic_mode == "delta_commentary_bow":
            return critic_text or "?"
        commentary = raw.get("commentary_text", "") if isinstance(raw, dict) else ""
        return str(commentary or "?")

    def _annotate_text_panel(
        frame: np.ndarray,
        raw: dict[str, Any] | None,
        *,
        reward_value: float,
        critic_text: str,
    ) -> np.ndarray:
        base = np.array(frame, copy=True)
        try:
            import cv2  # type: ignore

            h, w = base.shape[:2]
            font_scale = 0.26
            line_h = 13
            panel_h = max(72, line_h * 6)
            annotated = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
            annotated[:h, :, :] = _annotate_reward_corner(base, reward_value)
            # Bottom panel is already black via zeros; draw an explicit border line.
            cv2.line(annotated, (0, h), (w - 1, h), (255, 255, 255), 1)

            state = np.asarray(raw.get("state", []), dtype=np.float32).reshape(-1) if isinstance(raw, dict) else np.zeros((0,), dtype=np.float32)
            speed = float(state[15]) if state.size > 15 else 0.0
            routing = ""
            if isinstance(raw, dict):
                routing = str(raw.get("routing_command", "") or "").strip()
            prompt = f"The current speed is {speed:.2f} m/s. {routing or 'Follow the route.'}"
            reasoning = _format_text_field(raw, "reasoning_text") or _format_text_field(raw, "reasoning")
            subtask = _format_text_field(raw, "subtask_text") or _format_text_field(raw, "subtask")
            expert_action_str = ""
            agent_action_str = ""
            if isinstance(raw, dict):
                ea = raw.get("expert_action")
                if ea is not None:
                    ea = np.asarray(ea, dtype=np.float32).reshape(-1)
                    # First step is dims 0-3: [dx_speed, dy_speed, dx_route, dy_route]
                    first = ea[:4] if ea.size >= 4 else ea
                    expert_action_str = " ".join(f"{v:.3f}" for v in first)
                aa = raw.get("agent_action")
                if aa is not None:
                    aa = np.asarray(aa, dtype=np.float32).reshape(-1)
                    first = aa[:4] if aa.size >= 4 else aa
                    agent_action_str = " ".join(f"{v:.3f}" for v in first)

            def _clip_text(txt: str, max_chars: int = 120) -> str:
                return txt if len(txt) <= max_chars else (txt[: max_chars - 3] + "...")

            lines = [
                f"Expert: {_clip_text(critic_text) if critic_text else '?'}",
                f"ExpertAct[0]: {expert_action_str or '?'}",
                f"AgentAct[0]:  {agent_action_str or '?'}",
                f"Prompt: {_clip_text(prompt)}",
                f"Reasoning: {_clip_text(reasoning)}",
                f"Subtask: {_clip_text(subtask)}",
            ]
            y = h + line_h
            for line in lines:
                cv2.putText(
                    annotated,
                    line,
                    (4, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                y += line_h
            return annotated
        except Exception:
            return base

    def _maybe_log_episode_video(
        rollout_log: dict,
        final_frame: np.ndarray | None,
        final_raw: dict[str, Any] | None,
        *,
        final_reward: float,
        final_critic_text: str,
    ) -> None:
        if not log_images:
            return
        frames = list(episode_video_frames)
        if final_frame is not None:
            frames.append(
                _annotate_text_panel(
                    _as_video_frame(final_frame),
                    final_raw,
                    reward_value=final_reward,
                    critic_text=final_critic_text,
                )
            )
        if not frames:
            return
        video = np.stack(frames, axis=0)
        if video.ndim == 4:
            # W&B expects (T, C, H, W) for videos.
            video = np.transpose(video, (0, 3, 1, 2))
        rollout_log["rollout/episode_video"] = wandb.Video(video, fps=10, format="mp4")

    def _block_until_ready_tree(tree):
        return jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            tree,
        )

    last_update_info = None

    def _sample_agent_action(subkey):
        """Rollout policy (SteerVLA VLA path, DAgger, or DSRL flow); used in warmup and RL phases.

        Returns ``(action, base_action)``. ``base_action`` is the un-residualized
        Pi0 action chunk for ``sac_residual`` / ``dagger_residual`` mode; ``None`` otherwise.
        """
        if _online_training_mode == "dagger_direct" and hasattr(agent, "sample_actions_vla_direct"):
            return agent.sample_actions_vla_direct(obs[None], seed=subkey), None
        if _online_training_mode in {"sac_residual", "dagger_residual"} and hasattr(agent, "sample_actions_sac_residual"):
            if step <= _residual_warmup:
                # During warmup execute pure Pi0 with zero residual so random-init
                # MLP weights don't corrupt the base policy's driving quality.
                noise = jax.random.normal(subkey, (1, agent._flat_noise_dim()))
                base = agent._clip_actions_to_env(
                    jnp.asarray(agent.vla_sample_fn(obs[None], noise))
                )
                return base, base
            # Match rollout to the training objective:
            # - SAC residual explores stochastically.
            # - DAgger residual executes the deterministic mean residual that its MSE loss trains.
            temperature = 0.0 if _online_training_mode == "dagger_residual" else 1.0
            return agent.sample_actions_sac_residual(obs[None], seed=subkey, temperature=temperature)
        if getattr(agent, "vla_sample_fn", None) is not None:
            return agent.sample_actions_with_vla(obs[None], seed=subkey), None
        return agent.sample_actions(obs[None], seed=subkey), None

    _vla_steps_budget = int(np.random.randint(70, 201)) if FLAGS.expert_recover_debug else 0
    if FLAGS.expert_recover_debug:
        print(f"[expert_recover_debug] episode 0: VLA for {_vla_steps_budget} steps then expert", flush=True)

    for step in tqdm.tqdm(range(1, FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        t_sample_start = time.time()
        if raw_obs_holder is not None:
            raw_obs_holder["obs"] = obs_raw
        rng, sub = jax.random.split(rng)
        _in_expert_recovery = FLAGS.expert_recover_debug and (episode_steps >= _vla_steps_budget)
        if FLAGS.expert_recover_debug and (episode_steps == _vla_steps_budget):
            env.reinit_expert()
        in_warmup = warmup > 0 and step <= warmup
        base_action_np: np.ndarray | None = None
        if FLAGS.expert_debug or _in_expert_recovery or agent is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action_jax, base_action_jax = _sample_agent_action(sub)
            _block_until_ready_tree((action_jax, base_action_jax))
            action = np.asarray(action_jax[0])
            if base_action_jax is not None:
                base_action_np = np.asarray(base_action_jax[0], dtype=np.float32)
        t_sample_end = time.time()

        t_step_start = time.time()
        if FLAGS.expert_debug or _in_expert_recovery:
            next_obs_raw, reward, terminated, truncated, info = env.step_expert(obs_raw)
        else:
            next_obs_raw, reward, terminated, truncated, info = env.step(action)
            # Keep SimLingo warm: populates last_driving_data for live_expert labels
            # and pre-initializes the autopilot at the correct position for clean takeover
            env.tick_expert()
        if raw_obs_holder is not None:
            raw_obs_holder["next_obs"] = next_obs_raw
        drive_metrics = ego_drive_metrics_from_state_vec(next_obs_raw["state"])
        next_obs = _extract_agent_obs(env, next_obs_raw, obs_mode)
        done = bool(terminated or truncated)
        end_img = np.copy(_viz_image_from_raw(next_obs_raw)) if done and log_images else None

        # Compute critic language label for this transition.
        # raw_obs_holder["obs"] is still s_t here (not yet updated to s_{t+1}).
        _zero_label = np.zeros(_lang_dim, dtype=np.float32)
        _lang_text = ""
        if FLAGS.expert_debug or _in_expert_recovery or agent is None:
            _lang = _zero_label
            _next_lang = _zero_label
        elif _critic_feedback_mode == "none":
            _lang = _zero_label
            _next_lang = _zero_label
        elif _critic_feedback_mode == "action_delta":
            _lang = compute_action_delta(
                obs_raw, action, agent, agent_config,
                denormalize_kwargs=_agent_action_denorm_kwargs,
            )
            _next_lang = _zero_label  # bootstrap target sees zero delta (next action unknown)
        elif _critic_feedback_mode == "delta_commentary_bow":
            _lang_text, _lang = compute_action_delta_commentary(
                obs_raw, action, agent,
                denormalize_kwargs=_agent_action_denorm_kwargs,
            )
            _next_lang = _zero_label  # depends on current action-vs-expert comparison only
        else:
            _lang = np.asarray(obs_raw.get("language_label", _zero_label), dtype=np.float32)
            _next_lang = np.asarray(next_obs_raw.get("language_label", _zero_label), dtype=np.float32)
        _critic_text_for_video = _critic_input_text(_critic_feedback_mode, _lang, _lang_text, obs_raw)

        replay_action = action.astype(np.float32)
        if _online_training_mode in {"dagger_direct", "dagger_residual"} and not FLAGS.expert_debug:
            replay_action = np.asarray(obs_raw.get("expert_action", replay_action), dtype=np.float32)
            if _online_training_mode == "dagger_residual" and _agent_action_denorm_kwargs is not None:
                # expert_action is in physical env units (delta-xy meters); base_actions
                # are in Pi0 normalized space (divided by 7 for DELTA_XY format).
                # Normalize expert to the same space so the MSE loss gradient is correct.
                from vlas.steervla import _normalize_action_chunk_numpy
                replay_action = _normalize_action_chunk_numpy(
                    replay_action[None], **_agent_action_denorm_kwargs
                )[0]

        # Convert executed/base actions to env units so we can compare/log against
        # the expert (which is always in env units). Pi0 outputs are in normalized
        # model space when ``action_input_space="normalized"`` is set on the env.
        _action_flat_env = np.asarray(action, dtype=np.float32).reshape(-1)
        _base_flat_env = (
            np.asarray(base_action_np, dtype=np.float32).reshape(-1)
            if base_action_np is not None
            else None
        )
        if _agent_action_denorm_kwargs is not None:
            _expected_size = (
                int(_agent_action_denorm_kwargs["action_horizon"])
                * int(_agent_action_denorm_kwargs["action_dim"])
            )
            if _action_flat_env.size == _expected_size:
                _action_flat_env = denormalize_action_chunk(
                    _action_flat_env, **_agent_action_denorm_kwargs,
                )
            if _base_flat_env is not None and _base_flat_env.size == _expected_size:
                _base_flat_env = denormalize_action_chunk(
                    _base_flat_env, **_agent_action_denorm_kwargs,
                )

        # MSE between the actions and the expert label for the *current* state.
        # ``obs_raw`` here is still the state we acted on (it is reassigned to
        # ``next_obs_raw`` further below).
        expert_mse_metrics: dict[str, float] = {}
        _expert_action_raw = obs_raw.get("expert_action") if isinstance(obs_raw, dict) else None
        if _expert_action_raw is not None and not FLAGS.expert_debug:
            _expert_flat = np.asarray(_expert_action_raw, dtype=np.float32).reshape(-1)
            if _action_flat_env.shape == _expert_flat.shape:
                expert_mse_metrics["rollout/mse_action_to_expert"] = float(
                    np.mean(np.square(_action_flat_env - _expert_flat))
                )
            if _base_flat_env is not None and _base_flat_env.shape == _expert_flat.shape:
                expert_mse_metrics["rollout/mse_base_to_expert"] = float(
                    np.mean(np.square(_base_flat_env - _expert_flat))
                )

        residual_fields: dict[str, np.ndarray] = {}
        if _online_training_mode in {"sac_residual", "dagger_residual"}:
            # If the agent path was skipped (warmup with no agent, expert override),
            # fall back to the stored action so the residual = 0 implicitly.
            residual_fields["base_actions"] = (
                base_action_np if base_action_np is not None else replay_action
            )
        if _online_training_mode == "sac_residual":
            # Compute Pi0(next_obs) now at rollout time so the SAC training update
            # can look up base_next_actions from the batch without re-running Pi0
            # inference for every gradient step (eliminates VLA call from the hot path).
            _base_next = _sample_base_action_at(next_obs, next_obs_raw)
            residual_fields["base_next_actions"] = (
                _base_next if _base_next is not None
                else np.zeros(action_dim, dtype=np.float32)
            )
        buffer.add_transition(
            {
                "observations": np.asarray(obs),
                "actions": replay_action,
                "rewards": np.float32(reward),
                "next_observations": np.asarray(next_obs),
                "masks": np.float32(0.0 if terminated else 1.0),
                "terminals": np.float32(1.0 if done else 0.0),
                "language_label": _lang,
                "next_language_label": _next_lang,
                **residual_fields,
                **(_openpi_fields_from_raw(obs_raw) if steervla_actor is not None else {}),
                **(
                    {f"next_{k}": np.array(v) for k, v in _openpi_fields_from_raw(next_obs_raw).items()}
                    if steervla_actor is not None
                    else {}
                ),
            }
        )
        t_step_end = time.time()
        
        t_log_start = time.time()
        cot_obs_raw = dict(obs_raw)  # holds reasoning_text/subtask_text stashed by VLA
        # Stash the executed agent action (env units) for the video text panel so
        # AgentAct[0] is directly comparable to ExpertAct[0].
        cot_obs_raw["agent_action"] = _action_flat_env
        obs = next_obs
        obs_raw = next_obs_raw
        episode_return += float(reward)
        episode_steps += 1
        collision_count = int(info.get("collision_count", 0))
        episode_collision_count = max(episode_collision_count, collision_count)
        collision_delta = max(0, collision_count - prev_collision_count)
        episode_collision_events += collision_delta
        prev_collision_count = collision_count
        if log_images:
            should_sample_periodic = episode_steps % episode_video_every == 0
            had_collision_this_step = collision_delta > 0
            if should_sample_periodic or had_collision_this_step:
                frame = _as_video_frame(_viz_image_from_raw(obs_raw))
                frame = _annotate_text_panel(
                    frame,
                    cot_obs_raw,
                    reward_value=float(reward),
                    critic_text=_critic_text_for_video,
                )
                if had_collision_this_step:
                    frame = _annotate_collision_frame(
                        frame,
                        collision_count=collision_count,
                        collision_events=episode_collision_events,
                    )
                episode_video_frames.append(frame)
        last_video_reward = float(reward)
        last_video_critic_text = _critic_text_for_video
        t_log_end = time.time()

        step_wb = {f"rollout/{k}": v for k, v in drive_metrics.items()}
        step_wb["rollout/collision_count"] = float(collision_count)
        step_wb["rollout/collision_events"] = float(collision_delta)
        step_wb.update(_maybe_log_vla_action_support(step, obs_raw))
        if "reward_total" in info:
            step_wb["reward/total"] = float(info["reward_total"])
            step_wb["reward/terminal"] = float(info.get("reward_terminal", 0.0))
            step_wb["reward/penalty_collision"] = float(info.get("penalty_collision", 0.0))
            step_wb["reward/penalty_outside_route"] = float(info.get("penalty_outside_route", 0.0))
            step_wb["reward/penalty_steer"] = float(info.get("penalty_steer", 0.0))
            step_wb["reward/penalty_brake"] = float(info.get("penalty_brake", 0.0))
            step_wb["reward/penalty_speed_limit"] = float(info.get("penalty_speed_limit", 0.0))
            step_wb["reward/penalty_crash_stuck"] = float(info.get("penalty_crash_stuck", 0.0))
            step_wb["rollout/route_completion"] = float(info.get("route_completion", 0.0))
            step_wb["rollout/route_completion_delta"] = float(info.get("route_completion_delta", 0.0))
            step_wb["rollout/termination_reason"] = str(info.get("termination_reason", ""))
            step_wb["reward/soft_penalty_outside_lanes"] = float(info.get("soft_penalty_outside_lanes", 1.0))
            step_wb["reward/soft_penalty_lane_center"] = float(info.get("soft_penalty_lane_center", 1.0))
            step_wb["reward/soft_penalty_speeding"] = float(info.get("soft_penalty_speeding", 1.0))
            step_wb["reward/soft_penalty_ttc"] = float(info.get("soft_penalty_ttc", 1.0))
            step_wb["reward/soft_penalty_comfort"] = float(info.get("soft_penalty_comfort", 1.0))
            step_wb["reward/soft_penalty_product"] = float(info.get("soft_penalty_product", 1.0))
            step_wb["rollout/lane_offset_m"] = float(info.get("lane_offset_m", 0.0))
            step_wb["rollout/heading_error_rad"] = float(info.get("heading_error_rad", 0.0))

        # Log critic feedback signal (obs_raw is already next_obs_raw here)
        if _critic_feedback_mode == "action_delta":
            step_wb["label/action_delta_norm"] = float(np.linalg.norm(_lang))
        elif _critic_feedback_mode == "delta_commentary_bow":
            if _lang_text:
                step_wb["label/commentary_delta"] = wandb.Html(f"<p>{_lang_text}</p>")
        else:
            _commentary = obs_raw.get("commentary_text", "") if isinstance(obs_raw, dict) else ""
            if _commentary:
                step_wb["label/commentary"] = wandb.Html(f"<p>{_commentary}</p>")

        step_wb["time/sample_time"] = t_sample_end - t_sample_start
        step_wb["time/step_time"] = t_step_end - t_step_start
        step_wb["time/log_time"] = t_log_end - t_log_start
        step_wb["training/in_warmup"] = float(in_warmup)
        step_wb.update(expert_mse_metrics)

        wandb.log(step_wb, step=step)

        if done:
            episode_count += 1
            rollout_log = {
                "rollout/episode_return": episode_return,
                "rollout/episode_steps": episode_steps,
                "rollout/episodes": episode_count,
                "rollout/route": info.get("route", "?"),
                "rollout/episode_collision_count": float(episode_collision_count),
                "rollout/episode_collision_events": float(episode_collision_events),
                "rollout/collisions_over_episode": float(episode_collision_events) / max(float(episode_steps), 1.0),
            }
            if "reward_total" in info:
                rollout_log["rollout/final_step_reward"] = float(info["reward_total"])
                rollout_log["rollout/final_step_reward_terminal"] = float(info.get("reward_terminal", 0.0))
                rollout_log["rollout/route_completion"] = float(info.get("route_completion", 0.0))
                rollout_log["rollout/route_completion_delta"] = float(info.get("route_completion_delta", 0.0))
                rollout_log["rollout/termination_reason"] = str(info.get("termination_reason", ""))
                rollout_log["reward/soft_penalty_outside_lanes"] = float(info.get("soft_penalty_outside_lanes", 1.0))
                rollout_log["reward/soft_penalty_lane_center"] = float(info.get("soft_penalty_lane_center", 1.0))
                rollout_log["reward/soft_penalty_speeding"] = float(info.get("soft_penalty_speeding", 1.0))
                rollout_log["reward/soft_penalty_ttc"] = float(info.get("soft_penalty_ttc", 1.0))
                rollout_log["reward/soft_penalty_comfort"] = float(info.get("soft_penalty_comfort", 1.0))
                rollout_log["reward/soft_penalty_product"] = float(info.get("soft_penalty_product", 1.0))
                rollout_log["rollout/final_step_success"] = float(bool(info.get("success", False)))
            if FLAGS.expert_recover_debug:
                rollout_log["rollout/vla_steps_budget"] = float(_vla_steps_budget)
            _maybe_log_episode_video(
                rollout_log,
                end_img if log_images else None,
                cot_obs_raw if log_images else None,
                final_reward=last_video_reward,
                final_critic_text=last_video_critic_text,
            )
            wandb.log(rollout_log, step=step)
            obs_raw, _info = env.reset(seed=FLAGS.seed + episode_count)
            if raw_obs_holder is not None:
                raw_obs_holder["obs"] = obs_raw
                raw_obs_holder["next_obs"] = obs_raw
                
            if agent is not None:
                reset_vla_cache = getattr(getattr(agent, "vla_sample_fn", None), "reset_action_cache", None)
                if reset_vla_cache is not None:
                    reset_vla_cache()
            obs = _extract_agent_obs(env, obs_raw, obs_mode)
            episode_video_frames = []
            episode_return, episode_steps = 0.0, 0
            episode_collision_count = 0
            episode_collision_events = 0
            prev_collision_count = 0
            if FLAGS.expert_recover_debug:
                _vla_steps_budget = int(np.random.randint(70, 201))
                print(
                    f"[expert_recover_debug] episode {episode_count}: VLA for {_vla_steps_budget} steps then expert",
                    flush=True,
                )

        update_times = []
        if (
            (not FLAGS.expert_debug)
            and agent is not None
            and not in_warmup
            and buffer.size >= batch_size
        ):
            
            for _ in range(updates_per_step):
                t_update_start = time.time()
                batch = buffer.sample(batch_size)
                if _online_training_mode == "dagger_direct":
                    _, update_info = agent.update_dagger_direct(batch)
                elif _online_training_mode == "sac_residual":
                    agent, update_info = agent.update_sac_residual(batch)
                elif _online_training_mode == "dagger_residual":
                    agent, update_info = agent.update_dagger_residual(batch)
                elif getattr(agent, "vla_sample_fn", None) is not None:
                    _, update_info = agent.update_with_vla(batch)
                else:
                    _, update_info = agent.update(batch)
                _block_until_ready_tree((agent, update_info))
                t_update_end = time.time()
                update_times.append(t_update_end - t_update_start)
            last_update_info = update_info
            

        if step % FLAGS.log_interval == 0:
            metrics = {
                "time/steps_per_sec": FLAGS.log_interval / max(time.time() - last_log_time, 1e-6),
                "time/update_time": np.mean(update_times),
            }
            if last_update_info is not None:
                metrics.update({f"training/{k}": float(v) for k, v in last_update_info.items()})
                metrics["training/buffer_size"] = int(buffer.size)
            last_log_time = time.time()
            wandb.log(metrics, step=step)
            train_logger.log(metrics, step=step)

        if agent is not None and step % FLAGS.save_interval == 0:
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

    exp_name = os.environ.get("WANDB_RUN_NAME", get_exp_name(FLAGS.seed))
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
    online_training_mode = str(config.get("online_training_mode", "rl")).strip().lower()
    _VALID_TRAIN_MODES = {"rl", "dagger_direct", "sac_residual", "dagger_residual"}
    if online_training_mode not in _VALID_TRAIN_MODES:
        raise ValueError(
            f"Unsupported online_training_mode={online_training_mode!r}; "
            f"expected one of {sorted(_VALID_TRAIN_MODES)}."
        )
    use_steervla_rollout = bool(
        steervla_cfg is not None and steervla_cfg.get("enabled") and not FLAGS.expert_debug
    )
    if online_training_mode == "dagger_direct":
        if use_steervla_rollout:
            print(
                "[main_carla] DAgger mode: rolling out SteerVLA with expert relabels.",
                flush=True,
            )
        else:
            print(
                "[main_carla] DAgger mode requested but SteerVLA rollout is disabled; falling back to learner rollout for data collection.",
                flush=True,
            )
    if online_training_mode == "sac_residual":
        print(
            "[main_carla] SAC residual mode: Pi0 frozen; small residual MLP trained via "
            "Q-gradient from DSRL critic.",
            flush=True,
        )
    if online_training_mode == "dagger_residual":
        print(
            "[main_carla] DAgger residual mode: Pi0 frozen; small residual MLP supervised "
            "via MSE toward expert action.",
            flush=True,
        )
    if online_training_mode in {"sac_residual", "dagger_residual"} and bool(config.get("residual_use_pi_image_features", False)):
        residual_pi_feature_source = str(
            config.get("residual_pi_feature_source", "prefix")
        ).strip().lower()
        print(
            f"[main_carla] Residual actor input: frozen Pi {residual_pi_feature_source} features.",
            flush=True,
        )
    if bool(config.get("critic_use_pi_prefix_features", False)):
        print(
            "[main_carla] Critic input: frozen Pi prefix features.",
            flush=True,
        )
    critic_feedback_mode = str(config.get("critic_feedback_mode", "commentary_bow"))
    if critic_feedback_mode == "none":
        config.language_label_dim = 0
    elif critic_feedback_mode == "delta_commentary_bow":
        config.language_label_dim = NUM_DELTA_COMMENTARY_WORDS
    elif critic_feedback_mode == "commentary_bow":
        config.language_label_dim = NUM_COMMENTARY_WORDS
    extra_carla: dict[str, Any] = {}
    exec_cfg = _steervla_action_execution_cfg(steervla_cfg)
    if exec_cfg is not None:
        extra_carla["steervla_action_execution"] = exec_cfg
    if FLAGS.expert_debug or FLAGS.expert_recover_debug or online_training_mode == "dagger_direct":
        extra_carla["expert_controller"] = "simlingo_autopilot"

    # Leaderboard starts CARLA with subprocess (fork + exec). JAX initializes a native
    # thread pool; forking afterward triggers the stdlib warning and can deadlock the child,
    # which often surfaces as UE4 "RenderThread" timeouts. Bring the simulator up first.
    env = _make_carla_env(carla_yaml, FLAGS.route, extra_carla_config=extra_carla or None)
    try:
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
        if use_steervla_rollout or FLAGS.expert_debug or FLAGS.expert_recover_debug:
            raw_carla_holder = {"obs": obs_dict, "next_obs": obs_dict}

        steervla_actor = None
        agent = None
        if not FLAGS.expert_debug:
            agent_obs = _extract_agent_obs(env, obs_dict, obs_mode)
            ex_obs = np.expand_dims(agent_obs, 0)
            ex_actions = np.zeros((1,) + tuple(env.action_space.shape), dtype=np.float32)

            _configure_jax_training_device(int(config.get("training_gpu_rank", -1)))

            agent_class = agents[config["agent_name"]]
            create_kwargs = {}
            if config["agent_name"] == "dsrl":
                tr_rank = int(config.get("training_gpu_rank", -1))
                vla_bundle = None
                if use_steervla_rollout:
                    vla_bundle = _build_vla_sample_fn(
                        steervla_cfg, raw_carla_holder, training_gpu_rank=tr_rank
                    )
                if vla_bundle is not None:
                    vla_sample_fn, steervla_actor = vla_bundle
                    create_kwargs["vla_sample_fn"] = vla_sample_fn
                    url = steervla_cfg.get("actor_url") if steervla_cfg else None
                    if not (url and str(url).strip()):
                        # create_kwargs["vla_train_state"] = steervla_actor.train_state
                        create_kwargs["openpi_train_config"] = steervla_actor.train_cfg
                        create_kwargs["steervla_actor"] = steervla_actor
                if bool(config.get("critic_use_pi_prefix_features", False)):
                    if steervla_actor is None:
                        raise ValueError(
                            "critic_use_pi_prefix_features=True requires SteerVLA rollout."
                        )
                    if getattr(steervla_actor, "_remote", None) is not None:
                        raise ValueError(
                            "critic_use_pi_prefix_features=True requires local SteerVLA; "
                            "remote actor mode does not expose Pi prefix features."
                        )

            agent = agent_class.create(FLAGS.seed, ex_obs, ex_actions, config, **create_kwargs)

            if online_training_mode in {"sac_residual", "dagger_residual"}:
                if config["agent_name"] != "dsrl":
                    raise ValueError(
                        f"{online_training_mode} mode requires agent_name='dsrl'."
                    )
                if steervla_actor is None:
                    raise ValueError(
                        f"{online_training_mode} mode requires SteerVLA rollout (frozen Pi0 base policy)."
                    )
                if bool(config.get("residual_use_pi_image_features", False)):
                    if getattr(steervla_actor, "_remote", None) is not None:
                        raise ValueError(
                            "residual_use_pi_image_features=True requires local SteerVLA; "
                            "remote actor mode does not expose Pi residual features."
                        )
                    openpi_obs = steervla_actor.build_observation_batch_numpy(
                        batch_size=1, raw=obs_dict,
                    )
                    residual_pi_feature_source = str(
                        config.get("residual_pi_feature_source", "prefix")
                    ).strip().lower()
                    base_action_probe = np.zeros(
                        (1, int(config.get("vla_action_horizon", 10)) * int(config.get("vla_action_dim", 4))),
                        dtype=np.float32,
                    )
                    if residual_pi_feature_source == "prefix":
                        embed_dim = int(steervla_actor.encode_prefix_features(openpi_obs).shape[-1])
                    elif residual_pi_feature_source == "suffix":
                        embed_dim = int(
                            steervla_actor.encode_suffix_features(openpi_obs, base_action_probe).shape[-1]
                        )
                    else:
                        raise ValueError(
                            f"Unsupported residual_pi_feature_source={residual_pi_feature_source!r}; "
                            "expected 'prefix' or 'suffix'."
                        )
                else:
                    obs_mode_cfg = str(config.get("observation_mode", "state"))
                    if obs_mode_cfg == "state":
                        embed_dim = int(ex_obs.shape[-1])
                    else:
                        embed_dim = int(tuple(config.get("image_mlp_hidden_dims", (512,)))[-1])
                sac_residual_agent = SACResidualAgent.create(
                    FLAGS.seed, ex_obs, ex_actions, config, embed_dim=embed_dim,
                )
                agent = agent.attach_sac_residual(sac_residual_agent)

            if FLAGS.restore_path is not None:
                agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

        if FLAGS.eval_only:
            # No offline-eval pipeline yet for CARLA; do a single rollout.
            FLAGS.online_steps = max(FLAGS.online_steps, 200)
            FLAGS.save_buffer = FLAGS.save_buffer or False

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
