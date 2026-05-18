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
from pathlib import Path
from typing import Any, Optional

import numpy as np
import jax

from jax_agents import agents
from utils.flax_utils import restore_agent
from coaches.expert_label import NUM_COMMENTARY_WORDS, NUM_DELTA_COMMENTARY_WORDS
from coaches.critic_feedback import (
    compute_action_delta,
    compute_action_delta_commentary,
    critic_language_dim,
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
    
    # Get openpi fields from raw observation
    def _openpi_fields_from_raw(raw: dict | None) -> dict[str, np.ndarray]:
        if (
            steervla_actor is None
            or getattr(steervla_actor, "model_cfg", None) is None
            or raw is None
            or not isinstance(raw, dict)
        ):
            return {}

        obs_struct = steervla_actor.build_observation_batch_numpy(batch_size=1, raw=raw)
        out = {
            "openpi_image_base_0_rgb": np.asarray(obs_struct.images["base_0_rgb"][0], dtype=np.uint8),
            "openpi_image_mask_base_0_rgb": np.asarray(obs_struct.image_masks["base_0_rgb"][0], dtype=bool),
            "openpi_state": np.asarray(obs_struct.state[0], dtype=np.float32),
            "openpi_tokenized_prompt": np.asarray(obs_struct.tokenized_prompt[0], dtype=np.int32),
            "openpi_tokenized_prompt_mask": np.asarray(obs_struct.tokenized_prompt_mask[0], dtype=bool),
            "openpi_tokenized_reasoning": np.asarray(obs_struct.tokenized_reasoning[0], dtype=np.int32),
            "openpi_tokenized_reasoning_mask": np.asarray(obs_struct.tokenized_reasoning_mask[0], dtype=bool),
            "openpi_tokenized_subtask": np.asarray(obs_struct.tokenized_subtask[0], dtype=np.int32),
            "openpi_tokenized_subtask_mask": np.asarray(obs_struct.tokenized_subtask_mask[0], dtype=bool),
            # Keep legacy key names for CoT for backwards-compat with existing buffers.
            "reasoning": np.asarray(obs_struct.tokenized_reasoning[0], dtype=np.int32),
            "reasoning_mask": np.asarray(obs_struct.tokenized_reasoning_mask[0], dtype=bool),
            "subtask": np.asarray(obs_struct.tokenized_subtask[0], dtype=np.int32),
            "subtask_mask": np.asarray(obs_struct.tokenized_subtask_mask[0], dtype=bool),
        }
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

    _critic_feedback_mode = str(agent_config.get("critic_feedback_mode", "commentary_bow"))
    _online_training_mode = str(agent_config.get("online_training_mode", "rl")).strip().lower()
    _lang_dim = critic_language_dim(agent_config)

    action_dim = int(agent_config.get("action_dim", 4)*agent_config.get("actions_horizon", 10))
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
    if steervla_actor is not None:
        openpi0 = _openpi_fields_from_raw(obs_raw)
        example_transition.update(openpi0)
        example_transition.update({f"next_{k}": np.array(v) for k, v in openpi0.items()})
        
    # Create replay buffer
    buffer = ReplayBuffer.create(example_transition, size=capacity)
    
    rng = jax.random.PRNGKey(FLAGS.seed + 1)
    episode_return, episode_steps, episode_count = 0.0, 0, 0
    episode_collision_count = 0
    episode_collision_events = 0
    prev_collision_count = 0
    last_log_time = time.time()
    episode_video_every = 5
    episode_video_frames: list[np.ndarray] = []
    last_video_reward: float = 0.0
    last_video_critic_text: str = ""

    def _as_video_frame(image: np.ndarray) -> np.ndarray:
        frame = np.asarray(image)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def _annotate_collision_frame(
        frame: np.ndarray,
        *,
        collision_count: int,
        collision_events: int,
    ) -> np.ndarray:
        annotated = np.array(frame, copy=True)
        try:
            import cv2  # type: ignore

            h, w = annotated.shape[:2]
            bar_h = max(24, h // 12)
            cv2.rectangle(annotated, (0, 0), (w, bar_h), (0, 0, 255), thickness=-1)
            label = f"COLLISION count={collision_count} events={collision_events}"
            cv2.putText(
                annotated,
                label,
                (10, max(18, bar_h - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return annotated
        except Exception:
            # Fallback: mark top strip red even if cv2 text rendering is unavailable.
            h = annotated.shape[0]
            bar_h = max(8, h // 20)
            annotated[:bar_h, :, :] = np.array([255, 0, 0], dtype=np.uint8)
            return annotated

    def _annotate_reward_corner(frame: np.ndarray, reward_value: float) -> np.ndarray:
        annotated = np.array(frame, copy=True)
        try:
            import cv2  # type: ignore

            label = f"r={reward_value:+.3f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            pad = 6
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
                0.55,
                (255, 255, 255),
                2,
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
            panel_h = max(96, h // 4)  # 5 lines × 16px + 16px offset
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
            if isinstance(raw, dict):
                ea = raw.get("expert_action")
                if ea is not None:
                    ea = np.asarray(ea, dtype=np.float32).reshape(-1)
                    # First step is dims 0-3: [dx_speed, dy_speed, dx_route, dy_route]
                    first = ea[:4] if ea.size >= 4 else ea
                    expert_action_str = " ".join(f"{v:.3f}" for v in first)

            def _clip_text(txt: str, max_chars: int = 120) -> str:
                return txt if len(txt) <= max_chars else (txt[: max_chars - 3] + "...")

            lines = [
                f"Expert: {_clip_text(critic_text) if critic_text else '?'}",
                f"ExpertAct[0]: {expert_action_str or '?'}",
                f"Prompt: {_clip_text(prompt)}",
                f"Reasoning: {_clip_text(reasoning)}",
                f"Subtask: {_clip_text(subtask)}",
            ]
            font_scale = 0.32
            line_h = 16
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
        if FLAGS.expert_debug or _in_expert_recovery or agent is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        elif step <= warmup:
            action = env.action_space.sample()
        else:
            # [STEP 1] Sample the action from the agent
            if getattr(agent, "vla_sample_fn", None) is not None:
                action_jax = agent.sample_actions_with_vla(obs[None], seed=sub)
            elif _online_training_mode == "dagger" and hasattr(agent, "sample_actions_dagger"):
                action_jax = agent.sample_actions_dagger(obs[None])
            else:
                action_jax = agent.sample_actions(obs[None], seed=sub)
            _block_until_ready_tree(action_jax)
            action = np.asarray(action_jax[0])
        t_sample_end = time.time()

        t_step_start = time.time()
        if FLAGS.expert_debug or _in_expert_recovery:
            next_obs_raw, reward, terminated, truncated, info = env.step_expert(obs_raw)
        else:
            next_obs_raw, reward, terminated, truncated, info = env.step(action)
        if raw_obs_holder is not None:
            raw_obs_holder["next_obs"] = next_obs_raw
        drive_metrics = ego_drive_metrics_from_state_vec(next_obs_raw["state"])
        next_obs = _extract_agent_obs(env, next_obs_raw, obs_mode)
        done = bool(terminated or truncated)
        end_img = np.copy(next_obs) if done and log_images else None

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
            _lang = compute_action_delta(obs_raw, action, agent, agent_config)
            _next_lang = _zero_label  # bootstrap target sees zero delta (next action unknown)
        elif _critic_feedback_mode == "delta_commentary_bow":
            _lang_text, _lang = compute_action_delta_commentary(obs_raw, action, agent)
            _next_lang = _zero_label  # depends on current action-vs-expert comparison only
        else:
            _lang = np.asarray(obs_raw.get("language_label", _zero_label), dtype=np.float32)
            _next_lang = np.asarray(next_obs_raw.get("language_label", _zero_label), dtype=np.float32)
        _critic_text_for_video = _critic_input_text(_critic_feedback_mode, _lang, _lang_text, obs_raw)

        replay_action = action.astype(np.float32)
        if _online_training_mode == "dagger" and not FLAGS.expert_debug:
            replay_action = np.asarray(obs_raw.get("expert_action", replay_action), dtype=np.float32)

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
                frame = _as_video_frame(obs)
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
                    frame = _annotate_reward_corner(frame, float(reward))
                episode_video_frames.append(frame)
        last_video_reward = float(reward)
        last_video_critic_text = _critic_text_for_video
        t_log_end = time.time()

        step_wb = {f"rollout/{k}": v for k, v in drive_metrics.items()}
        step_wb["rollout/collision_count"] = float(collision_count)
        step_wb["rollout/collision_events"] = float(collision_delta)
        if "reward_total" in info:
            step_wb["reward/total"] = float(info["reward_total"])
            step_wb["reward/progress"] = float(info.get("reward_progress", 0.0))
            step_wb["reward/centering"] = float(info.get("reward_centering", 0.0))
            step_wb["reward/heading"] = float(info.get("reward_heading", 0.0))
            step_wb["reward/terminal"] = float(info.get("reward_terminal", 0.0))
            step_wb["reward/penalty_collision"] = float(info.get("penalty_collision", 0.0))
            step_wb["reward/penalty_outside_route"] = float(info.get("penalty_outside_route", 0.0))
            step_wb["reward/penalty_steer"] = float(info.get("penalty_steer", 0.0))
            step_wb["reward/penalty_brake"] = float(info.get("penalty_brake", 0.0))
            step_wb["reward/penalty_crash_stuck"] = float(info.get("penalty_crash_stuck", 0.0))
            step_wb["rollout/lane_offset_m"] = float(info.get("lane_offset_m", 0.0))
            step_wb["rollout/heading_error_rad"] = float(info.get("heading_error_rad", 0.0))
            step_wb["rollout/speed_norm"] = float(info.get("speed_norm", 0.0))
            step_wb["rollout/centering_factor"] = float(info.get("centering_factor", 0.0))
            step_wb["rollout/heading_factor"] = float(info.get("heading_factor", 0.0))

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
                rollout_log["rollout/final_step_reward_progress"] = float(info.get("reward_progress", 0.0))
                rollout_log["rollout/final_step_reward_centering"] = float(info.get("reward_centering", 0.0))
                rollout_log["rollout/final_step_reward_heading"] = float(info.get("reward_heading", 0.0))
                rollout_log["rollout/final_step_reward_terminal"] = float(info.get("reward_terminal", 0.0))
                rollout_log["rollout/final_step_penalty_collision"] = float(info.get("penalty_collision", 0.0))
                rollout_log["rollout/final_step_penalty_outside_route"] = float(info.get("penalty_outside_route", 0.0))
                rollout_log["rollout/final_step_penalty_steer"] = float(info.get("penalty_steer", 0.0))
                rollout_log["rollout/final_step_penalty_brake"] = float(info.get("penalty_brake", 0.0))
                rollout_log["rollout/final_step_penalty_crash_stuck"] = float(info.get("penalty_crash_stuck", 0.0))
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

        update_time = 0.0
        if (not FLAGS.expert_debug) and agent is not None and step > warmup and buffer.size >= batch_size:
            t_update_start = time.time()
            for _ in range(updates_per_step):
                batch = buffer.sample(batch_size)
                if _online_training_mode == "dagger":
                    _, update_info = agent.update_dagger(batch)
                elif getattr(agent, "vla_sample_fn", None) is not None:
                    _, update_info = agent.update_with_vla(batch)
                else:
                    _, update_info = agent.update(batch)
                _block_until_ready_tree((agent, update_info))
            last_update_info = update_info
            t_update_end = time.time()
            update_time = t_update_end - t_update_start

        if step % FLAGS.log_interval == 0:
            metrics = {
                "time/steps_per_sec": FLAGS.log_interval / max(time.time() - last_log_time, 1e-6),
                "time/update_time": update_time,
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
    online_training_mode = str(config.get("online_training_mode", "rl")).strip().lower()
    if online_training_mode not in {"rl", "dagger"}:
        raise ValueError(
            f"Unsupported online_training_mode={online_training_mode!r}; expected 'rl' or 'dagger'."
        )
    use_steervla_rollout = bool(
        steervla_cfg is not None and steervla_cfg.get("enabled") and not FLAGS.expert_debug
    )
    if online_training_mode == "dagger":
        if use_steervla_rollout:
            print(
                "[main_carla] DAgger mode: rolling out SteerVLA and training the internal flow policy with relabeled expert actions.",
                flush=True,
            )
        else:
            print(
                "[main_carla] DAgger mode requested but SteerVLA rollout is disabled; falling back to learner rollout for data collection.",
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
    if FLAGS.expert_debug or FLAGS.expert_recover_debug:
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

            agent = agent_class.create(FLAGS.seed, ex_obs, ex_actions, config, **create_kwargs)

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
