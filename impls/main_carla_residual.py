"""Online residual RL on CARLA Bench2Drive with a frozen SteerVLA base policy.

    SteerVLA proposes a base action chunk. The residual SAC agent adds a small correction.
    the env executes one tick. Transitions feed a replay buffer.

JAX RL agents live under ``jax_agents/`` so the top-level ``agents`` name stays
free for CARLA's ``PythonAPI/carla/agents``.
"""

from __future__ import annotations

import json
import os
import random
import re
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
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

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
flags.DEFINE_bool(
    "base_only",
    None,
    "No-RL baseline: roll out the frozen base policy only (no residual agent, encoder, "
    "buffer, or updates). Overrides config.base_only.",
)

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
    url = steervla_cfg.get("actor_url")
    remote = bool(url and str(url).strip())
    return {
        "output_action_format": steervla_cfg.get("output_action_format") or "DELTA_XY_T_DELTA_XY_SPACE",
        "action_horizon": int(steervla_cfg.get("action_horizon", 10)),
        "action_dim": int(steervla_cfg.get("action_dim", 4)),
        "action_input_space": "policy_output" if remote else "normalized",
    }


def _make_carla_env(carla_config_path, route, *, extra_carla_config=None):
    from ogbench.carla.carla_utils import CarlaBench2DriveWrapper, load_carla_config

    cfg = load_carla_config(carla_config_path)
    if extra_carla_config:
        cfg = {**cfg, **extra_carla_config}
    return CarlaBench2DriveWrapper(cfg, route=route)


def _base_chunk(vla_sample_fn, raw_holder: dict, obs: dict, base_noise: jnp.ndarray) -> np.ndarray:
    """Frozen SteerVLA base action chunk for one flow-noise draw -> float32 [action_dim]."""
    raw_holder["obs"] = obs
    out = vla_sample_fn(jnp.zeros((1, 1), dtype=jnp.float32), base_noise)
    return np.asarray(jax.device_get(out), dtype=np.float32).reshape(-1)


# --------------------------------------------------------------------------- #
# Logging helpers                                                             #
# --------------------------------------------------------------------------- #

# Reward components emitted by ogbench/carla/carla_utils.py::_compute_reward_and_info.
# Logged under reward/* so the W&B "reward" section breaks total into its parts.
_REWARD_KEYS = (
    "reward_total",
    "reward_progress",
    "reward_terminal",
    "penalty_collision",
    "penalty_outside_route",
    "penalty_traffic_violation",
    "penalty_steer",
    "penalty_brake",
    "penalty_speed_limit",
    "penalty_crash_stuck",
)
# Per-step driving diagnostics (logged under rollout/*).
_DRIVE_KEYS = (
    "route_progress_pct",
    "route_progress_delta",
    "lane_offset_m",
    "heading_error_rad",
    "speed_norm",
    "overspeed_frac",
    "collision_count",
)


# Gym state-vector indices (mirror ogbench.carla.carla_utils.EGO_STATE_IDX_*); defined
# locally so logging doesn't import carla_utils (and pull in CARLA) at module load.
_EGO_STATE_IDX_SPEED = 15
_EGO_STATE_IDX_THROTTLE = 16
_EGO_STATE_IDX_STEER = 17
_EGO_STATE_IDX_BRAKE = 18


def _ego_control_log(obs: dict) -> dict[str, float]:
    """Last-applied CARLA control + speed from the gym state vector (drive/* line charts).

    These are the signals that actually say whether the car is moving: a dead
    ``drive/control_throttle`` with ~zero ``drive/ego_speed_mps`` is the "inching"
    failure mode, independent of what the raw action chunk looks like.
    """
    if not isinstance(obs, dict):
        return {}
    s = np.asarray(obs.get("state"), dtype=np.float32).reshape(-1)
    if s.size <= _EGO_STATE_IDX_BRAKE:
        return {}
    return {
        "drive/ego_speed_mps": float(s[_EGO_STATE_IDX_SPEED]),
        "drive/control_throttle": float(s[_EGO_STATE_IDX_THROTTLE]),
        "drive/control_steer": float(s[_EGO_STATE_IDX_STEER]),
        "drive/control_brake": float(s[_EGO_STATE_IDX_BRAKE]),
    }


def _ego_speed_mps(obs: dict) -> np.float32:
    """CARLA ego speed (m/s) from the gym state vector; 0.0 if unavailable.

    Used only for the ``debug_task`` stop reward (reward = -ego_speed).
    """
    if not isinstance(obs, dict):
        return np.float32(0.0)
    s = np.asarray(obs.get("state"), dtype=np.float32).reshape(-1)
    if s.size <= _EGO_STATE_IDX_SPEED:
        return np.float32(0.0)
    return np.float32(s[_EGO_STATE_IDX_SPEED])


def _chunk_stats_log(name: str, chunk_flat: np.ndarray, action_dim: int) -> dict[str, float]:
    """Per-component scalar summaries of a flattened ``(H, action_dim)`` chunk.

    Splits the SimLingo layout into speed deltas (cols 0:2) and route deltas
    (cols 2:4) so W&B shows readable line charts instead of a per-step histogram
    heatmap. ``*_speed_cumnorm`` is the magnitude of the cumulative speed waypoint
    -- the quantity the PID converts into desired speed -- so it directly flags a
    near-stationary (collapsed) base chunk vs. one that actually commands motion.
    """
    arr = np.asarray(chunk_flat, dtype=np.float32).reshape(-1)
    if action_dim <= 0 or arr.size == 0 or arr.size % action_dim != 0:
        return {f"action/{name}_absmean": float(np.abs(arr).mean()) if arr.size else 0.0}
    chunk = arr.reshape(-1, action_dim)
    speed = chunk[:, :2]
    out = {
        f"action/{name}_absmean": float(np.abs(arr).mean()),
        f"action/{name}_speed_absmean": float(np.abs(speed).mean()),
        f"action/{name}_speed_cumnorm": float(np.linalg.norm(np.cumsum(speed, axis=0)[-1])),
    }
    if action_dim >= 4:
        route = chunk[:, 2:4]
        out[f"action/{name}_route_absmean"] = float(np.abs(route).mean())
        out[f"action/{name}_route_cumnorm"] = float(np.linalg.norm(np.cumsum(route, axis=0)[-1]))
    return out


def _accel_steer_stats_log(name: str, vec: np.ndarray) -> dict[str, float]:
    """Scalar summaries of a 2-D ``[accel, steer]`` action for W&B line charts.

    Used in ``residual_action_space='accel_steer'`` mode, where base/final/residual
    actions live in the bounded control space rather than the waypoint chunk.
    """
    a = np.asarray(vec, dtype=np.float32).reshape(-1)
    out: dict[str, float] = {}
    if a.size >= 1:
        out[f"action/{name}_accel"] = float(a[0])
    if a.size >= 2:
        out[f"action/{name}_steer"] = float(a[1])
    return out


def _reward_breakdown_log(info: dict) -> dict[str, float]:
    """Flatten the env reward components + drive diagnostics into a W&B log dict."""
    out: dict[str, float] = {}
    if "reward_total" in info:
        for k in _REWARD_KEYS:
            if k in info:
                out[f"reward/{k}"] = float(info[k])
    for k in _DRIVE_KEYS:
        if k in info:
            out[f"rollout/{k}"] = float(info[k])
    return out


def _episode_summary_log(info: dict) -> dict[str, Any]:
    """Per-episode terminal summary (success / score / progress / collisions)."""
    out: dict[str, Any] = {}
    if "success" in info:
        out["rollout/success"] = float(bool(info["success"]))
    if "driving_score" in info:
        out["rollout/driving_score"] = float(info["driving_score"])
    if "route_progress_pct" in info:
        out["rollout/final_route_progress_pct"] = float(info["route_progress_pct"])
    if "collision_count" in info:
        out["rollout/episode_collision_count"] = float(info["collision_count"])
    if "termination_reason" in info:
        out["rollout/termination_reason"] = str(info["termination_reason"])
    return out


def _viz_frame(obs: dict) -> Optional[np.ndarray]:
    """uint8 camera frame for the rollout video (prefers high-res ``image_viz``)."""
    if not isinstance(obs, dict):
        return None
    img = obs.get("image_viz")
    if img is None:
        img = obs.get("image")
    if img is None:
        return None
    frame = np.asarray(img)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _annotate_frame(frame: np.ndarray, obs: dict, *, reward: float, action_flat=None, exec_cfg=None) -> np.ndarray:
    """Overlay predicted waypoints + a bottom text panel (reward, prompt, reasoning, subtask).

    Prompt/reasoning/subtask are the SteerVLA CoT strings the actor stashes back onto the
    gym obs dict. No-op passthrough if cv2 is unavailable.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return frame
    out = np.ascontiguousarray(frame)
    if exec_cfg is not None and action_flat is not None:
        try:
            from ogbench.carla.waypoint_viz import annotate_waypoints_on_frame

            out = annotate_waypoints_on_frame(out, action_flat=np.asarray(action_flat), exec_cfg=exec_cfg)
        except Exception:
            pass

    def _s(key: str) -> str:
        v = obs.get(key) if isinstance(obs, dict) else None
        return v.strip()[:110] if isinstance(v, str) else ""

    lines = [
        f"r={reward:+.3f}",
        f"Prompt: {_s('openpi_prompt_text')}",
        f"Reasoning: {_s('reasoning_text') or _s('reasoning')}",
        f"Subtask: {_s('subtask_text') or _s('subtask')}",
    ]
    h, w = out.shape[:2]
    canvas = np.vstack([out, np.zeros((13 * len(lines) + 4, w, 3), dtype=np.uint8)])
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (4, h + 13 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _episode_video(frames: list[np.ndarray], fps: float):
    """Stack captured frames into a W&B video (T, C, H, W); None if empty."""
    if not frames:
        return None
    video = np.stack(frames, axis=0)
    if video.ndim == 4:  # (T, H, W, C) -> (T, C, H, W) as W&B expects.
        video = np.transpose(video, (0, 3, 1, 2))
    return wandb.Video(video, fps=fps, format="mp4")


def _maybe_capture_frame(
    frames: list[np.ndarray], obs: dict, reward: float, *, episode_steps: int, done: bool,
    log_video: bool, video_every: int, action_flat=None, exec_cfg=None,
) -> None:
    """Append an annotated frame on capture steps (every Nth step + the terminal one)."""
    if log_video and (episode_steps % video_every == 0 or done):
        frame = _viz_frame(obs)
        if frame is not None:
            frames.append(_annotate_frame(frame, obs, reward=float(reward), action_flat=action_flat, exec_cfg=exec_cfg))


def _log_episode_end(
    info: dict, *, episode_return: float, episode_steps: int, episode_index: int,
    frames: list[np.ndarray], log_video: bool, video_fps: float, step: int, train_logger: CsvLogger,
) -> None:
    """Log per-episode rollout metrics (+ video) to W&B and CSV, then clear ``frames``."""
    rollout_log: dict[str, Any] = {
        "rollout/episode_return": episode_return,
        "rollout/episode_length": episode_steps,
        "rollout/episodes": episode_index,
    }
    rollout_log.update(_episode_summary_log(info))
    if log_video:
        video = _episode_video(frames, video_fps)
        if video is not None:
            rollout_log["rollout/episode_video"] = video
    wandb.log(rollout_log, step=step)
    train_logger.log(rollout_log, step=step)
    frames.clear()


# --------------------------------------------------------------------------- #
# Online loop                                                                  #
# --------------------------------------------------------------------------- #


def run_online(env, agent, config, obs, *, vla_sample_fn, steervla_actor, raw_holder, state_encoder, exec_cfg=None):
    """Residual-SAC online loop. One env.step == one CARLA tick == one transition."""
    base_only = agent is None
    vla_action_dim = int(config["steervla"]["action_dim"])
    action_dim = int(config["steervla"]["action_horizon"]) * vla_action_dim
    warmup = int(config["residual_warmup_steps"])
    batch_size = int(config["batch_size"])
    updates_per_step = int(config["updates_per_step"])
    capacity = int(config["buffer_capacity"])
    # Debug task: store per-step ego speed so the agent can relabel reward to -ego_speed
    # (learn to stop). Keeps the real env reward in the buffer for logging.
    debug_task = bool(config.get("debug_task", False))
    enable_updates = bool(FLAGS.enable_updates) if FLAGS.enable_updates is not None else bool(config["enable_updates"])
    if base_only:
        print("[main_carla_residual] base_only=True: rolling out the frozen base policy (no RL).", flush=True)
    elif not enable_updates:
        print("[main_carla_residual] enable_updates=False: rollout-only (no RL gradient updates).", flush=True)

    log_video = bool(config.get("log_episode_video", True))
    video_fps = float(config.get("episode_video_fps", 10.0))
    video_every = max(1, int(config.get("episode_video_every", 2)))
    episode_frames: list[np.ndarray] = []

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    last_residual: Optional[np.ndarray] = None

    rng = jax.random.PRNGKey(FLAGS.seed)

    # Base flow noise is sized to the MODEL action tensor (action_horizon x
    # model_action_dim, e.g. 10 x 32), not the env chunk (action_horizon x
    # vla_action_dim, 10 x 4). This mirrors main_carla.py's rollout
    # (DSRLAgent._flat_noise_dim): a model-dim draw is written into horizon step 0 of
    # the flow-noise tensor, whereas a short env-dim draw is broadcast across all
    # horizon steps. Fall back to the env dim only when the model isn't loaded
    # (e.g. remote actor, where the noise is ignored server-side anyway).
    _base_model = getattr(steervla_actor, "model", None)
    base_noise_dim = (
        int(_base_model.action_horizon) * int(_base_model.action_dim)
        if _base_model is not None
        else action_dim
    )

    def _draw_base_noise(key: jax.Array) -> jnp.ndarray:
        """Fresh Gaussian seed for the base flow ODE (``x ~ N(0, I)``)."""
        return jax.random.normal(key, (1, base_noise_dim), dtype=jnp.float32)

    # Residual action space (see steervla_residual_config.py):
    #   "waypoint_chunk" (default): the residual acts on the raw normalized 40-D chunk;
    #       the env decodes the corrected chunk -> waypoints -> PID. The residual thus
    #       reshapes the waypoints (visible per-step jitter once active).
    #   "accel_steer": the frozen base chunk is PID-decoded to a bounded 2-D
    #       [accel, steer] BEFORE the residual (matches routing-commands). The residual
    #       acts on that 2-D control, the env executes it via _action_to_control, and the
    #       drawn waypoints stay the (smooth) base plan. Better-conditioned RL problem.
    # The frozen VLA still produces the 40-D chunk either way; only what the agent /
    # buffer / env see downstream changes.
    residual_space = str(config.get("residual_action_space", "waypoint_chunk")).strip().lower()
    accel_steer = residual_space == "accel_steer"
    accel_steer_decoder = None
    if accel_steer:
        if exec_cfg is None:
            raise ValueError(
                "residual_action_space='accel_steer' requires the SteerVLA action-execution "
                "config (chunk layout) to PID-decode the base chunk into [accel, steer]."
            )
        from ogbench.carla.steervla_simlingo_control import SimlingoStyleWaypointDecoder

        accel_steer_decoder = SimlingoStyleWaypointDecoder()

    def _agent_base_action(o: dict, base_chunk: np.ndarray) -> np.ndarray:
        """Frozen VLA chunk -> agent-facing base action (chunk, or PID-decoded [accel, steer])."""
        if not accel_steer:
            return base_chunk
        return np.asarray(
            accel_steer_decoder.flat_action_to_accel_steer(
                base_chunk,
                state_vec=np.asarray(o["state"], dtype=np.float32),
                output_action_format=str(exec_cfg["output_action_format"]),
                action_horizon=int(exec_cfg["action_horizon"]),
                action_dim=int(exec_cfg["action_dim"]),
                action_input_space=str(exec_cfg.get("action_input_space", "normalized")),
            ),
            dtype=np.float32,
        )

    # Base chunk before encode: it samples + stashes the CoT that the rl_token
    # encoder needs to reproduce the prefix the policy acted on (no-op for others).
    # ``base_chunk`` is always the raw 40-D VLA chunk (used for the waypoint overlay);
    # ``base`` is what the agent conditions on (chunk or 2-D [accel, steer]).
    rng, nk = jax.random.split(rng)
    base_chunk = _base_chunk(vla_sample_fn, raw_holder, obs, _draw_base_noise(nk))
    base = _agent_base_action(obs, base_chunk)
    x = None if state_encoder is None else state_encoder.encode(obs)

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
        if agent is None:
            return
        save_agent(agent, FLAGS.save_dir, step_tag)
        if FLAGS.save_buffer and buffer is not None:
            path = FLAGS.buffer_path or os.path.join(FLAGS.save_dir, "buffer.npz")
            saved = buffer.save(path)
            print(f"[main_carla_residual] Saved replay buffer ({buffer.size} transitions) -> {saved}", flush=True)

    last_step = 0
    try:
        for step in tqdm.tqdm(range(1, FLAGS.online_steps + 1), dynamic_ncols=True):
            last_step = step
            residual_active = (
                agent is not None and step > warmup and buffer is not None and buffer.size >= batch_size
            )
            if residual_active:
                rng, sample_key = jax.random.split(rng)
                final_b, residual_b = agent.sample_actions(x[None], base[None], seed=sample_key)
                final = np.asarray(jax.device_get(final_b), dtype=np.float32).reshape(-1)
                last_residual = np.asarray(jax.device_get(residual_b), dtype=np.float32).reshape(-1)
            else:
                final = base

            next_obs, reward, terminated, truncated, info = env.step(final)
            done = bool(terminated or truncated)

            rng, nk = jax.random.split(rng)
            next_base_chunk = _base_chunk(vla_sample_fn, raw_holder, next_obs, _draw_base_noise(nk))
            next_base = _agent_base_action(next_obs, next_base_chunk)
            next_x = None if state_encoder is None else state_encoder.encode(next_obs)

            if agent is not None:
                transition = dict(
                    observations=x,
                    actions=final,
                    base_actions=base,
                    rewards=np.float32(reward),
                    next_observations=next_x,
                    next_base_actions=next_base,
                    masks=np.float32(1.0 - float(terminated)),
                )
                if debug_task:
                    # Speed at the state the action was taken from (matches main_carla.py).
                    transition["ego_speed"] = _ego_speed_mps(obs)
                if buffer is None:
                    buffer = ReplayBuffer.create(transition, size=capacity)
                buffer.add_transition(transition)

            episode_return += float(reward)
            episode_steps += 1
            # In accel_steer mode ``final`` is a 2-D control and can't be projected as
            # waypoints, so draw the (smooth) base chunk plan -- matching how
            # main_carla.py overlays base waypoints. In chunk mode draw the executed
            # (residual-perturbed) chunk.
            viz_action = base_chunk if accel_steer else final
            _maybe_capture_frame(
                episode_frames, next_obs, reward, episode_steps=episode_steps, done=done,
                log_video=log_video, video_every=video_every,
                action_flat=viz_action, exec_cfg=exec_cfg,
            )

            train_info: dict[str, Any] = {}
            if agent is not None and enable_updates and buffer is not None and buffer.size >= batch_size:
                for _ in range(updates_per_step):
                    agent, train_info = agent.update(buffer.sample(batch_size))

            if step % FLAGS.log_interval == 0:
                log = {
                    "env/reward": float(reward),
                    "env/episode_count": episode_count,
                    "env/sps": step / max(time.time() - start_time, 1e-6),
                }
                if agent is not None:
                    log["env/buffer_size"] = int(buffer.size) if buffer is not None else 0
                    log["env/residual_active"] = int(residual_active)
                log.update(_reward_breakdown_log(info))
                if train_info:
                    log.update({k: float(jax.device_get(v)) for k, v in train_info.items()})
                # Executed control + per-component chunk stats as scalar line charts.
                # (A per-step histogram of the 40-value flattened chunk renders as an
                # unreadable time-heatmap and mixes 10 horizon steps x 4 heterogeneous
                # dims; the splits below are the readable, diagnostic signals.)
                log.update(_ego_control_log(next_obs))
                if accel_steer:
                    # No waypoint viz for the executed 2-D control, so log it explicitly:
                    # base = frozen-policy [accel, steer], final = executed, residual =
                    # applied delta (final - base, i.e. post-clip; 0 during warmup). The
                    # base_chunk_* stats still track the underlying (drawn) waypoint plan.
                    log.update(_accel_steer_stats_log("base", base))
                    log.update(_chunk_stats_log("base_chunk", base_chunk, vla_action_dim))
                    if agent is not None:
                        log.update(_accel_steer_stats_log("final", final))
                        log.update(_accel_steer_stats_log(
                            "residual",
                            np.asarray(final, dtype=np.float32) - np.asarray(base, dtype=np.float32),
                        ))
                else:
                    log.update(_chunk_stats_log("base", base, vla_action_dim))
                    if agent is not None:
                        log.update(_chunk_stats_log("final", final, vla_action_dim))
                        if last_residual is not None:
                            log.update(_chunk_stats_log("residual", last_residual, vla_action_dim))
                wandb.log(log, step=step)
                train_logger.log(log, step=step)

            if agent is not None and FLAGS.save_interval > 0 and step % FLAGS.save_interval == 0:
                save_agent(agent, FLAGS.save_dir, step)

            if done:
                _log_episode_end(
                    info, episode_return=episode_return, episode_steps=episode_steps,
                    episode_index=episode_count + 1, frames=episode_frames, log_video=log_video,
                    video_fps=video_fps, step=step, train_logger=train_logger,
                )
                episode_count += 1
                episode_return = 0.0
                episode_steps = 0
                obs, _info = env.reset(seed=FLAGS.seed + episode_count)
                steervla_actor.reset_action_cache()
                rng, nk = jax.random.split(rng)
                base_chunk = _base_chunk(vla_sample_fn, raw_holder, obs, _draw_base_noise(nk))
                base = _agent_base_action(obs, base_chunk)
                x = None if state_encoder is None else state_encoder.encode(obs)
            else:
                obs, x, base, base_chunk = next_obs, next_x, next_base, next_base_chunk
    finally:
        train_logger.close()
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

    base_only = bool(FLAGS.base_only) if FLAGS.base_only is not None else bool(config.get("base_only", False))

    wandb_mode = FLAGS.wandb_mode or os.environ.get("WANDB_MODE", "online")
    # Descriptive run name so runs are distinguishable in W&B: <route>-<mode>-sd###_<ts>,
    # where <mode> is the state encoder (rl runs) or "base" (no-RL baseline).
    route_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", str(FLAGS.route)).strip("-")
    mode_tag = "base" if base_only else str(config.get("state_encoder", "pi_prefix"))
    exp_name = f"{route_tag}-{mode_tag}-{get_exp_name(FLAGS.seed)}"
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

        # base_only -> no residual agent / encoder; run_online executes the frozen
        # base chunk every step. Otherwise build the encoder + residual SAC agent.
        agent = None
        state_encoder = None
        if not base_only:
            from encoders import build_state_encoder

            state_encoder = build_state_encoder(config, steervla_actor)
            # accel_steer -> the residual acts on the 2-D [accel, steer] control the env
            # executes; otherwise on the full flattened waypoint chunk.
            residual_space = str(config.get("residual_action_space", "waypoint_chunk")).strip().lower()
            if residual_space == "accel_steer":
                if exec_cfg is None:
                    raise ValueError(
                        "residual_action_space='accel_steer' requires the SteerVLA action-execution "
                        "config (set carla_config['steervla_action_execution'])."
                    )
                action_dim = 2
            else:
                action_dim = int(steervla_cfg["action_horizon"]) * int(steervla_cfg["action_dim"])
            # Probe the encoded-state width once so the agent's MLPs are sized correctly
            # (encoder output dim is not known until the SteerVLA model loads).
            x_dim = int(state_encoder.encode(obs).shape[-1])
            ex_obs = np.zeros((1, x_dim), dtype=np.float32)
            ex_base = np.zeros((1, action_dim), dtype=np.float32)
            print(
                f"[main_carla_residual] state_encoder={state_encoder.name}; x_dim={x_dim}; "
                f"action_dim={action_dim}; residual_action_space={residual_space}",
                flush=True,
            )

            from jax_agents.sac_residual import SACResidualAgent

            agent = SACResidualAgent.create(FLAGS.seed, ex_obs, ex_base, config)
            if FLAGS.restore_path is not None:
                agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

        run_online(
            env, agent, config, obs,
            vla_sample_fn=vla_sample_fn,
            steervla_actor=steervla_actor,
            raw_holder=raw_holder,
            state_encoder=state_encoder,
            exec_cfg=exec_cfg,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
