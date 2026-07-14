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


def _expo_candidate_log(base_cands: np.ndarray, q_all: np.ndarray, winner_idx: int, n: int) -> dict[str, float]:
    """EXPO diagnostics for one step's 2N pool: Q spread, base-vs-edit winner + margin, and
    base-action diversity as per-dim Gaussian entropy 0.5*log(2*pi*e*var). q_all is (2N,) min-
    ensemble Q (first N base, last N edits); base_cands is (N, adim)."""
    out: dict[str, float] = {}
    q = np.asarray(q_all, dtype=np.float64).reshape(-1)
    if q.size:
        out["expo/q_max"] = float(q.max())
        out["expo/q_min"] = float(q.min())
        out["expo/q_mean"] = float(q.mean())
        if q.size >= 2 * n and n > 0:
            out["expo/q_base_mean"] = float(q[:n].mean())
            out["expo/q_edit_mean"] = float(q[n:].mean())
        srt = np.sort(q)[::-1]
        out["expo/q_margin"] = float(srt[0] - srt[1]) if q.size > 1 else 0.0
    out["expo/winner_is_edit"] = float(int(winner_idx) >= n)
    out["expo/winner_idx"] = float(int(winner_idx))
    bc = np.asarray(base_cands, dtype=np.float64).reshape(n, -1)
    var = bc.var(axis=0)  # (adim,) diversity of the N base actions per dim
    ent = 0.5 * np.log(2.0 * np.pi * np.e * np.maximum(var, 1e-12))
    for d in range(ent.shape[0]):
        out[f"expo/base_entropy/dim_{d}"] = float(ent[d])
    out["expo/base_entropy_mean"] = float(ent.mean())
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
    # Warm-start ramp: residual authority (scale) is 0 through warmup, then linearly rises
    # to the target ``residual_scale`` over ``residual_ramp_steps`` env steps. Starting the
    # ramp at 0 exactly at the warmup boundary avoids any magnitude jump at handover, and the
    # gradual rise keeps the critic in-distribution as the executed policy drifts off base.
    ramp_steps = max(0, int(config.get("residual_ramp_steps", 0)))
    target_scale = float(config["residual_scale"])

    def _residual_scale(step: int) -> float:
        """Annealed residual authority at ``step`` (0 during warmup, linear ramp, then target)."""
        if step <= warmup:
            return 0.0
        if ramp_steps <= 0 or step >= warmup + ramp_steps:
            return target_scale
        return target_scale * float(step - warmup) / float(ramp_steps)

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

    # EXPO best-of-N: each step samples n_cand base actions from distinct SteerVLA CoTs, the
    # agent edits each 1:1, and executes the argmax-Q of the 2N pool. During warmup/base_only we
    # take the cheap single-base path tiled to N so the buffer schema stays uniform. expo=False ->
    # plain residual SAC (n_cand=1, execute the single edit, no critic selection).
    use_otf = agent is not None and bool(config.get("expo", True))
    n_cand = max(1, int(config.get("best_of_n", 8))) if use_otf else 1
    cot_temp = float(config.get("vla_cot_temperature", 1.0))
    # Token encoders (pi_prefix / rl_token) fold the CoT into the state -> one state per candidate
    # (K=N). siglip_pool is image-only -> one shared state (K=1) broadcast across candidates. K is
    # fixed per run so next_obs_cands has a uniform shape.
    state_cot_dependent = bool(getattr(state_encoder, "cot_dependent", True)) if state_encoder is not None else False
    k_state = n_cand if state_cot_dependent else 1

    def _encode_state(o: dict, cands) -> Optional[np.ndarray]:
        """Encode the (K, embed) state(s) for one obs. cands=None (warmup) -> encode once, tile to
        K. cot-dependent -> one row per candidate, each with that candidate's CoT via _last_cot_out."""
        if state_encoder is None:
            return None
        if not state_cot_dependent or cands is None:
            xi = np.asarray(state_encoder.encode(o), dtype=np.float32).reshape(-1)
            return np.tile(xi[None, :], (k_state, 1))
        rows = []
        for i in range(n_cand):
            steervla_actor._last_cot_out = {kk: vv[i : i + 1] for kk, vv in cands["cot_out"].items()}
            rows.append(np.asarray(state_encoder.encode(o), dtype=np.float32).reshape(-1))
        return np.stack(rows, axis=0)

    def _compute_base(o: dict, otf_mode: bool, key: jax.Array):
        """Returns (base_cands (N, adim), x_cands (K, embed)|None, chunks (N, 40), cands|None).
        base_cands is what the agent conditions on; chunks are the raw VLA chunks (waypoint overlay).
        otf_mode -> N diverse CoT samples; else one base tiled to N."""
        if otf_mode:
            cands = steervla_actor.sample_candidates(n_cand, temperature=cot_temp, rng=key)
            chunks = np.asarray(cands["actions"], dtype=np.float32).reshape(n_cand, -1)
            base_cands = np.stack(
                [np.asarray(_agent_base_action(o, chunks[i]), dtype=np.float32) for i in range(n_cand)],
                axis=0,
            )
            return base_cands, _encode_state(o, cands), chunks, cands
        bc = _base_chunk(vla_sample_fn, raw_holder, o, _draw_base_noise(key))
        b = np.asarray(_agent_base_action(o, bc), dtype=np.float32)
        base_cands = np.tile(b[None, :], (n_cand, 1))
        chunks = np.tile(np.asarray(bc, dtype=np.float32).reshape(1, -1), (n_cand, 1))
        return base_cands, _encode_state(o, None), chunks, None

    # Initial base rep (step 1 is in warmup -> single-base). cands feeds subtask-diversity logging.
    rng, nk = jax.random.split(rng)
    base_cands, x_cands, base_chunks, cands = _compute_base(obs, use_otf and 1 > warmup, nk)

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
            # Annealed residual authority for this step (0 during warmup -> target after ramp).
            scale_now = _residual_scale(step)
            scale_j = jnp.asarray(scale_now, dtype=jnp.float32)
            # OTF selection needs a training critic -> active only past warmup (else execute base).
            residual_active = agent is not None and step > warmup
            q_all: Optional[np.ndarray] = None
            winner_idx = 0
            if residual_active and use_otf:
                rng, sample_key = jax.random.split(rng)
                exe_b, wbase_b, wx_b, q_b, widx_b = agent.select_action_otf(
                    jnp.asarray(x_cands), jnp.asarray(base_cands), scale_j, sample_key
                )
                final = np.asarray(jax.device_get(exe_b), dtype=np.float32).reshape(-1)
                winner_base = np.asarray(jax.device_get(wbase_b), dtype=np.float32).reshape(-1)
                winner_x = np.asarray(jax.device_get(wx_b), dtype=np.float32).reshape(-1)
                q_all = np.asarray(jax.device_get(q_b), dtype=np.float32).reshape(-1)
                winner_idx = int(jax.device_get(widx_b))
                last_residual = final - winner_base
            elif residual_active:
                # Regular residual SAC: execute the single edit, no best-of-N selection.
                rng, sample_key = jax.random.split(rng)
                final_b, _res_b = agent.sample_actions(
                    jnp.asarray(x_cands[0][None]), jnp.asarray(base_cands[0][None]), scale_j, seed=sample_key
                )
                final = np.asarray(jax.device_get(final_b), dtype=np.float32).reshape(-1)
                winner_base = base_cands[0]
                winner_x = None if x_cands is None else x_cands[0]
                last_residual = final - winner_base
            else:
                final = base_cands[0]
                winner_base = base_cands[0]
                winner_x = None if x_cands is None else x_cands[0]
            # Winner's raw VLA chunk drives the waypoint overlay in accel_steer mode.
            base_chunk = base_chunks[winner_idx % n_cand]

            next_obs, reward, terminated, truncated, info = env.step(final)
            done = bool(terminated or truncated)

            # Next base cands are diverse iff the next step runs OTF (step >= warmup).
            rng, nk = jax.random.split(rng)
            next_base_cands, next_x_cands, next_base_chunks, next_cands = _compute_base(
                next_obs, use_otf and step >= warmup, nk
            )

            if agent is not None:
                transition = dict(
                    observations=winner_x,
                    actions=final,
                    base_actions=winner_base,
                    rewards=np.float32(reward),
                    next_obs_cands=next_x_cands,  # (K, embed)
                    next_base_cands=next_base_cands,  # (N, adim)
                    masks=np.float32(1.0 - float(terminated)),
                )
                if debug_task:
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
            # Hold updates until warmup ends: warmup is a pure-base baseline phase, and at
            # scale=0 the residual can't affect the executed action so its gradient is
            # degenerate. The ramp then starts near 0, giving the critic room to warm up.
            if (
                agent is not None and enable_updates and step > warmup
                and buffer is not None and buffer.size >= batch_size
            ):
                for _ in range(updates_per_step):
                    agent, train_info = agent.update(buffer.sample(batch_size), scale_j)

            if step % FLAGS.log_interval == 0:
                log = {
                    "env/reward": float(reward),
                    "env/episode_count": episode_count,
                    "env/sps": step / max(time.time() - start_time, 1e-6),
                }
                if agent is not None:
                    log["env/buffer_size"] = int(buffer.size) if buffer is not None else 0
                    log["env/residual_active"] = int(residual_active)
                    log["env/residual_scale"] = float(scale_now)
                log.update(_reward_breakdown_log(info))
                if train_info:
                    log.update({k: float(jax.device_get(v)) for k, v in train_info.items()})
                # Executed control + per-component chunk stats as scalar line charts.
                # (A per-step histogram of the 40-value flattened chunk renders as an
                # unreadable time-heatmap and mixes 10 horizon steps x 4 heterogeneous
                # dims; the splits below are the readable, diagnostic signals.)
                log.update(_ego_control_log(next_obs))
                if accel_steer:
                    # 2-D control has no waypoint viz, so log base/final/residual explicitly;
                    # base_chunk_* still tracks the underlying waypoint plan.
                    log.update(_accel_steer_stats_log("base", winner_base))
                    log.update(_chunk_stats_log("base_chunk", base_chunk, vla_action_dim))
                    if agent is not None:
                        log.update(_accel_steer_stats_log("final", final))
                        log.update(_accel_steer_stats_log(
                            "residual",
                            np.asarray(final, dtype=np.float32) - np.asarray(winner_base, dtype=np.float32),
                        ))
                else:
                    log.update(_chunk_stats_log("base", winner_base, vla_action_dim))
                    if agent is not None:
                        log.update(_chunk_stats_log("final", final, vla_action_dim))
                        if last_residual is not None:
                            log.update(_chunk_stats_log("residual", last_residual, vla_action_dim))
                # EXPO 2N-pool diagnostics (only meaningful once OTF is active).
                if residual_active and q_all is not None:
                    log.update(_expo_candidate_log(base_cands, q_all, winner_idx, n_cand))
                    if cands is not None:
                        subtasks = cands["subtask_texts"]
                        log["expo/n_unique_subtasks"] = float(len(set(subtasks)))
                        won = "edit" if winner_idx >= n_cand else "base"
                        print(
                            f"[expo] step={step} winner={won} q={float(q_all[winner_idx]):.3f} "
                            f"subtask={subtasks[winner_idx % n_cand]!r} ({len(set(subtasks))}/{n_cand} unique)",
                            flush=True,
                        )
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
                base_cands, x_cands, base_chunks, cands = _compute_base(
                    obs, use_otf and step + 1 > warmup, nk
                )
            else:
                obs = next_obs
                base_cands, x_cands, base_chunks, cands = (
                    next_base_cands, next_x_cands, next_base_chunks, next_cands,
                )
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
