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

   Set ``enable_updates=false`` in the agent config (or ``--enable_updates=false``)
   to run rollout-only: collect transitions and log videos without gradient updates.

   Live rollout video (single overwriting MP4 + browser UI)::

     --live_policy_view=true --live_policy_port=8765 --live_policy_interval=5

   Open ``http://<host>:8765/`` while training; the file is also written to
   ``<save_dir>/live_policy.mp4``.

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

import faulthandler
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

# CARLA's C++ rpclib I/O thread calls std::terminate() -> SIGABRT when concurrent RPC
# from multiple threads corrupts the msgpack socket framing (surfaces as a bogus
# "Actor could not be found in the registry ... set_actor_simulate_physics" + "Fatal
# Python error: Aborted"). See ogbench/carla/carla_utils._patch_speedometer_no_rpc.
# faulthandler.enable() installs an all-threads handler for SIGABRT (among SIGSEGV/
# SIGFPE/SIGBUS/SIGILL) by default, so every thread's Python stack is dumped on abort,
# revealing which background thread was mid-CARLA-RPC. (SIGABRT cannot be passed to
# faulthandler.register(); enable() is the supported path.)
faulthandler.enable()

import numpy as np
import jax
import jax.numpy as jnp

from jax_agents import agents
from utils.flax_utils import restore_agent
from coaches.expert_label import NUM_COMMENTARY_WORDS, NUM_DELTA_COMMENTARY_WORDS
from coaches.critic_feedback import (
    compute_action_delta,
    compute_action_delta_commentary,
    critic_language_dim,
    resolve_critic_feedback_mode,
)
from impls.coaches.online_vlm_coach import OnlineVLMSession

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
from utils.live_policy_viewer import LivePolicyViewer

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
    "Bench2Drive or Fail2Drive route. Accepts scenario-name "
    "(parking-cut-in-001, base-pedestrians-on-road-0085), file basename "
    "(bench2drive_007, Base_PedestriansOnRoad_0085), or route id "
    "(1711 for bench2drive; f2d:85 for fail2drive). See --list_routes=true.",
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

# flags.DEFINE_string("save_dir", "/raid/users/celine/carla_exps", "Save directory.")
flags.DEFINE_string("save_dir", "/home/carla/exps", "Save directory.")
flags.DEFINE_string("restore_path", None, "Restore path for JAX agents.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")

flags.DEFINE_integer("online_steps", 1000, "Number of online environment steps to run.")
flags.DEFINE_integer("log_interval", 1, "Logging interval (env steps).")
flags.DEFINE_integer("save_interval", 100_000, "Agent-checkpoint interval (env steps).")
flags.DEFINE_integer(
    "resume_interval",
    1000,
    "Crash-recovery checkpoint interval (env steps) for the residual loop: how often the agent + "
    "replay buffer + resume_state are committed so --resume can continue near the crash point. "
    "Smaller = less lost progress per crash, more I/O. 0 disables periodic recovery checkpoints.",
)
flags.DEFINE_bool("save_buffer", False, "Dump the replay buffer to <save_dir>/buffer.npz at the end.")
flags.DEFINE_string("buffer_path", None, "Optional explicit path for the saved buffer.")

flags.DEFINE_string(
    "exp_name",
    None,
    "Fixed experiment/run name (overrides the timestamped default). The run_carla.sh crash "
    "supervisor sets this so save_dir + the W&B run stay stable across relaunches.",
)
flags.DEFINE_bool(
    "resume",
    False,
    "Resume the residual run from <save_dir>/resume_state.json (restores agent + replay buffer + "
    "step counter). Used by the restart supervisor to recover from CARLA native crashes.",
)

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
flags.DEFINE_bool(
    "enable_updates",
    None,
    "If false, skip RL gradient updates (rollout/buffer logging only). "
    "Default: agent config ``enable_updates`` (true).",
)
flags.DEFINE_bool(
    "live_policy_view",
    False,
    "Serve a single overwriting live_policy.mp4 updated during rollouts.",
)
flags.DEFINE_integer(
    "live_policy_port",
    8765,
    "HTTP port for --live_policy_view (GET / and /live.mp4).",
)
flags.DEFINE_integer(
    "live_policy_interval",
    5,
    "Rewrite live_policy.mp4 every N env steps.",
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


# Must match :data:`ogbench.carla.carla_utils.EGO_STATE_IDX_*`.
_EGO_STATE_IDX_SPEED = 15
_EGO_STATE_IDX_THROTTLE = 16
_EGO_STATE_IDX_STEER = 17
_EGO_STATE_IDX_BRAKE = 18


def _ego_speed_mps_from_raw(raw: dict) -> np.float32:
    state = np.asarray(raw["state"], dtype=np.float32).reshape(-1)
    if state.size <= _EGO_STATE_IDX_SPEED:
        return np.float32(0.0)
    return np.float32(state[_EGO_STATE_IDX_SPEED])


def _coerce_language_label(value, dim: int, fallback: np.ndarray) -> np.ndarray:
    """Return ``value`` as a length-``dim`` float32 vector, or ``fallback`` if mismatched."""
    if dim <= 0:
        return np.zeros(0, dtype=np.float32)
    if value is None:
        return fallback
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != int(dim):
        return fallback
    return arr


def _steervla_prompt_subtask_strings(raw: dict, steervla_actor=None) -> tuple[str, str]:
    """Extract SteerVLA prompt and subtask strings from a CARLA raw obs dict."""
    prompt = str(raw.get("openpi_prompt_text") or "").strip()
    if not prompt:
        from vlas.steervla import (
            carla_state_vec_to_steervla_state,
            format_steervla_cot_prompt,
            routing_instruction_prompt,
            steervla_prompt_state_dim,
        )

        state = np.asarray(raw.get("state", []), dtype=np.float32).reshape(-1)
        speed = float(state[_EGO_STATE_IDX_SPEED]) if state.size > _EGO_STATE_IDX_SPEED else 0.0
        routing = str(raw.get("routing_command", "") or "").strip() or "Follow the route."
        include_hist = bool(getattr(steervla_actor, "include_ego_history", False)) if steervla_actor else False
        proprio_norm = bool(getattr(steervla_actor, "proprio_norm", True)) if steervla_actor else True
        state_pad = carla_state_vec_to_steervla_state(
            state,
            include_ego_history=include_hist,
            proprio_norm=proprio_norm,
        )
        prompt = format_steervla_cot_prompt(
            routing_instruction_prompt(routing_command=routing, current_speed_mps=speed),
            state_pad,
            state_dim=steervla_prompt_state_dim(include_ego_history=include_hist),
        )

    subtask = ""
    for key in ("subtask_text", "subtask"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            subtask = value.strip()
            break
    return prompt, subtask


def _extract_agent_obs(
    env,
    env_obs: dict,
    mode: str,
    *,
    image_encoder: str = "impala",
    siglip_encoder=None,
    siglip_include_prompt_subtask: bool = False,
    steervla_actor=None,
) -> np.ndarray:
    """Pick the tensor the RL agent trains on (env always exposes both keys).

    The language label (BOW or delta) is stored separately in the replay buffer
    and concatenated to the encoded observation ONLY inside the critic (dsrl.py).
    When ``image_encoder='siglip'`` and ``siglip_include_prompt_subtask=True``,
    returns ``[image_embed, prompt_embed, subtask_embed]`` for actor and critics.
    """
    if mode == "state":
        return np.asarray(env_obs["state"], dtype=np.float32)
    if mode == "image":
        if image_encoder == "siglip":
            if siglip_encoder is None:
                raise ValueError("image_encoder='siglip' requires a SigLIPEncoder instance.")
            if siglip_include_prompt_subtask:
                prompt, subtask = _steervla_prompt_subtask_strings(env_obs, steervla_actor)
                return np.asarray(
                    siglip_encoder.encode_observation(
                        env_obs["image"],
                        prompt=prompt,
                        subtask=subtask,
                        include_prompt_subtask=True,
                    ),
                    dtype=np.float32,
                )
            return np.asarray(siglip_encoder.encode(env_obs["image"]), dtype=np.float32)
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
    b2d_count = sum(1 for e in entries if e.source == "bench2drive")
    f2d_count = sum(1 for e in entries if e.source == "fail2drive")
    print(f"# {len(entries)} routes ({b2d_count} bench2drive, {f2d_count} fail2drive)")
    header = (
        f"{'source':<12} {'scenario_name':<48} {'file_name':<32} "
        f"{'route_id':<10} {'town':<10} {'scenario_type'}"
    )
    print(header)
    for e in entries:
        print(
            f"{e.source:<12} {e.scenario_name:<48} {e.file_name:<32} "
            f"{e.route_id:<10} {e.town:<10} {e.scenario_type}"
        )


def _steervla_action_execution_cfg(steervla_cfg, *, residual: bool = False) -> dict[str, Any] | None:
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
    if residual and not remote:
        action_input_space = "normalized"
    else:
        # SteerVLA actor applies OpenPI Unnormalize + fixed steervla denormalize_actions.
        action_input_space = "policy_output"
    return {
        "output_action_format": fmt,
        "action_horizon": ah,
        "action_dim": ad,
        "action_input_space": action_input_space,
        # PID brake threshold (m/s): desired speeds below this brake. Lower it to avoid the
        # cold-start brake trap. See SimlingoStyleWaypointDecoder.control_pid.
        "brake_speed": float(steervla_cfg.get("brake_speed", 0.1)),
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
    noise_scale: float = 1.0,
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
            noise_scale=noise_scale,
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
        noise_scale=noise_scale,
    )


# --------------------------------------------------------------------------- #
# Residual-stack helpers (used by run_online_residual)                        #
# --------------------------------------------------------------------------- #

# Reward components emitted by ogbench/carla/carla_utils.py::_compute_reward_and_info.
_REWARD_KEYS = (
    "reward_total",
    "reward_progress",
    "reward_centering",
    "reward_heading",
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
    "centering_factor",
    "heading_factor",
    "collision_count",
)


def _base_chunk(vla_sample_fn, raw_holder: dict, obs: dict, base_noise: jnp.ndarray) -> np.ndarray:
    """Frozen SteerVLA base action chunk for one flow-noise draw -> float32 [action_dim]."""
    raw_holder["obs"] = obs
    out = vla_sample_fn(jnp.zeros((1, 1), dtype=jnp.float32), base_noise)
    return np.asarray(jax.device_get(out), dtype=np.float32).reshape(-1)


def _ego_control_log(obs: dict) -> dict[str, float]:
    """Last-applied CARLA control + speed from the gym state vector (drive/* line charts)."""
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
    """CARLA ego speed (m/s) from the gym state vector; used only for the debug_task reward."""
    if not isinstance(obs, dict):
        return np.float32(0.0)
    s = np.asarray(obs.get("state"), dtype=np.float32).reshape(-1)
    if s.size <= _EGO_STATE_IDX_SPEED:
        return np.float32(0.0)
    return np.float32(s[_EGO_STATE_IDX_SPEED])


def _chunk_stats_log(name: str, chunk_flat: np.ndarray, action_dim: int) -> dict[str, float]:
    """Per-component scalar summaries of a flattened ``(H, action_dim)`` chunk.

    Splits the SimLingo layout into speed deltas (cols 0:2) and route deltas (cols 2:4);
    ``*_speed_cumnorm`` is the cumulative speed-waypoint magnitude the PID turns into
    desired speed, so it flags a collapsed (near-stationary) base chunk.
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
    """Scalar summaries of a 2-D ``[accel, steer]`` action for W&B line charts."""
    a = np.asarray(vec, dtype=np.float32).reshape(-1)
    out: dict[str, float] = {}
    if a.size >= 1:
        out[f"action/{name}_accel"] = float(a[0])
    if a.size >= 2:
        out[f"action/{name}_steer"] = float(a[1])
    return out


def _expo_candidate_log(base_cands: np.ndarray, q_all: np.ndarray, winner_idx: int, n: int) -> dict[str, float]:
    """EXPO diagnostics for one step's 2N pool: Q spread, base-vs-edit winner + margin, and
    base-action diversity as per-dim Gaussian entropy. q_all is (2N,) min-ensemble Q (first N
    base, last N edits); base_cands is (N, adim)."""
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
    var = bc.var(axis=0)
    ent = 0.5 * np.log(2.0 * np.pi * np.e * np.maximum(var, 1e-12))
    for d in range(ent.shape[0]):
        out[f"expo/base_entropy/dim_{d}"] = float(ent[d])
    out["expo/base_entropy_mean"] = float(ent.mean())
    return out


def _drive_diagnostics_log(info: dict) -> dict[str, float]:
    """Per-step rollout diagnostics (route progress, centering, heading). Objective-agnostic, so
    these are logged in debug runs too."""
    out: dict[str, float] = {}
    for k in _DRIVE_KEYS:
        if k in info:
            out[f"rollout/{k}"] = float(info[k])
    return out


def _reward_breakdown_log(info: dict) -> dict[str, float]:
    """Flatten the env reward components + drive diagnostics into a W&B log dict."""
    out: dict[str, float] = {}
    if "reward_total" in info:
        for k in _REWARD_KEYS:
            if k in info:
                out[f"reward/{k}"] = float(info[k])
    out.update(_drive_diagnostics_log(info))
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
    # termination_reason is a string (not chartable); emit it plus a one-hot over the
    # reasons the env can end on so W&B can plot per-reason frequency. The env only sets
    # termination_reason for crash_stuck / episode_max_steps; a scenario-tree end (success
    # or collision/route failure) leaves it unset but populates ``success``.
    reason = str(info.get("termination_reason") or "").strip()
    if not reason:
        reason = "scenario_end" if "success" in info else "unknown"
    out["rollout/termination_reason"] = reason
    for r in ("crash_stuck", "episode_max_steps", "scenario_end", "unknown"):
        out[f"rollout/term_{r}"] = 1.0 if reason == r else 0.0
    return out


def _final_step_reward_log(info: dict) -> dict[str, float]:
    """Terminal-step reward breakdown (rollout/final_step_*); shared by both episode-end paths."""
    if "reward_total" not in info:
        return {}
    return {
        "rollout/final_step_reward": float(info["reward_total"]),
        "rollout/final_step_reward_progress": float(info.get("reward_progress", 0.0)),
        "rollout/final_step_reward_centering": float(info.get("reward_centering", 0.0)),
        "rollout/final_step_reward_heading": float(info.get("reward_heading", 0.0)),
        "rollout/final_step_reward_terminal": float(info.get("reward_terminal", 0.0)),
        "rollout/final_step_penalty_collision": float(info.get("penalty_collision", 0.0)),
        "rollout/final_step_penalty_outside_route": float(info.get("penalty_outside_route", 0.0)),
        "rollout/final_step_penalty_steer": float(info.get("penalty_steer", 0.0)),
        "rollout/final_step_penalty_brake": float(info.get("penalty_brake", 0.0)),
        "rollout/final_step_penalty_crash_stuck": float(info.get("penalty_crash_stuck", 0.0)),
        "rollout/final_step_success": float(bool(info.get("success", False))),
    }


# --------------------------------------------------------------------------- #
# Rollout-video annotation
# --------------------------------------------------------------------------- #


def _viz_image_from_raw(raw) -> Optional[np.ndarray]:
    """High-res camera frame for the rollout video (prefers ``image_viz``, falls back to ``image``)."""
    if isinstance(raw, dict):
        img = raw.get("image_viz")
        if img is None:
            img = raw.get("image")
        return None if img is None else np.asarray(img, dtype=np.uint8)
    return None if raw is None else np.asarray(raw, dtype=np.uint8)


def _draw_corner_badge(frame: np.ndarray, label: str, *, corner: str = "tl", bg=(0, 0, 0)) -> np.ndarray:
    """Boxed text badge in a top corner (``'tl'``/``'tr'``); no-op if cv2 is unavailable."""
    annotated = np.array(frame, copy=True)
    try:
        import cv2  # type: ignore

        font_scale, thickness, pad = 0.38, 1, 4
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        bw, bh = tw + 2 * pad, th + baseline + 2 * pad
        x0 = max(6, annotated.shape[1] - 6 - bw) if corner == "tr" else 6
        y0 = 6
        x1, y1 = x0 + bw, y0 + bh
        cv2.rectangle(annotated, (x0, y0), (x1, y1), bg, thickness=-1)
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 255, 255), thickness=1)
        cv2.putText(annotated, label, (x0 + pad, y1 - baseline - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    except Exception:
        pass
    return annotated


def _format_text_field(raw, key: str) -> str:
    """Stringify a raw obs field (text passthrough, else leading token ids) for the video panel."""
    if not isinstance(raw, dict) or raw.get(key) is None:
        return ""
    value = raw.get(key)
    if isinstance(value, str):
        return value
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return ""
    if arr.dtype == bool:
        return " ".join(map(str, arr.astype(np.int32)[:16].tolist()))
    return " ".join(map(str, arr.astype(np.int32)[:24].tolist()))


def _critic_input_text(critic_mode: str, critic_label: np.ndarray, critic_text: str, raw) -> str:
    """Human-readable critic-feedback string for the DSRL video panel."""
    if critic_mode == "none":
        return "none"
    if critic_mode == "action_delta":
        arr = np.asarray(critic_label, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return "[]"
        show = " ".join(f"{v:+.3f}" for v in arr[:8])
        return show if arr.size <= 8 else f"{show} ..."
    if critic_mode in ("delta_commentary_bow", "vlm_chunk_bow"):
        return critic_text or "?"
    commentary = raw.get("commentary_text", "") if isinstance(raw, dict) else ""
    return str(commentary or "?")


def _annotate_waypoints_frame(frame: np.ndarray, action_flat, exec_cfg) -> np.ndarray:
    """Project the SteerVLA chunk's waypoints onto the frame (no-op without cfg/action)."""
    if exec_cfg is None or action_flat is None:
        return frame
    try:
        from ogbench.carla.waypoint_viz import annotate_waypoints_on_frame

        return annotate_waypoints_on_frame(frame, action_flat=np.asarray(action_flat), exec_cfg=exec_cfg)
    except Exception:
        return frame


def _annotate_text_panel(
    frame, raw, *, reward: float, critic_text: str = "", steervla_actor=None, hud: dict | None = None
) -> np.ndarray:
    """Bottom text panel: reward corner + prompt/reasoning/subtask (+ optional expert/critic lines).

    ``critic_text`` / ``expert_action`` lines are omitted when unavailable, so residual videos stay
    clean while the DSRL loop (which always supplies critic text) keeps its full panel. ``hud`` (when
    given) adds a residual-RL header line: base/residual/final actions + progress%/step/return/scale.
    """
    base = np.array(frame, copy=True)
    try:
        import cv2  # type: ignore

        h, w = base.shape[:2]
        font_scale, line_h = 0.26, 13
        annotated_top = _draw_corner_badge(base, f"r={reward:+.3f}", corner="tl")

        prompt = str(raw.get("openpi_prompt_text") or "").strip() if isinstance(raw, dict) else ""
        if not prompt and isinstance(raw, dict):
            from vlas.steervla import (
                carla_state_vec_to_steervla_state,
                format_steervla_cot_prompt,
                routing_instruction_prompt,
                steervla_prompt_state_dim,
            )

            state = np.asarray(raw.get("state", []), dtype=np.float32).reshape(-1)
            speed = float(state[15]) if state.size > 15 else 0.0
            routing = str(raw.get("routing_command", "") or "").strip() or "Follow the route."
            include_hist = bool(getattr(steervla_actor, "include_ego_history", False))
            proprio_norm = bool(getattr(steervla_actor, "proprio_norm", True)) if steervla_actor is not None else True
            state_pad = carla_state_vec_to_steervla_state(
                state, include_ego_history=include_hist, proprio_norm=proprio_norm
            )
            prompt = format_steervla_cot_prompt(
                routing_instruction_prompt(routing_command=routing, current_speed_mps=speed),
                state_pad,
                state_dim=steervla_prompt_state_dim(include_ego_history=include_hist),
            )
        reasoning = _format_text_field(raw, "reasoning_text") or _format_text_field(raw, "reasoning")
        subtask = _format_text_field(raw, "subtask_text") or _format_text_field(raw, "subtask")
        expert_action_str = ""
        if isinstance(raw, dict) and raw.get("expert_action") is not None:
            ea = np.asarray(raw["expert_action"], dtype=np.float32).reshape(-1)
            first = ea[:4] if ea.size >= 4 else ea
            expert_action_str = " ".join(f"{v:.3f}" for v in first)

        def _clip(txt: str, n: int = 120) -> str:
            return txt if len(txt) <= n else (txt[: n - 3] + "...")

        lines = []
        if hud:
            lines.append(
                f"Action base={hud.get('base', '-')} residual={hud.get('residual', '-')} "
                f"final={hud.get('final', '-')} scale={hud.get('scale', '-')}"
            )
            lines.append(
                f"progress={float(hud.get('progress', 0.0)):.1f}% step={int(hud.get('ep_step', 0))} "
                f"return={float(hud.get('ep_return', 0.0)):+.1f}"
            )
        if critic_text:
            lines.append(f"Expert: {_clip(critic_text)}")
        if expert_action_str:
            lines.append(f"ExpertAct[0]: {expert_action_str}")
        lines += [
            f"Prompt: {_clip(prompt)}",
            f"Reasoning: {_clip(reasoning)}",
            f"Subtask: {_clip(subtask)}",
        ]
        panel_h = max(72, line_h * (len(lines) + 1))
        annotated = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
        annotated[:h, :, :] = annotated_top
        cv2.line(annotated, (0, h), (w - 1, h), (255, 255, 255), 1)
        y = h + line_h
        for line in lines:
            cv2.putText(annotated, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            y += line_h
        return annotated
    except Exception:
        return base


def _annotate_full_frame(
    frame, raw, *, reward: float, action_flat=None, exec_cfg=None,
    critic_text: str = "", collision=None, steervla_actor=None, expert_debug: bool = False,
    hud: dict | None = None,
) -> np.ndarray:
    """Unified rollout-video frame: waypoint overlay + text panel + optional collision badge."""
    out = np.asarray(frame)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if not expert_debug:
        out = _annotate_waypoints_frame(out, action_flat, exec_cfg)
    out = _annotate_text_panel(
        out, raw, reward=reward, critic_text=critic_text, steervla_actor=steervla_actor, hud=hud
    )
    if collision is not None:
        out = _draw_corner_badge(out, f"COLL c={int(collision[0])} e={int(collision[1])}", corner="tr", bg=(255, 0, 0))
    return out


def _episode_video(frames: list[np.ndarray], fps: float):
    """Stack captured frames into a W&B video (T, C, H, W); None if empty."""
    if not frames:
        return None
    video = np.stack(frames, axis=0)
    if video.ndim == 4:
        video = np.transpose(video, (0, 3, 1, 2))
    return wandb.Video(video, fps=fps, format="mp4")


def _maybe_capture_frame(
    frames: list[np.ndarray], obs: dict, reward: float, *, episode_steps: int, done: bool,
    log_video: bool, video_every: int, action_flat=None, exec_cfg=None,
    collision=None, steervla_actor=None, hud: dict | None = None,
) -> Optional[np.ndarray]:
    """Append an annotated frame on capture steps (every Nth step + the terminal one).

    Uses the shared :func:`_annotate_full_frame` (waypoints + text panel + optional collision
    badge). Returns the annotated frame that was appended (or ``None``) so callers can also feed
    a live viewer without re-annotating.
    """
    if log_video and (episode_steps % video_every == 0 or done):
        frame = _viz_image_from_raw(obs)
        if frame is not None:
            annotated = _annotate_full_frame(
                frame, obs, reward=float(reward), action_flat=action_flat, exec_cfg=exec_cfg,
                collision=collision, steervla_actor=steervla_actor, hud=hud,
            )
            frames.append(annotated)
            return annotated
    return None


def _log_episode_end(
    info: dict, *, episode_return: float, episode_steps: int, episode_index: int,
    frames: list[np.ndarray], log_video: bool, video_fps: float, step: int, train_logger: CsvLogger,
    collision_events: int = 0, debug_task: bool = False, debug_return: float = 0.0,
) -> None:
    """Log per-episode rollout metrics (+ video) to W&B and CSV, then clear ``frames``."""
    rollout_log: dict[str, Any] = {
        "rollout/episode_length": episode_steps,
        "rollout/episodes": episode_index,
        "rollout/episode_collision_events": float(collision_events),
        "rollout/collisions_over_episode": float(collision_events) / max(float(episode_steps), 1.0),
    }
    rollout_log.update(_episode_summary_log(info))
    if debug_task:
        rollout_log["debug/episode_return"] = debug_return
    else:
        rollout_log["rollout/episode_return"] = episode_return
        rollout_log.update(_final_step_reward_log(info))
    if log_video:
        video = _episode_video(frames, video_fps)
        if video is not None:
            rollout_log["rollout/episode_video"] = video
    wandb.log(rollout_log, step=step)
    train_logger.log(rollout_log, step=step)
    frames.clear()


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
    *,
    image_encoder: str = "impala",
    siglip_encoder=None,
    siglip_include_prompt_subtask: bool = False,
) -> None:

    obs_mode = str(agent_config.get("observation_mode", "state"))
    _extract_obs_kwargs = dict(
        image_encoder=image_encoder,
        siglip_encoder=siglip_encoder,
        siglip_include_prompt_subtask=siglip_include_prompt_subtask,
        steervla_actor=steervla_actor,
    )

    capacity = int(agent_config.get("buffer_capacity", 5_000))
    warmup = int(agent_config.get("warmup_steps", 1000))
    warmup_expo = int(agent_config.get("warmup_expo_steps", 0))
    updates_per_step = int(agent_config.get("updates_per_step", 1))
    update_interval = int(agent_config.get("update_interval", 1))
    if update_interval < 1:
        raise ValueError(f"update_interval must be >= 1, got {update_interval}")
    batch_size = int(agent_config.get("batch_size", 256))
    enable_updates = bool(agent_config.get("enable_updates", True))
    if FLAGS.enable_updates is not None:
        enable_updates = bool(FLAGS.enable_updates)

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    if not enable_updates:
        print("[main_carla] enable_updates=False: rollout-only (no RL gradient updates)", flush=True)
    if warmup > 0 and agent is not None and not FLAGS.expert_debug:
        print(f"[main_carla] warmup: no RL updates while step < {warmup}", flush=True)
    if update_interval > 1 and agent is not None and enable_updates:
        print(f"[main_carla] RL updates every {update_interval} env steps", flush=True)
    if warmup_expo > 0:
        if warmup_expo <= warmup:
            raise ValueError(
                f"warmup_expo_steps ({warmup_expo}) must be greater than warmup_steps ({warmup})."
            )
        if agent is not None and hasattr(agent, "set_edit_actor_rollout_enabled"):
            print(
                f"[main_carla] EXPO warmup: RL updates from step {warmup}, "
                f"VLA-only rollout until step > {warmup_expo}",
                flush=True,
            )
    
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
        from vlas.steervla import (
            openpi_replay_fields_from_observation,
            openpi_replay_fields_with_fast_placeholders,
        )

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
            if fk is not None and fmk is not None:
                out["openpi_tokenized_fast"] = np.asarray(fk, dtype=np.int32)
                out["openpi_tokenized_fast_mask"] = np.asarray(fmk, dtype=bool)
                out["fast"] = out["openpi_tokenized_fast"]
                out["fast_mask"] = out["openpi_tokenized_fast_mask"]
        return openpi_replay_fields_with_fast_placeholders(
            out,
            model_cfg=getattr(steervla_actor, "model_cfg", None),
        )
    
    raw_obs_holder = raw_carla_obs_holder

    if raw_obs_holder is not None and raw_obs_holder.get("obs") is not None:
        obs_raw = raw_obs_holder["obs"]
    else:
        obs_raw, _info = env.reset(seed=FLAGS.seed)
    if raw_obs_holder is not None:
        raw_obs_holder["obs"] = obs_raw
        raw_obs_holder["next_obs"] = obs_raw
    obs = _extract_agent_obs(env, obs_raw, obs_mode, **_extract_obs_kwargs)
    if siglip_include_prompt_subtask:
        print(
            "[main_carla] SigLIP observations = [image_embed, prompt_embed, subtask_embed]",
            flush=True,
        )
    log_images = obs_mode == "image"
    capture_rollout_video = log_images or bool(FLAGS.live_policy_view)

    live_viewer: LivePolicyViewer | None = None
    if FLAGS.live_policy_view:
        live_viewer = LivePolicyViewer(
            os.path.join(FLAGS.save_dir, "live_policy.mp4"),
            port=int(FLAGS.live_policy_port),
            fps=10.0,
            publish_every_n_steps=int(FLAGS.live_policy_interval),
        )
        live_viewer.start()

    _critic_feedback_mode = resolve_critic_feedback_mode(agent_config)
    _vlm_coach: OnlineVLMSession | None = None
    lang_fb = agent_config.get("language_feedback")
    if _critic_feedback_mode == "vlm_chunk_bow":
        vlm_cfg = agent_config.get("vlm_coach")
        if vlm_cfg is None:
            raise ValueError(
                "language_feedback.source='vlm' requires agent config block 'vlm_coach'."
            )
        _vlm_coach = OnlineVLMSession(
            vlm_cfg,
            save_dir=FLAGS.save_dir,
            action_chunk_steps=int(agent_config.get("action_horizon", 10)),
        )
        print(
            f"[main_carla] VLM coach enabled (provider={_vlm_coach.provider}, "
            f"query_every={_vlm_coach.query_every_n_episode_steps} ep steps)",
            flush=True,
        )
        capture_rollout_video = True
    _online_training_mode = str(agent_config.get("online_training_mode", "rl")).strip().lower()
    _lang_dim = critic_language_dim(agent_config)
    _steervla_exec_cfg = _steervla_action_execution_cfg(agent_config.get("steervla") or {})

    steervla_cfg = agent_config.get("steervla") or {}
    env_ah = int(steervla_cfg.get("action_horizon", agent_config.get("vla_action_horizon", 10)))
    env_ad = int(steervla_cfg.get("action_dim", agent_config.get("vla_action_dim", 4)))
    action_dim = env_ah * env_ad
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
    if agent_config.get("debug_task", False):
        example_transition["ego_speed"] = _ego_speed_mps_from_raw(obs_raw)
    if steervla_actor is not None:
        openpi0 = _openpi_fields_from_raw(obs_raw)
        example_transition.update(openpi0)
        example_transition.update({f"next_{k}": np.array(v) for k, v in openpi0.items()})
        
    # Create replay buffer
    buffer = ReplayBuffer.create(example_transition, size=capacity)
    _buffer_keys = frozenset(example_transition.keys())

    def _buffer_transition(raw_obs: dict, next_raw_obs: dict, **core) -> dict[str, np.ndarray]:
        """Build a transition dict whose keys match the replay buffer schema exactly."""
        transition = dict(core)
        if steervla_actor is not None:
            transition.update(_openpi_fields_from_raw(raw_obs))
            transition.update(
                {f"next_{k}": np.array(v) for k, v in _openpi_fields_from_raw(next_raw_obs).items()}
            )
        extra = set(transition.keys()) - _buffer_keys
        if extra:
            raise KeyError(
                f"Transition has keys not in replay buffer schema: {sorted(extra)}. "
                f"Recreate the buffer or extend example_transition."
            )
        missing = _buffer_keys - set(transition.keys())
        if missing:
            raise KeyError(f"Transition missing replay buffer keys: {sorted(missing)}")
        return transition
    
    rng = jax.random.PRNGKey(FLAGS.seed + 1)
    episode_return, episode_steps, episode_count = 0.0, 0, 0
    _last_buf_idx: int | None = None
    episode_collision_count = 0
    episode_collision_events = 0
    prev_collision_count = 0
    last_log_time = time.time()
    episode_video_every = 2
    episode_video_fps = 10.0
    episode_video_frames: list[np.ndarray] = []
    episode_trajectory: list[dict[str, Any]] = []
    episode_video_frame_index = 0
    last_video_reward: float = 0.0
    last_policy_action: np.ndarray | None = None
    last_video_critic_text: str = ""
    trajectory_dir = os.path.join(FLAGS.save_dir, "trajectories")
    os.makedirs(trajectory_dir, exist_ok=True)

    def _sync_steervla_debug_noise_context(route_name: str) -> None:
        if steervla_actor is None or not steervla_actor.debug_noise:
            return
        steervla_actor.set_debug_noise_context(
            run_name=exp_name,
            save_root=FLAGS.save_dir,
            route_name=route_name,
            episode=max(0, episode_count),
        )

    _sync_steervla_debug_noise_context(
        str(FLAGS.route or obs_raw.get("routing_command", "?") if isinstance(obs_raw, dict) else "?")
    )
    if _vlm_coach is not None:
        _vlm_coach.begin_episode(
            episode_count=max(1, episode_count),
            route_name=str(obs_raw.get("routing_command", "?") if isinstance(obs_raw, dict) else "?"),
        )

    def _maybe_log_episode_video(
        rollout_log: dict,
        final_frame: np.ndarray | None,
        final_raw: dict[str, Any] | None,
        *,
        final_reward: float,
        final_critic_text: str,
    ) -> None:
        """Assemble the episode video (already-annotated per-step frames + the final frame)."""
        if not log_images:
            return
        frames = list(episode_video_frames)
        if final_frame is not None:
            frames.append(
                _annotate_full_frame(
                    final_frame,
                    final_raw,
                    reward=final_reward,
                    action_flat=last_policy_action,
                    exec_cfg=_steervla_exec_cfg,
                    critic_text=final_critic_text,
                    steervla_actor=steervla_actor,
                    expert_debug=FLAGS.expert_debug,
                )
            )
        video = _episode_video(frames, episode_video_fps)
        if video is not None:
            rollout_log["rollout/episode_video"] = video

    def _save_episode_trajectory_json(
        *,
        episode_index: int,
        route_name: str,
        episode_step_count: int,
        done_info: dict[str, Any],
    ) -> str | None:
        if not episode_trajectory:
            return None
        out_path = os.path.join(trajectory_dir, f"episode_{episode_index:04d}.json")
        payload = {
            "episode": int(episode_index),
            "route": route_name,
            "episode_steps": int(episode_step_count),
            "video_fps": float(episode_video_fps),
            "video_frame_stride": int(episode_video_every),
            "success": bool(done_info.get("success", False)),
            "termination_reason": done_info.get("termination_reason"),
            "scenario_tree_status": done_info.get("scenario_tree_status"),
            "steps": episode_trajectory,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out_path

    def _append_trajectory_step(
        *,
        global_step: int,
        episode_step: int,
        state_raw: dict[str, Any],
        step_info: dict[str, Any],
        collision_occurred: bool,
        in_video: bool,
        video_frame_index: int | None,
    ) -> None:
        drive = ego_drive_metrics_from_state_vec(state_raw["state"])
        record: dict[str, Any] = {
            "step": int(global_step),
            "episode_step": int(episode_step),
            "ego_speed_mps": float(drive["ego_speed"]),
            "control_throttle": float(drive["control_throttle"]),
            "control_steer": float(drive["control_steer"]),
            "control_brake": float(drive["control_brake"]),
            "collision": bool(collision_occurred),
            "route_progress_pct": float(step_info.get("route_progress_pct", 0.0)),
            "in_video": bool(in_video),
        }
        if in_video and video_frame_index is not None:
            record["video_frame_index"] = int(video_frame_index)
            record["video_timestamp_sec"] = round(float(video_frame_index) / episode_video_fps, 3)
        else:
            record["video_frame_index"] = None
            record["video_timestamp_sec"] = None
        episode_trajectory.append(record)

    def _block_until_ready_tree(tree):
        return jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            tree,
        )

    last_update_info = None

    def _sample_agent_action(subkey):
        """Rollout policy (SteerVLA VLA path, DAgger, or DSRL flow); used in warmup and RL phases."""
        uses_vla = getattr(agent, "vla_sample_fn", None) is not None
        pause_env = uses_vla and hasattr(env, "pause_for_vla_inference")
        if pause_env:
            env.pause_for_vla_inference()
        t0 = time.time()
        try:
            if uses_vla:
                return agent.sample_actions_with_vla(obs[None], seed=subkey)
            if _online_training_mode == "dagger" and hasattr(agent, "sample_actions_dagger"):
                return agent.sample_actions_dagger(obs[None])
            return agent.sample_actions(obs[None], seed=subkey)
        finally:
            if uses_vla:
                elapsed = time.time() - t0
                if elapsed > 5.0:
                    print(f"[main_carla] VLA sample took {elapsed:.1f}s", flush=True)
            if pause_env:
                env.resume_after_vla_inference()

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
        in_warmup = warmup > 0 and step < warmup
        in_expo_warmup = (
            warmup_expo > 0
            and step <= warmup_expo
            and agent is not None
            and hasattr(agent, "set_edit_actor_rollout_enabled")
        )
        if agent is not None and hasattr(agent, "set_edit_actor_rollout_enabled"):
            agent.set_edit_actor_rollout_enabled(step > warmup_expo)
        if (
            obs_mode == "image"
            and image_encoder == "siglip"
            and siglip_include_prompt_subtask
        ):
            obs = _extract_agent_obs(env, obs_raw, obs_mode, **_extract_obs_kwargs)
        if steervla_actor is not None and steervla_actor.debug_noise:
            steervla_actor.debug_noise_episode_step = episode_steps + 1
        if FLAGS.expert_debug or _in_expert_recovery or agent is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            step_obs = obs
        else:
            action_jax = _sample_agent_action(sub)
            _block_until_ready_tree(action_jax)
            action = np.asarray(action_jax[0])
            last_policy_action = action
            if (
                obs_mode == "image"
                and image_encoder == "siglip"
                and siglip_include_prompt_subtask
            ):
                step_obs = _extract_agent_obs(env, obs_raw, obs_mode, **_extract_obs_kwargs)
            else:
                step_obs = obs
            if steervla_actor is not None and _last_buf_idx is not None:
                from vlas.steervla import openpi_cot_replay_fields_from_raw

                _cot_for_next = openpi_cot_replay_fields_from_raw(obs_raw)
                if _cot_for_next:
                    buffer.update_at(
                        _last_buf_idx,
                        **{
                            f"next_{k}": v
                            for k, v in _cot_for_next.items()
                            if f"next_{k}" in _buffer_keys
                        },
                    )
        t_sample_end = time.time()

        t_step_start = time.time()
        if FLAGS.expert_debug or _in_expert_recovery:
            next_obs_raw, reward, terminated, truncated, info = env.step_expert(obs_raw)
        else:
            next_obs_raw, reward, terminated, truncated, info = env.step(action)
        if raw_obs_holder is not None:
            raw_obs_holder["next_obs"] = next_obs_raw
        next_obs = _extract_agent_obs(env, next_obs_raw, obs_mode, **_extract_obs_kwargs)
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
            _lang = compute_action_delta(obs_raw, action, agent, agent_config)
            _next_lang = _zero_label  # bootstrap target sees zero delta (next action unknown)
        elif _critic_feedback_mode == "delta_commentary_bow":
            _lang_text, _lang = compute_action_delta_commentary(obs_raw, action, agent)
            _next_lang = _zero_label  # depends on current action-vs-expert comparison only
        elif _critic_feedback_mode == "vlm_chunk_bow":
            _lang_text, _lang = (
                _vlm_coach.language_label_for_episode_step(episode_steps + 1)
                if _vlm_coach is not None
                else ("", _zero_label)
            )
            _next_lang = _zero_label
        elif _critic_feedback_mode == "subtask_siglip":
            # Best-of-N stashes a SigLIP subtask embedding on ``obs_raw`` after sampling.
            # ``env._obs_dict()`` also puts a 119-dim expert commentary BoW in the same
            # key — ignore mismatched sizes and never copy env labels into ``next_*``.
            _lang = _coerce_language_label(obs_raw.get("language_label"), _lang_dim, _zero_label)
            _next_lang = _zero_label
        else:
            _lang = _coerce_language_label(obs_raw.get("language_label"), _lang_dim, _zero_label)
            _next_lang = _coerce_language_label(
                next_obs_raw.get("language_label"), _lang_dim, _zero_label
            )
        _critic_text_for_video = _critic_input_text(_critic_feedback_mode, _lang, _lang_text, obs_raw)

        replay_action = action.astype(np.float32)
        if _online_training_mode == "dagger" and not FLAGS.expert_debug:
            replay_action = np.asarray(obs_raw.get("expert_action", replay_action), dtype=np.float32)

        buf_idx = buffer.add_transition(
            _buffer_transition(
                obs_raw,
                next_obs_raw,
                observations=np.asarray(step_obs),
                actions=replay_action,
                rewards=np.float32(reward),
                next_observations=np.asarray(next_obs),
                masks=np.float32(0.0 if done else 1.0),
                terminals=np.float32(1.0 if done else 0.0),
                language_label=_lang,
                next_language_label=_next_lang,
                **(
                    {"ego_speed": _ego_speed_mps_from_raw(obs_raw)}
                    if agent_config.get("debug_task", False)
                    else {}
                ),
            )
        )
        _last_buf_idx = int(buf_idx)
        t_step_end = time.time()
        
        t_log_start = time.time()
        cot_obs_raw = dict(obs_raw)  # holds reasoning_text/subtask_text stashed by VLA
        obs = next_obs
        obs_raw = next_obs_raw
        episode_return += float(reward)
        episode_steps += 1
        if _vlm_coach is not None:
            _vlm_coach.track_buffer_transition(buffer_index=buf_idx, episode_step=episode_steps)
        collision_count = int(info.get("collision_count", 0))
        episode_collision_count = max(episode_collision_count, collision_count)
        collision_delta = max(0, collision_count - prev_collision_count)
        episode_collision_events += collision_delta
        prev_collision_count = collision_count
        step_in_video = False
        step_video_frame_index: int | None = None
        if capture_rollout_video:
            should_sample_periodic = episode_steps % episode_video_every == 0
            had_collision_this_step = collision_delta > 0
            step_in_video = should_sample_periodic or had_collision_this_step
            if step_in_video:
                step_video_frame_index = episode_video_frame_index
            raw_frame = _viz_image_from_raw(obs_raw) if step_in_video else None
            if raw_frame is not None:
                frame = _annotate_full_frame(
                    raw_frame,
                    cot_obs_raw,
                    reward=float(reward),
                    action_flat=last_policy_action,
                    exec_cfg=_steervla_exec_cfg,
                    critic_text=_critic_text_for_video,
                    collision=(collision_count, episode_collision_events) if had_collision_this_step else None,
                    steervla_actor=steervla_actor,
                    expert_debug=FLAGS.expert_debug,
                )
                episode_video_frames.append(frame)
                if _vlm_coach is not None:
                    _vlm_coach.record_frame(frame)
                episode_video_frame_index += 1
            else:
                step_in_video = False
                step_video_frame_index = None
            if live_viewer is not None and episode_video_frames:
                live_viewer.publish_frames(episode_video_frames, step)
        _append_trajectory_step(
            global_step=step,
            episode_step=episode_steps,
            state_raw=next_obs_raw,
            step_info=info,
            collision_occurred=collision_delta > 0,
            in_video=step_in_video,
            video_frame_index=step_video_frame_index,
        )
        if _vlm_coach is not None and episode_trajectory:
            _vlm_coach.record_trajectory_step(episode_trajectory[-1])
            if _vlm_coach.maybe_query(episode_step=episode_steps, done_info=info):
                _vlm_coach.backfill_buffer(buffer)
        last_video_reward = float(reward)
        last_video_critic_text = _critic_text_for_video
        t_log_end = time.time()

        step_wb: dict[str, Any] = {}
        step_wb.update(_reward_breakdown_log(info))
        step_wb.update(_ego_control_log(next_obs_raw))
        step_wb["rollout/collision_events"] = float(collision_delta)

        # Log critic feedback signal (obs_raw is already next_obs_raw here)
        if _critic_feedback_mode == "action_delta":
            step_wb["label/action_delta_norm"] = float(np.linalg.norm(_lang))
        elif _critic_feedback_mode == "delta_commentary_bow":
            if _lang_text:
                step_wb["label/commentary_delta"] = wandb.Html(f"<p>{_lang_text}</p>")
        elif _critic_feedback_mode == "vlm_chunk_bow":
            if _lang_text:
                step_wb["label/vlm_chunk_feedback"] = wandb.Html(f"<p>{_lang_text}</p>")
        else:
            _commentary = obs_raw.get("commentary_text", "") if isinstance(obs_raw, dict) else ""
            if _commentary:
                step_wb["label/commentary"] = wandb.Html(f"<p>{_commentary}</p>")

        step_wb["time/sample_time"] = t_sample_end - t_sample_start
        step_wb["time/step_time"] = t_step_end - t_step_start
        step_wb["time/log_time"] = t_log_end - t_log_start
        step_wb["training/in_warmup"] = float(in_warmup)
        step_wb["training/in_expo_warmup"] = float(in_expo_warmup)
        step_wb["training/enable_updates"] = float(enable_updates)
        if "episode_step_count" in info:
            step_wb["rollout/episode_step"] = float(info["episode_step_count"])

        wandb.log(step_wb, step=step)

        if done:
            episode_count += 1
            done_info = dict(info)
            done_episode_return = float(episode_return)
            done_episode_steps = int(episode_steps)
            done_collision_count = int(episode_collision_count)
            done_collision_events = int(episode_collision_events)
            done_route = str(done_info.get("route", "?"))

            # Finish pending JAX work, then reset CARLA immediately.
            if agent is not None:
                _block_until_ready_tree(agent)
                reset_vla_cache = getattr(getattr(agent, "vla_sample_fn", None), "reset_action_cache", None)
                if reset_vla_cache is not None:
                    reset_vla_cache()
            obs_raw, _info = env.reset(seed=FLAGS.seed + episode_count)
            if raw_obs_holder is not None:
                raw_obs_holder["obs"] = obs_raw
                raw_obs_holder["next_obs"] = obs_raw
            obs = _extract_agent_obs(env, obs_raw, obs_mode, **_extract_obs_kwargs)
            _last_buf_idx = None

            rollout_log = {
                "rollout/episode_return": done_episode_return,
                "rollout/episode_steps": done_episode_steps,
                "rollout/episodes": episode_count,
                "rollout/route": done_route,
                "rollout/episode_collision_count": float(done_collision_count),
                "rollout/episode_collision_events": float(done_collision_events),
                "rollout/collisions_over_episode": float(done_collision_events)
                / max(float(done_episode_steps), 1.0),
            }
            rollout_log.update(_final_step_reward_log(done_info))
            if FLAGS.expert_recover_debug:
                rollout_log["rollout/vla_steps_budget"] = float(_vla_steps_budget)
            traj_path = _save_episode_trajectory_json(
                episode_index=episode_count,
                route_name=done_route,
                episode_step_count=done_episode_steps,
                done_info=done_info,
            )
            if traj_path is not None:
                rollout_log["rollout/trajectory_json"] = traj_path
            n_video_frames = len(episode_video_frames) + (1 if end_img is not None else 0)
            _maybe_log_episode_video(
                rollout_log,
                end_img if log_images else None,
                cot_obs_raw if log_images else None,
                final_reward=last_video_reward,
                final_critic_text=last_video_critic_text,
            )
            if live_viewer is not None and episode_video_frames:
                live_viewer.publish_frames(episode_video_frames, step, force=True)
            wandb.log(rollout_log, step=step)
            train_logger.log(
                {
                    k: v
                    for k, v in rollout_log.items()
                    if k.startswith("rollout/") and k != "rollout/route"
                },
                step=step,
            )
            print(
                f"[main_carla] episode {episode_count} done: "
                f"return={done_episode_return:.3f} steps={done_episode_steps} "
                f"route={done_route!r} "
                f"video={'yes' if 'rollout/episode_video' in rollout_log else 'no'} "
                f"({n_video_frames} frames)",
                flush=True,
            )
            if _vlm_coach is not None:
                _vlm_coach.maybe_query(
                    episode_step=done_episode_steps, done_info=done_info, force=True
                )
                _vlm_coach.backfill_buffer(buffer)
            episode_video_frames = []
            episode_trajectory = []
            episode_video_frame_index = 0
            episode_return, episode_steps = 0.0, 0
            if _vlm_coach is not None:
                _vlm_coach.reset_episode()
                _vlm_coach.begin_episode(
                    episode_count=episode_count,
                    route_name=done_route,
                )
            _sync_steervla_debug_noise_context(done_route)
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
            enable_updates
            and (not FLAGS.expert_debug)
            and agent is not None
            and not in_warmup
            and buffer.size >= batch_size
            and step % update_interval == 0
        ):
            for _ in range(updates_per_step):
                t_update_start = time.time()
                use_vla_update = getattr(agent, "vla_sample_fn", None) is not None
                batch = buffer.sample(batch_size)
                if _online_training_mode == "dagger":
                    agent, update_info = agent.update_dagger(batch)
                elif use_vla_update:
                    agent, update_info = agent.update_with_vla(batch)
                else:
                    agent, update_info = agent.update(batch)
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

        if agent is not None and enable_updates and step % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, step)

    train_logger.close()

    if FLAGS.save_buffer:
        buffer_path = FLAGS.buffer_path or os.path.join(FLAGS.save_dir, "buffer.npz")
        path = buffer.save(buffer_path)
        print(f"[buffer] saved {buffer.size} transitions -> {path}", flush=True)


# --------------------------------------------------------------------------- #
# Residual-SAC / EXPO online loop                                             #
# --------------------------------------------------------------------------- #


def run_online_residual(
    env, agent, config, obs, *, vla_sample_fn, steervla_actor, raw_holder, state_encoder, exec_cfg=None,
    start_step: int = 0, episode_count_start: int = 0, buffer: Optional[ReplayBuffer] = None,
) -> None:
    """Residual-SAC online loop. One env.step == one CARLA tick == one transition.

    SteerVLA proposes a base action chunk; the residual SAC agent (optionally EXPO best-of-N)
    edits it; the env executes one tick. Shares the module's SteerVLA/env/wandb/live-viewer
    scaffolding with :func:`run_online_carla`.

    ``start_step`` / ``episode_count_start`` / ``buffer`` are non-defaults only when resuming after
    a crash (see the run_carla.sh supervisor): the loop continues from ``start_step + 1`` with the
    restored replay buffer, and the warmup/ramp schedule (keyed on absolute step) stays consistent.
    """
    base_only = agent is None
    vla_action_dim = int(config["steervla"]["action_dim"])
    action_dim = int(config["steervla"]["action_horizon"]) * vla_action_dim
    warmup = int(config["residual_warmup_steps"])
    # Warm-start ramp: residual authority (scale) is 0 through warmup, then linearly rises to
    # ``residual_scale`` over ``residual_ramp_steps`` env steps, avoiding a magnitude jump at
    # handover and keeping the critic in-distribution as the executed policy drifts off base.
    ramp_steps = max(0, int(config.get("residual_ramp_steps", 0)))
    _accel_steer_space = str(config.get("residual_action_space", "waypoint_chunk")).strip().lower() == "accel_steer"
    _accel_scale = float(config["residual_scale"])
    _steer_scale = float(config.get("residual_steer_scale", -1.0))
    # Per-dim [accel, steer] authority in accel_steer mode (steer<0 -> reuse accel); scalar otherwise.
    target_scale = (
        np.array([_accel_scale, _steer_scale if _steer_scale >= 0.0 else _accel_scale], dtype=np.float32)
        if _accel_steer_space else np.float32(_accel_scale)
    )

    def _residual_scale(step: int):
        """Annealed residual authority at ``step`` (0 in warmup, linear ramp, then target); per-dim in accel_steer."""
        if step <= warmup:
            return target_scale * 0.0
        if ramp_steps <= 0 or step >= warmup + ramp_steps:
            return target_scale
        return target_scale * (float(step - warmup) / float(ramp_steps))

    batch_size = int(config["batch_size"])
    updates_per_step = int(config["updates_per_step"])
    capacity = int(config["buffer_capacity"])
    debug_task = bool(config.get("debug_task", False))
    enable_updates = bool(FLAGS.enable_updates) if FLAGS.enable_updates is not None else bool(config["enable_updates"])
    if base_only:
        print("[main_carla] base_only=True: rolling out the frozen base policy (no RL).", flush=True)
    elif not enable_updates:
        print("[main_carla] enable_updates=False: rollout-only (no RL gradient updates).", flush=True)

    log_video = bool(config.get("log_episode_video", True))
    video_fps = float(config.get("episode_video_fps", 10.0))
    video_every = max(1, int(config.get("episode_video_every", 2)))
    episode_frames: list[np.ndarray] = []

    live_viewer: LivePolicyViewer | None = None
    if FLAGS.live_policy_view:
        live_viewer = LivePolicyViewer(
            os.path.join(FLAGS.save_dir, "live_policy.mp4"),
            port=int(FLAGS.live_policy_port),
            fps=video_fps,
            publish_every_n_steps=int(FLAGS.live_policy_interval),
        )
        live_viewer.start()

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    last_residual: Optional[np.ndarray] = None

    rng = jax.random.PRNGKey(FLAGS.seed)
    if start_step:
        # Resume: fold the start step into the key so the post-crash sampling stream differs from
        # the pre-crash one (avoids replaying the exact same noise/CoT draws).
        rng = jax.random.fold_in(rng, start_step)

    # Base flow noise is sized to the MODEL action tensor (action_horizon x model_action_dim),
    # not the env chunk. Mirrors DSRLAgent._flat_noise_dim; falls back to the env dim only when
    # the model isn't loaded (remote actor, where noise is ignored server-side).
    _base_model = getattr(steervla_actor, "model", None)
    base_noise_dim = (
        int(_base_model.action_horizon) * int(_base_model.action_dim)
        if _base_model is not None
        else action_dim
    )

    def _draw_base_noise(key: jax.Array) -> jnp.ndarray:
        """Fresh Gaussian seed for the base flow ODE (``x ~ N(0, I)``)."""
        return jax.random.normal(key, (1, base_noise_dim), dtype=jnp.float32)

    # Residual action space: "waypoint_chunk" (residual acts on the raw normalized 40-D chunk,
    # env decodes corrected chunk -> waypoints -> PID) or "accel_steer" (base chunk is
    # PID-decoded to a bounded 2-D [accel, steer] BEFORE the residual, which acts on that
    # control). The frozen VLA still produces the 40-D chunk either way.
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

        accel_steer_decoder = SimlingoStyleWaypointDecoder(
            brake_speed=float(exec_cfg.get("brake_speed", 0.1)),
        )

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
    # (K=N). siglip_pool is image-only -> one shared state (K=1) broadcast across candidates.
    state_cot_dependent = bool(getattr(state_encoder, "cot_dependent", True)) if state_encoder is not None else False
    k_state = n_cand if state_cot_dependent else 1

    # Per-step state-encoder timing (summed over the K encodes done for the step; ``calls`` = K).
    # Cleared before each next-obs encode; logged as encode/<name>/<phase>. Phases are per-encoder
    # (rl_token: vlm/ae/total, pi_prefix: vlm/total, siglip_pool: vision/text/total).
    enc_time_acc: dict[str, float] = {}

    # Black-image sanity check: encoders see a zeroed image (the base policy still gets the real one),
    # to confirm the encoded state actually depends on the image. VLM encoders then need a fresh prefix
    # forward, so flag the actor to bypass its real-image cache (a second VLM pass, test-only).
    sanity_black_image = bool(config.get("sanity_black_image", False))
    if sanity_black_image and steervla_actor is not None:
        steervla_actor._recompute_prefix_from_obs = True

    def _encode_one(o: dict) -> np.ndarray:
        if sanity_black_image:
            o = {**o, "image": np.zeros_like(o["image"])}
        vec, breakdown = state_encoder.encode_timed(o)
        for k, v in breakdown.items():
            enc_time_acc[k] = enc_time_acc.get(k, 0.0) + float(v)
        enc_time_acc["calls"] = enc_time_acc.get("calls", 0.0) + 1.0
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    def _encode_state(o: dict, cands) -> Optional[np.ndarray]:
        """Encode the (K, embed) state(s) for one obs. cands=None (warmup) -> encode once, tile to
        K. cot-dependent -> one row per candidate, each with that candidate's CoT via _last_cot_out."""
        if state_encoder is None:
            return None
        if not state_cot_dependent or cands is None:
            steervla_actor._prefix_cache_row = 0  # batch-1 cached prefix
            xi = _encode_one(o)
            return np.tile(xi[None, :], (k_state, 1))
        rows = []
        for i in range(n_cand):
            # Line up candidate i's CoT and cached prefix row so the encoded state matches the base.
            steervla_actor._last_cot_out = {kk: vv[i : i + 1] for kk, vv in cands["cot_out"].items()}
            steervla_actor._prefix_cache_row = i
            rows.append(_encode_one(o))
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

    # Initial base rep. cands feeds subtask-diversity logging. On resume the first step is
    # start_step+1, so gate OTF on the absolute step (not a hardcoded 1).
    first_step = start_step + 1
    rng, nk = jax.random.split(rng)
    base_cands, x_cands, base_chunks, cands = _compute_base(obs, use_otf and first_step > warmup, nk)

    episode_return = 0.0
    debug_return = 0.0  # sum of the debug objective (-ego_speed) over the episode; only used when debug_task.
    episode_steps = 0
    episode_count = episode_count_start
    episode_collision_count = 0
    episode_collision_events = 0
    prev_collision_count = 0
    start_time = time.time()

    def _flush_checkpoint(step_tag: int) -> None:
        """Persist the latest agent (+ optional buffer) on normal exit / exception / Ctrl-C."""
        if agent is None:
            return
        save_agent(agent, FLAGS.save_dir, step_tag)
        if FLAGS.save_buffer and buffer is not None:
            path = _buffer_persist_path(FLAGS.save_dir)
            _atomic_buffer_save(buffer, path)
            print(f"[main_carla] Saved replay buffer ({buffer.size} transitions) -> {path}", flush=True)

    last_step = start_step
    try:
        for step in tqdm.tqdm(range(first_step, FLAGS.online_steps + 1), dynamic_ncols=True):
            last_step = step
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

            t_step_start = time.time()
            next_obs, reward, terminated, truncated, info = env.step(final)
            t_step_end = time.time()
            done = bool(terminated or truncated)

            collision_count = int(info.get("collision_count", 0))
            collision_delta = max(0, collision_count - prev_collision_count)
            episode_collision_count = max(episode_collision_count, collision_count)
            episode_collision_events += collision_delta
            prev_collision_count = collision_count

            # Next base cands are diverse iff the next step runs OTF (step >= warmup).
            rng, nk = jax.random.split(rng)
            enc_time_acc.clear()  # collect this step's per-encoder timing breakdown
            t_base_start = time.time()
            next_base_cands, next_x_cands, next_base_chunks, next_cands = _compute_base(
                next_obs, use_otf and step >= warmup, nk
            )
            t_base_end = time.time()

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
            # Debug objective: reward = -ego_speed (matches the relabel the agent trains on).
            debug_step_reward = -float(_ego_speed_mps(obs)) if debug_task else 0.0
            debug_return += debug_step_reward
            episode_steps += 1
            # In accel_steer mode ``final`` is a 2-D control (no waypoint projection), so draw the
            # (smooth) base chunk plan; in chunk mode draw the executed (perturbed) chunk.
            viz_action = base_chunk if accel_steer else final
            # accel_steer HUD: show base/residual/final controls + progress/step/return on the frame
            # (the waypoint overlay can't convey the residual once it acts post-PID).
            hud = None
            if accel_steer:
                def _fmt2(v) -> str:
                    a = np.asarray(v, dtype=np.float32).reshape(-1)
                    return f"({a[0]:+.2f},{a[1]:+.2f})" if a.size >= 2 else f"({a[0]:+.2f})"

                show_resid = residual_active and last_residual is not None
                hud = {
                    "base": _fmt2(winner_base),
                    "residual": _fmt2(last_residual) if show_resid else "(--)",
                    "final": _fmt2(final),
                    "scale": _fmt2(scale_now),
                    "progress": float(info.get("route_progress_pct", 0.0)),
                    "ep_step": episode_steps,
                    "ep_return": episode_return,
                }
            _maybe_capture_frame(
                episode_frames, next_obs, debug_step_reward if debug_task else reward,
                episode_steps=episode_steps, done=done,
                log_video=log_video, video_every=video_every,
                action_flat=viz_action, exec_cfg=exec_cfg,
                collision=(collision_count, episode_collision_events) if collision_delta > 0 else None,
                steervla_actor=steervla_actor, hud=hud,
            )
            if live_viewer is not None and episode_frames:
                live_viewer.publish_frames(episode_frames, step)

            train_info: dict[str, Any] = {}
            # Hold updates until warmup ends: at scale=0 the residual can't affect the executed
            # action so its gradient is degenerate; the ramp then starts near 0.
            t_update_start = time.time()
            if (
                agent is not None and enable_updates and step > warmup
                and buffer is not None and buffer.size >= batch_size
            ):
                for _ in range(updates_per_step):
                    agent, train_info = agent.update(buffer.sample(batch_size), scale_j)
            t_update_end = time.time()

            if step % FLAGS.log_interval == 0:
                log = {
                    "env/episode_count": episode_count,
                    "env/sps": step / max(time.time() - start_time, 1e-6),
                }

                log["rollout/collision_events"] = float(collision_delta)
                if debug_task:
                    log["debug/step_reward"] = debug_step_reward
                    log.update(_drive_diagnostics_log(info))
                else:
                    log["env/reward"] = float(reward)
                    log.update(_reward_breakdown_log(info))  # reward components + drive diagnostics
                if agent is not None:
                    log["env/buffer_size"] = int(buffer.size) if buffer is not None else 0
                    log["env/residual_active"] = int(residual_active)
                    _sn = np.asarray(scale_now, dtype=np.float32).reshape(-1)
                    if _sn.size >= 2:
                        log["env/residual_scale_accel"] = float(_sn[0])
                        log["env/residual_scale_steer"] = float(_sn[1])
                    else:
                        log["env/residual_scale"] = float(_sn[0])
                log["training/in_warmup"] = float(step <= warmup)
                log["training/in_ramp"] = float(ramp_steps > 0 and warmup < step < warmup + ramp_steps)
                log["training/enable_updates"] = float(enable_updates)
                log["time/step_time"] = t_step_end - t_step_start
                log["time/update_time"] = t_update_end - t_update_start
                # base = SteerVLA chunk sample + state encode (the encoder VLM pass). Compare against
                # step_time (CARLA tick): whichever dominates is the real bottleneck.
                log["time/base_encode_time"] = t_base_end - t_base_start
                # State-encoder timing broken down by phase (summed over the step's K encodes), so
                # encoder speeds (rl_token vs pi_prefix vs siglip_pool) are directly comparable.
                if state_encoder is not None and enc_time_acc:
                    for k, v in enc_time_acc.items():
                        log[f"encode/{state_encoder.name}/{'calls' if k == 'calls' else k}"] = float(v)
                if train_info:
                    log.update({k: float(jax.device_get(v)) for k, v in train_info.items()})
                # Executed control + per-component chunk stats as scalar line charts.
                log.update(_ego_control_log(next_obs))
                if accel_steer:
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

            # Permanent milestone checkpoint (sparse; for eval / analysis).
            milestone = FLAGS.save_interval > 0 and step % FLAGS.save_interval == 0
            # Crash-recovery checkpoint (frequent): commit agent + buffer + resume_state together so
            # --resume continues near this step. A SIGSEGV skips the finally block, so this periodic
            # write is the only thing that survives; resume_state is written LAST as the commit
            # marker (a crash mid-save leaves the previous consistent checkpoint intact).
            recover = agent is not None and FLAGS.resume_interval > 0 and step % FLAGS.resume_interval == 0
            if agent is not None and (milestone or recover):
                save_agent(agent, FLAGS.save_dir, step)
            if recover:
                if FLAGS.save_buffer and buffer is not None:
                    _atomic_buffer_save(buffer, _buffer_persist_path(FLAGS.save_dir))
                _write_resume_state(
                    FLAGS.save_dir, step=step, episode_count=episode_count, agent_epoch=step,
                )

            if done:
                if live_viewer is not None and episode_frames:
                    live_viewer.publish_frames(episode_frames, step, force=True)
                _log_episode_end(
                    info, episode_return=episode_return, episode_steps=episode_steps,
                    episode_index=episode_count + 1, frames=episode_frames, log_video=log_video,
                    video_fps=video_fps, step=step, train_logger=train_logger,
                    collision_events=episode_collision_events,
                    debug_task=debug_task, debug_return=debug_return,
                )
                episode_count += 1
                episode_return = 0.0
                debug_return = 0.0
                episode_steps = 0
                episode_collision_count = 0
                episode_collision_events = 0
                prev_collision_count = 0
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
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #

_RESUME_STATE_NAME = "resume_state.json"


def _resume_state_path(save_dir: str) -> str:
    return os.path.join(save_dir, _RESUME_STATE_NAME)


def _buffer_persist_path(save_dir: str) -> str:
    """Where the replay buffer is checkpointed for crash-resume (same as the end-of-run dump).

    Always ``.npz``-terminated so the periodic save and the resume-time load reference the exact
    same file (ReplayBuffer.save force-appends ``.npz``, so an un-suffixed path would mismatch).
    """
    path = FLAGS.buffer_path or os.path.join(save_dir, "buffer.npz")
    return path if path.endswith(".npz") else path + ".npz"


def _atomic_buffer_save(buffer, path: str) -> None:
    """Save the buffer to a temp file then os.replace into place, so a crash mid-write never
    leaves a truncated .npz that would fail to load on resume."""
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    written = buffer.save(path + ".tmp")  # ReplayBuffer.save appends .npz -> <path>.tmp.npz
    os.replace(written, path)


def _write_resume_state(save_dir: str, *, step: int, episode_count: int, agent_epoch: int) -> None:
    """Atomically persist the minimal state a relaunch needs to continue (step/episode/ckpt)."""
    tmp = _resume_state_path(save_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"step": int(step), "episode_count": int(episode_count), "agent_epoch": int(agent_epoch)}, f)
    os.replace(tmp, _resume_state_path(save_dir))


def _read_resume_state(save_dir: str) -> Optional[dict]:
    path = _resume_state_path(save_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[main_carla] ignoring unreadable resume state {path}: {e}", flush=True)
        return None


def _run_residual_entry(config):
    """Residual-SAC / EXPO online CARLA RL entry (frozen SteerVLA base + state encoder)."""
    if FLAGS.route is None:
        raise ValueError("--route is required (see --list_routes=true).")

    base_only = bool(config.get("base_only", False))

    wandb_mode = _resolve_wandb_mode()
    # Descriptive run name: <route>-<mode>-sd###_<ts>, where <mode> is the state encoder
    # (rl runs) or "base" (no-RL baseline). --exp_name pins this across restarts so save_dir and
    # the W&B run id stay stable (the crash supervisor relies on this to resume, not fork).
    route_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", str(FLAGS.route)).strip("-")
    mode_tag = "base" if base_only else str(config.get("state_encoder", "pi_prefix"))
    exp_name = FLAGS.exp_name or f"{route_tag}-{mode_tag}-{get_exp_name(FLAGS.seed)}"
    # W&B run id must be a stable, id-safe token; resume='allow' continues the same run on relaunch.
    wandb_id = re.sub(r"[^A-Za-z0-9_-]+", "-", exp_name).strip("-")[:128] or None
    setup_wandb(
        project="OGBench-CARLA-Residual", group=FLAGS.run_group, name=exp_name, mode=wandb_mode,
        id=wandb_id, resume=("allow" if FLAGS.resume else None),
    )
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(get_flag_dict(), f)

    # Resume bookkeeping: if --resume and a prior resume_state exists, we continue from the last
    # checkpointed step (restore agent + buffer below). resume_state is None -> fresh run.
    resume_state = _read_resume_state(FLAGS.save_dir) if FLAGS.resume else None
    if resume_state is not None:
        print(
            f"[main_carla] resuming from step {resume_state['step']} "
            f"(episode {resume_state['episode_count']}, ckpt epoch {resume_state['agent_epoch']}).",
            flush=True,
        )

    carla_yaml = FLAGS.carla_config or str(_IMPLS_ROOT / "configs" / "carla_config.yaml")

    steervla_cfg = config.get("steervla", None)
    if steervla_cfg is None or not steervla_cfg.get("enabled"):
        raise ValueError("config.steervla.enabled must be true: residual RL needs a base policy.")

    extra_carla: dict[str, Any] = {}
    exec_cfg = _steervla_action_execution_cfg(steervla_cfg, residual=True)
    if exec_cfg is not None:
        extra_carla["steervla_action_execution"] = exec_cfg

    # Bring CARLA up before JAX initializes its thread pool (forking afterwards can deadlock
    # the UE4 RenderThread). The reset below starts the simulator.
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

        # base_only -> no residual agent / encoder; run_online_residual executes the frozen base
        # chunk every step. Otherwise build the encoder + residual SAC agent.
        agent = None
        state_encoder = None
        if not base_only:
            from encoders import build_state_encoder

            state_encoder = build_state_encoder(config, steervla_actor)
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
            # Prefix (VLM) encoders read the cache the base sampler populates, so warm it with one base
            # sample before probing the encoded-state width; reset after so the loop starts clean.
            if getattr(state_encoder, "cot_dependent", False) and getattr(steervla_actor, "model", None) is not None:
                raw_holder["obs"] = obs
                _bm = steervla_actor.model
                vla_sample_fn(
                    jnp.zeros((1, 1), dtype=jnp.float32),
                    jax.random.normal(
                        jax.random.PRNGKey(FLAGS.seed),
                        (1, int(_bm.action_horizon) * int(_bm.action_dim)),
                        dtype=jnp.float32,
                    ),
                )
            # Probe the encoded-state width once so the agent's MLPs are sized correctly.
            x_dim = int(state_encoder.encode(obs).shape[-1])
            if getattr(steervla_actor, "reset_action_cache", None) is not None:
                steervla_actor.reset_action_cache()
            ex_obs = np.zeros((1, x_dim), dtype=np.float32)
            ex_base = np.zeros((1, action_dim), dtype=np.float32)
            print(
                f"[main_carla] state_encoder={state_encoder.name}; x_dim={x_dim}; "
                f"action_dim={action_dim}; residual_action_space={residual_space}",
                flush=True,
            )

            from jax_agents.sac_residual import SACResidualAgent

            agent = SACResidualAgent.create(FLAGS.seed, ex_obs, ex_base, config)
            if FLAGS.restore_path is not None:
                agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

        if FLAGS.eval_only:
            FLAGS.online_steps = max(FLAGS.online_steps, 200)

        # Restore agent + replay buffer from the last checkpoint when resuming after a crash.
        start_step = 0
        episode_count_start = 0
        resumed_buffer: Optional[ReplayBuffer] = None
        if resume_state is not None:
            start_step = int(resume_state["step"])
            episode_count_start = int(resume_state["episode_count"])
            if agent is not None and int(resume_state["agent_epoch"]) >= 0:
                agent = restore_agent(agent, FLAGS.save_dir, int(resume_state["agent_epoch"]))
                bpath = _buffer_persist_path(FLAGS.save_dir)
                if os.path.exists(bpath):
                    resumed_buffer = ReplayBuffer.load(bpath, max_size=int(config["buffer_capacity"]))
                    print(f"[main_carla] restored replay buffer ({resumed_buffer.size}) from {bpath}", flush=True)

        run_online_residual(
            env, agent, config, obs,
            vla_sample_fn=vla_sample_fn,
            steervla_actor=steervla_actor,
            raw_holder=raw_holder,
            state_encoder=state_encoder,
            exec_cfg=exec_cfg,
            start_step=start_step,
            episode_count_start=episode_count_start,
            buffer=resumed_buffer,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass
        wandb.finish()


def main(_):
    if FLAGS.list_routes:
        _list_routes_and_exit()
        return

    config = FLAGS.agent
    if str(config.get("agent_name", "")) == "sac_residual":
        _run_residual_entry(config)
    else:
        _run_dsrl_entry(config)


def _run_dsrl_entry(config):
    """DSRL / generic online CARLA RL entry (SteerVLA noise-actor, coaches, DAgger, VLM, ...)."""
    wandb_mode = _resolve_wandb_mode()

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
    critic_feedback_mode = resolve_critic_feedback_mode(config)
    if critic_feedback_mode == "none":
        config.language_label_dim = 0
    elif critic_feedback_mode in ("delta_commentary_bow", "vlm_chunk_bow"):
        config.language_label_dim = NUM_DELTA_COMMENTARY_WORDS
    elif critic_feedback_mode == "commentary_bow":
        config.language_label_dim = NUM_COMMENTARY_WORDS
    elif critic_feedback_mode == "action_delta":
        config.language_label_dim = int(config.get("critic_action_dim", 4))
    elif critic_feedback_mode == "subtask_siglip":
        # Subtask text embedded with SigLIP; agent.create syncs siglip_embed_dim to the encoder.
        config.language_label_dim = int(config.get("siglip_embed_dim", 1152))
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
        image_encoder = str(config.get("image_encoder", "impala")).lower()
        obs_dict, _info = env.reset(seed=FLAGS.seed)
        if not isinstance(obs_dict, dict) or "state" not in obs_dict or "image" not in obs_dict:
            raise ValueError(
                "CARLA env must return a Dict observation with 'state' and 'image'; "
                f"got {type(obs_dict).__name__}."
            )

        siglip_encoder = None
        siglip_include_prompt_subtask = bool(config.get("siglip_include_prompt_subtask", False))
        if obs_mode == "image" and image_encoder == "siglip":
            from utils.siglip_encoder import SigLIPEncoder

            siglip_encoder = SigLIPEncoder(
                model_id=str(config.get("siglip_model_id", "google/siglip2-so400m-patch14-384")),
                device=config.get("siglip_device"),
            )
            siglip_encoder.setup()
            config.siglip_embed_dim = int(siglip_encoder.embedding_dim)
            obs_dim = siglip_encoder.observation_dim(
                include_prompt_subtask=siglip_include_prompt_subtask
            )
            print(
                f"[main_carla] SigLIP encoder {siglip_encoder.model_id} "
                f"embed_dim={config.siglip_embed_dim} obs_dim={obs_dim}",
                flush=True,
            )

        raw_carla_holder: dict | None = None
        if use_steervla_rollout or FLAGS.expert_debug or FLAGS.expert_recover_debug:
            raw_carla_holder = {"obs": obs_dict, "next_obs": obs_dict}

        steervla_actor = None
        vla_sample_fn = None
        agent = None
        if not FLAGS.expert_debug:
            tr_rank = int(config.get("training_gpu_rank", -1))
            if use_steervla_rollout:
                vla_bundle = _build_vla_sample_fn(
                    steervla_cfg,
                    raw_carla_holder,
                    training_gpu_rank=tr_rank,
                    noise_scale=float(config.get("noise_scale", 1.0)),
                )
                if vla_bundle is None:
                    raise ValueError("SteerVLA rollout enabled but vla_sample_fn could not be built.")
                vla_sample_fn, steervla_actor = vla_bundle
                steervla_actor.debug_noise = bool(config.get("debug_noise", False))
                steervla_actor.debug_noise_samples = int(config.get("debug_noise_samples", 15))
                steervla_actor.use_best_noise = bool(config.get("use_best_noise", True))
                steervla_actor.debug_noise_log_every_n_steps = int(
                    config.get("debug_noise_log_every_n_steps", 5)
                )
                steervla_actor.noise_scale = float(config.get("noise_scale", 1.0))
                if steervla_actor.debug_noise:
                    print(
                        f"[main_carla] debug_noise enabled: "
                        f"{steervla_actor.debug_noise_samples} random noises per fresh VLA query, "
                        f"log plots every {steervla_actor.debug_noise_log_every_n_steps} env steps, "
                        f"use_best_noise={steervla_actor.use_best_noise}",
                        flush=True,
                    )

            agent_obs = _extract_agent_obs(
                env,
                obs_dict,
                obs_mode,
                image_encoder=image_encoder,
                siglip_encoder=siglip_encoder,
                siglip_include_prompt_subtask=siglip_include_prompt_subtask,
                steervla_actor=steervla_actor,
            )
            ex_obs = np.expand_dims(agent_obs, 0)
            ex_actions = np.zeros((1,) + tuple(env.action_space.shape), dtype=np.float32)

            _configure_jax_training_device(tr_rank)

            agent_class = agents[config["agent_name"]]
            create_kwargs = {}
            if config["agent_name"] in ("dsrl", "best_of_n") and vla_sample_fn is not None:
                create_kwargs["vla_sample_fn"] = vla_sample_fn
                url = steervla_cfg.get("actor_url") if steervla_cfg else None
                if not (url and str(url).strip()):
                    create_kwargs["openpi_train_config"] = steervla_actor.train_cfg
                    create_kwargs["steervla_actor"] = steervla_actor

            if config["agent_name"] == "best_of_n":
                # Subtask -> critic language label uses SigLIP text features (shared encoder if present).
                create_kwargs["siglip_encoder"] = siglip_encoder

            if config["agent_name"] == "expo" and vla_sample_fn is not None:
                create_kwargs["vla_sample_fn"] = vla_sample_fn
                url = steervla_cfg.get("actor_url") if steervla_cfg else None
                if not (url and str(url).strip()):
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
            image_encoder=image_encoder,
            siglip_encoder=siglip_encoder,
            siglip_include_prompt_subtask=siglip_include_prompt_subtask,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
