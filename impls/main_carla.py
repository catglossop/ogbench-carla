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

import base64
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import jax

from jax_agents import agents
from jax_agents.sac_residual import SACResidualAgent
from utils.flax_utils import restore_agent
from coaches.expert_label import NUM_COMMENTARY_WORDS, NUM_DELTA_COMMENTARY_WORDS
from coaches.critic_feedback import (
    compute_action_delta,
    compute_action_delta_commentary,
    compute_expert_target,
    critic_language_dim,
    resolve_critic_feedback_mode,
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

# flags.DEFINE_string("save_dir", "/raid/users/celine/carla_exps", "Save directory.")
flags.DEFINE_string("save_dir", "/home/celinet/carla_exps", "Save directory.")
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


_EGO_STATE_IDX_SPEED = 15


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
    returns ``[image_embed, prompt_embed, subtask_embed]`` for actor and critic.
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
        # SteerVLA actor applies OpenPI Unnormalize + fixed denormalize_actions before returning actions.
        "action_input_space": "policy_output",
    }


class CarlaEnvSubprocess:
    """Runs carla_env_server.py in a subprocess (Python 3.10 + carla 0.9.15).

    Communicates over JSON stdin/stdout so main_carla.py (Python 3.11 + JAX)
    never loads the carla 0.9.15 shared library directly.  Set the env var
    ``CARLA_ENV_SUBPROCESS_PYTHON`` to the Python executable to use, e.g.
    ``/home/celinet/ogbench-carla/.venv-carla-0915/bin/python``.
    """

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _SERVER_SCRIPT = str(_REPO_ROOT / "impls" / "carla_env_server.py")

    def __init__(
        self,
        carla_config_path: Optional[str],
        route: str,
        python_exe: str,
        extra_carla_config: Optional[dict] = None,
    ):
        self._python_exe = python_exe
        self._carla_config_path = carla_config_path
        self._route = route
        self._extra_carla_config = extra_carla_config or {}
        self._proc = None
        self.action_space: Any = None

    def setup(self):
        rebuttal = str(self._REPO_ROOT / "simlingo-rebuttal")
        carla_root = os.environ.get("CARLA_ROOT", "/home/celinet/VLA_driving/software")
        # Bench2Drive leaderboard must come BEFORE simlingo-rebuttal/leaderboard.
        # PYTHONPATH is prepended to sys.path; carla_env_server.py uses sys.path.insert(0)
        # but skips paths already in sys.path, so PYTHONPATH order is authoritative.
        pythonpath_parts = [
            f"{rebuttal}/Bench2Drive/leaderboard/leaderboard",
            f"{rebuttal}/Bench2Drive/leaderboard",
            f"{rebuttal}/Bench2Drive/scenario_runner",
            str(self._REPO_ROOT),
            str(self._REPO_ROOT / "impls"),
            rebuttal,
            f"{rebuttal}/leaderboard/leaderboard",
            f"{rebuttal}/leaderboard",
            f"{rebuttal}/scenario_runner",
            f"{carla_root}/PythonAPI/carla",
        ]
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ":".join(pythonpath_parts) + (":" + existing if existing else "")
        env["CARLA_ROOT"] = carla_root
        env.setdefault("WORK_DIR", rebuttal)
        env["SCENARIO_RUNNER_ROOT"] = f"{rebuttal}/Bench2Drive/scenario_runner"

        cmd = [self._python_exe, self._SERVER_SCRIPT, f"--route={self._route}"]
        # Keep the agent from carla_config.yaml (observation_only) — it registers the
        # rgb_front camera that SteerVLA policy images are decoded from.  The server's
        # default ("simlingo") registers only rgb_simlingo, leaving obs["image"] all zeros.
        cmd.append("--leaderboard_agent=config")
        if self._carla_config_path:
            cmd.append(f"--carla_config={self._carla_config_path}")
        if self._extra_carla_config:
            cmd.append(f"--extra_config_json={json.dumps(self._extra_carla_config)}")

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        startup_line = self._proc.stdout.readline()
        startup = json.loads(startup_line)
        assert startup.get("ready"), f"Unexpected startup: {startup}"
        shape = tuple(startup["action_space_shape"])
        lo = float(startup.get("action_space_low", -1.0))
        hi = float(startup.get("action_space_high", 1.0))
        import gymnasium as _gym
        self.action_space = _gym.spaces.Box(
            low=lo, high=hi, shape=shape, dtype=np.float32
        )

    @staticmethod
    def _decode_obs(wire: dict) -> dict:
        def _img(key: str) -> np.ndarray | None:
            b64 = wire.get(f"{key}_b64")
            if b64 is None:
                return None
            return np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(wire[f"{key}_shape"])

        image = _img("image")
        viz = _img("viz_image")
        simlingo = _img("simlingo_image")
        if viz is None:
            viz = simlingo if simlingo is not None else image
        ea = wire.get("expert_action")
        return {
            "state": np.array(wire["state"], dtype=np.float32),
            "image": image,
            "image_viz": viz,  # native rgb_front (or simlingo camera) for rollout video logging
            "simlingo_image": simlingo,
            "routing_command": wire["routing_command"],
            "target_points": np.array(wire["target_points"], dtype=np.float32),
            "expert_action": np.array(ea, dtype=np.float32) if ea is not None else None,
        }

    def _read_obs_msg(self):
        line = self._proc.stdout.readline()
        return json.loads(line)

    def reset(self, seed=None):
        if self._proc is None:
            raise RuntimeError("Call setup() before reset().")
        self._proc.stdin.write(json.dumps({"reset": True}) + "\n")
        self._proc.stdin.flush()
        msg = self._read_obs_msg()
        return self._decode_obs(msg["obs"]), msg.get("info", {})

    def step(self, action):
        self._proc.stdin.write(json.dumps({"action": action.tolist()}) + "\n")
        self._proc.stdin.flush()
        msg = self._read_obs_msg()
        return self._decode_obs(msg["obs"]), msg["reward"], msg["terminated"], msg["truncated"], msg["info"]

    def step_expert(self, obs_raw=None):
        self._proc.stdin.write(json.dumps({"expert_step": True}) + "\n")
        self._proc.stdin.flush()
        msg = self._read_obs_msg()
        return self._decode_obs(msg["obs"]), msg["reward"], msg["terminated"], msg["truncated"], msg["info"]

    def reinit_expert(self):
        if self._proc is None:
            return
        self._proc.stdin.write(json.dumps({"reinit_expert": True}) + "\n")
        self._proc.stdin.flush()
        self._proc.stdout.readline()  # consume ack

    def close(self):
        if self._proc is not None:
            try:
                self._proc.stdin.write(json.dumps({"shutdown": True}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None


def _make_carla_env(
    carla_config_path: Optional[str],
    route: Optional[str],
    *,
    extra_carla_config: Optional[dict[str, Any]] = None,
):
    subprocess_python = os.environ.get("CARLA_ENV_SUBPROCESS_PYTHON", "").strip()
    if subprocess_python:
        if not Path(subprocess_python).exists():
            raise FileNotFoundError(
                f"CARLA_ENV_SUBPROCESS_PYTHON={subprocess_python!r} not found."
            )
        if route is None:
            raise ValueError("--route is required when using CARLA_ENV_SUBPROCESS_PYTHON.")
        print(
            f"[main_carla] Using carla_env_server subprocess: {subprocess_python}",
            flush=True,
        )
        env = CarlaEnvSubprocess(
            carla_config_path, route, subprocess_python,
            extra_carla_config=extra_carla_config,
        )
        env.setup()
        return env

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
    updates_per_step = int(agent_config.get("updates_per_step", 1))
    batch_size = int(agent_config.get("batch_size", 256))
    _online_training_mode = str(agent_config.get("online_training_mode", "rl")).strip().lower()
    _residual_warmup = int(agent_config.get("residual_warmup_steps", 0))
    _residual_append_state = bool(agent_config.get("residual_append_state", False))
    _residual_obs_dim = int(agent_config.get("residual_obs_dim", 25))
    # Welford online stats for residual obs normalization (updated during residual warmup).
    _res_norm_count = 0
    _res_norm_mean = np.zeros(_residual_obs_dim, dtype=np.float64)
    _res_norm_M2 = np.zeros(_residual_obs_dim, dtype=np.float64)

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
    log_images = True

    _critic_feedback_mode = resolve_critic_feedback_mode(agent_config)
    _lang_dim = critic_language_dim(agent_config)

    steervla_cfg = agent_config.get("steervla") or {}
    _steervla_exec_cfg = _steervla_action_execution_cfg(steervla_cfg)
    env_ah = int(steervla_cfg.get("action_horizon", agent_config.get("vla_action_horizon", 10)))
    env_ad = int(steervla_cfg.get("action_dim", agent_config.get("vla_action_dim", 4)))
    action_dim = env_ah * env_ad

    # accel_steer residual mode: PID-decode the Pi0 waypoint chunk to a 2-D
    # [accel, steer] in [-1, 1] BEFORE the residual (torch residual_sac parity).
    # The replay buffer / critic see 2-D actions; the env executes them via the
    # legacy _action_to_control path.
    _residual_2d = (
        _online_training_mode in {"sac_residual", "dagger_residual"}
        and str(agent_config.get("residual_action_space", "waypoint_chunk")).strip().lower() == "accel_steer"
    )
    if _residual_2d:
        if _steervla_exec_cfg is None:
            raise ValueError(
                "residual_action_space='accel_steer' requires the SteerVLA action "
                "execution config (waypoint chunk layout) to PID-decode base actions."
            )
        action_dim = 2

    def _make_accel_steer_decoder():
        from ogbench.carla.steervla_simlingo_control import SimlingoStyleWaypointDecoder

        return SimlingoStyleWaypointDecoder()

    _accel_steer_decoder = _make_accel_steer_decoder() if _residual_2d else None
    _expert_accel_steer_decoder = (
        _make_accel_steer_decoder()
        if (_residual_2d and _online_training_mode == "dagger_residual")
        else None
    )
    # Dedicated decoder for the critic's privileged expert label in accel_steer
    # mode: the expert waypoint chunk is PID-decoded to [accel, steer] controls so
    # the label lives in the same 2-D space as the critic's action inputs. A
    # separate instance keeps its PID state tracking the episode at exactly one
    # decode per step, independent of the DAgger replay decoder.
    _critic_expert_decoder = (
        _make_accel_steer_decoder()
        if (_residual_2d and _critic_feedback_mode in ("expert_action", "action_delta"))
        else None
    )

    def _critic_expert_first(raw: dict) -> np.ndarray | None:
        """PID-decoded 2-D expert controls for the critic label (None outside accel_steer mode)."""
        if _critic_expert_decoder is None:
            return None
        expert_raw = raw.get("expert_action")
        if expert_raw is None or raw.get("state") is None:
            return None
        try:
            return _decode_chunk_to_accel_steer(_critic_expert_decoder, expert_raw, raw["state"])
        except Exception as e:
            print(f"[critic_expert_first] decode failed: {e}", flush=True)
            return None

    def _decode_chunk_to_accel_steer(decoder, chunk_flat: np.ndarray, state_vec) -> np.ndarray:
        return decoder.flat_action_to_accel_steer(
            np.asarray(chunk_flat, dtype=np.float32).reshape(-1),
            state_vec=np.asarray(state_vec, dtype=np.float32),
            output_action_format=str(_steervla_exec_cfg["output_action_format"]),
            action_horizon=int(_steervla_exec_cfg["action_horizon"]),
            action_dim=int(_steervla_exec_cfg["action_dim"]),
            action_input_space=str(_steervla_exec_cfg.get("action_input_space", "policy_output")),
        )

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
        example_transition["base_actions"] = np.zeros((action_dim,), dtype=np.float32)
    if _online_training_mode == "sac_residual":
        example_transition["base_next_actions"] = np.zeros((action_dim,), dtype=np.float32)
    if _residual_append_state and _online_training_mode in {"sac_residual", "dagger_residual"}:
        example_transition["residual_obs"] = np.zeros((_residual_obs_dim,), dtype=np.float32)
        example_transition["next_residual_obs"] = np.zeros((_residual_obs_dim,), dtype=np.float32)
    if steervla_actor is not None:
        openpi0 = _openpi_fields_from_raw(obs_raw)
        example_transition.update(openpi0)
        example_transition.update({f"next_{k}": np.array(v) for k, v in openpi0.items()})
    _uses_pi_prefix: bool = (
        steervla_actor is not None
        and (
            (
                bool(agent_config.get("residual_use_pi_image_features", False))
                and str(agent_config.get("residual_pi_feature_source", "prefix")).strip().lower() == "prefix"
            )
            or bool(agent_config.get("critic_use_pi_prefix_features", False))
        )
    )
    if _uses_pi_prefix:
        _ex_openpi_obs = steervla_actor.build_observation_batch_numpy(batch_size=1, raw=obs_raw)
        _pi_prefix_dim = int(steervla_actor.encode_prefix_features(_ex_openpi_obs).shape[-1])
        example_transition["pi_prefix_obs_e"] = np.zeros((_pi_prefix_dim,), dtype=np.float32)
        example_transition["pi_prefix_next_obs_e"] = np.zeros((_pi_prefix_dim,), dtype=np.float32)

    # Create replay buffer
    buffer = ReplayBuffer.create(example_transition, size=capacity)

    def _compute_pi_prefix_e(raw_obs: dict) -> np.ndarray:
        openpi_obs = steervla_actor.build_observation_batch_numpy(batch_size=1, raw=raw_obs)
        return np.asarray(steervla_actor.encode_prefix_features(openpi_obs)[0], dtype=np.float32)

    _pi_prefix_e: np.ndarray | None = _compute_pi_prefix_e(obs_raw) if _uses_pi_prefix else None
    
    rng = jax.random.PRNGKey(FLAGS.seed + 1)
    episode_return, episode_steps, episode_count = 0.0, 0, 0
    episode_collision_count = 0
    episode_collision_events = 0
    prev_collision_count = 0
    episode_traffic_violations = 0
    prev_traffic_violation_count = 0
    last_log_time = time.time()
    episode_video_every = 2
    episode_video_frames: list[np.ndarray] = []
    last_video_reward: float = 0.0
    last_video_critic_text: str = ""
    last_policy_action: np.ndarray | None = None
    _last_base_action_np: np.ndarray | None = None
    _last_vla_chunk_holder: list = [None]  # raw Pi0 waypoint chunk (before 2-D decode)

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
            if raw.get("simlingo_image") is not None:
                return np.asarray(raw["simlingo_image"], dtype=np.uint8)
            if raw.get("image") is not None:
                return np.asarray(raw["image"], dtype=np.uint8)
        return np.asarray(raw, dtype=np.uint8)

    def _annotate_waypoints(
        frame: np.ndarray,
        action_flat: np.ndarray | None,
        base_action_flat: np.ndarray | None = None,
        *,
        vla_chunk: np.ndarray | None = None,
        target_points: np.ndarray | None = None,
    ) -> np.ndarray:
        if _steervla_exec_cfg is None:
            return frame
        # vla_chunk: raw Pi0 waypoint chunk before 2-D accel/steer decoding.
        # In accel_steer residual mode base_action_flat is 2-D and can't be projected.
        proj_action = vla_chunk if vla_chunk is not None else (
            base_action_flat if base_action_flat is not None else action_flat
        )
        if proj_action is None:
            return frame
        try:
            from ogbench.carla.waypoint_viz import annotate_waypoints_on_frame

            return annotate_waypoints_on_frame(
                frame,
                action_flat=proj_action,
                exec_cfg=_steervla_exec_cfg,
                target_points=target_points,
            )
        except Exception:
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

    def _annotate_traffic_violation_frame(
        frame: np.ndarray,
        *,
        violation_count: int,
        episode_violations: int,
    ) -> np.ndarray:
        annotated = np.array(frame, copy=True)
        try:
            import cv2  # type: ignore

            _, w = annotated.shape[:2]
            label = f"TRAF VIOL v={violation_count} e={episode_violations}"
            font_scale = 0.38
            thickness = 1
            pad = 4
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x1 = w - 6
            x0 = max(6, x1 - tw - 2 * pad)
            y0 = 22  # below collision banner
            y1 = y0 + th + baseline + 2 * pad
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (200, 140, 0), thickness=-1)
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
        if critic_mode in ("expert_action", "action_delta"):
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
        base_action: np.ndarray | None = None,
        composed_action: np.ndarray | None = None,
    ) -> np.ndarray:
        base = np.array(frame, copy=True)
        try:
            import cv2  # type: ignore

            h, w = base.shape[:2]
            font_scale = 0.26
            line_h = 13
            n_extra = 1 if (base_action is not None or composed_action is not None) else 0
            panel_h = max(72, line_h * (6 + n_extra))
            annotated = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
            annotated[:h, :, :] = _annotate_reward_corner(base, reward_value)
            cv2.line(annotated, (0, h), (w - 1, h), (255, 255, 255), 1)

            state = np.asarray(raw.get("state", []), dtype=np.float32).reshape(-1) if isinstance(raw, dict) else np.zeros((0,), dtype=np.float32)
            speed = float(state[_EGO_STATE_IDX_SPEED]) if state.size > _EGO_STATE_IDX_SPEED else 0.0
            routing = ""
            if isinstance(raw, dict):
                routing = str(raw.get("routing_command", "") or "").strip()
            prompt = f"spd={speed:.2f}m/s {routing or 'Follow the route.'}"
            reasoning = _format_text_field(raw, "reasoning_text") or _format_text_field(raw, "reasoning")
            subtask = _format_text_field(raw, "subtask_text") or _format_text_field(raw, "subtask")
            expert_action_str = ""
            if isinstance(raw, dict):
                ea = raw.get("expert_action")
                if ea is not None:
                    ea = np.asarray(ea, dtype=np.float32).reshape(-1)
                    first = ea[:4] if ea.size >= 4 else ea
                    expert_action_str = " ".join(f"{v:.3f}" for v in first)

            def _clip_text(txt: str, max_chars: int = 120) -> str:
                return txt if len(txt) <= max_chars else (txt[: max_chars - 3] + "...")

            def _fmt_action(arr: np.ndarray | None) -> str:
                if arr is None:
                    return "?"
                a = np.asarray(arr, dtype=np.float32).reshape(-1)
                return " ".join(f"{v:+.3f}" for v in a[:min(a.size, 6)])

            lines = [
                f"Expert: {_clip_text(critic_text) if critic_text else '?'}",
                f"ExpertAct[0]: {expert_action_str or '?'}",
                f"Prompt: {_clip_text(prompt)}",
                f"Reasoning: {_clip_text(reasoning)}",
                f"Subtask: {_clip_text(subtask)}",
            ]
            if base_action is not None or composed_action is not None:
                res_np = (np.asarray(composed_action) - np.asarray(base_action)) if (base_action is not None and composed_action is not None) else None
                lines.append(
                    f"Base: {_fmt_action(base_action)}  Res: {_fmt_action(res_np)}  Comp: {_fmt_action(composed_action)}"
                )
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
            final_viz = _as_video_frame(final_frame)
            if not FLAGS.expert_debug:
                _ftp = final_raw.get("target_points") if isinstance(final_raw, dict) else None
                final_viz = _annotate_waypoints(
                    final_viz, last_policy_action, base_action_flat=_last_base_action_np,
                    vla_chunk=_last_vla_chunk_holder[0], target_points=_ftp,
                )
            _f_base = _last_base_action_np if _online_training_mode in {"sac_residual", "dagger_residual"} else None
            frames.append(
                _annotate_text_panel(
                    final_viz,
                    final_raw,
                    reward_value=final_reward,
                    critic_text=final_critic_text,
                    base_action=_f_base,
                    composed_action=last_policy_action if _f_base is not None else None,
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
        """Rollout policy; returns ``(action, base_action_or_None)``."""
        if _online_training_mode in {"sac_residual", "dagger_residual"} and hasattr(agent, "sample_actions_sac_residual"):
            noise = jax.random.normal(subkey, (1, agent._flat_noise_dim()))
            if _residual_2d:
                chunk = agent._clip_actions_to_env(
                    jax.numpy.asarray(agent.vla_sample_fn(obs[None], noise))
                )
                _last_vla_chunk_holder[0] = np.asarray(chunk[0], dtype=np.float32)
                base2d = _decode_chunk_to_accel_steer(
                    _accel_steer_decoder, np.asarray(chunk)[0], obs_raw["state"]
                )[None]
                if step <= _residual_warmup:
                    base = jax.numpy.asarray(base2d)
                    return base, base
                temperature = 0.0 if _online_training_mode == "dagger_residual" else 1.0
                return agent.sample_actions_sac_residual(
                    obs[None], seed=subkey, temperature=temperature, base_action=base2d
                )
            if step <= _residual_warmup:
                # During warmup execute pure Pi0 with zero residual.
                base = jax.numpy.asarray(agent.vla_sample_fn(obs[None], noise))
                base = agent._clip_actions_to_env(base)
                return base, base
            temperature = 0.0 if _online_training_mode == "dagger_residual" else 1.0
            return agent.sample_actions_sac_residual(obs[None], seed=subkey, temperature=temperature)
        if getattr(agent, "vla_sample_fn", None) is not None:
            return agent.sample_actions_with_vla(obs[None], seed=subkey), None
        if _online_training_mode == "dagger" and hasattr(agent, "sample_actions_dagger"):
            return agent.sample_actions_dagger(obs[None]), None
        return agent.sample_actions(obs[None], seed=subkey), None

    _vla_steps_budget = int(np.random.randint(70, 201)) if FLAGS.expert_recover_debug else 0
    if FLAGS.expert_recover_debug:
        print(f"[expert_recover_debug] episode 0: VLA for {_vla_steps_budget} steps then expert", flush=True)

    # True when the previous buffer slot belongs to a finished episode — guards the
    # next-step backfills below from writing the new episode's data into it.
    _prev_transition_done = True

    for step in tqdm.tqdm(range(1, FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        t_sample_start = time.time()
        if raw_obs_holder is not None:
            raw_obs_holder["obs"] = obs_raw
        rng, sub = jax.random.split(rng)
        _in_expert_recovery = FLAGS.expert_recover_debug and (episode_steps >= _vla_steps_budget)
        if FLAGS.expert_recover_debug and (episode_steps == _vla_steps_budget):
            env.reinit_expert()
        in_warmup = warmup > 0 and step <= warmup
        if image_encoder == "siglip" and siglip_include_prompt_subtask:
            obs = _extract_agent_obs(env, obs_raw, obs_mode, **_extract_obs_kwargs)
        base_action_np: np.ndarray | None = None
        if FLAGS.expert_debug or _in_expert_recovery or agent is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action_jax, base_action_jax = _sample_agent_action(sub)
            _block_until_ready_tree((action_jax, base_action_jax))
            action = np.asarray(action_jax[0])
            last_policy_action = action
            if base_action_jax is not None:
                base_action_np = np.asarray(base_action_jax[0], dtype=np.float32)
                _last_base_action_np = base_action_np
        t_sample_end = time.time()

        t_step_start = time.time()
        if FLAGS.expert_debug or _in_expert_recovery:
            next_obs_raw, reward, terminated, truncated, info = env.step_expert(obs_raw)
        else:
            print(f"[RC-STEP] step={step} action={np.round(action, 4).tolist()}", flush=True)
            next_obs_raw, reward, terminated, truncated, info = env.step(action)
        if raw_obs_holder is not None:
            raw_obs_holder["next_obs"] = next_obs_raw
        drive_metrics = ego_drive_metrics_from_state_vec(next_obs_raw["state"])
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
        elif _critic_feedback_mode == "expert_action":
            # State-only privileged label: expert first-step target (+validity).
            # In accel_steer mode the expert chunk is PID-decoded to 2-D controls;
            # the decoder is stateful, so the next label is backfilled one step
            # later instead of decoding next_obs_raw's chunk a second time.
            _expert_first = _critic_expert_first(obs_raw)
            if _residual_2d and _expert_first is None:
                _lang = _zero_label  # expert unavailable this step (validity flag stays 0)
            else:
                _lang = compute_expert_target(obs_raw, agent, agent_config, expert_first=_expert_first)
            if _residual_2d:
                _next_lang = _zero_label  # backfilled next step (see below)
            else:
                _next_lang = compute_expert_target(next_obs_raw, agent, agent_config)
        elif _critic_feedback_mode == "action_delta":
            _expert_first = _critic_expert_first(obs_raw)
            if _residual_2d and _expert_first is None:
                _lang = _zero_label
            else:
                _lang = compute_action_delta(
                    obs_raw, action, agent, agent_config,
                    expert_first=_expert_first,
                    agent_first=action if _residual_2d else None,
                )
            # Placeholder: the true next label depends on the *next* logged action,
            # which doesn't exist yet. Backfilled one step later (below) — a zero
            # delta means "agent matched the expert", so leaving it zero would
            # bias the bootstrap optimistically.
            _next_lang = _zero_label
        elif _critic_feedback_mode == "delta_commentary_bow":
            _lang_text, _lang = compute_action_delta_commentary(obs_raw, action, agent)
            # Placeholder: backfilled one step later (below), same reason as above.
            _next_lang = _zero_label
        else:
            _lang = np.asarray(obs_raw.get("language_label", _zero_label), dtype=np.float32)
            _next_lang = np.asarray(next_obs_raw.get("language_label", _zero_label), dtype=np.float32)
        _critic_text_for_video = _critic_input_text(_critic_feedback_mode, _lang, _lang_text, obs_raw)

        replay_action = action.astype(np.float32)
        if _online_training_mode in {"dagger", "dagger_residual"} and not FLAGS.expert_debug:
            replay_action = np.asarray(obs_raw.get("expert_action", replay_action), dtype=np.float32)
            if _residual_2d and replay_action.size == env_ah * env_ad:
                # Expert provides a waypoint chunk; decode to 2-D [accel, steer] with a
                # dedicated PID so its state tracks the episode like the policy's PID.
                replay_action = _decode_chunk_to_accel_steer(
                    _expert_accel_steer_decoder, replay_action, obs_raw["state"]
                )

        residual_fields: dict[str, np.ndarray] = {}
        if _online_training_mode in {"sac_residual", "dagger_residual"}:
            residual_fields["base_actions"] = (
                base_action_np if base_action_np is not None else replay_action
            )
        if _online_training_mode == "sac_residual":
            residual_fields["base_next_actions"] = np.zeros_like(
                residual_fields["base_actions"]
            )
        if (
            _online_training_mode == "sac_residual"
            and buffer.size > 0
            and base_action_np is not None
            and not _prev_transition_done
        ):
            # Backfill: base_action_np = Pi0(obs) = Pi0(s') for the *previous* transition.
            buffer._dict["base_next_actions"][(buffer.pointer - 1) % buffer.max_size] = base_action_np
        _needs_next_label_backfill = _critic_feedback_mode in ("action_delta", "delta_commentary_bow") or (
            _critic_feedback_mode == "expert_action" and _residual_2d
        )
        if _needs_next_label_backfill and buffer.size > 0 and not _prev_transition_done:
            # Backfill: _lang = label at s_t (under the logged a_t for the delta
            # modes), which is the next-state label for the previous transition.
            # This keeps the bootstrap conditioned on the label the critic will see
            # when that next transition is trained as a "current" state (instead of
            # a zero label, which means "agent matched the expert" in delta space).
            buffer._dict["next_language_label"][(buffer.pointer - 1) % buffer.max_size] = _lang
        if _uses_pi_prefix and _pi_prefix_e is not None:
            _pi_prefix_next_e = _compute_pi_prefix_e(next_obs_raw)
            residual_fields["pi_prefix_obs_e"] = _pi_prefix_e
            residual_fields["pi_prefix_next_obs_e"] = _pi_prefix_next_e
        if _residual_append_state and _online_training_mode in {"sac_residual", "dagger_residual"}:
            _res_state = np.asarray(obs_raw.get("state", np.zeros(25)), dtype=np.float32)[6:]
            _res_next_state = np.asarray(next_obs_raw.get("state", np.zeros(25)), dtype=np.float32)[6:]
            residual_fields["residual_obs"] = _res_state
            residual_fields["next_residual_obs"] = _res_next_state
            # Welford update during residual warmup to build normalizer stats.
            if step <= _residual_warmup:
                _res_norm_count += 1
                _delta = _res_state.astype(np.float64) - _res_norm_mean
                _res_norm_mean += _delta / _res_norm_count
                _res_norm_M2 += _delta * (_res_state.astype(np.float64) - _res_norm_mean)
            # Freeze normalizer at the end of warmup.
            if (
                step == _residual_warmup
                and _res_norm_count > 0
                and agent is not None
                and getattr(agent, "sac_residual_agent", None) is not None
            ):
                _res_std = np.sqrt(_res_norm_M2 / max(_res_norm_count - 1, 1) + 1e-8).astype(np.float32)
                agent = agent.replace(
                    sac_residual_agent=agent.sac_residual_agent.set_obs_norm(
                        _res_norm_mean.astype(np.float32), _res_std
                    )
                )
                print(
                    f"[main_carla] residual obs normalizer frozen from {_res_norm_count} warmup samples "
                    f"(mean={_res_norm_mean[:3].round(3)}, std={_res_std[:3].round(3)})",
                    flush=True,
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
        _prev_transition_done = done
        if _uses_pi_prefix and _pi_prefix_e is not None:
            _pi_prefix_e = _pi_prefix_next_e
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
        traffic_violation_count = int(info.get("traffic_violation_count", 0))
        traffic_violation_delta = max(0, traffic_violation_count - prev_traffic_violation_count)
        episode_traffic_violations += traffic_violation_delta
        prev_traffic_violation_count = traffic_violation_count
        if traffic_violation_delta > 0:
            print(
                f"[main_carla] TRAFFIC VIOLATION at step {step}: "
                f"count={traffic_violation_count} episode_total={episode_traffic_violations}",
                flush=True,
            )
        if log_images:
            should_sample_periodic = episode_steps % episode_video_every == 0
            had_collision_this_step = collision_delta > 0
            had_violation_this_step = traffic_violation_delta > 0
            if should_sample_periodic or had_collision_this_step or had_violation_this_step:
                frame = _as_video_frame(_viz_image_from_raw(obs_raw))
                if not FLAGS.expert_debug:
                    _tp = obs_raw.get("target_points") if isinstance(obs_raw, dict) else None
                    frame = _annotate_waypoints(
                        frame, last_policy_action, base_action_flat=_last_base_action_np,
                        vla_chunk=_last_vla_chunk_holder[0], target_points=_tp,
                    )
                _vid_base = _last_base_action_np if _online_training_mode in {"sac_residual", "dagger_residual"} else None
                _vid_comp = last_policy_action if _vid_base is not None else None
                frame = _annotate_text_panel(
                    frame,
                    cot_obs_raw,
                    reward_value=float(reward),
                    critic_text=_critic_text_for_video,
                    base_action=_vid_base,
                    composed_action=_vid_comp,
                )
                if had_collision_this_step:
                    frame = _annotate_collision_frame(
                        frame,
                        collision_count=collision_count,
                        collision_events=episode_collision_events,
                    )
                if had_violation_this_step:
                    frame = _annotate_traffic_violation_frame(
                        frame,
                        violation_count=traffic_violation_count,
                        episode_violations=episode_traffic_violations,
                    )
                episode_video_frames.append(frame)
        last_video_reward = float(reward)
        last_video_critic_text = _critic_text_for_video
        t_log_end = time.time()

        step_wb = {f"rollout/{k}": v for k, v in drive_metrics.items()}
        step_wb["rollout/collision_count"] = float(collision_count)
        step_wb["rollout/collision_events"] = float(collision_delta)
        step_wb["rollout/traffic_violation_count"] = float(traffic_violation_count)
        step_wb["rollout/traffic_violation_events"] = float(traffic_violation_delta)
        step_wb["reward/penalty_traffic_violation"] = float(info.get("penalty_traffic_violation", 0.0))
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
            step_wb["reward/penalty_speed_limit"] = float(info.get("penalty_speed_limit", 0.0))
            step_wb["reward/penalty_crash_stuck"] = float(info.get("penalty_crash_stuck", 0.0))
            step_wb["rollout/lane_offset_m"] = float(info.get("lane_offset_m", 0.0))
            step_wb["rollout/heading_error_rad"] = float(info.get("heading_error_rad", 0.0))
            step_wb["rollout/speed_norm"] = float(info.get("speed_norm", 0.0))
            step_wb["rollout/centering_factor"] = float(info.get("centering_factor", 0.0))
            step_wb["rollout/heading_factor"] = float(info.get("heading_factor", 0.0))

        # Log critic feedback signal (obs_raw is already next_obs_raw here)
        if _critic_feedback_mode == "expert_action":
            step_wb["label/expert_target_norm"] = float(np.linalg.norm(_lang[:-1]))
            step_wb["label/expert_target_valid"] = float(_lang[-1])
        elif _critic_feedback_mode == "action_delta":
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
        if (
            _online_training_mode in {"sac_residual", "dagger_residual"}
            and base_action_np is not None
            and not in_warmup
        ):
            residual_np = action - base_action_np
            step_wb["rollout/base_action_abs_mean"] = float(np.abs(base_action_np).mean())
            step_wb["rollout/base_action_abs_max"] = float(np.abs(base_action_np).max())
            step_wb["rollout/residual_abs_mean"] = float(np.abs(residual_np).mean())
            step_wb["rollout/residual_abs_max"] = float(np.abs(residual_np).max())
            step_wb["rollout/composed_action_abs_mean"] = float(np.abs(action).mean())
            if _residual_2d and base_action_np.shape[-1] == 2:
                step_wb["rollout/base_accel"] = float(base_action_np[0])
                step_wb["rollout/base_steer"] = float(base_action_np[1])
                step_wb["rollout/residual_accel"] = float(residual_np[0])
                step_wb["rollout/residual_steer"] = float(residual_np[1])
                step_wb["rollout/composed_accel"] = float(action[0])
                step_wb["rollout/composed_steer"] = float(action[1])
        if step % 10 == 0:
            print(
                f"[main_carla] step {step}: sample={t_sample_end - t_sample_start:.3f}s "
                f"env_step={t_step_end - t_step_start:.3f}s log={t_log_end - t_log_start:.3f}s",
                flush=True,
            )

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
                "rollout/episode_traffic_violations": float(episode_traffic_violations),
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
            if _residual_2d:
                # Fresh PID state for the new episode (the controllers integrate error).
                _accel_steer_decoder = _make_accel_steer_decoder()
                if _expert_accel_steer_decoder is not None:
                    _expert_accel_steer_decoder = _make_accel_steer_decoder()
            obs = _extract_agent_obs(env, obs_raw, obs_mode, **_extract_obs_kwargs)
            if _uses_pi_prefix:
                _pi_prefix_e = _compute_pi_prefix_e(obs_raw)
            episode_video_frames = []
            episode_return, episode_steps = 0.0, 0
            episode_collision_count = 0
            episode_collision_events = 0
            prev_collision_count = 0
            episode_traffic_violations = 0
            prev_traffic_violation_count = 0
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
                if _online_training_mode == "sac_residual":
                    agent, update_info = agent.update_sac_residual(batch)
                elif _online_training_mode == "dagger_residual":
                    agent, update_info = agent.update_dagger_residual(batch)
                elif _online_training_mode == "dagger":
                    agent, update_info = agent.update_dagger(batch)
                elif getattr(agent, "vla_sample_fn", None) is not None:
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
    if FLAGS.route:
        exp_name = f"{exp_name}_{FLAGS.route}"
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
    _VALID_TRAIN_MODES = {"rl", "dagger", "sac_residual", "dagger_residual"}
    if online_training_mode not in _VALID_TRAIN_MODES:
        raise ValueError(
            f"Unsupported online_training_mode={online_training_mode!r}; "
            f"expected one of {sorted(_VALID_TRAIN_MODES)}."
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
    _residual_2d_mode = (
        online_training_mode in {"sac_residual", "dagger_residual"}
        and str(config.get("residual_action_space", "waypoint_chunk")).strip().lower() == "accel_steer"
    )
    critic_feedback_mode = resolve_critic_feedback_mode(config)
    if _residual_2d_mode and critic_feedback_mode in ("expert_action", "action_delta"):
        # accel_steer residual: the critic action space is the 2-D PID controls,
        # so the expert label is PID-decoded [accel, steer] (see
        # _critic_expert_first), not the 4-D waypoint first step.
        config.critic_action_dim = 2
    if critic_feedback_mode == "none":
        config.language_label_dim = 0
    elif critic_feedback_mode == "expert_action":
        # first_step(expert) + trailing validity flag.
        config.language_label_dim = int(config.get("critic_action_dim", 4)) + 1
    elif critic_feedback_mode == "action_delta":
        config.language_label_dim = int(config.get("critic_action_dim", 4))
    elif critic_feedback_mode in ("delta_commentary_bow", "vlm_chunk_bow"):
        config.language_label_dim = NUM_DELTA_COMMENTARY_WORDS
    elif critic_feedback_mode == "commentary_bow":
        config.language_label_dim = NUM_COMMENTARY_WORDS
    if critic_feedback_mode in ("action_delta", "delta_commentary_bow") and online_training_mode == "sac_residual":
        print(
            "[main_carla] NOTE: critic_feedback_mode="
            f"'{critic_feedback_mode}' labels depend on the logged action; the "
            "residual actor's Q-query pairs freshly sampled actions with that "
            "logged-action label. critic_feedback_mode='expert_action' is the "
            "state-only (fully Bellman-consistent) alternative.",
            flush=True,
        )
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
        tr_rank = int(config.get("training_gpu_rank", -1))
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
            siglip_device = config.get("siglip_device") or (f"cuda:{tr_rank}" if tr_rank >= 0 else "cuda:0")
            siglip_encoder = SigLIPEncoder(
                model_id=str(config.get("siglip_model_id", "google/siglip2-so400m-patch14-384")),
                device=siglip_device,
            )
            siglip_encoder.setup()
            config.siglip_embed_dim = int(siglip_encoder.embedding_dim)
            obs_dim = siglip_encoder.observation_dim(include_prompt_subtask=siglip_include_prompt_subtask)
            print(
                f"[main_carla] SigLIP encoder {siglip_encoder.model_id} "
                f"embed_dim={config.siglip_embed_dim} obs_dim={obs_dim} device={siglip_device}",
                flush=True,
            )

        raw_carla_holder: dict | None = None
        if use_steervla_rollout or FLAGS.expert_debug or FLAGS.expert_recover_debug:
            raw_carla_holder = {"obs": obs_dict, "next_obs": obs_dict}

        steervla_actor = None
        agent = None
        if not FLAGS.expert_debug:
            agent_obs = _extract_agent_obs(
                env, obs_dict, obs_mode,
                image_encoder=image_encoder,
                siglip_encoder=siglip_encoder,
                siglip_include_prompt_subtask=siglip_include_prompt_subtask,
                steervla_actor=None,
            )
            ex_obs = np.expand_dims(agent_obs, 0)
            ex_actions = np.zeros((1,) + tuple(env.action_space.shape), dtype=np.float32)

            _configure_jax_training_device(tr_rank)

            agent_class = agents[config["agent_name"]]
            create_kwargs = {}
            if config["agent_name"] == "dsrl":
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

            if online_training_mode in {"sac_residual", "dagger_residual"}:
                if config["agent_name"] != "dsrl":
                    raise ValueError(
                        f"{online_training_mode} mode requires agent_name='dsrl'."
                    )
                if steervla_actor is None:
                    raise ValueError(
                        f"{online_training_mode} mode requires SteerVLA rollout (frozen Pi0 base policy)."
                    )
                obs_mode_cfg = str(config.get("observation_mode", "state"))
                if obs_mode_cfg == "state":
                    embed_dim = int(ex_obs.shape[-1])
                elif str(config.get("image_encoder", "impala")).lower() == "siglip":
                    embed_dim = int(ex_obs.shape[-1])  # precomputed SigLIP embedding
                else:
                    embed_dim = int(tuple(config.get("image_mlp_hidden_dims", (512,)))[-1])
                if bool(config.get("residual_append_base_action", False)):
                    embed_dim += 2 if _residual_2d_mode else int(ex_actions.shape[-1])
                if bool(config.get("residual_append_state", False)):
                    embed_dim += int(config.get("residual_obs_dim", 19))
                sac_residual_agent = SACResidualAgent.create(
                    FLAGS.seed, ex_obs, ex_actions, config, embed_dim=embed_dim,
                )
                agent = agent.attach_sac_residual(sac_residual_agent)
                print(
                    f"[main_carla] SACResidualAgent created (embed_dim={embed_dim}, "
                    f"action_dim={2 if _residual_2d_mode else ex_actions.shape[-1]}, "
                    f"action_space={config.get('residual_action_space', 'waypoint_chunk')}).",
                    flush=True,
                )

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
