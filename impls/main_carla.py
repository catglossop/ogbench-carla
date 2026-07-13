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
   Individual update kinds can be toggled with ``enable_updates_rl`` (DSRL critic/actor),
   ``enable_updates_bc`` (full DAgger imitation path), and ``enable_updates_bc_hl`` (the
   high-level VLM backbone update on relabeled data) — each is ANDed with ``enable_updates``.

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
import traceback
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
from impls.coaches.cast_relabel import OnlineCastRelabelSession

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
flags.DEFINE_string("save_dir", "/home/cglossop/exps", "Save directory.")
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
flags.DEFINE_bool(
    "enable_updates",
    None,
    "Master switch: if false, skip ALL gradient updates (rollout/buffer logging only). "
    "Default: agent config ``enable_updates`` (true).",
)
flags.DEFINE_bool(
    "enable_updates_rl",
    None,
    "If false, skip DSRL critic/actor (RL) gradient updates. ANDed with ``enable_updates``. "
    "Default: agent config ``enable_updates_rl`` (true).",
)
flags.DEFINE_bool(
    "enable_updates_bc",
    None,
    "If false, skip the full BC / DAgger imitation update (``update_dagger``). "
    "ANDed with ``enable_updates``. Default: agent config ``enable_updates_bc`` (true).",
)
flags.DEFINE_bool(
    "enable_updates_bc_hl",
    None,
    "If false, skip the high-level VLM backbone update (``update_hl`` on relabeled data). "
    "ANDed with ``enable_updates``. Default: agent config ``enable_updates_bc_hl`` (true).",
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


# Must match :data:`ogbench.carla.carla_utils.EGO_STATE_IDX_SPEED`.
_EGO_STATE_IDX_SPEED = 15


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
        # SteerVLA actor applies OpenPI Unnormalize + fixed steervla denormalize_actions.
        "action_input_space": "policy_output",
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
    # Master switch (all updates) plus per-kind switches. Each per-kind flag is ANDed with the
    # master, so ``enable_updates=False`` still disables everything (back-compat). CLI flags
    # override the config values when provided.
    enable_updates = bool(agent_config.get("enable_updates", True))
    if FLAGS.enable_updates is not None:
        enable_updates = bool(FLAGS.enable_updates)
    enable_updates_rl = bool(agent_config.get("enable_updates_rl", True))
    if FLAGS.enable_updates_rl is not None:
        enable_updates_rl = bool(FLAGS.enable_updates_rl)
    enable_updates_bc = bool(agent_config.get("enable_updates_bc", True))
    if FLAGS.enable_updates_bc is not None:
        enable_updates_bc = bool(FLAGS.enable_updates_bc)
    enable_updates_bc_hl = bool(agent_config.get("enable_updates_bc_hl", True))
    if FLAGS.enable_updates_bc_hl is not None:
        enable_updates_bc_hl = bool(FLAGS.enable_updates_bc_hl)
    # Effective per-kind gates: RL = DSRL critic/actor, BC = full DAgger imitation path,
    # HL = high-level VLM backbone update (``update_hl`` on cast_relabel data).
    rl_updates_on = enable_updates and enable_updates_rl
    bc_updates_on = enable_updates and enable_updates_bc
    hl_updates_on = enable_updates and enable_updates_bc_hl
    any_updates_on = rl_updates_on or bc_updates_on or hl_updates_on

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    if not any_updates_on:
        print("[main_carla] all updates disabled: rollout-only (no gradient updates)", flush=True)
    else:
        print(
            f"[main_carla] updates enabled -> rl={rl_updates_on} bc={bc_updates_on} hl={hl_updates_on}",
            flush=True,
        )
    if warmup > 0 and agent is not None and not FLAGS.expert_debug:
        print(f"[main_carla] warmup: no updates while step < {warmup}", flush=True)
    if update_interval > 1 and agent is not None and any_updates_on:
        print(f"[main_carla] updates every {update_interval} env steps", flush=True)
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

    # CAST relabel observer: window rollout -> VLM good/bad review -> per-chunk credit
    # assignment -> suggested subtasks. Artifacts + wandb only (no buffer backfill).
    _cast_relabel: OnlineCastRelabelSession | None = None
    cast_cfg = agent_config.get("cast_relabel")
    if cast_cfg is not None and bool(cast_cfg.get("enabled", False)):
        _cast_relabel = OnlineCastRelabelSession(
            cast_cfg,
            save_dir=FLAGS.save_dir,
            action_chunk_steps=int(agent_config.get("action_horizon", 10)),
        )
        print(
            f"[main_carla] CAST relabel enabled (provider={_cast_relabel.provider}, "
            f"window={_cast_relabel.window_env_steps} env steps, debug={_cast_relabel.debug})",
            flush=True,
        )
        # Point the (trainable) SteerVLA actor at the CAST-relabel HL dataset so its high-level
        # update (run from DSRL ``update_with_vla``) trains on the stored steervla_hl_dataset_format
        # samples as they accumulate.
        if steervla_actor is not None and getattr(steervla_actor, "load_trainable_params", False):
            steervla_actor.hl_dataset_dir = _cast_relabel.hl_dataset_dir
            print(
                f"[main_carla] SteerVLA high-level update wired to HL dataset dir "
                f"{steervla_actor.hl_dataset_dir} (every {steervla_actor.hl_update_every} vla updates, "
                f"batch {steervla_actor.hl_update_batch_size}).",
                flush=True,
            )
        capture_rollout_video = True
    _online_training_mode = str(agent_config.get("online_training_mode", "rl")).strip().lower()
    _lang_dim = critic_language_dim(agent_config)
    _bon_viz_interval = int(agent_config.get("bon_viz_interval", 0))
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
    if _cast_relabel is not None:
        _cast_relabel.begin_episode(
            episode_count=max(1, episode_count),
            route_name=str(obs_raw.get("routing_command", "?") if isinstance(obs_raw, dict) else "?"),
            route_command_plan=(
                obs_raw.get("route_command_plan") if isinstance(obs_raw, dict) else None
            ),
        )

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

    def _plot_bon_candidates(frame: np.ndarray, cand: dict, step: int):
        """W&B image: the frame + all best-of-N candidate action trajectories.

        Each candidate's first two action dims are speed-waypoint deltas; their cumsum gives a
        planned (forward, lateral) path. The legend keys every candidate to its subtask and
        critic Q value, with the executed (best) candidate highlighted.
        """
        import textwrap
        from io import BytesIO

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        actions = np.asarray(cand["actions"], dtype=np.float32)  # (n, env_flat)
        q = np.asarray(cand["q"], dtype=np.float32)  # (n,)
        subtasks = list(cand.get("subtasks", []))
        best = int(cand.get("best", int(np.argmax(q)) if q.size else 0))
        ah, ad = int(cand["ah"]), int(cand["ad"])
        n = actions.shape[0]
        chunks = actions.reshape(n, ah, ad)

        fig, (ax_img, ax_traj) = plt.subplots(1, 2, figsize=(16, 8))
        ax_img.imshow(frame)
        ax_img.set_title(f"frame @ step {step}")
        ax_img.axis("off")

        cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
        handles = []
        for i in range(n):
            wps = np.cumsum(chunks[i, :, :2], axis=0)
            wps = np.concatenate([np.zeros((1, 2), dtype=wps.dtype), wps], axis=0)
            is_best = i == best
            # Show the full subtask, wrapped so long strings stay readable in the legend.
            sub = subtasks[i] if i < len(subtasks) else ""
            sub_wrapped = "\n     ".join(textwrap.wrap(sub, width=60)) or "(none)"
            (h,) = ax_traj.plot(
                wps[:, 1],  # lateral on x-axis
                wps[:, 0],  # forward on y-axis
                marker="o",
                markersize=3,
                color=cmap(i % cmap.N),
                linewidth=3.0 if is_best else 1.5,
                alpha=1.0 if is_best else 0.6,
                zorder=3 if is_best else 2,
                label=f"{i}{'*' if is_best else ''}: Q={q[i]:.2f} | {sub_wrapped}",
            )
            handles.append(h)
        ego = ax_traj.scatter([0], [0], c="k", marker="s", s=40, zorder=4, label="ego")
        handles.append(ego)
        ax_traj.set_title("best-of-N candidate action trajectories (cumsum dx, dy)")
        ax_traj.set_xlabel("lateral (action dim 1, cumsum)")
        ax_traj.set_ylabel("forward (action dim 0, cumsum)")
        ax_traj.set_aspect("equal", adjustable="datalim")
        ax_traj.grid(True, alpha=0.3)
        # Legend below the plots so full (wrapped) subtasks have horizontal room.
        legend = fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            fontsize=8,
            ncol=2,
            title="candidate: Q | subtask  (* = executed)",
        )
        # Render with bbox_inches='tight' so the out-of-axes legend is never clipped.
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", bbox_extra_artists=(legend,))
        plt.close(fig)
        buf.seek(0)
        img = wandb.Image(plt.imread(buf))
        return img

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
        if critic_mode == "delta_commentary_bow" or critic_mode == "vlm_chunk_bow":
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

            prompt = ""
            if isinstance(raw, dict):
                prompt = str(raw.get("openpi_prompt_text") or "").strip()
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
                include_hist = bool(getattr(steervla_actor, "include_ego_history", False)) if steervla_actor is not None else False
                proprio_norm = bool(getattr(steervla_actor, "proprio_norm", True)) if steervla_actor is not None else True
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

    def _annotate_waypoints(
        frame: np.ndarray,
        action_flat: np.ndarray | None,
    ) -> np.ndarray:
        if _steervla_exec_cfg is None or action_flat is None:
            return frame
        try:
            from ogbench.carla.waypoint_viz import annotate_waypoints_on_frame

            return annotate_waypoints_on_frame(
                frame,
                action_flat=action_flat,
                exec_cfg=_steervla_exec_cfg,
            )
        except Exception:
            return frame

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
                final_viz = _annotate_waypoints(final_viz, last_policy_action)
            frames.append(
                _annotate_text_panel(
                    final_viz,
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
        rollout_log["rollout/episode_video"] = wandb.Video(video, fps=episode_video_fps, format="mp4")

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
        _bon_viz_img = None
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
            # Best-of-N candidate visualization (frame is still s_t here), every N env steps.
            if (
                _bon_viz_interval > 0
                and step % _bon_viz_interval == 0
                and steervla_actor is not None
                and getattr(steervla_actor, "last_bon_candidates", None) is not None
            ):
                try:
                    _bon_viz_img = _plot_bon_candidates(
                        _viz_image_from_raw(obs_raw),
                        steervla_actor.last_bon_candidates,
                        step,
                    )
                except Exception as e:
                    print(f"[main_carla] best-of-n viz failed: {e}", flush=True)
        t_sample_end = time.time()

        t_step_start = time.time()
        if FLAGS.expert_debug or _in_expert_recovery:
            next_obs_raw, reward, terminated, truncated, info = env.step_expert(obs_raw)
        else:
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
            if should_sample_periodic or had_collision_this_step:
                frame = _as_video_frame(_viz_image_from_raw(obs_raw))
                if not FLAGS.expert_debug:
                    frame = _annotate_waypoints(frame, last_policy_action)
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
                if _vlm_coach is not None:
                    _vlm_coach.record_frame(frame)
                if _cast_relabel is not None:
                    _cast_relabel.record_frame(
                        frame,
                        subtask_text=(
                            _format_text_field(cot_obs_raw, "subtask_text")
                            or _format_text_field(cot_obs_raw, "subtask")
                        ),
                        episode_step=episode_steps,
                    )
                if step_in_video:
                    episode_video_frame_index += 1
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
        if _cast_relabel is not None and episode_trajectory:
            # Enrich the recorded step with the executed subtask / CoT reasoning / prompt
            # (stashed on ``cot_obs_raw`` by the VLA) so the CAST relabel window review can
            # key them by timestamp for the VLM coach prompt.
            _cast_step_record = dict(episode_trajectory[-1])
            _cast_step_record["subtask"] = (
                _format_text_field(cot_obs_raw, "subtask_text")
                or _format_text_field(cot_obs_raw, "subtask")
            )
            _cast_step_record["reasoning"] = (
                _format_text_field(cot_obs_raw, "reasoning_text")
                or _format_text_field(cot_obs_raw, "reasoning")
            )
            _cast_step_record["prompt"] = _format_text_field(cot_obs_raw, "openpi_prompt_text")
            _cast_relabel.record_trajectory_step(_cast_step_record)
            # Stash the raw SteerVLA model input (pre-step obs the action was taken from) so the
            # window review can persist BAD/relabeled chunks as high-level dataset samples. The
            # session keeps only chunk-start steps, so calling this every step is cheap.
            _cast_relabel.record_model_input(
                episode_step=episode_steps,
                image=cot_obs_raw.get("image"),
                state=cot_obs_raw.get("state"),
                current_speed=float(_ego_speed_mps_from_raw(cot_obs_raw)),
                prompt=_cast_step_record["prompt"],
                subtask=_cast_step_record["subtask"],
                reasoning=_cast_step_record["reasoning"],
                action_chunk=replay_action,
            )
            # A mid-route window review makes blocking Gemini calls (video upload + two
            # model queries) that can exceed the CARLA leaderboard watchdog timeout. Pause
            # the watchdogs/pseudo-sensors across the query so the route isn't stopped for
            # inactivity (same mechanism SteerVLA inference uses per step).
            if _cast_relabel.should_query(episode_steps):
                _pause_offtick = hasattr(env, "pause_for_vla_inference")
                if _pause_offtick:
                    env.pause_for_vla_inference()
                try:
                    _cast_relabel.maybe_query(
                        episode_step=episode_steps, done_info=info, global_step=step
                    )
                finally:
                    if _pause_offtick and hasattr(env, "resume_after_vla_inference"):
                        env.resume_after_vla_inference()
        last_video_reward = float(reward)
        last_video_critic_text = _critic_text_for_video
        t_log_end = time.time()

        step_wb = {f"rollout/{k}": v for k, v in drive_metrics.items()}
        step_wb["rollout/collision_count"] = float(collision_count)
        step_wb["rollout/collision_events"] = float(collision_delta)
        # Best-of-N candidate action entropy (per-dim + mean), stashed by the agent at sample time.
        _bon_actor = getattr(agent, "steervla_actor", None)
        _bon_metrics = getattr(_bon_actor, "last_bon_metrics", None) if _bon_actor is not None else None
        if _bon_metrics:
            step_wb.update(_bon_metrics)
        if _bon_viz_img is not None:
            step_wb["rollout/bon_candidates"] = _bon_viz_img
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
        step_wb["training/rl_updates_on"] = float(rl_updates_on)
        step_wb["training/bc_updates_on"] = float(bc_updates_on)
        step_wb["training/hl_updates_on"] = float(hl_updates_on)
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
            if "reward_total" in done_info:
                rollout_log["rollout/final_step_reward"] = float(done_info["reward_total"])
                rollout_log["rollout/final_step_reward_progress"] = float(
                    done_info.get("reward_progress", 0.0)
                )
                rollout_log["rollout/final_step_reward_centering"] = float(
                    done_info.get("reward_centering", 0.0)
                )
                rollout_log["rollout/final_step_reward_heading"] = float(
                    done_info.get("reward_heading", 0.0)
                )
                rollout_log["rollout/final_step_reward_terminal"] = float(
                    done_info.get("reward_terminal", 0.0)
                )
                rollout_log["rollout/final_step_penalty_collision"] = float(
                    done_info.get("penalty_collision", 0.0)
                )
                rollout_log["rollout/final_step_penalty_outside_route"] = float(
                    done_info.get("penalty_outside_route", 0.0)
                )
                rollout_log["rollout/final_step_penalty_steer"] = float(
                    done_info.get("penalty_steer", 0.0)
                )
                rollout_log["rollout/final_step_penalty_brake"] = float(
                    done_info.get("penalty_brake", 0.0)
                )
                rollout_log["rollout/final_step_penalty_crash_stuck"] = float(
                    done_info.get("penalty_crash_stuck", 0.0)
                )
                rollout_log["rollout/final_step_success"] = float(bool(done_info.get("success", False)))
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
            if _cast_relabel is not None:
                _cast_relabel.maybe_query(
                    episode_step=done_episode_steps, done_info=done_info, force=True, global_step=step
                )
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
            if _cast_relabel is not None:
                _cast_relabel.reset_episode()
                _cast_relabel.begin_episode(
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
            any_updates_on
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
                update_info = None
                if _online_training_mode == "dagger":
                    # Full BC / DAgger imitation path.
                    if bc_updates_on:
                        agent, update_info = agent.update_dagger(batch)
                elif use_vla_update:
                    # ``update_with_vla`` runs the DSRL critic/actor (RL) step and the HL VLM
                    # backbone update; each is gated independently.
                    if rl_updates_on or hl_updates_on:
                        agent, update_info = agent.update_with_vla(
                            batch, run_rl=rl_updates_on, run_hl=hl_updates_on,
                        )
                else:
                    if rl_updates_on:
                        agent, update_info = agent.update(batch)
                if update_info is not None:
                    _block_until_ready_tree((agent, update_info))
                    last_update_info = update_info
                t_update_end = time.time()
                update_times.append(t_update_end - t_update_start)
            

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

        if agent is not None and any_updates_on and step % FLAGS.save_interval == 0:
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

    def _slug(s: str) -> str:
        return re.sub(r"[^0-9A-Za-z._-]+", "-", str(s)).strip("-") or "na"

    _agent_name = str(config.get("agent_name", "agent"))
    _route_name = str(FLAGS.route or "all-routes")
    _exp_name_parts = [_slug(_agent_name)]
    if _agent_name == "best_of_n":
        _exp_name_parts.append(f"n{int(config.get('best_of_n', 10))}")
    _exp_name_parts.extend([_slug(_route_name), get_exp_name(FLAGS.seed)])
    exp_name = "_".join(_exp_name_parts)
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
