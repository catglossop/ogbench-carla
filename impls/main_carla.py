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
import dataclasses
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
flags.DEFINE_string(
    "pretrained_critic", None,
    "Path to a pretrained critic .pkl checkpoint from pretrain_critic.py. "
    "Injects the obs_encoder and critic params into the freshly created agent, "
    "leaving the actor/noise modules at random init. "
    "The checkpoint must have been created with a compatible config (same "
    "image_encoder, critic_hidden_dims, action_mode, and critic_feedback_mode=none).",
)

flags.DEFINE_string(
    "qgf_critic_ckpt", None,
    "Path to a pretrain_critic.py .pkl checkpoint for QGF inference-time guidance. "
    "When set together with --qgf_guidance_weight, injects Q-gradient guidance into "
    "each pi0 denoising step at test time (no actor training). The checkpoint must "
    "use action_mode=waypoints (40-D) and SigLIP image-only obs (1152-D).",
)
flags.DEFINE_float(
    "qgf_guidance_weight", 0.0,
    "Scale for Q-gradient added to pi0 flow velocity at each denoising step. "
    "0.0 disables QGF (default). Typical range: 0.01-0.5; sweep to find best value.",
)

flags.DEFINE_string(
    "bon_critic_ckpt", None,
    "Path to a pretrain_critic.py .pkl checkpoint used to score N candidate action "
    "chunks sampled straight from the frozen pi0 base policy (no residual actor) and "
    "execute the highest-Q candidate each step. Checkpoint must use action_mode=waypoints "
    "(40-D). Its SigLIP obs_enc width (image-only 1152-D, or 3456-D image+zero-padded "
    "prompt/subtask if trained with --siglip_include_prompt_subtask) is auto-detected "
    "from the checkpoint's first critic layer. Requires --train_mode not in "
    "{sac_residual, dagger_residual} (best-of-N replaces the residual pipeline entirely "
    "rather than feeding it).",
)
flags.DEFINE_integer(
    "bon_num_candidates", 8,
    "Number of pi0 action-chunk candidates sampled per env step when --bon_critic_ckpt "
    "or --bon_online_critic is set; the candidate with the highest-Q value is executed.",
)
flags.DEFINE_bool(
    "bon_online_critic", False,
    "Best-of-N action selection (same as --bon_critic_ckpt) but scored with the live "
    "online agent's own critic (modules_critic) instead of a separate frozen checkpoint, "
    "so the critic keeps learning from collected rollout transitions via the normal "
    "update_with_vla() Bellman backup. Warm-start it from a pretrained checkpoint with "
    "--pretrained_critic (same .pkl format as --bon_critic_ckpt). Mutually exclusive "
    "with --bon_critic_ckpt. Requires --eval_only=false (otherwise the critic never "
    "updates) and --train_mode not in {sac_residual, dagger_residual}.",
)
flags.DEFINE_bool(
    "bon_gemini_select", False,
    "Replace critic-based best-of-N scoring with Gemini: sample N candidates as usual, "
    "project each candidate's route (red) + speed (green) waypoints onto its own copy "
    "of the first-person frame, then send all N images together in a single "
    "multiple-choice prompt asking Gemini to pick the best one. Requires "
    "GEMINI_API_KEY in the environment. Mutually exclusive with --bon_critic_ckpt / "
    "--bon_online_critic.",
)
flags.DEFINE_string(
    "gemini_model", "gemini-3.6-flash",
    "Gemini model used for --bon_gemini_select.",
)
flags.DEFINE_bool(
    "bon_gemini_rollout_chunk", False,
    "When --bon_gemini_select is active, execute the FULL selected action chunk "
    "(all vla_action_horizon steps, shifted one step at a time) across consecutive "
    "env steps instead of re-sampling N candidates and re-querying Gemini on every "
    "single step. Only re-queries Gemini once the chunk is exhausted (or on the "
    "first step / after an episode reset).",
)
flags.DEFINE_bool(
    "bon_critic_rollout_chunk", False,
    "When --bon_critic_ckpt or --bon_online_critic is active, execute the FULL "
    "selected action chunk (all vla_action_horizon steps, shifted one step at a time) "
    "across consecutive env steps instead of re-sampling N candidates and re-scoring "
    "with the critic on every single step. Only re-scores once the chunk is exhausted "
    "(or on the first step / after an episode reset). Mirrors --bon_gemini_rollout_chunk.",
)
flags.DEFINE_integer(
    "bon_candidates_log_every", 20,
    "When best-of-N is active (--bon_critic_ckpt or --bon_online_critic), log an "
    "overlay frame every N env steps showing every candidate's subtask text and Q "
    "value, with the selected candidate marked. 0 disables this.",
)
flags.DEFINE_integer(
    "bon_max_sample_attempts", 6,
    "Best-of-N candidate diversity search: max resample attempts per candidate slot "
    "(beyond the first) when hunting for a subtask that's diverse from already-accepted "
    "candidates. Worst-case draws per step is roughly 1 + (N-1) * this. Lower this (e.g. "
    "2-3) to trade diversity guarantee for speed -- each draw is a full VLA forward pass.",
)
flags.DEFINE_bool(
    "bon_shadow_only", False,
    "When --bon_critic_ckpt is set, sample and Q-score N candidates every step (for the "
    "usual bon/q_best, bon/q_mean, and periodic candidates-panel logging) but do NOT use "
    "them to pick the executed action -- execute the plain single-sample (greedy) rollout "
    "action instead. Lets you see what best-of-N would have selected without it actually "
    "driving.",
)

flags.DEFINE_bool(
    "bon_include_brake_candidate", False,
    "When best-of-N is active (--bon_critic_ckpt or --bon_online_critic), append one "
    "extra synthetic all-zero (full-stop) action chunk to the N sampled candidates each "
    "step, so the critic always has the option to brake regardless of what the pi0 base "
    "policy actually sampled. Shown in candidate logging as '[synthetic] full brake'.",
)

flags.DEFINE_float(
    "pid_brake_speed", None,
    "Override SimlingoStyleWaypointDecoder.brake_speed (m/s) -- the controller brakes "
    "outright whenever the predicted desired_speed falls below this. Default (unset) "
    "keeps the class's own default of 0.1.",
)
flags.DEFINE_integer(
    "pid_stuck_threshold", None,
    "Override SimlingoStyleWaypointDecoder.stuck_threshold (consecutive near-stopped "
    "control ticks before forcing a creep_throttle burst). Default (unset) keeps the "
    "class's own default of 800, which is far larger than the ~10-11 ticks of sustained "
    "throttle CARLA's vehicle physics actually need after a fresh spawn/reset to build up "
    "enough engine RPM/torque to overcome initial stiction -- confirmed empirically via "
    "impls/debug_raw_control.py (hardcoded throttle=1.0 stays near-zero speed for ticks "
    "0-10, then breaks through at tick 11). The VLA/PID path recomputes throttle from a "
    "fresh independent noise draw every step, so it never reliably sustains throttle "
    "through that window on its own; a low stuck_threshold (e.g. 15) lets the existing "
    "creep-recovery safety net do it instead.",
)
flags.DEFINE_integer(
    "pid_creep_duration", None,
    "Override SimlingoStyleWaypointDecoder.creep_duration (ticks the forced creep_throttle "
    "burst lasts once stuck_threshold is hit). Default (unset) keeps the class's own "
    "default of 15.",
)
flags.DEFINE_float(
    "pid_creep_throttle", None,
    "Override SimlingoStyleWaypointDecoder.creep_throttle (the forced throttle floor "
    "during a stuck-recovery burst). Default (unset) keeps the class's own default of "
    "0.4 -- half the throttle=1.0 that impls/debug_raw_control.py empirically confirmed "
    "is needed to reliably break the ~11-tick post-reset stiction window; 0.4 may take "
    "much longer to break through or not reliably break through at all.",
)

flags.DEFINE_bool(
    "save_video_local", True,
    "Write each episode's annotated rollout video to <save_dir>/videos/epNNNN.mp4 "
    "in addition to (or instead of, if --wandb_mode=disabled) uploading to W&B.",
)

flags.DEFINE_integer("online_steps", 1000, "Number of online environment steps to run.")
flags.DEFINE_integer(
    "max_episodes", 0,
    "Stop after this many completed episodes (0 = unlimited, bounded only by "
    "--online_steps). Checked right after an episode ends, before the next reset.",
)
flags.DEFINE_bool(
    "debug_log_traffic", False,
    "Print ground-truth id/type/speed/location for every non-ego vehicle/walker actor "
    "each step (env.traffic_actor_states()) -- for directly verifying whether "
    "background traffic is actually moving, independent of what best-of-N candidate "
    "visualizations suggest.",
)
flags.DEFINE_integer(
    "debug_freeze_after_step", 0,
    "Once the episode reaches this step, force a hard stop (all-zero action, brake "
    "engaged) for every subsequent step instead of running the policy -- holds the ego "
    "at a fixed vantage point so background traffic motion can be judged without ego "
    "motion/parallax as a confound. 0 disables this.",
)
flags.DEFINE_bool(
    "terminate_on_collision", False,
    "Force the current episode to end (as if terminated) the moment any collision is "
    "detected, instead of continuing to drive after impact.",
)
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


def _detect_siglip_include_prompt_subtask(checkpoint_path: str, action_dim: int, embedding_dim: int) -> bool:
    """Auto-detect whether a pretrain_critic.py checkpoint's obs_enc includes the
    (zero-padded) prompt/subtask SigLIP text slots, by comparing the critic's first-layer
    input width against action_dim + embedding_dim vs. action_dim + 3*embedding_dim.

    Mirrors the auto-detection already done for --bon_critic_ckpt (see the bon_critic_ckpt
    block in run_online_carla) but runs earlier, before the online agent's network is
    built, so --pretrained_critic warm-starts don't crash with a ScopeParamShapeError from
    a siglip_include_prompt_subtask mismatch between the checkpoint and the fresh agent.
    """
    import pickle

    with open(checkpoint_path, "rb") as f:
        state = pickle.load(f)
    kernel = state["params"]["modules_critic"]["value_net"]["Dense_0"]["kernel"]
    critic_in_dim = int(kernel.shape[1])
    obs_enc_dim = critic_in_dim - action_dim
    if obs_enc_dim == embedding_dim:
        return False
    if obs_enc_dim == embedding_dim * 3:
        return True
    raise ValueError(
        f"--pretrained_critic checkpoint obs_enc width {obs_enc_dim} (= critic input "
        f"{critic_in_dim} - action_dim {action_dim}) matches neither SigLIP embedding_dim "
        f"{embedding_dim} nor 3x that; cannot auto-configure prompt/subtask padding for "
        "this checkpoint."
    )


def _load_pretrained_critic(agent, checkpoint_path: str):
    """Inject pretrained critic (and obs_encoder) params from a pretrain_critic.py checkpoint.

    Loads modules_obs_encoder and modules_critic from the checkpoint and injects them
    into the agent's network params.  modules_target_critic is synced to modules_critic.
    All other modules (noise_actor, noise_critic, flow) keep their random-init params.

    The checkpoint must have been created with a compatible config (same image_encoder,
    critic_hidden_dims, action_dim, and critic_feedback_mode=none / language_label_dim=0).
    """
    import pickle
    import jax

    with open(checkpoint_path, "rb") as f:
        state = pickle.load(f)
    pretrained_params = state["params"]

    new_params = dict(agent.network.params)
    injected = []
    for key in ("modules_obs_encoder", "modules_critic"):
        if key in pretrained_params:
            new_params[key] = jax.tree_util.tree_map(
                lambda x: jax.device_put(x), pretrained_params[key]
            )
            injected.append(key)
    if "modules_critic" in pretrained_params:
        new_params["modules_target_critic"] = new_params["modules_critic"]
        injected.append("modules_target_critic (= modules_critic)")

    agent = agent.replace(network=agent.network.replace(params=new_params))
    print(f"[main_carla] Pretrained critic loaded from {checkpoint_path}")
    print(f"[main_carla]   Injected: {injected}")
    return agent


def _setup_qgf_guidance(steervla_actor, ckpt_path: str, guidance_weight: float, siglip_encoder) -> None:
    """Wire QGF inference-time Q-gradient guidance into SteerVLAActor.

    Loads the pretrained critic from ckpt_path and calls steervla_actor.setup_qgf().
    siglip_encoder must be a SigLIPEncoder that supports .encode(image) → 1152-D embedding.
    """
    from qgf_guidance import load_pretrained_critic

    critic_def, critic_params = load_pretrained_critic(ckpt_path)
    steervla_actor.setup_qgf(critic_def, critic_params, guidance_weight, siglip_encoder)
    print(
        f"[main_carla] QGF guidance enabled: ckpt={ckpt_path} weight={guidance_weight}",
        flush=True,
    )


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
    if mode == "policy_embed":
        if steervla_actor is None:
            raise ValueError("observation_mode='policy_embed' requires a SteerVLAActor (steervla.enabled=True).")
        embed = np.asarray(steervla_actor.ensure_policy_embedding(1, raw=env_obs), dtype=np.float32)
        return embed[0] if embed.ndim > 1 else embed
    raise ValueError(f"Unknown observation_mode {mode!r}; expected 'state', 'image', or 'policy_embed'.")


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
    out = {
        "output_action_format": fmt,
        "action_horizon": ah,
        "action_dim": ad,
        # SteerVLA actor applies OpenPI Unnormalize + fixed denormalize_actions before returning actions.
        "action_input_space": "policy_output",
    }
    if FLAGS.pid_brake_speed is not None:
        out["brake_speed"] = float(FLAGS.pid_brake_speed)
    if FLAGS.pid_stuck_threshold is not None:
        out["stuck_threshold"] = int(FLAGS.pid_stuck_threshold)
    if FLAGS.pid_creep_duration is not None:
        out["creep_duration"] = int(FLAGS.pid_creep_duration)
    if FLAGS.pid_creep_throttle is not None:
        out["creep_throttle"] = float(FLAGS.pid_creep_throttle)
    return out


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
        ea = None
        if obs_raw is not None and obs_raw.get("expert_action") is not None:
            ea = obs_raw["expert_action"].tolist()
        self._proc.stdin.write(json.dumps({"expert_step": True, "expert_action": ea}) + "\n")
        self._proc.stdin.flush()
        msg = self._read_obs_msg()
        return self._decode_obs(msg["obs"]), msg["reward"], msg["terminated"], msg["truncated"], msg["info"]

    def reinit_expert(self):
        if self._proc is None:
            return
        self._proc.stdin.write(json.dumps({"reinit_expert": True}) + "\n")
        self._proc.stdin.flush()
        self._proc.stdout.readline()  # consume ack

    def checkpoint(self) -> bool:
        """Snapshot the subprocess's live world state (server-side; nothing crosses the wire).

        The returned sentinel is opaque -- pass it straight to :meth:`restore`. The
        server only tracks one checkpoint slot at a time; a second ``checkpoint()``
        call overwrites it.
        """
        self._proc.stdin.write(json.dumps({"checkpoint": True}) + "\n")
        self._proc.stdin.flush()
        msg = self._read_obs_msg()
        if not msg.get("ack"):
            raise RuntimeError(f"carla_env_server checkpoint failed: {msg}")
        return True

    def restore(self, ckpt: bool) -> None:
        """Teleport the subprocess's world back to the last :meth:`checkpoint`."""
        self._proc.stdin.write(json.dumps({"restore": True}) + "\n")
        self._proc.stdin.flush()
        msg = self._read_obs_msg()
        if not msg.get("ack"):
            raise RuntimeError(f"carla_env_server restore failed: {msg}")

    def traffic_actor_states(self) -> list[dict[str, Any]]:
        """Ground-truth (id/type/speed/loc) of every non-ego vehicle/walker actor.
        Debug-only -- see --debug_log_traffic."""
        self._proc.stdin.write(json.dumps({"traffic_states": True}) + "\n")
        self._proc.stdin.flush()
        msg = self._read_obs_msg()
        return list(msg.get("states") or [])

    def teleport_to_obstacle(self, offset_m: float = 1.0) -> bool:
        """Teleport the ego to sit ``offset_m`` in front of the nearest scenario obstacle prop.

        Debug-only (main_carla_teleop.py wall_snapshot mode). Returns False if no
        obstacle prop is currently in the world. Camera/state observations are stale
        until the caller issues a normal step() afterward.
        """
        self._proc.stdin.write(json.dumps({"teleport_to_obstacle": True, "offset_m": offset_m}) + "\n")
        self._proc.stdin.flush()
        msg = self._read_obs_msg()
        return bool(msg.get("ack"))

    def drive_straight_until_close(
        self,
        *,
        target_distance_m: float = 10.0,
        slowdown_distance_m: float = 30.0,
        max_ticks: int = 2000,
        throttle: float = 0.4,
        slow_throttle: float = 0.12,
    ) -> Optional[dict]:
        """Drive straight via raw VehicleControl (bypassing the policy) until within
        target_distance_m of the nearest scenario obstacle prop (braking to a stop
        right there), or a collision happens first, or max_ticks elapses.

        Debug-only (main_carla_teleop.py wall_snapshot mode). Blocks server-side for
        up to max_ticks CARLA ticks -- this call has no client-side timeout, so a
        large max_ticks can take a while. On success, returns a dict with a *fresh*
        observation read directly server-side (no extra tick -- some obstacle props
        get removed from the world shortly after a collision resolves, so avoiding
        an extra env.step() here maximizes the chance it's still present/visible):
        ``{"obs": <decoded obs dict>, "nearest_obstacle_distance_m": float,
        "collision_count": int}``. Returns None on failure (target never reached).
        """
        self._proc.stdin.write(
            json.dumps({
                "drive_straight_until_close": True,
                "target_distance_m": target_distance_m,
                "slowdown_distance_m": slowdown_distance_m,
                "max_ticks": max_ticks,
                "throttle": throttle,
                "slow_throttle": slow_throttle,
            }) + "\n"
        )
        self._proc.stdin.flush()
        msg = self._read_obs_msg()
        if not msg.get("ack"):
            return None
        return {
            "obs": self._decode_obs(msg["obs"]),
            "nearest_obstacle_distance_m": float(msg.get("nearest_obstacle_distance_m", -1.0)),
            "collision_count": int(msg.get("collision_count", -1)),
        }

    def step_raw_control(self, throttle: float, steer: float, brake: float = 0.0) -> dict:
        """One env tick with a raw VehicleControl, bypassing the policy/action-format
        decoding. Debug-only (main_carla_teleop.py wall_snapshot video recording) --
        for interleaving driving ticks with client-side policy queries, unlike
        drive_straight_until_close()'s single blocking call. Returns
        ``{"obs": <decoded obs dict>, "running": bool,
        "nearest_obstacle_distance_m": float, "collision_count": int}``.
        """
        self._proc.stdin.write(
            json.dumps({
                "step_raw_control": True, "throttle": throttle, "steer": steer, "brake": brake,
            }) + "\n"
        )
        self._proc.stdin.flush()
        msg = self._read_obs_msg()
        return {
            "obs": self._decode_obs(msg["obs"]),
            "running": bool(msg.get("running", False)),
            "nearest_obstacle_distance_m": float(msg.get("nearest_obstacle_distance_m", -1.0)),
            "collision_count": int(msg.get("collision_count", -1)),
        }

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
    # Rollout flow-latent scale for the frozen Pi0 base policy (see _sample_agent_action).
    _vla_noise_scale = float(agent_config.get("vla_noise_scale", 1.0))
    if _vla_noise_scale != 1.0:
        print(f"[main_carla] VLA rollout noise: tanh(N(0,1)) * {_vla_noise_scale}", flush=True)
    _residual_append_state = bool(agent_config.get("residual_append_state", False))
    _residual_obs_dim = int(agent_config.get("residual_obs_dim", 25))

    # Best-of-N: score N candidate action chunks sampled straight from the frozen pi0
    # base policy with a pretrained critic and execute the highest-Q one. Bypasses the
    # residual pipeline entirely (see _sample_agent_action).
    _bon_n = max(1, int(FLAGS.bon_num_candidates))
    _bon_q_fn = None
    if FLAGS.bon_critic_ckpt:
        if _online_training_mode in {"sac_residual", "dagger_residual"}:
            raise ValueError(
                "--bon_critic_ckpt selects actions straight from the pi0 base policy and "
                f"is incompatible with --train_mode={_online_training_mode!r}; use "
                "--train_mode=rl (no residual actor) instead."
            )
        if siglip_encoder is None:
            raise ValueError("--bon_critic_ckpt requires image_encoder='siglip' (for critic obs_enc).")
        from qgf_guidance import load_pretrained_critic, make_q_fn_batched

        _bon_critic_def, _bon_critic_params = load_pretrained_critic(FLAGS.bon_critic_ckpt)
        _bon_q_fn = make_q_fn_batched(_bon_critic_def, _bon_critic_params)

        # Auto-detect whether this checkpoint's obs_enc includes the (zero-padded, per
        # pretrain_critic.py's encode_all_parallel) prompt/subtask SigLIP text slots, by
        # comparing the critic's first-layer input width against action_dim + embedding_dim
        # vs. action_dim + 3*embedding_dim. Checkpoints trained with different
        # --siglip_include_prompt_subtask settings need different obs_enc shapes at eval time.
        _bon_critic_in_dim = int(_bon_critic_params["value_net"]["Dense_0"]["kernel"].shape[1])
        _bon_action_dim = int(agent._flat_env_action_dim())
        _bon_obs_enc_dim = _bon_critic_in_dim - _bon_action_dim
        _bon_img_dim = siglip_encoder.embedding_dim
        if _bon_obs_enc_dim == _bon_img_dim:
            _bon_include_prompt_subtask = False
        elif _bon_obs_enc_dim == _bon_img_dim * 3:
            _bon_include_prompt_subtask = True
        else:
            raise ValueError(
                f"--bon_critic_ckpt obs_enc width {_bon_obs_enc_dim} (= critic input "
                f"{_bon_critic_in_dim} - action_dim {_bon_action_dim}) matches neither "
                f"SigLIP embedding_dim {_bon_img_dim} nor 3x that; cannot auto-configure "
                "prompt/subtask padding for this checkpoint."
            )
        print(
            f"[main_carla] Best-of-N action selection enabled: ckpt={FLAGS.bon_critic_ckpt} "
            f"N={_bon_n} obs_enc_dim={_bon_obs_enc_dim} include_prompt_subtask={_bon_include_prompt_subtask}",
            flush=True,
        )

    # Best-of-N with the live online critic (modules_critic), instead of a separate frozen
    # checkpoint: scoring re-reads agent.network.params on every call, so as update_with_vla()
    # trains modules_critic each step (standard Bellman backup on collected transitions), the
    # candidates get scored by whatever the critic currently believes. Warm-start via
    # --pretrained_critic (same .pkl format as --bon_critic_ckpt).
    _bon_online = bool(FLAGS.bon_online_critic)
    if _bon_online:
        if FLAGS.bon_critic_ckpt:
            raise ValueError("--bon_online_critic and --bon_critic_ckpt are mutually exclusive.")
        if _online_training_mode in {"sac_residual", "dagger_residual"}:
            raise ValueError(
                "--bon_online_critic selects actions straight from the pi0 base policy and "
                f"is incompatible with --train_mode={_online_training_mode!r}; use "
                "--train_mode=rl (no residual actor) instead."
            )
        if FLAGS.eval_only:
            raise ValueError(
                "--bon_online_critic requires --eval_only=false; with eval_only=true the "
                "critic never receives training updates and stays frozen at its initial "
                "(random or --pretrained_critic warm-start) weights."
            )
        print(f"[main_carla] Best-of-N action selection enabled with LIVE online critic: N={_bon_n}", flush=True)

    # Best-of-N scored by Gemini instead of a learned critic: see gemini_bon_selector.py.
    _gemini_selector = None
    if FLAGS.bon_gemini_select:
        if FLAGS.bon_critic_ckpt or _bon_online:
            raise ValueError(
                "--bon_gemini_select is mutually exclusive with --bon_critic_ckpt / "
                "--bon_online_critic."
            )
        if _online_training_mode in {"sac_residual", "dagger_residual"}:
            raise ValueError(
                "--bon_gemini_select selects actions straight from the pi0 base policy and "
                f"is incompatible with --train_mode={_online_training_mode!r}; use "
                "--train_mode=rl (no residual actor) instead."
            )
        from gemini_bon_selector import GeminiActionSelector

        _gemini_selector = GeminiActionSelector(model=FLAGS.gemini_model)
        print(
            f"[main_carla] Best-of-N action selection enabled with Gemini: "
            f"model={FLAGS.gemini_model} N={_bon_n}",
            flush=True,
        )

    _bon_last_q_best: list[float] = [0.0]
    _bon_last_q_mean: list[float] = [0.0]
    _bon_last_candidates: list[dict | None] = [None]  # {"q_vals", "best_idx", "subtasks"}
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
    # --bon_gemini_rollout_chunk state: the flat (1, ah*ad) chunk Gemini most recently
    # selected, plus how many of its steps have already been executed. See
    # _sample_agent_action's Gemini branch, which shifts this chunk one step at a time
    # (mirrors SteerVLAActor._shift_cached_action_chunk) instead of re-querying Gemini
    # on every env step.
    _gemini_rollout_chunk: list = [None]  # flat np.ndarray (ah*ad,) selected by the last Gemini call
    _gemini_rollout_step: list = [0]  # steps of _gemini_rollout_chunk already executed
    # --bon_critic_rollout_chunk state: same idea as the Gemini rollout state above, but
    # for the critic-scored best-of-N paths (--bon_critic_ckpt / --bon_online_critic).
    _bon_rollout_chunk: list = [None]  # flat np.ndarray (ah*ad,) selected by the last critic score
    _bon_rollout_step: list = [0]  # steps of _bon_rollout_chunk already executed

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
        critic_mode: str = "none",
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

            if critic_mode == "expert_action":
                _critic_line = f"CriticIn[exp+valid]: {_clip_text(critic_text) if critic_text else '?'}"
            elif critic_mode == "action_delta":
                _critic_line = f"CriticIn[delta]: {_clip_text(critic_text) if critic_text else '?'}"
            else:
                _critic_line = f"Expert: {_clip_text(critic_text) if critic_text else '?'}"
            lines = [
                _critic_line,
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

    # Distinct per-candidate colors for waypoint projection + matching text swatches
    # (cycles if there are more candidates than colors).
    _CANDIDATE_PALETTE = [
        (255, 0, 0), (0, 200, 0), (30, 144, 255), (255, 140, 0),
        (200, 0, 200), (0, 220, 220), (255, 215, 0), (150, 75, 0),
    ]

    def _annotate_candidates_panel(
        frame: np.ndarray, candidates: dict, target_points: np.ndarray | None = None,
    ) -> np.ndarray:
        """Project every best-of-N candidate's predicted waypoints onto ``frame`` in a
        distinct color, then append a text panel below listing each candidate's Q value
        + subtask (color swatch matching its projected path), with the selected
        candidate highlighted. ``candidates`` is a
        ``{"q_vals": [...], "best_idx": int, "subtasks": [...], "chunks": (N, D)}``
        dict, see ``_bon_last_candidates``."""
        base = np.array(frame, copy=True)
        try:
            import cv2  # type: ignore
            from ogbench.carla.waypoint_viz import annotate_waypoints_on_frame

            q_vals = candidates.get("q_vals") or []
            best_idx = candidates.get("best_idx", -1)
            subtasks = candidates.get("subtasks") or []
            chunks = candidates.get("chunks")
            n = len(q_vals)
            if n == 0:
                return base

            colors = [_CANDIDATE_PALETTE[i % len(_CANDIDATE_PALETTE)] for i in range(n)]
            if chunks is not None and _steervla_exec_cfg is not None:
                # Draw the selected candidate last (on top) so its path isn't occluded.
                order = sorted(range(n), key=lambda i: i == best_idx)
                for i in order:
                    base = annotate_waypoints_on_frame(
                        base,
                        action_flat=chunks[i],
                        exec_cfg=_steervla_exec_cfg,
                        target_points=target_points if i == 0 else None,
                        route_color=colors[i],
                        speed_color=colors[i],
                        label=str(i),
                    )

            h, w = base.shape[:2]
            font_scale = 0.26
            line_h = 13
            panel_h = max(20, line_h * (n + 1))
            annotated = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
            annotated[:h, :, :] = base
            cv2.line(annotated, (0, h), (w - 1, h), (255, 255, 255), 1)

            y = h + line_h
            cv2.putText(
                annotated, "Best-of-N candidates:", (4, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1, cv2.LINE_AA,
            )
            y += line_h
            for i in range(n):
                selected = i == best_idx
                subtask = subtasks[i] if i < len(subtasks) else "?"
                subtask = subtask if len(subtask) <= 90 else subtask[:87] + "..."
                marker = "[SELECTED] " if selected else "            "
                cv2.rectangle(annotated, (4, y - 8), (12, y), colors[i], thickness=-1)
                line = f"  {marker}cand {i}: Q={q_vals[i]:+.3f}  {subtask}"
                color = (80, 255, 80) if selected else (255, 255, 255)
                cv2.putText(
                    annotated, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA,
                )
                y += line_h
            return annotated
        except Exception:
            return base

    _candidates_dir = Path(FLAGS.save_dir) / "videos" / "candidates"

    def _maybe_log_episode_video(
        rollout_log: dict,
        final_frame: np.ndarray | None,
        final_raw: dict[str, Any] | None,
        *,
        final_reward: float,
        final_critic_text: str,
        video_suffix: str = "",
    ) -> None:
        if not log_images:
            return
        frames = list(episode_video_frames)
        if final_frame is not None:
            final_viz = _as_video_frame(final_frame)
            if not FLAGS.expert_debug:
                _ftp = final_raw.get("target_points") if isinstance(final_raw, dict) else None
                if _bon_last_candidates[0] is not None:
                    # Must match the per-step branch in the main loop (same multi-
                    # candidate panel height) or frames.append() below produces a
                    # differently-shaped last frame and np.stack()/video writing fails.
                    final_viz = _annotate_candidates_panel(final_viz, _bon_last_candidates[0], target_points=_ftp)
                else:
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
                    critic_mode=_critic_feedback_mode,
                    base_action=_f_base,
                    composed_action=last_policy_action if _f_base is not None else None,
                )
            )
        if not frames:
            return
        if FLAGS.save_video_local:
            _save_local_video(frames, episode_count, suffix=video_suffix)
        video = np.stack(frames, axis=0)
        if video.ndim == 4:
            # W&B expects (T, C, H, W) for videos.
            video = np.transpose(video, (0, 3, 1, 2))
        rollout_log["rollout/episode_video"] = wandb.Video(video, fps=10, format="mp4")

    _video_dir = Path(FLAGS.save_dir) / "videos"
    if FLAGS.save_video_local:
        _video_dir.mkdir(parents=True, exist_ok=True)

    def _save_local_video(
        frames: list[np.ndarray], ep_idx: int, fps: float = 10.0, suffix: str = "",
    ) -> None:
        import cv2  # type: ignore

        h, w = frames[0].shape[:2]
        tag = f"ep{ep_idx:04d}{suffix}"
        path = str(_video_dir / f"{tag}.mp4")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        try:
            for frame in frames:
                writer.write(np.asarray(frame, dtype=np.uint8)[:, :, ::-1])  # RGB -> BGR
        finally:
            writer.release()
        print(f"[video] wrote {len(frames)} frames -> {path}", flush=True)

        # Individual frame images alongside the compiled video, for frame-level
        # inspection without decoding the mp4.
        frames_dir = _video_dir / f"{tag}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            cv2.imwrite(
                str(frames_dir / f"{i:04d}.jpg"),
                np.asarray(frame, dtype=np.uint8)[:, :, ::-1],  # RGB -> BGR
            )
        print(f"[video] wrote {len(frames)} frame images -> {frames_dir}", flush=True)

    def _block_until_ready_tree(tree):
        return jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            tree,
        )

    def _append_brake_candidate(
        chunks_np: np.ndarray, candidate_subtasks: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Append a synthetic all-zero (full-stop) action chunk as an extra best-of-N
        candidate. Zero deltas in DELTA_XY_T_DELTA_XY_SPACE decode to zero desired_speed
        (see pretrain_critic.py's _waypoints_action / SimlingoStyleWaypointDecoder.
        control_pid), i.e. a genuine brake -- this just guarantees the critic always has
        that option on the table, regardless of what the pi0 base policy sampled.
        """
        brake_chunk = np.zeros((1,) + chunks_np.shape[1:], dtype=chunks_np.dtype)
        chunks_np = np.concatenate([chunks_np, brake_chunk], axis=0)
        candidate_subtasks = candidate_subtasks + ["[synthetic] full brake"]
        return chunks_np, candidate_subtasks

    def _sample_diverse_candidates(subkey, n: int) -> tuple[np.ndarray, list[str]]:
        """Sample ``n`` candidate action chunks from the frozen pi0 base policy for
        best-of-N, one at a time (not batched), so each draw gets a fresh CoT.

        For each slot beyond the first, resamples up to
        ``subtask_diversity.MAX_SAMPLE_ATTEMPTS_PER_CANDIDATE`` times and keeps
        whichever draw is most different (by subtask keyword category, see
        ``subtask_diversity.py``) from the candidates already accepted, stopping
        early once one clears ``DIVERSITY_JACCARD_THRESHOLD``. Mirrors
        main_carla_teleop.py's ``_sample_candidates``. Some states genuinely only
        afford one sensible action (e.g. stopped at a red light) -- in that case
        this just falls back to the best of the attempted draws rather than
        forcing artificial diversity that isn't there.

        Much slower than a single batched (N, ...) vla_sample_fn call (up to
        N * MAX_SAMPLE_ATTEMPTS_PER_CANDIDATE individual forward passes instead of
        one) -- the cost of actually getting distinct subtasks per candidate.
        """
        from subtask_diversity import DIVERSITY_JACCARD_THRESHOLD, diversity_score, subtask_categories

        max_attempts = max(1, int(FLAGS.bon_max_sample_attempts))
        reset_cache = getattr(getattr(agent, "vla_sample_fn", None), "reset_action_cache", None)
        chunks: list[np.ndarray] = []
        subtasks: list[str] = []
        accepted_cats: list[frozenset] = []
        rng_local = subkey
        for slot in range(n):
            best = None
            best_score = -1.0
            attempts = 1 if slot == 0 else max_attempts
            for _attempt in range(attempts):
                rng_local, sub = jax.random.split(rng_local)
                if reset_cache is not None:
                    # Force a fresh CoT + action sample per draw; the actor otherwise
                    # caches one action chunk per env-step cadence, which would make
                    # every draw identical.
                    reset_cache()
                noise = jax.random.normal(sub, (1, agent._flat_noise_dim()))
                noise = jax.numpy.tanh(noise) * _vla_noise_scale
                chunk_jax = agent._clip_actions_to_env(
                    jax.numpy.asarray(agent.vla_sample_fn(obs[None], noise))
                )
                chunk_np = np.asarray(jax.device_get(chunk_jax[0]), dtype=np.float32)
                _decoded = steervla_actor.decode_last_batch_subtasks() if steervla_actor is not None else []
                subtask = _decoded[0] if _decoded else ""
                cats = subtask_categories(subtask)
                score = diversity_score(cats, accepted_cats)
                if score > best_score:
                    best, best_score = (chunk_np, subtask, cats), score
                if score >= (1.0 - DIVERSITY_JACCARD_THRESHOLD):
                    break
            chunk_np, subtask, cats = best
            chunks.append(chunk_np)
            subtasks.append(subtask)
            accepted_cats.append(cats)
        return np.stack(chunks, axis=0), subtasks

    def _score_candidates_with_critic(rng_key):
        """Sample N diverse candidates and Q-score them with the frozen best-of-N critic.

        Populates ``_bon_last_candidates``/``_bon_last_q_best``/``_bon_last_q_mean`` (for
        the usual wandb + candidates-panel logging) and returns ``(chunks, q_vals,
        best_idx)``. Does not touch ``_last_vla_chunk_holder`` -- callers decide whether
        the critic's pick is actually executed or this is purely diagnostic (see
        ``--bon_shadow_only``).
        """
        chunks_np, candidate_subtasks = _sample_diverse_candidates(rng_key, _bon_n)
        if FLAGS.bon_include_brake_candidate:
            chunks_np, candidate_subtasks = _append_brake_candidate(chunks_np, candidate_subtasks)
        chunks = jax.numpy.asarray(chunks_np)  # (N, vla_action_horizon * vla_action_dim), e.g. (N, 40)
        img = obs_raw.get("image") if isinstance(obs_raw, dict) else None
        if img is None:
            raise RuntimeError("[best-of-N] obs_raw['image'] is None; cannot compute obs_enc for critic.")
        if _bon_include_prompt_subtask:
            # prompt/subtask left empty -> zero text embeddings, matching how
            # pretrain_critic.py's encode_all_parallel padded this checkpoint's dataset.
            obs_enc_np = siglip_encoder.encode_observation(
                img, prompt="", subtask="", include_prompt_subtask=True
            )
        else:
            obs_enc_np = siglip_encoder.encode(img)
        obs_enc = jax.numpy.asarray(obs_enc_np, dtype=jax.numpy.float32)[None]
        obs_enc = jax.numpy.broadcast_to(obs_enc, (chunks.shape[0], obs_enc.shape[-1]))
        q_vals = np.asarray(_bon_q_fn(obs_enc, chunks))  # (N,)
        best_idx = int(np.argmax(q_vals))
        _bon_last_q_best[0] = float(q_vals[best_idx])
        _bon_last_q_mean[0] = float(q_vals.mean())
        _bon_last_candidates[0] = {
            "q_vals": q_vals.tolist(),
            "best_idx": best_idx,
            "subtasks": candidate_subtasks,
            "chunks": chunks_np,
        }
        _ah = int(agent._env_action_horizon())
        _ad = int(agent._env_action_dim())
        _chunks_first2 = chunks_np.reshape(chunks_np.shape[0], _ah, _ad)[:, :2, :]
        print(
            f"[BON-DEBUG] step={step} q_vals={np.array2string(q_vals, precision=4)} "
            f"best_idx={best_idx}",
            flush=True,
        )
        print(
            f"[BON-DEBUG] first-2-waypoints per candidate (N,2,{_ad}):\n"
            f"{np.array2string(_chunks_first2, precision=4, suppress_small=False)}",
            flush=True,
        )
        print(
            f"[BON-DEBUG] subtasks: {candidate_subtasks}",
            flush=True,
        )
        return chunks, q_vals, best_idx

    def _score_candidates_with_gemini(rng_key):
        """Sample N diverse candidates and pick the best one with a single Gemini
        multiple-choice call (see gemini_bon_selector.py). Populates the same
        ``_bon_last_candidates``/``_bon_last_q_best``/``_bon_last_q_mean`` fields as
        ``_score_candidates_with_critic`` (Gemini's one-hot choice standing in for Q
        values) so the existing wandb/candidates-panel logging works unmodified. Also
        saves each candidate's projected frame + Gemini's raw response to
        ``<save_dir>/videos/candidates/`` for inspection.
        """
        chunks_np, candidate_subtasks = _sample_diverse_candidates(rng_key, _bon_n)
        base_frame = _as_video_frame(_viz_image_from_raw(obs_raw))
        cand_frames = []
        for i in range(chunks_np.shape[0]):
            frame_i = base_frame
            if _steervla_exec_cfg is not None:
                from ogbench.carla.waypoint_viz import annotate_waypoints_on_frame

                frame_i = annotate_waypoints_on_frame(
                    base_frame,
                    action_flat=chunks_np[i],
                    exec_cfg=_steervla_exec_cfg,
                    route_color=(255, 0, 0),
                    speed_color=(0, 255, 0),
                    label=str(i),
                )
            cand_frames.append(frame_i)

        best_idx, critical_agents, reasoning, scores = _gemini_selector.select_candidate(
            cand_frames, candidate_subtasks
        )
        _bon_last_q_best[0] = float(scores[best_idx])
        _bon_last_q_mean[0] = float(scores.mean())
        _bon_last_candidates[0] = {
            "q_vals": scores.tolist(),
            "best_idx": best_idx,
            "subtasks": candidate_subtasks,
            "chunks": chunks_np,
        }
        print(
            f"[GEMINI-BON] step={step} choice={best_idx} "
            f"critical_agents={critical_agents!r} reasoning={reasoning!r}",
            flush=True,
        )
        print(f"[GEMINI-BON] subtasks: {candidate_subtasks}", flush=True)

        if FLAGS.save_video_local:
            import cv2  # type: ignore

            _candidates_dir.mkdir(parents=True, exist_ok=True)
            for i, frame_i in enumerate(cand_frames):
                cv2.imwrite(
                    str(_candidates_dir / f"step{step:06d}_cand{i}.jpg"),
                    frame_i[:, :, ::-1],  # RGB -> BGR
                )
            response_lines = [
                f"step={step}",
                f"choice={best_idx}",
                f"critical_agents={critical_agents}",
                f"reasoning={reasoning}",
                "",
                "subtasks:",
                *[f"  [{i}] {t}" for i, t in enumerate(candidate_subtasks)],
            ]
            (_candidates_dir / f"step{step:06d}_gemini.txt").write_text(
                "\n".join(response_lines) + "\n"
            )

        chunks = jax.numpy.asarray(chunks_np)
        return chunks, scores, best_idx

    def _shift_flat_chunk(flat: np.ndarray, step: int) -> np.ndarray:
        """Shift a flat (ah*ad,) action chunk forward by ``step`` timesteps, padding the
        tail by repeating the last waypoint. Mirrors
        SteerVLAActor._shift_cached_action_chunk (steervla.py) so --bon_gemini_rollout_chunk
        executes the remaining steps of a Gemini-selected chunk the same way the VLA's own
        actions_per_model_query cache would.
        """
        ah = int(_steervla_exec_cfg["action_horizon"])
        ad = int(_steervla_exec_cfg["action_dim"])
        if step <= 0:
            return flat
        chunk = flat.reshape(ah, ad)
        shifted = np.zeros_like(chunk)
        keep = max(0, ah - step)
        if keep > 0:
            shifted[:keep, :] = chunk[step: step + keep, :]
            shifted[keep:, :] = shifted[keep - 1: keep, :]
        else:
            shifted[:, :] = chunk[-1:, :]
        return shifted.reshape(ah * ad)

    last_update_info = None
    def _sample_agent_action(subkey):
        """Rollout policy; returns ``(action, base_action_or_None)``."""
        if (
            _bon_q_fn is not None
            and FLAGS.bon_shadow_only
            and getattr(agent, "vla_sample_fn", None) is not None
        ):
            # Diagnostic-only best-of-N: score candidates (populates the usual bon/*
            # logging) but fall through to the plain greedy branch below to pick the
            # actually-executed action, so we can see what best-of-N *would* have
            # selected without it actually driving.
            subkey, shadow_key = jax.random.split(subkey)
            _score_candidates_with_critic(shadow_key)
        if (
            _bon_q_fn is not None
            and not FLAGS.bon_shadow_only
            and getattr(agent, "vla_sample_fn", None) is not None
        ):
            _ah = int(_steervla_exec_cfg["action_horizon"]) if _steervla_exec_cfg is not None else 1
            if (
                FLAGS.bon_critic_rollout_chunk
                and _bon_rollout_chunk[0] is not None
                and _bon_rollout_step[0] < _ah
            ):
                # Chunk still in flight: execute the next shifted step of the last
                # critic-selected chunk instead of re-sampling candidates / re-scoring.
                shifted = _shift_flat_chunk(_bon_rollout_chunk[0], _bon_rollout_step[0])
                _bon_rollout_step[0] += 1
                best_chunk = jax.numpy.asarray(shifted[None])  # (1, 40)
                _last_vla_chunk_holder[0] = shifted
                return best_chunk, None
            chunks, q_vals, best_idx = _score_candidates_with_critic(subkey)
            best_chunk = chunks[best_idx][None]  # (1, 40)
            _last_vla_chunk_holder[0] = np.asarray(best_chunk[0], dtype=np.float32)
            if FLAGS.bon_critic_rollout_chunk:
                _bon_rollout_chunk[0] = np.asarray(best_chunk[0], dtype=np.float32)
                _bon_rollout_step[0] = 1  # step 0 was just returned by the fresh score
            return best_chunk, None
        if _gemini_selector is not None and getattr(agent, "vla_sample_fn", None) is not None:
            _ah = int(_steervla_exec_cfg["action_horizon"]) if _steervla_exec_cfg is not None else 1
            if (
                FLAGS.bon_gemini_rollout_chunk
                and _gemini_rollout_chunk[0] is not None
                and _gemini_rollout_step[0] < _ah
            ):
                # Chunk still in flight: execute the next shifted step of the last
                # Gemini-selected chunk instead of re-sampling candidates / re-querying
                # Gemini. Mirrors actions_per_model_query caching in steervla.py, but
                # applied at the best-of-N selection level.
                shifted = _shift_flat_chunk(_gemini_rollout_chunk[0], _gemini_rollout_step[0])
                _gemini_rollout_step[0] += 1
                best_chunk = jax.numpy.asarray(shifted[None])  # (1, 40)
                _last_vla_chunk_holder[0] = shifted
                return best_chunk, None
            chunks, scores, best_idx = _score_candidates_with_gemini(subkey)
            best_chunk = chunks[best_idx][None]  # (1, 40)
            _last_vla_chunk_holder[0] = np.asarray(best_chunk[0], dtype=np.float32)
            if FLAGS.bon_gemini_rollout_chunk:
                _gemini_rollout_chunk[0] = np.asarray(best_chunk[0], dtype=np.float32)
                _gemini_rollout_step[0] = 1  # step 0 was just returned by the fresh query
            return best_chunk, None
        if _bon_online and getattr(agent, "vla_sample_fn", None) is not None:
            _ah = int(_steervla_exec_cfg["action_horizon"]) if _steervla_exec_cfg is not None else 1
            if (
                FLAGS.bon_critic_rollout_chunk
                and _bon_rollout_chunk[0] is not None
                and _bon_rollout_step[0] < _ah
            ):
                # Chunk still in flight: execute the next shifted step of the last
                # live-critic-selected chunk instead of re-sampling candidates / re-scoring.
                # Note the online critic keeps training on collected transitions regardless
                # (see update_with_vla() below) -- this only affects action *selection*
                # cadence, not the Bellman backup.
                shifted = _shift_flat_chunk(_bon_rollout_chunk[0], _bon_rollout_step[0])
                _bon_rollout_step[0] += 1
                best_chunk = jax.numpy.asarray(shifted[None])  # (1, 40)
                _last_vla_chunk_holder[0] = shifted
                return best_chunk, None
            chunks_np, candidate_subtasks = _sample_diverse_candidates(subkey, _bon_n)
            if FLAGS.bon_include_brake_candidate:
                chunks_np, candidate_subtasks = _append_brake_candidate(chunks_np, candidate_subtasks)
            chunks = jax.numpy.asarray(chunks_np)  # (N, 40)
            # Live agent critic: re-reads agent.network.params every call, so this tracks
            # whatever update_with_vla() has trained modules_critic to as of this step.
            obs_e = agent._encode_obs(agent.network.params, obs[None])  # (1, obs_enc_dim)
            obs_e = jax.numpy.broadcast_to(obs_e, (chunks.shape[0], obs_e.shape[-1]))
            qs = agent.network.select("critic")(obs_e, chunks, params=agent.network.params)  # (ensemble, N)
            q_vals = np.asarray(jax.numpy.min(qs, axis=0))  # (N,)
            best_idx = int(np.argmax(q_vals))
            best_chunk = chunks[best_idx][None]  # (1, 40)
            _last_vla_chunk_holder[0] = np.asarray(best_chunk[0], dtype=np.float32)
            _bon_last_q_best[0] = float(q_vals[best_idx])
            _bon_last_q_mean[0] = float(q_vals.mean())
            _bon_last_candidates[0] = {
                "q_vals": q_vals.tolist(),
                "best_idx": best_idx,
                "subtasks": candidate_subtasks,
                "chunks": chunks_np,
            }
            if FLAGS.bon_critic_rollout_chunk:
                _bon_rollout_chunk[0] = np.asarray(best_chunk[0], dtype=np.float32)
                _bon_rollout_step[0] = 1  # step 0 was just returned by the fresh score
            return best_chunk, None
        if _online_training_mode in {"sac_residual", "dagger_residual"} and hasattr(agent, "sample_actions_sac_residual"):
            noise = jax.random.normal(subkey, (1, agent._flat_noise_dim()))
            # Match master's rollout noise (bounded tanh noise-actor output, not raw
            # unit Gaussian): the pi05 checkpoint barely creeps from standstill, and only
            # the variance of a *bounded* flow latent stochastically kicks the car out of
            # the brake-ratio stall (verified on generalization-wall-1095: raw unit-Gaussian
            # noise = permanent 0 m/s forever; tanh-squashed noise = takeoff and cruising).
            # Must run unconditionally -- gating this behind `!= 1.0` means the default
            # vla_noise_scale=1.0 silently falls back to the broken raw-Gaussian case.
            noise = jax.numpy.tanh(noise) * _vla_noise_scale
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
            # Bounded tanh-squashed noise (not sample_actions_with_vla's learned
            # noise_actor distribution, which is uncalibrated/near-zero-variance on a
            # freshly-initialized eval-only agent): the pi0 checkpoint tends to stall
            # at standstill, and only a *bounded* flow-latent's variance reliably
            # kicks it back out (see the sac_residual branch above / brake-ratio-stall
            # note). Matches that branch's rollout-noise convention exactly.
            noise = jax.random.normal(subkey, (1, agent._flat_noise_dim()))
            noise = jax.numpy.tanh(noise) * _vla_noise_scale
            base = jax.numpy.asarray(agent.vla_sample_fn(obs[None], noise))
            base = agent._clip_actions_to_env(base)
            return base, None
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
        _dump_obs_path = os.environ.get("DUMP_OBS_PATH")
        if _dump_obs_path and step == 1:
            import pickle as _pkl_dump
            with open(_dump_obs_path, "wb") as _f_dump:
                _pkl_dump.dump(obs_raw, _f_dump)
            print(f"[DUMP_OBS] wrote raw obs to {_dump_obs_path}", flush=True)
        rng, sub = jax.random.split(rng)
        _in_expert_recovery = FLAGS.expert_recover_debug and (episode_steps >= _vla_steps_budget)
        if FLAGS.expert_recover_debug and (episode_steps == _vla_steps_budget):
            env.reinit_expert()
        in_warmup = warmup > 0 and step <= warmup
        if image_encoder == "siglip" and siglip_include_prompt_subtask:
            obs = _extract_agent_obs(env, obs_raw, obs_mode, **_extract_obs_kwargs)
        # Figure capture: record every env step's action + Q-value for the whole trajectory.
        # Saved to disk at episode end (or run end). The figure script picks the
        # interesting window post-hoc rather than guessing it upfront.
        _qgf_cfg = getattr(steervla_actor, "_qgf_config", None) if steervla_actor else None
        if steervla_actor is not None:
            if not hasattr(steervla_actor, "_traj_capture"):
                steervla_actor._traj_capture = []
            # For QGF runs, arm per-step denoising capture every step.
            if _qgf_cfg is not None:
                _qgf_cfg["capture_step"] = step
                _qgf_cfg["capture_data"] = []   # reset each step; denoising fills it
            # For baseline, arm the one-shot holder every step.
            else:
                steervla_actor._baseline_capture_holder = {"ready": False}

        base_action_np: np.ndarray | None = None
        if FLAGS.expert_debug or _in_expert_recovery or agent is None:
            action = np.zeros((action_dim,), dtype=np.float32)
            if (
                (FLAGS.expert_debug or _in_expert_recovery)
                and _bon_q_fn is not None
                and getattr(agent, "vla_sample_fn", None) is not None
            ):
                # Shadow-score policy candidates against the critic while the expert
                # actually drives, so bon/* logging + the candidates panel reflect real
                # in-distribution route states (reached by expert driving) instead of
                # wherever the policy's own (possibly-stalled) execution would end up.
                _score_candidates_with_critic(sub)
        else:
            action_jax, base_action_jax = _sample_agent_action(sub)
            _block_until_ready_tree((action_jax, base_action_jax))
            action = np.asarray(action_jax[0])
            last_policy_action = action
            if base_action_jax is not None:
                base_action_np = np.asarray(base_action_jax[0], dtype=np.float32)
                _last_base_action_np = base_action_np
            print(
                f"[ACTION_DEBUG] step={step} base={base_action_np.tolist() if base_action_np is not None else None} "
                f"final={action.tolist()} speed={float(np.asarray(obs_raw.get('state', np.zeros(1))).reshape(-1)[0]) if isinstance(obs_raw, dict) else None}",
                flush=True,
            )

            # Accumulate per-step capture into the trajectory buffer.
            import pickle as _pkl
            if steervla_actor is not None and hasattr(steervla_actor, "_traj_capture"):
                state_vec_cap = obs_raw.get("state", np.zeros(25))
                if _qgf_cfg is not None and _qgf_cfg.get("capture_data"):
                    # QGF: store the denoising trace + final action from this env step.
                    denoising = _qgf_cfg["capture_data"]
                    final_action = denoising[-1]["x_t_after"] if denoising else None
                    steervla_actor._traj_capture.append({
                        "step": step,
                        "state": np.array(state_vec_cap),
                        "final_action_flat": final_action,
                        "q_at_final": denoising[-1]["q_bc"] if denoising else None,
                        "qgrad_norm": denoising[-1]["qgrad_norm"] if denoising else None,
                        "obs_enc": denoising[-1]["obs_enc"] if denoising else None,
                        "denoising": denoising,  # full 10-step trace
                    })
                    _qgf_cfg["capture_data"] = None
                else:
                    # Baseline: store the final unguided action from this env step.
                    holder = getattr(steervla_actor, "_baseline_capture_holder", None)
                    if holder is not None and holder.get("ready"):
                        steervla_actor._traj_capture.append({
                            "step": step,
                            "state": np.array(state_vec_cap),
                            "final_action_flat": holder["action_flat"],
                        })

            # Every 100 steps, flush trajectory to disk (episode-end flush happens below).
            _traj = getattr(steervla_actor, "_traj_capture", None) if steervla_actor else None
            if step % 100 == 0 and _traj and len(_traj) > 0:
                guidance_weight = _qgf_cfg["guidance_weight"] if _qgf_cfg else 0.0
                _traj_path = os.path.join(
                    FLAGS.save_dir, f"traj_capture_w{guidance_weight}_ep{episode_count}.pkl"
                )
                with open(_traj_path, "wb") as _f:
                    _pkl.dump({"guidance_weight": guidance_weight, "steps": _traj}, _f)
                print(f"[traj-capture] flushed {len(_traj)} steps → {_traj_path}", flush=True)
                steervla_actor._traj_capture = []

        t_sample_end = time.time()

        if FLAGS.debug_freeze_after_step > 0 and step >= FLAGS.debug_freeze_after_step and not FLAGS.expert_debug:
            # All-zero waypoint chunk decodes to desired_speed=0 (cumsum of zeros),
            # which forces brake=True in the RC-PID -- a reliable hard stop regardless
            # of action encoding, reusing the same zero-action convention as the
            # expert_debug/no-agent branch above. For directly observing whether
            # background traffic is actually moving from a fixed ego vantage point.
            action = np.zeros((action_dim,), dtype=np.float32)

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
                _lang = compute_expert_target(obs_raw, agent, agent_config, expert_first=_expert_first)[:_lang_dim]
            if _residual_2d:
                _next_lang = _zero_label  # backfilled next step (see below)
            else:
                _next_lang = compute_expert_target(next_obs_raw, agent, agent_config)[:_lang_dim]
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
        if FLAGS.debug_log_traffic:
            _traffic_states = env.traffic_actor_states() if hasattr(env, "traffic_actor_states") else []
            print(
                f"[TRAFFIC-DEBUG] step={step} n_actors={len(_traffic_states)} "
                f"{[(s['id'], s['type'].split('.')[-1], round(s['speed'], 3), tuple(round(x, 1) for x in s['loc'])) for s in _traffic_states]}",
                flush=True,
            )
            if FLAGS.save_video_local:
                import cv2  # type: ignore

                _traffic_frame_dir = Path(FLAGS.save_dir) / "videos" / "traffic_debug"
                _traffic_frame_dir.mkdir(parents=True, exist_ok=True)
                _traffic_frame = _as_video_frame(_viz_image_from_raw(obs_raw))
                cv2.imwrite(
                    str(_traffic_frame_dir / f"step{step:06d}.jpg"),
                    _traffic_frame[:, :, ::-1],  # RGB -> BGR
                )
        episode_return += float(reward)
        episode_steps += 1
        collision_count = int(info.get("collision_count", 0))
        episode_collision_count = max(episode_collision_count, collision_count)
        collision_delta = max(0, collision_count - prev_collision_count)
        episode_collision_events += collision_delta
        prev_collision_count = collision_count
        if FLAGS.terminate_on_collision and collision_delta > 0:
            # Reassigned after the replay-buffer terminal-flag write above already ran
            # for this transition -- fine for eval_only runs (no training), but the
            # buffer's "terminals" entry for this exact step would be stale if training.
            print(
                f"[main_carla] Collision at step {step}; forcing episode end "
                "(--terminate_on_collision).",
                flush=True,
            )
            done = True
            terminated = True
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
                    if _bon_last_candidates[0] is not None:
                        # Best-of-N (critic or Gemini): show every candidate in its own
                        # color + a caption panel with each subtask, selected one
                        # highlighted -- same view used for the periodic bon/candidates
                        # snapshot, but baked into every frame of the compiled episode
                        # video instead of just occasional standalone snapshots.
                        frame = _annotate_candidates_panel(frame, _bon_last_candidates[0], target_points=_tp)
                    else:
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
                    critic_mode=_critic_feedback_mode,
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

        if (
            log_images
            and _bon_last_candidates[0] is not None
            and FLAGS.bon_candidates_log_every > 0
            and step % FLAGS.bon_candidates_log_every == 0
        ):
            _cand_frame = _as_video_frame(_viz_image_from_raw(obs_raw))
            _cand_tp = obs_raw.get("target_points") if isinstance(obs_raw, dict) else None
            _cand_frame = _annotate_candidates_panel(_cand_frame, _bon_last_candidates[0], target_points=_cand_tp)
            wandb.log({"bon/candidates": wandb.Image(_cand_frame)}, step=step)
            if FLAGS.save_video_local:
                import cv2  # type: ignore

                _candidates_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(_candidates_dir / f"step{step:06d}.jpg"),
                    _cand_frame[:, :, ::-1],  # RGB -> BGR
                )

        last_video_reward = float(reward)
        last_video_critic_text = _critic_text_for_video
        t_log_end = time.time()

        step_wb = {f"rollout/{k}": v for k, v in drive_metrics.items()}
        if _bon_q_fn is not None or _bon_online or _gemini_selector is not None:
            step_wb["bon/q_best"] = _bon_last_q_best[0]
            step_wb["bon/q_mean"] = _bon_last_q_mean[0]
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

            # Flush any remaining trajectory capture for this episode.
            _traj_ep = getattr(steervla_actor, "_traj_capture", None) if steervla_actor else None
            if _traj_ep and len(_traj_ep) > 0:
                import pickle as _pkl2
                _gw = _qgf_cfg["guidance_weight"] if _qgf_cfg else 0.0
                _ep_path = os.path.join(FLAGS.save_dir, f"traj_capture_w{_gw}_ep{episode_count}.pkl")
                with open(_ep_path, "wb") as _f2:
                    _pkl2.dump({"guidance_weight": _gw, "steps": _traj_ep}, _f2)
                print(f"[traj-capture] episode end flush: {len(_traj_ep)} steps → {_ep_path}", flush=True)
                steervla_actor._traj_capture = []

            if FLAGS.max_episodes > 0 and episode_count >= FLAGS.max_episodes:
                print(f"[main_carla] Reached --max_episodes={FLAGS.max_episodes}; stopping.", flush=True)
                break

            obs_raw, _info = env.reset(seed=FLAGS.seed + episode_count)
            if raw_obs_holder is not None:
                raw_obs_holder["obs"] = obs_raw
                raw_obs_holder["next_obs"] = obs_raw
                
            if agent is not None:
                reset_vla_cache = getattr(getattr(agent, "vla_sample_fn", None), "reset_action_cache", None)
                if reset_vla_cache is not None:
                    reset_vla_cache()
            _gemini_rollout_chunk[0] = None
            _gemini_rollout_step[0] = 0
            _bon_rollout_chunk[0] = None
            _bon_rollout_step[0] = 0
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
            and (not FLAGS.eval_only)
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
                "time/update_time": float(np.mean(update_times)) if update_times else 0.0,
            }
            if last_update_info is not None:
                metrics.update({
                    f"training/{k}": float(v)
                    for k, v in last_update_info.items()
                    if np.asarray(v).ndim == 0
                })
                metrics["training/buffer_size"] = int(buffer.size)
            last_log_time = time.time()
            wandb.log(metrics, step=step)
            train_logger.log(metrics, step=step)

        if agent is not None and step % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, step)

    # online_steps was exhausted without the current episode ever hitting `done`
    # (no collision/success/off-route termination) -- the video/frame logging
    # inside the `if done:` block above never ran for it. Flush whatever frames
    # were collected so far so the rollout isn't silently un-logged.
    if episode_video_frames:
        _tail_rollout_log: dict = {}
        _maybe_log_episode_video(
            _tail_rollout_log,
            None,
            None,
            final_reward=last_video_reward,
            final_critic_text=last_video_critic_text,
            video_suffix="_incomplete",
        )
        wandb.log(_tail_rollout_log, step=step)

    train_logger.close()

    if FLAGS.save_buffer:
        buffer_path = FLAGS.buffer_path or os.path.join(FLAGS.save_dir, "buffer.npz")
        path = buffer.save(buffer_path)
        print(f"[buffer] saved {buffer.size} transitions -> {path}", flush=True)


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #


def _resolve_carla_env_config(config) -> tuple[Optional[str], Optional[dict], Optional[dict]]:
    """Compute ``(carla_yaml_path, extra_carla_config, exec_cfg)`` for constructing the CARLA env.

    Shared by ``main()`` and other entrypoints (e.g. ``main_carla_teleop.py``) so env
    construction stays consistent with the agent config's SteerVLA / expert-debug settings.
    """
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
    if FLAGS.expert_debug or FLAGS.expert_recover_debug:
        extra_carla["expert_controller"] = "simlingo_autopilot"
    return carla_yaml, (extra_carla or None), exec_cfg


@dataclasses.dataclass
class CarlaSession:
    """Bundle of everything :func:`build_carla_session` constructs for a CARLA env."""

    env: Any
    agent: Any
    steervla_actor: Any
    raw_carla_holder: Optional[dict]
    obs_dict: dict
    obs_mode: str
    image_encoder: str
    siglip_encoder: Any
    siglip_include_prompt_subtask: bool
    exec_cfg: Optional[dict]


def build_carla_session(config, env, exec_cfg: Optional[dict] = None) -> CarlaSession:
    """Construct the agent + SteerVLA actor + SigLIP encoder for an already-built CARLA env.

    Mirrors ``main()``'s original inline setup (config mutation, agent creation,
    residual/QGF wiring) so ``run_online_carla`` and other loops (e.g.
    ``main_carla_teleop.py``) see identical agent/env state. ``config`` is mutated in
    place (``critic_action_dim``, ``language_label_dim``), matching the prior behavior.
    """
    steervla_cfg = config.get("steervla", None)
    online_training_mode = str(config.get("online_training_mode", "rl")).strip().lower()
    _VALID_TRAIN_MODES = {"rl", "dagger", "sac_residual", "dagger_residual"}
    if online_training_mode not in _VALID_TRAIN_MODES:
        raise ValueError(
            f"Unsupported online_training_mode={online_training_mode!r}; "
            f"expected one of {sorted(_VALID_TRAIN_MODES)}."
        )
    # Normally expert_debug means the expert drives, so there's no need to build the VLA.
    # But with a best-of-N critic also configured, we still build it to shadow-sample and
    # Q-score candidates against real expert-driven states (see the agent-construction
    # gate above and _score_candidates_with_critic in run_online_carla).
    use_steervla_rollout = bool(
        steervla_cfg is not None
        and steervla_cfg.get("enabled")
        and (not FLAGS.expert_debug or FLAGS.bon_critic_ckpt or FLAGS.bon_gemini_select)
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
        if not config.get("language_label_dim"):
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
    if exec_cfg is None:
        exec_cfg = _steervla_action_execution_cfg(steervla_cfg)

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
        if FLAGS.pretrained_critic:
            _pc_action_dim = int(np.prod(env.action_space.shape))
            _pc_detected = _detect_siglip_include_prompt_subtask(
                FLAGS.pretrained_critic, action_dim=_pc_action_dim,
                embedding_dim=int(siglip_encoder.embedding_dim),
            )
            if _pc_detected != siglip_include_prompt_subtask:
                print(
                    f"[main_carla] --pretrained_critic checkpoint requires "
                    f"siglip_include_prompt_subtask={_pc_detected} (auto-detected); "
                    f"overriding config (was {siglip_include_prompt_subtask}) so the "
                    "online agent's critic is built with a matching obs_enc width.",
                    flush=True,
                )
                siglip_include_prompt_subtask = _pc_detected
                config.siglip_include_prompt_subtask = _pc_detected
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
    # Normally expert_debug skips building the policy/agent entirely (the expert drives
    # via step_expert()). But when a best-of-N critic is also configured, we still want
    # the frozen VLA + agent built so _score_candidates_with_critic can shadow-sample and
    # Q-score policy candidates at each expert-driven step (see run_online_carla) --
    # purely diagnostic, the expert action still actually drives.
    if not FLAGS.expert_debug or FLAGS.bon_critic_ckpt or FLAGS.bon_gemini_select:
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

        # "policy_embed" isn't one of _extract_agent_obs's modes (it's populated via
        # SteerVLAActor.ensure_policy_embedding, not the env's raw obs dict), so it needs
        # steervla_actor built first (above) to get a real example embedding for agent init.
        if obs_mode == "policy_embed":
            if steervla_actor is None:
                raise ValueError(
                    "observation_mode='policy_embed' requires SteerVLA rollout (steervla.enabled=True)."
                )
            agent_obs = np.asarray(steervla_actor.ensure_policy_embedding(1, raw=obs_dict), dtype=np.float32)
            if agent_obs.ndim > 1:
                agent_obs = agent_obs[0]
        else:
            agent_obs = _extract_agent_obs(
                env, obs_dict, obs_mode,
                image_encoder=image_encoder,
                siglip_encoder=siglip_encoder,
                siglip_include_prompt_subtask=siglip_include_prompt_subtask,
                steervla_actor=None,
            )
        ex_obs = np.expand_dims(agent_obs, 0)
        ex_actions = np.zeros((1,) + tuple(env.action_space.shape), dtype=np.float32)

        # Best-of-N (frozen or online) selects the executed action from raw random
        # noise + critic scoring (see _sample_diverse_candidates) -- the learned
        # noise_actor/noise_critic never influence behavior in this mode. Tell the
        # agent to skip training them: it's wasted compute, and (found 2026-07-30)
        # the noise_actor's tanh-Gaussian log_prob is numerically unstable and can
        # NaN the *shared* combined gradient (see total_loss_vla), corrupting the
        # critic that best-of-N actually depends on. Decoupling removes that risk
        # entirely rather than just clipping around it.
        config.skip_noise_actor_training = bool(FLAGS.bon_critic_ckpt) or bool(FLAGS.bon_online_critic)
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

        if FLAGS.pretrained_critic is not None:
            agent = _load_pretrained_critic(agent, FLAGS.pretrained_critic)

        # QGF: inject inference-time Q-gradient guidance into pi0 denoising.
        if FLAGS.qgf_critic_ckpt and FLAGS.qgf_guidance_weight > 0.0 and steervla_actor is not None:
            _setup_qgf_guidance(steervla_actor, FLAGS.qgf_critic_ckpt, FLAGS.qgf_guidance_weight, siglip_encoder)

    return CarlaSession(
        env=env,
        agent=agent,
        steervla_actor=steervla_actor,
        raw_carla_holder=raw_carla_holder,
        obs_dict=obs_dict,
        obs_mode=obs_mode,
        image_encoder=image_encoder,
        siglip_encoder=siglip_encoder,
        siglip_include_prompt_subtask=siglip_include_prompt_subtask,
        exec_cfg=exec_cfg,
    )


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

    carla_yaml, extra_carla, exec_cfg = _resolve_carla_env_config(config)

    # Leaderboard starts CARLA with subprocess (fork + exec). JAX initializes a native
    # thread pool; forking afterward triggers the stdlib warning and can deadlock the child,
    # which often surfaces as UE4 "RenderThread" timeouts. Bring the simulator up first.
    env = _make_carla_env(carla_yaml, FLAGS.route, extra_carla_config=extra_carla)
    try:
        session = build_carla_session(config, env, exec_cfg=exec_cfg)

        if FLAGS.eval_only:
            # No offline-eval pipeline yet for CARLA; do a single rollout.
            FLAGS.online_steps = max(FLAGS.online_steps, 200)
            FLAGS.save_buffer = FLAGS.save_buffer or False

        run_online_carla(
            env,
            session.agent,
            config,
            exp_name,
            raw_carla_obs_holder=session.raw_carla_holder,
            steervla_actor=session.steervla_actor,
            image_encoder=session.image_encoder,
            siglip_encoder=session.siglip_encoder,
            siglip_include_prompt_subtask=session.siglip_include_prompt_subtask,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
