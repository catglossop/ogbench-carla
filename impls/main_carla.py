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

import base64
import dataclasses
import faulthandler
import json
import os
import random
import re
import subprocess
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
# routing-commands' DSRL residual sub-agent (attached to DSRLAgent). surya/rl-token's
# standalone agents["sac_residual"] lives in jax_agents.sac_residual and is imported
# locally in _run_residual_entry -- they are different agents, see dsrl_residual.py.
from jax_agents.dsrl_residual import SACResidualAgent
from utils.flax_utils import restore_agent
from coaches.expert_label import NUM_COMMENTARY_WORDS, NUM_DELTA_COMMENTARY_WORDS
from coaches.critic_feedback import (
    compute_action_delta,
    compute_action_delta_commentary,
    compute_expert_target,
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

from utils.live_policy_viewer import LivePolicyViewer

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

# Run artifacts root; the run's own dir is <save_dir>/<project>/<run_group>/<exp_name>, holding
# videos/, trajectories/, checkpoints/, cast_relabel/ and flags.json.
# The previous default (/home/celinet/carla_exps) pointed at a home directory that does not exist
# on this box, so any launch that did not pass --save_dir died at startup with PermissionError
# before CARLA even connected. Overridable per run with --save_dir, or globally with
# OGBENCH_SAVE_DIR for a machine whose scratch lives somewhere else.
flags.DEFINE_string(
    "save_dir", os.environ.get("OGBENCH_SAVE_DIR", "/raid/users/cglossop/exps"), "Save directory."
)
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
    # Who applies the fixed RLDS scaling (``denormalize_actions``: *7 on the speed-waypoint
    # deltas for DELTA_XY_T_DELTA_XY_SPACE)? It has to happen exactly once.
    #
    # * **remote** (``steervla.actor_url`` set): steervla_server.py calls
    #   ``steervla_physical_denormalize_actions`` before returning, so the chunk on the wire is
    #   already physical -> ``policy_output`` (env must not scale again).
    # * **local**: SteerVLAActor returns the raw model output. OpenPI Normalize/Unnormalize are
    #   deliberately disabled for these checkpoints (see ``impls/vlas/steervla.py``:
    #   ``STEERVLA_ENABLE_OPENPI_NORM``), and the rollout ``sample_actions`` path does *not*
    #   denormalize, so the env has to -> ``normalized``.
    #
    # ``residual`` is NOT the discriminator: every local entrypoint gets the same raw chunk from
    # the same ``vla_sample_fn``. Keying on it left DSRL / cast_relabel / best_of_n on
    # ``policy_output``, silently dropping the *7 -- PID desired speed came out ~7x too low and
    # the car crawled (<1 m/s) while the residual stack on the identical checkpoint drove at
    # ~10 m/s. See dev_testrun.md.
    action_input_space = "policy_output" if remote else "normalized"
    out = {
        "output_action_format": fmt,
        "action_horizon": ah,
        "action_dim": ad,
        "action_input_space": action_input_space,
        # PID brake threshold (m/s): desired speeds below this brake. Lower it to avoid the
        # cold-start brake trap. See SimlingoStyleWaypointDecoder.control_pid.
        "brake_speed": float(steervla_cfg.get("brake_speed", 0.1)),
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
    if "traffic_violation_count" in info:
        out["rollout/episode_traffic_violations"] = float(info["traffic_violation_count"])
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


def _draw_corner_badge(
    frame: np.ndarray, label: str, *, corner: str = "tl", bg=(0, 0, 0), row: int = 0
) -> np.ndarray:
    """Boxed text badge in a top corner (``'tl'``/``'tr'``); no-op if cv2 is unavailable.

    ``row`` stacks badges downwards in the same corner (row 1 sits just below row 0), so
    several event banners -- collision, traffic violation -- can be drawn on one frame
    without overlapping.
    """
    annotated = np.array(frame, copy=True)
    try:
        import cv2  # type: ignore

        font_scale, thickness, pad = 0.38, 1, 4
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        bw, bh = tw + 2 * pad, th + baseline + 2 * pad
        x0 = max(6, annotated.shape[1] - 6 - bw) if corner == "tr" else 6
        y0 = 6 + 16 * int(row)
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

        white = (255, 255, 255)
        lines: list[str] = []
        colors: list[tuple] = []  # per-line BGR; parallel to ``lines``
        if hud and hud.get("lines"):
            # Generic header lines (GRPO overlay): rendered verbatim, with optional per-line colors.
            hud_lines = [str(x) for x in hud["lines"]]
            hud_colors = hud.get("line_colors") or [white] * len(hud_lines)
            lines.extend(hud_lines)
            colors.extend(hud_colors[i] if i < len(hud_colors) and hud_colors[i] else white for i in range(len(hud_lines)))
        elif hud:
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
        colors += [white] * (len(lines) - len(colors))  # remaining lines default white
        panel_h = max(72, line_h * (len(lines) + 1))
        annotated = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
        annotated[:h, :, :] = annotated_top
        cv2.line(annotated, (0, h), (w - 1, h), (255, 255, 255), 1)
        y = h + line_h
        for line, color in zip(lines, colors):
            cv2.putText(annotated, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
            y += line_h
        return annotated
    except Exception:
        return base


def _annotate_full_frame(
    frame, raw, *, reward: float, action_flat=None, exec_cfg=None,
    critic_text: str = "", collision=None, traffic_violation=None, steervla_actor=None,
    expert_debug: bool = False, hud: dict | None = None,
) -> np.ndarray:
    """Unified rollout-video frame: waypoint overlay + text panel + optional violation badges.

    ``collision`` / ``traffic_violation`` are ``(count, episode_events)`` pairs, or ``None``
    when no such event fired on this step. They render as the same red / amber banners the
    ``run_online_carla`` loop draws, so videos from either rollout loop read the same way.
    """
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
    if traffic_violation is not None:
        out = _draw_corner_badge(
            out,
            f"TRAF VIOL v={int(traffic_violation[0])} e={int(traffic_violation[1])}",
            corner="tr", bg=(200, 140, 0), row=1,
        )
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
    collision=None, traffic_violation=None, steervla_actor=None, hud: dict | None = None,
) -> Optional[np.ndarray]:
    """Append an annotated frame on capture steps (every Nth step + the terminal one).

    Uses the shared :func:`_annotate_full_frame` (waypoints + text panel + optional violation
    badges). Returns the annotated frame that was appended (or ``None``) so callers can also feed
    a live viewer without re-annotating.

    A step on which a collision or traffic violation fired is *always* captured, on top of the
    periodic schedule -- otherwise the badge would only ever show up when the event happened to
    land on a multiple of ``video_every``, which is how these annotations go missing from the
    video even though the event was logged.
    """
    event = collision is not None or traffic_violation is not None
    if log_video and (episode_steps % video_every == 0 or done or event):
        frame = _viz_image_from_raw(obs)
        if frame is not None:
            annotated = _annotate_full_frame(
                frame, obs, reward=float(reward), action_flat=action_flat, exec_cfg=exec_cfg,
                collision=collision, traffic_violation=traffic_violation,
                steervla_actor=steervla_actor, hud=hud,
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
    # Best-of-N selects with a frozen pretrained critic and has no noise actor, so its
    # action-selection path is valid regardless of whether RL updates run. Keep the two
    # concerns separate: enable_updates_rl should decide whether the critic *trains*, not
    # whether selection happens at all.
    _selection_needs_no_rl = str(agent_config.get("agent_name", "")) == "best_of_n"
    bc_updates_on = enable_updates and enable_updates_bc
    hl_updates_on = enable_updates and enable_updates_bc_hl
    any_updates_on = rl_updates_on or bc_updates_on or hl_updates_on

    # Export the HL-fine-tuned SteerVLA backbone as a redeployable params-only checkpoint every
    # ``hl_checkpoint_every_steps`` env steps and at exit. Only when the HL update is actually running.
    _steervla_cfg = agent_config.get("steervla", None)
    _hl_ckpt_every = int(_steervla_cfg.get("hl_checkpoint_every_steps", 0)) if _steervla_cfg is not None else 0
    _hl_ckpt_dir = str(_steervla_cfg.get("hl_checkpoint_dir", "") or "") if _steervla_cfg is not None else ""
    # Retain only the newest N step dirs (0 = keep every one). Each is ~10 GB.
    _hl_ckpt_keep = int(_steervla_cfg.get("hl_checkpoint_keep_last", 0)) if _steervla_cfg is not None else 0
    _hl_ckpt_on = (
        hl_updates_on
        and _hl_ckpt_every > 0
        and steervla_actor is not None
        and bool(getattr(steervla_actor, "load_trainable_params", False))
    )
    if _hl_ckpt_on:
        print(
            f"[main_carla] SteerVLA policy checkpoints -> "
            f"{_hl_ckpt_dir or os.path.join(FLAGS.save_dir, 'checkpoints')} "
            f"every {_hl_ckpt_every} env steps (+ at exit), "
            f"keep_last={_hl_ckpt_keep or 'all'} (~10 GB each, params-only/inference).",
            flush=True,
        )
    elif _hl_ckpt_every > 0:
        # A configured interval that will never fire is worth one line: the usual cause is a
        # rollout-only run, where the weights never change so there is nothing to checkpoint.
        print(
            f"[main_carla] SteerVLA policy checkpointing configured (every {_hl_ckpt_every} steps) "
            f"but inactive: hl_updates={hl_updates_on}, trainable_actor="
            f"{steervla_actor is not None and bool(getattr(steervla_actor, 'load_trainable_params', False))}.",
            flush=True,
        )

    def _save_steervla_ckpt(step_tag: int, *, final: bool = False) -> None:
        if not _hl_ckpt_on or (not final and step_tag % _hl_ckpt_every != 0):
            return
        out_root = _hl_ckpt_dir or os.path.join(FLAGS.save_dir, "checkpoints")
        try:
            steervla_actor.save_checkpoint(out_root, int(step_tag), keep_last=_hl_ckpt_keep)
        except Exception as exc:  # noqa: BLE001 - checkpoint export must never kill training.
            import traceback

            print(f"[main_carla] SteerVLA HL checkpoint save failed (non-fatal): {exc}", flush=True)
            traceback.print_exc()
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
        if _online_training_mode in {"sac_residual", "dagger_residual"} and (
            str(agent_config.get("residual_action_space", "waypoint_chunk")).strip().lower()
            != "accel_steer"
        ):
            raise ValueError(
                "--bon_critic_ckpt selects a chunk straight from the pi0 base policy, which the "
                "residual actor can only consume once it has been PID-decoded to [accel, "
                f"steer]; --train_mode={_online_training_mode!r} therefore needs "
                "residual_action_space='accel_steer' (best-of-N picks the base action, the "
                "residual corrects it -- see _bon_selected_to_action). Use --train_mode=rl "
                "for no residual actor at all."
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
        if _online_training_mode in {"sac_residual", "dagger_residual"} and (
            str(agent_config.get("residual_action_space", "waypoint_chunk")).strip().lower()
            != "accel_steer"
        ):
            raise ValueError(
                "--bon_online_critic selects a chunk straight from the pi0 base policy, which the "
                "residual actor can only consume once it has been PID-decoded to [accel, "
                f"steer]; --train_mode={_online_training_mode!r} therefore needs "
                "residual_action_space='accel_steer' (best-of-N picks the base action, the "
                "residual corrects it -- see _bon_selected_to_action). Use --train_mode=rl "
                "for no residual actor at all."
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
        if _online_training_mode in {"sac_residual", "dagger_residual"} and (
            str(agent_config.get("residual_action_space", "waypoint_chunk")).strip().lower()
            != "accel_steer"
        ):
            raise ValueError(
                "--bon_gemini_select selects a chunk straight from the pi0 base policy, which the "
                "residual actor can only consume once it has been PID-decoded to [accel, "
                f"steer]; --train_mode={_online_training_mode!r} therefore needs "
                "residual_action_space='accel_steer' (best-of-N picks the base action, the "
                "residual corrects it -- see _bon_selected_to_action). Use --train_mode=rl "
                "for no residual actor at all."
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
    _lang_dim = critic_language_dim(agent_config)

    _vlm_coach: OnlineVLMSession | None = None
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
            # Only used when ``cast_relabel.hl_dataset_root`` points several runs at one shared
            # corpus (offline collection): it namespaces this run's window dirs under that root.
            run_tag=exp_name,
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
            action_input_space=str(_steervla_exec_cfg.get("action_input_space", "normalized")),
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
    if agent_config.get("debug_task", False):
        example_transition["ego_speed"] = _ego_speed_mps_from_raw(obs_raw)
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
    # Index of the transition added last step, so the next step can backfill its
    # next-state CoT fields (master 63e19f7). Reset to None on episode boundaries.
    _last_buf_idx: int | None = None
    episode_return, episode_steps, episode_count = 0.0, 0, 0
    episode_collision_count = 0
    episode_collision_events = 0
    prev_collision_count = 0
    episode_traffic_violations = 0
    prev_traffic_violation_count = 0
    last_log_time = time.time()
    episode_video_every = 2
    episode_video_fps = 10.0
    episode_video_frames: list[np.ndarray] = []
    episode_trajectory: list[dict[str, Any]] = []
    episode_video_frame_index = 0
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
            # ``route_name`` above is the current routing *command* (scene context for the VLM
            # prompt), not the scenario. The stored HL samples need the real route so a corpus
            # merged across routes can be split/weighted by it.
            route_id=str(FLAGS.route or "?"),
        )

    # Video annotation for this loop (routing-commands / cast_relabel): candidates panel,
    # waypoint overlay, text panel and the violation banners. These nested helpers shadow the
    # module-level _viz_image_from_raw / _annotate_text_panel / _annotate_full_frame pipeline
    # from surya/rl-token inside run_online_carla, on purpose -- the module-level pipeline is
    # what the residual loop uses.
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

    # Violation banners, top-right, stacked: collision (red) on row 0, traffic violation
    # (amber) on row 1. Both delegate to the module-level ``_draw_corner_badge`` so the
    # residual loop's ``_annotate_full_frame`` path renders them identically.
    def _annotate_collision_frame(
        frame: np.ndarray,
        *,
        collision_count: int,
        collision_events: int,
    ) -> np.ndarray:
        return _draw_corner_badge(
            frame,
            f"COLL c={int(collision_count)} e={int(collision_events)}",
            corner="tr",
            bg=(255, 0, 0),
        )

    def _annotate_traffic_violation_frame(
        frame: np.ndarray,
        *,
        violation_count: int,
        episode_violations: int,
    ) -> np.ndarray:
        return _draw_corner_badge(
            frame,
            f"TRAF VIOL v={int(violation_count)} e={int(episode_violations)}",
            corner="tr",
            bg=(200, 140, 0),
            row=1,
        )

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

    def _maybe_log_episode_video(
        rollout_log: dict,
        final_frame: np.ndarray | None,
        final_raw: dict[str, Any] | None,
        *,
        final_reward: float,
        final_critic_text: str,
        video_suffix: str = "",
    ) -> None:
        """Assemble the episode video (already-annotated per-step frames + the final frame)."""
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
            # rc's nested _annotate_text_panel (not the module-level _annotate_full_frame):
            # this whole function is routing-commands' video path -- _annotate_candidates_panel,
            # _annotate_waypoints, _save_local_video and video_suffix all come from it.
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
            # Env reward for this step (``_compute_reward_and_info``). Recorded so the VLM coaches
            # (window review + CAST credit assignment) can see the actual objective the RL side is
            # optimizing, instead of inferring it from speed / progress / collisions.
            "reward_total": float(step_info.get("reward_total", 0.0)),
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
    def _bon_selected_to_action(best_chunk, subkey):
        """Hand a best-of-N selected chunk to the residual actor, or execute it directly.

        Best-of-N and the residual actor answer different questions -- selection picks *which*
        pi0 proposal to follow, the residual then *corrects* the control it decodes to -- so in
        a residual run the selected chunk is the residual's base action rather than the executed
        action. Without this the BoN branches returned the chunk straight out of
        ``_sample_agent_action`` and the residual actor never ran at all.

        Only the ``accel_steer`` residual space is wired up: the chunk is PID-decoded to
        ``[accel, steer]`` and the residual perturbs those two numbers. Returns the usual
        ``(action, base_action_or_None)`` pair so the caller stores the base alongside the
        executed action in the replay buffer.
        """
        if not _residual_2d:
            return best_chunk, None
        _last_vla_chunk_holder[0] = np.asarray(best_chunk[0], dtype=np.float32)
        base2d = _decode_chunk_to_accel_steer(
            _accel_steer_decoder, np.asarray(best_chunk)[0], obs_raw["state"]
        )[None]
        if step <= _residual_warmup:
            # Warmup executes the selected candidate unmodified (zero residual), so the
            # residual's critic sees on-distribution base actions before it starts steering.
            base = jax.numpy.asarray(base2d)
            return base, base
        temperature = 0.0 if _online_training_mode == "dagger_residual" else 1.0
        return agent.sample_actions_sac_residual(
            obs[None], seed=subkey, temperature=temperature, base_action=base2d
        )

    def _sample_agent_action(subkey):
        """Rollout policy; returns ``(action, base_action_or_None)``."""
        # master 63e19f7: pause the CARLA sim across the (slow) VLA forward so sim time
        # does not advance during inference, and warn on unusually slow samples.
        uses_vla = getattr(agent, "vla_sample_fn", None) is not None
        pause_env = uses_vla and hasattr(env, "pause_for_vla_inference")
        if pause_env:
            env.pause_for_vla_inference()
        t0 = time.time()
        try:
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
                    return _bon_selected_to_action(best_chunk, subkey)
                chunks, q_vals, best_idx = _score_candidates_with_critic(subkey)
                best_chunk = chunks[best_idx][None]  # (1, 40)
                _last_vla_chunk_holder[0] = np.asarray(best_chunk[0], dtype=np.float32)
                if FLAGS.bon_critic_rollout_chunk:
                    _bon_rollout_chunk[0] = np.asarray(best_chunk[0], dtype=np.float32)
                    _bon_rollout_step[0] = 1  # step 0 was just returned by the fresh score
                return _bon_selected_to_action(best_chunk, subkey)
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
                    return _bon_selected_to_action(best_chunk, subkey)
                chunks, scores, best_idx = _score_candidates_with_gemini(subkey)
                best_chunk = chunks[best_idx][None]  # (1, 40)
                _last_vla_chunk_holder[0] = np.asarray(best_chunk[0], dtype=np.float32)
                if FLAGS.bon_gemini_rollout_chunk:
                    _gemini_rollout_chunk[0] = np.asarray(best_chunk[0], dtype=np.float32)
                    _gemini_rollout_step[0] = 1  # step 0 was just returned by the fresh query
                return _bon_selected_to_action(best_chunk, subkey)
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
                    return _bon_selected_to_action(best_chunk, subkey)
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
                return _bon_selected_to_action(best_chunk, subkey)
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
                if (rl_updates_on or _selection_needs_no_rl) and hasattr(agent, "sample_actions_with_vla"):
                    # DSRL proper: the learned noise actor picks the flow latent, so the RL
                    # updates actually steer the executed policy. Bypassing this (as the
                    # fixed-noise branch below does) makes ``enable_updates_rl`` a no-op on
                    # behaviour -- the critic/actor train but never drive.
                    #
                    # ``_selection_needs_no_rl`` covers best-of-N, whose selection does NOT
                    # depend on RL updates: it has no noise actor, and its critic is a frozen
                    # pretrained ranker. Gating it on ``rl_updates_on`` forced a choice between
                    # "best-of-N fires" and "the critic stays static", which are meant to be
                    # independent -- and silently degraded the run to a single sample when RL
                    # was off.
                    return agent.sample_actions_with_vla(obs[None], seed=subkey), None
                # Rollout-only / eval (no RL updates): the noise actor is still at its random
                # init, so its samples are uncalibrated and near-zero-variance. Draw a bounded
                # tanh-squashed latent instead -- the pi0 checkpoint tends to stall at
                # standstill and only a *bounded* flow latent's variance reliably kicks it back
                # out (see the sac_residual branch above / brake-ratio-stall note).
                noise = jax.random.normal(subkey, (1, agent._flat_noise_dim()))
                noise = jax.numpy.tanh(noise) * _vla_noise_scale
                base = jax.numpy.asarray(agent.vla_sample_fn(obs[None], noise))
                base = agent._clip_actions_to_env(base)
                return base, None
            if _online_training_mode == "dagger" and hasattr(agent, "sample_actions_dagger"):
                return agent.sample_actions_dagger(obs[None]), None
            return agent.sample_actions(obs[None], seed=subkey), None
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

    # True when the previous buffer slot belongs to a finished episode — guards the
    # next-step backfills below from writing the new episode's data into it.
    _prev_transition_done = True

    for step in tqdm.tqdm(range(1, FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        t_sample_start = time.time()
        _bon_viz_img = None
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
        # master's off-by-one fix: `<` so exactly `warmup` steps are collected.
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
            # action_dim (not env.action_space.shape) so accel_steer residual mode gets 2-D.
            action = np.zeros((action_dim,), dtype=np.float32)
            step_obs = obs
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

        # master's _buffer_transition validates against the buffer schema (and fills the
        # openpi / next_openpi fields); residual_fields is routing-commands' sac_residual
        # extension. masks uses ``terminated`` (not ``done``) so a truncation would still
        # bootstrap -- the RL-correct semantics.
        buf_idx = buffer.add_transition(
            _buffer_transition(
                obs_raw,
                next_obs_raw,
                observations=np.asarray(step_obs),
                actions=replay_action,
                rewards=np.float32(reward),
                next_observations=np.asarray(next_obs),
                masks=np.float32(0.0 if terminated else 1.0),
                terminals=np.float32(1.0 if done else 0.0),
                language_label=_lang,
                next_language_label=_next_lang,
                **residual_fields,
                **(
                    {"ego_speed": _ego_speed_mps_from_raw(obs_raw)}
                    if agent_config.get("debug_task", False)
                    else {}
                ),
            )
        )
        _last_buf_idx = int(buf_idx)
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
            step_in_video = (
                should_sample_periodic or had_collision_this_step or had_violation_this_step
            )
            if step_in_video:
                step_video_frame_index = episode_video_frame_index
            # Frame construction is routing-commands' (candidates panel / waypoint overlay /
            # text panel); the step_in_video + episode_video_frame_index bookkeeping is
            # master's, needed by _append_trajectory_step below.
            if step_in_video:
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
                # Unconditional: inside this branch ``step_in_video`` is necessarily True
                # (``raw_frame`` is only computed when it is).
                episode_video_frame_index += 1
            else:
                # Frame extraction failed even though this step was selected for video, so
                # clear the reservation made above — otherwise _append_trajectory_step records
                # a video_frame_index pointing at a frame that was never appended.
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
            # Raw routing instruction, NOT the wrapped ``Prompt:...;State:...;`` display string:
            # this value is re-tokenized by ``SteerVLAActor._build_hl_observation_batch``, which
            # applies the wrapper itself. Storing the wrapped form double-wraps every online HL
            # sample ("Prompt:Prompt:...;State:..;;State:..;") — a prefix the model never sees at
            # inference. ``openpi_prompt_text`` is kept as the fallback for older raw dicts.
            _cast_step_record["prompt"] = _format_text_field(
                cot_obs_raw, "openpi_prompt_raw_text"
            ) or _format_text_field(cot_obs_raw, "openpi_prompt_text")
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
                # Bare routing instruction (no "The current speed is X m/s. " prefix): the offline
                # RLDS converter stores it verbatim as the dataset's ``routing_command`` field and
                # lets the OpenPI loader rebuild the prompt from it exactly as the actor did.
                routing_command=_format_text_field(cot_obs_raw, "routing_command"),
                global_step=step,
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

        step_wb: dict[str, Any] = {f"rollout/{k}": v for k, v in drive_metrics.items()}
        step_wb.update(_reward_breakdown_log(info))
        if _bon_q_fn is not None or _bon_online or _gemini_selector is not None:
            step_wb["bon/q_best"] = _bon_last_q_best[0]
            step_wb["bon/q_mean"] = _bon_last_q_mean[0]
        step_wb["rollout/collision_count"] = float(collision_count)
        step_wb["rollout/traffic_violation_events"] = float(traffic_violation_delta)
        step_wb.update(_ego_control_log(next_obs_raw))
        step_wb["rollout/collision_events"] = float(collision_delta)
        # Bench2Drive route completion, per step. The dominant positive reward term and the
        # clearest read on whether the policy is actually advancing rather than stalling.
        if "route_progress_pct" in info:
            step_wb["rollout/route_progress_pct"] = float(info["route_progress_pct"])
            step_wb["rollout/route_progress_delta"] = float(info.get("route_progress_delta", 0.0))
        # Leaderboard infraction counters (only meaningful as running totals).
        if "traffic_violation_count" in info:
            step_wb["rollout/traffic_violation_count"] = float(info["traffic_violation_count"])
        if "outside_route_value" in info:
            step_wb["rollout/outside_route_value"] = float(info["outside_route_value"])
        # Best-of-N candidate action entropy (per-dim + mean), stashed by the agent at sample time.
        _bon_actor = getattr(agent, "steervla_actor", None)
        _bon_metrics = getattr(_bon_actor, "last_bon_metrics", None) if _bon_actor is not None else None
        if _bon_metrics:
            step_wb.update(_bon_metrics)
        if _bon_viz_img is not None:
            step_wb["rollout/bon_candidates"] = _bon_viz_img
        # The reward/* and rollout/{lane_offset_m,heading_error_rad,speed_norm,centering_factor,
        # heading_factor} keys this block used to build inline are now emitted by
        # ``_reward_breakdown_log(info)`` (called above), which also covers penalty_traffic_violation
        # and overspeed_frac. NOTE: the helper keeps the raw info key, so the five reward_* metrics
        # are renamed -- reward/total -> reward/reward_total, reward/progress -> reward/reward_progress,
        # and likewise centering / heading / terminal. penalty_* names are unchanged. Update any
        # saved W&B panels keyed on the old names.

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
            # Snapshot end-of-episode state for the logging block below. The env reset
            # itself stays in the HEAD block further down (routing-commands also resets
            # the gemini/BoN rollout chunks and the PID decoders) -- doing it here too
            # would reset CARLA twice per episode.
            done_info = dict(info)
            done_episode_return = float(episode_return)
            done_episode_steps = int(episode_steps)
            done_collision_count = int(episode_collision_count)
            done_collision_events = int(episode_collision_events)
            done_route = str(done_info.get("route", "?"))
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
            # ``_final_step_reward_log`` (surya/rl-token) emits exactly the ``rollout/final_step_*``
            # keys this block used to build inline — same names, same defaults — so both the DSRL
            # and residual paths stay on one namespace.
            rollout_log.update(_final_step_reward_log(done_info))
            # Bench2Drive end-of-route statistics. ``driving_score`` is the leaderboard's composed
            # score (route completion x infraction multiplier), computed by
            # ``statistics_manager.compute_route_statistics`` when the wrapper finalizes the route,
            # so it exists only on the terminal step — log it per episode, not per step.
            if "driving_score" in done_info:
                rollout_log["rollout/driving_score"] = float(done_info["driving_score"])
            if "route_progress_pct" in done_info:
                rollout_log["rollout/episode_route_progress_pct"] = float(
                    done_info["route_progress_pct"]
                )
                rollout_log["rollout/episode_route_completed"] = float(
                    float(done_info["route_progress_pct"]) >= 99.5
                )
            if "traffic_violation_count" in done_info:
                rollout_log["rollout/episode_traffic_violations"] = float(
                    done_info["traffic_violation_count"]
                )
            rollout_log["rollout/episode_termination_reason"] = str(
                done_info.get("termination_reason", "?")
            )
            if FLAGS.expert_recover_debug:
                rollout_log["rollout/vla_steps_budget"] = float(_vla_steps_budget)
            # NOTE: master's per-episode trajectory JSON is intentionally omitted here --
            # routing-commands has no _append_trajectory_step/episode_trajectory accumulation,
            # so the payload would always be empty. It keeps its own richer per-step capture
            # (steervla_actor._traj_capture -> traj_capture_*.pkl) instead.
            n_video_frames = len(episode_video_frames) + (1 if end_img is not None else 0)
            _maybe_log_episode_video(
                rollout_log,
                end_img if log_images else None,
                cot_obs_raw if log_images else None,
                final_reward=last_video_reward,
                final_critic_text=last_video_critic_text,
            )
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
                    # ``done_route`` is the env's own scenario name (info["route"]); fall back to
                    # the launched route if the leaderboard did not report one.
                    route_id=str(done_route if done_route != "?" else (FLAGS.route or "?")),
                )
            _sync_steervla_debug_noise_context(done_route)
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
            any_updates_on
            and (not FLAGS.expert_debug)
            and (not FLAGS.eval_only)
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
                if _online_training_mode == "sac_residual":
                    agent, update_info = agent.update_sac_residual(batch)
                elif _online_training_mode == "dagger_residual":
                    agent, update_info = agent.update_dagger_residual(batch)
                elif _online_training_mode == "dagger":
                    agent, update_info = agent.update_dagger(batch)
                elif use_vla_update:
                    # ``update_with_vla`` runs the DSRL critic/actor (RL) step and the HL VLM
                    # backbone update; each is gated independently.
                    if rl_updates_on or hl_updates_on:
                        agent, update_info = agent.update_with_vla(
                            batch, run_rl=rl_updates_on, run_hl=hl_updates_on, global_step=step,
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

        if agent is not None and any_updates_on and step % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, step)

        _save_steervla_ckpt(step)

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

    # Final export of the fine-tuned backbone at exit (in addition to the periodic saves above).
    _save_steervla_ckpt(FLAGS.online_steps, final=True)

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
    episode_traffic_violations = 0
    prev_traffic_violation_count = 0
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
                traffic_violation=(
                    (traffic_violation_count, episode_traffic_violations)
                    if traffic_violation_delta > 0 else None
                ),
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
                log["rollout/traffic_violation_events"] = float(traffic_violation_delta)
                log["rollout/traffic_violation_count"] = float(traffic_violation_count)
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
                episode_traffic_violations = 0
                prev_traffic_violation_count = 0
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
    exec_cfg = _steervla_action_execution_cfg(steervla_cfg)
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


def run_online_grpo(env, steervla_actor, vla_sample_fn, coach, config, obs_raw, *, raw_holder, exec_cfg):
    """GRPO on the SteerVLA high-level (CoT/subtask) policy, scored by a VLM critic (action expert frozen).

    At each scored decision state the actor samples ``group_size`` candidate CoTs for the current
    observation (one batched forward); the VLM critic scores each candidate subtask in [0, 1] given the
    current frame + env-reward context (speed / route progress / cumulative + last-step reward). The
    scores form the group: advantage ``A_k=(s_k-mean)/std``. The K candidate CoTs are recorded with
    those advantages, and the argmax-score candidate's action chunk is executed to advance the episode.
    Records accumulate across ``update_every_states`` scored states, then :meth:`update_hl_grpo` takes
    the (minibatched) policy-gradient step with a KL penalty to a frozen reference. Runs until the
    ``online_steps`` env-step budget is spent.
    """
    from coaches.cast_relabel import build_candidate_score_prompt, parse_candidate_scores

    grpo = config.get("grpo") or {}
    n_cand = max(2, int(grpo.get("group_size", 8)))
    score_temp = float(grpo.get("score_temperature", 1.0))
    beta_kl = float(grpo.get("beta_kl", 0.01))
    num_epochs = int(grpo.get("num_update_steps", 1))
    score_every = max(1, int(grpo.get("score_every", 1)))  # env steps between scored decision states
    update_every = max(1, int(grpo.get("update_every_states", 4)))  # scored states pooled per update
    adv_eps = float(grpo.get("advantage_eps", 1e-6))
    ckpt_every_steps = int(grpo.get("checkpoint_every_steps", 2000))
    ckpt_root = os.path.join(FLAGS.save_dir, "steervla_hl_ckpt")
    routing_command = str((config.get("steervla") or {}).get("routing_command", "Follow the route."))

    # Debug stop task: env reward -> -ego_speed (surfaced to the VLM) and the scoring objective flips to
    # "prefer stopping". inject_stop_candidate swaps one candidate for a canned stop CoT + zero chunk.
    debug_task = bool(grpo.get("debug_task", False))
    inject_stop = bool(grpo.get("inject_stop_candidate", False))
    stop_reasoning = str(grpo.get("stop_reasoning", "The vehicle must come to a stop."))
    stop_subtask = str(grpo.get("stop_subtask", "The vehicle comes to a complete stop and remains stationary."))
    score_objective = (
        "how much it reduces speed and brings the vehicle to a complete stop (reward = -speed; a fully "
        "stopped vehicle scores highest)" if debug_task else None
    )

    log_video = bool(config.get("log_episode_video", True))
    video_fps = float(config.get("episode_video_fps", 10.0))
    video_every = max(1, int(config.get("episode_video_every", 2)))
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))

    base_model = getattr(steervla_actor, "model", None)
    base_noise_dim = int(base_model.action_horizon) * int(base_model.action_dim) if base_model is not None else 40
    rng = jax.random.PRNGKey(FLAGS.seed)
    total_steps = int(FLAGS.online_steps)

    pooled: list[dict] = []
    pooled_adv: list[float] = []
    score_stats: list[np.ndarray] = []
    winner_scores: list[float] = []  # top score picked per scored state (for update metrics)
    inject_wins: list[float] = []    # 1.0 when the injected stop candidate won, else 0.0
    states_since_update = 0
    update_idx = 0

    obs = obs_raw
    info: dict[str, Any] = {}
    steervla_actor.reset_action_cache()
    episode_return, episode_steps, episode_count, last_reward = 0.0, 0, 0, 0.0
    episode_speed_sum = 0.0
    frames: list[np.ndarray] = []

    def _do_update(step_tag: int) -> None:
        nonlocal pooled, pooled_adv, score_stats, winner_scores, inject_wins, states_since_update, update_idx
        if not pooled:
            return
        uinfo = steervla_actor.update_hl_grpo(
            pooled, np.asarray(pooled_adv, dtype=np.float32),
            beta=beta_kl, num_epochs=num_epochs, global_step=step_tag,
        )
        all_scores = np.concatenate(score_stats) if score_stats else np.zeros((1,), dtype=np.float32)
        metrics = {f"grpo/{k}": float(v) for k, v in uinfo.items()}
        metrics.update({
            "grpo/update": float(update_idx),
            "grpo/beta_kl": beta_kl,
            "grpo/n_states": float(states_since_update),
            "grpo/score_mean": float(all_scores.mean()),
            "grpo/score_std": float(all_scores.std()),
            "grpo/winner_score_mean": float(np.mean(winner_scores)) if winner_scores else 0.0,
        })
        if inject_stop:
            metrics["grpo/inject_stop_won_frac"] = float(np.mean(inject_wins)) if inject_wins else 0.0
        wandb.log(metrics, step=step_tag)
        train_logger.log({k: v for k, v in metrics.items() if np.isscalar(v)}, step=step_tag)
        pooled, pooled_adv, score_stats, winner_scores, inject_wins, states_since_update = [], [], [], [], [], 0
        update_idx += 1

    for step in tqdm.tqdm(range(1, total_steps + 1), dynamic_ncols=True):
        raw_holder["obs"] = obs
        hud = None
        if step % score_every == 0:
            # Sample K candidate CoTs, VLM-score them, record them with advantages; execute the top one.
            rng, ck = jax.random.split(rng)
            cands = steervla_actor.sample_candidates(n_cand, temperature=score_temp, raw=obs, rng=ck)
            subtasks = list(cands["subtask_texts"])
            chunks = np.asarray(cands["actions"], dtype=np.float32).reshape(n_cand, -1)
            recs = steervla_actor.grpo_records_from_candidates(cands, obs)
            if inject_stop:
                # Swap candidate 0 for a canned stop: zero chunk (car brakes) + stop CoT record.
                chunks[0] = 0.0
                subtasks[0] = stop_subtask
                recs[0] = steervla_actor.grpo_stop_record(obs, reasoning=stop_reasoning, subtask=stop_subtask)
            context = {
                "routing_command": routing_command,
                "current_speed_mps": round(float(_ego_speed_mps(obs)), 2),
                "route_progress_pct": round(float(info.get("route_progress_pct", 0.0)), 2),
                "episode_return_so_far": round(episode_return, 3),
                "last_step_reward": round(last_reward, 3),
                "episode_step": episode_steps,
            }
            frame = _viz_image_from_raw(obs)
            if frame is None:
                frame = obs.get("image")
            text = coach.complete_image_text(
                frame, build_candidate_score_prompt(context, subtasks, objective=score_objective)
            )
            scores = np.asarray(parse_candidate_scores(text, num=n_cand), dtype=np.float32)
            adv = (scores - scores.mean()) / (scores.std() + adv_eps)
            pooled.extend(recs)
            pooled_adv.extend(float(a) for a in adv)
            score_stats.append(scores)
            winner = int(np.argmax(scores))
            winner_scores.append(float(scores[winner]))
            inject_wins.append(1.0 if (inject_stop and winner == 0) else 0.0)
            states_since_update += 1
            action = chunks[winner]
            # HUD: one line per candidate (score + subtask); the executed winner is drawn in green
            # ('>' marker + WIN tag), the injected stop candidate in cyan.
            hud_lines = [f"step={episode_steps} ret={episode_return:+.1f} win=c{winner}"]
            hud_colors: list[tuple] = [(255, 255, 255)]
            for i in range(n_cand):
                is_win = i == winner
                is_stop = inject_stop and i == 0
                hud_lines.append(
                    f"{'>' if is_win else ' '}c{i} s={scores[i]:.2f}"
                    f"{' STOP' if is_stop else ''} {subtasks[i][:44]}{'  <-WIN' if is_win else ''}"
                )
                hud_colors.append((0, 255, 0) if is_win else ((255, 255, 0) if is_stop else (255, 255, 255)))
            hud = {"lines": hud_lines, "line_colors": hud_colors}
        else:
            rng, nk = jax.random.split(rng)
            action = _base_chunk(
                vla_sample_fn, raw_holder, obs, jax.random.normal(nk, (1, base_noise_dim), dtype=jnp.float32)
            )

        next_obs, reward, terminated, truncated, info = env.step(action)
        speed = float(_ego_speed_mps(next_obs))
        # Debug stop task: optimize -speed instead of the env reward (surfaced to the VLM via context).
        reward_used = -speed if debug_task else float(reward)
        episode_return += reward_used
        episode_steps += 1
        episode_speed_sum += speed
        last_reward = reward_used
        done = bool(terminated or truncated)
        if log_video:
            _maybe_capture_frame(
                frames, next_obs, reward_used, episode_steps=episode_steps, done=done,
                log_video=log_video, video_every=video_every,
                action_flat=action, exec_cfg=exec_cfg, steervla_actor=steervla_actor, hud=hud,
            )
        obs = next_obs

        if done:
            ep_metrics: dict[str, Any] = {
                "grpo/episode": float(episode_count),
                "grpo/episode_return": episode_return,
                "grpo/episode_steps": float(episode_steps),
                "grpo/episode_mean_speed_mps": episode_speed_sum / max(episode_steps, 1),
            }
            if log_video and frames:
                video = _episode_video(frames, video_fps)
                if video is not None:
                    ep_metrics["grpo/rollout_video"] = video
            wandb.log(ep_metrics, step=step)
            print(
                f"[grpo] episode {episode_count}: return={episode_return:.3f} steps={episode_steps} "
                f"mean_speed={episode_speed_sum / max(episode_steps, 1):.3f} (env_step {step}/{total_steps})",
                flush=True,
            )
            episode_count += 1
            episode_return, episode_steps, last_reward, episode_speed_sum = 0.0, 0, 0.0, 0.0
            frames = []
            obs, info = env.reset(seed=FLAGS.seed + episode_count)
            steervla_actor.reset_action_cache()

        if states_since_update >= update_every:
            _do_update(step)
        if ckpt_every_steps > 0 and step % ckpt_every_steps == 0:
            steervla_actor.save_checkpoint(ckpt_root, step)

    _do_update(total_steps)
    try:
        steervla_actor.save_checkpoint(ckpt_root, total_steps)  # final export
    except Exception as exc:  # noqa: BLE001
        print(f"[grpo] final checkpoint save failed (non-fatal): {exc}", flush=True)


def _run_grpo_entry(config):
    """GRPO-on-HL-policy online CARLA entry (frozen SteerVLA base + action expert; only CoT trained)."""
    if FLAGS.route is None:
        raise ValueError("--route is required (see --list_routes=true).")

    steervla_cfg = config.get("steervla", None)
    if steervla_cfg is None or not steervla_cfg.get("enabled"):
        raise ValueError("config.steervla.enabled must be true: GRPO trains the HL policy of a base VLA.")
    if not bool(steervla_cfg.get("load_trainable_params", False)):
        raise ValueError("GRPO needs steervla.load_trainable_params=true to update the HL backbone.")

    wandb_mode = _resolve_wandb_mode()
    # Match the residual scheme: <route>-<mode>-sd###_<ts>, where <mode> encodes the GRPO variant.
    grpo_cfg = config.get("grpo") or {}
    mode_parts = ["grpo"]
    if bool(grpo_cfg.get("debug_task", False)):
        mode_parts.append("dbg")
    if bool(grpo_cfg.get("inject_stop_candidate", False)):
        mode_parts.append("inject")
    mode_tag = "-".join(mode_parts)
    route_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", str(FLAGS.route)).strip("-")
    exp_name = FLAGS.exp_name or f"{route_tag}-{mode_tag}-{get_exp_name(FLAGS.seed)}"
    wandb_id = re.sub(r"[^A-Za-z0-9_-]+", "-", exp_name).strip("-")[:128] or None
    setup_wandb(
        project="OGBench-CARLA-GRPO", group=FLAGS.run_group, name=exp_name, mode=wandb_mode,
        id=wandb_id, resume=("allow" if FLAGS.resume else None),
    )
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(get_flag_dict(), f)

    carla_yaml, extra_carla, exec_cfg = _resolve_carla_env_config(config)
    env = _make_carla_env(carla_yaml, FLAGS.route, extra_carla_config=extra_carla)
    try:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        obs, _info = env.reset(seed=FLAGS.seed)
        if not isinstance(obs, dict) or "state" not in obs or "image" not in obs:
            raise ValueError("CARLA env must return a Dict obs with 'state' and 'image'.")

        raw_holder: dict = {"obs": obs}
        from vlas.steervla import create_steervla_pi0_cot_sample_fn

        vla_sample_fn, steervla_actor = create_steervla_pi0_cot_sample_fn(
            steervla_cfg, raw_holder, training_gpu_rank=int(config.get("training_gpu_rank", -1))
        )
        _configure_jax_training_device(int(config.get("training_gpu_rank", -1)))

        # VLM critic: reuse the coach the DSRL/CAST path uses (provider/model from the vlm_coach block).
        from coaches.vlm_feedback import create_coach

        vlm_cfg = config.get("vlm_coach") or {}
        coach = create_coach(
            str(vlm_cfg.get("provider", "gemini")),
            model=str(vlm_cfg.get("gemini_model", "gemini-2.0-flash")),
        )

        run_online_grpo(
            env, steervla_actor, vla_sample_fn, coach, config, obs,
            raw_holder=raw_holder, exec_cfg=exec_cfg,
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
    if str(config.get("online_training_mode", "")).strip().lower() == "grpo_hl":
        _run_grpo_entry(config)
    elif str(config.get("agent_name", "")) == "sac_residual":
        _run_residual_entry(config)
    else:
        _run_dsrl_entry(config)


def _run_dsrl_entry(config):
    """DSRL / generic online CARLA RL entry (SteerVLA noise-actor, coaches, DAgger, VLM, ...)."""
    wandb_mode = _resolve_wandb_mode()

    def _slug(s: str) -> str:
        return re.sub(r"[^0-9A-Za-z._-]+", "-", str(s)).strip("-") or "na"

    def _coach_tag() -> str:
        """What form the coaching takes, e.g. ``cast-gemini-hl+good_critic-none``.

        Two independent sources can be active: the CAST-relabel session (window video -> VLM
        -> per-chunk GOOD/BAD credit -> corrected subtask, optionally persisted as high-level
        VLM-backbone training samples) and the DSRL critic's language label.
        """
        parts = []
        cast_cfg = config.get("cast_relabel", None)
        if cast_cfg is not None and bool(cast_cfg.get("enabled", False)):
            tag = f"cast-{_slug(cast_cfg.get('provider', 'vlm'))}"
            if bool(cast_cfg.get("store_hl_dataset", False)):
                tag += "-hl"
            if bool(cast_cfg.get("store_good_chunks", False)):
                tag += "+good"
            parts.append(tag)
        lf_cfg = config.get("language_feedback", None)
        if lf_cfg is not None:
            src = str(lf_cfg.get("source", ""))
            critic_mode = "vlm" if src == "vlm" else str(lf_cfg.get("expert_mode", "none"))
        else:
            critic_mode = str(config.get("critic_feedback_mode", "none"))
        parts.append(f"critic-{_slug(critic_mode)}")
        return "_".join(parts)

    def _updates_tag() -> str:
        """Which gradient updates are on, after the CLI flags override the config values."""

        def _on(flag_value, cfg_value) -> bool:
            return bool(cfg_value if flag_value is None else flag_value)

        if not _on(FLAGS.enable_updates, config.get("enable_updates", True)):
            return "noupd"
        kinds = [
            name
            for name, flag_value, cfg_value in (
                ("rl", FLAGS.enable_updates_rl, config.get("enable_updates_rl", True)),
                ("bc", FLAGS.enable_updates_bc, config.get("enable_updates_bc", True)),
                ("hl", FLAGS.enable_updates_bc_hl, config.get("enable_updates_bc_hl", True)),
            )
            if _on(flag_value, cfg_value)
        ]
        return "upd-" + "-".join(kinds) if kinds else "noupd"

    _agent_name = str(config.get("agent_name", "agent"))
    _route_name = str(FLAGS.route or "all-routes")
    # The run name carries what the run *is*, not just which agent class it used: an optional
    # CARLA_RUN_TAG (the experiment arm -- branch, ablation, ...), the coach form, which
    # updates are enabled, and the route. A wandb sidebar full of concurrent runs is then
    # readable without opening each one.
    _run_tag = os.environ.get("CARLA_RUN_TAG", "").strip()
    _exp_name_parts = [_slug(_run_tag)] if _run_tag else []
    _exp_name_parts.append(_slug(_agent_name))
    if _agent_name == "best_of_n":
        _exp_name_parts.append(f"n{int(config.get('best_of_n', 10))}")
    _exp_name_parts.extend(
        [_coach_tag(), _updates_tag(), _slug(_route_name), get_exp_name(FLAGS.seed)]
    )
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
        # best_of_n takes the same frozen-VLA wiring as dsrl (master 63e19f7).
        if config["agent_name"] in ("dsrl", "best_of_n"):
            vla_bundle = None
            if use_steervla_rollout:
                vla_bundle = _build_vla_sample_fn(
                    steervla_cfg,
                    raw_carla_holder,
                    training_gpu_rank=tr_rank,
                    noise_scale=float(config.get("noise_scale", 1.0)),
                )
            if vla_bundle is not None:
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
                create_kwargs["vla_sample_fn"] = vla_sample_fn
                url = steervla_cfg.get("actor_url") if steervla_cfg else None
                if not (url and str(url).strip()):
                    # create_kwargs["vla_train_state"] = steervla_actor.train_state
                    create_kwargs["openpi_train_config"] = steervla_actor.train_cfg
                    create_kwargs["steervla_actor"] = steervla_actor

            if config["agent_name"] == "best_of_n":
                # Subtask -> critic language label uses SigLIP text features (shared encoder).
                create_kwargs["siglip_encoder"] = siglip_encoder

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


if __name__ == "__main__":
    app.run(main)
