"""Residual SAC training on CARLA Bench2Drive using a frozen SimLingo base policy.

Runs under the SimLingo Python 3.8 conda environment.  The CARLA environment
is managed by a separate process (carla_env_server.py) running in the OGBench
Python 3.11 uv environment.  Communication is via newline-delimited JSON over
stdin/stdout of the subprocess.

The frozen SimLingo VLM provides:
  - base_action (accel, steer): from PID control over predicted waypoints
  - vlm_features (896,): mean-pooled last-layer hidden states for driving tokens

A small PyTorch residual SAC actor/critic trains on top of these features:
  final_action = clip(base_action + res_scale * residual_action, -1, 1)

Usage::

    # Eval-only (verify base policy matches expected scores)
    /home/celinet/miniconda3/envs/simlingo/bin/python impls/main_carla_simlingo.py \\
        --simlingo_checkpoint=/home/celinet/simlingo_checkpoints/simlingo/checkpoints/epoch=013.ckpt \\
        --route=bench2drive_007 \\
        --eval_only

    # Residual SAC training
    /home/celinet/miniconda3/envs/simlingo/bin/python impls/main_carla_simlingo.py \\
        --simlingo_checkpoint=/home/celinet/simlingo_checkpoints/simlingo/checkpoints/epoch=013.ckpt \\
        --route=parking-cut-in-001 \\
        --total_steps=10000
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_IMPLS_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _IMPLS_ROOT.parent
_REBUTTAL_ROOT = _REPO_ROOT / "simlingo-rebuttal"

for _p in [str(_IMPLS_ROOT), str(_REPO_ROOT), str(_REBUTTAL_ROOT), str(_REBUTTAL_ROOT / "team_code")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from absl import app, flags

import wandb
from utils.log_utils import get_exp_name, setup_wandb  # type: ignore

# Keep this file importable in the SimLingo conda env.  Importing
# ogbench.carla.carla_utils here pulls in Bench2Drive leaderboard modules that
# only the env-server subprocess needs.
_STATE_DIM = 25
_EGO_STATE_IDX_SPEED = 15
_EGO_STATE_IDX_THROTTLE = 16
_EGO_STATE_IDX_STEER = 17
_EGO_STATE_IDX_BRAKE = 18


def _load_pretrained_critic_simlingo(agent, checkpoint_path: str) -> None:
    """Inject pretrained critic weights from a pretrain_critic_simlingo.py checkpoint.

    Only loads critic and critic_target; skips actor, optimizer, state_normalizer.

    When the online agent uses state_dim > 0 (include_ego_state=True) but the
    pretrained checkpoint used state_dim=0, exact load_state_dict would fail due
    to first-layer shape mismatch.  In this case we do a partial injection:
    the vlm-features + action columns of the first layer are copied from the
    pretrained checkpoint; the state columns remain at random init.

    For all layers beyond the first (hidden layers, output) shapes are always
    identical (same hidden_dims), so they are loaded exactly.

    The pretrained checkpoint must have been created with the same vlm_feature_dim,
    action_dim, and hidden_dims as the online agent.
    """
    import copy
    import torch

    state = torch.load(checkpoint_path, map_location=agent.device)
    if "critic" not in state:
        raise KeyError(f"Checkpoint at {checkpoint_path} has no 'critic' key; got {list(state.keys())}")

    def _inject_critic(online_critic, pretrained_sd):
        online_sd = online_critic.state_dict()
        if all(v.shape == online_sd[k].shape for k, v in pretrained_sd.items() if k in online_sd):
            online_critic.load_state_dict(pretrained_sd, strict=True)
            return "exact"
        # Shapes differ (state_dim mismatch in first layer).  Inject shared columns.
        new_sd = copy.deepcopy(online_sd)
        for key in pretrained_sd:
            if key not in online_sd:
                continue
            pt_w, on_w = pretrained_sd[key], online_sd[key]
            if pt_w.shape == on_w.shape:
                new_sd[key] = pt_w.clone()
            elif pt_w.ndim == 2 and on_w.ndim == 2 and pt_w.shape[0] == on_w.shape[0]:
                # First-layer weight: on_w is (out, in_online), pt_w is (out, in_pretrain).
                # in_pretrain < in_online; copy first in_pretrain columns.
                cols = pt_w.shape[1]
                new_sd[key] = on_w.clone()
                new_sd[key][:, :cols] = pt_w
            # bias and other tensors: skip if shapes differ (shouldn't happen)
        online_critic.load_state_dict(new_sd, strict=True)
        return "partial (first-layer column injection)"

    mode = _inject_critic(agent.critic, state["critic"])
    print(f"[main_carla_simlingo] Pretrained critic loaded from {checkpoint_path} ({mode})")

    pretrained_target_sd = state.get("critic_target", state["critic"])
    mode_t = _inject_critic(agent.critic_target, pretrained_target_sd)
    print(f"[main_carla_simlingo] Pretrained critic_target loaded ({mode_t})")


def _ego_drive_metrics_from_state_vec(state: Any) -> Dict[str, float]:
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    if s.size < _STATE_DIM:
        return {
            "ego_speed": 0.0,
            "control_throttle": 0.0,
            "control_steer": 0.0,
            "control_brake": 0.0,
        }
    return {
        "ego_speed": float(s[_EGO_STATE_IDX_SPEED]),
        "control_throttle": float(s[_EGO_STATE_IDX_THROTTLE]),
        "control_steer": float(s[_EGO_STATE_IDX_STEER]),
        "control_brake": float(s[_EGO_STATE_IDX_BRAKE]),
    }

# Indices into the 19-dim ego-state slice (obs["state"][6:], orientation dropped).
# Layout: vel(3), avel(3), acc(3), speed(1), throttle(1), steer(1), brake(1), routing_onehot(6)
_NORM_IDX_SPEED      = 9   # speed        (m/s)
_NORM_IDX_VEL_X      = 0   # longitudinal velocity (m/s)
_NORM_IDX_AVEL_Z     = 5   # yaw rate     (deg/s)
_NORM_IDX_STEER_CTRL = 11  # last-applied steer


def _state_norm_wandb_log(agent: Any) -> Dict[str, float]:
    """Return a flat dict of state-normalizer diagnostics for wandb.

    Logs the count (to see when freezing occurs), aggregate std statistics
    across all 22 dims, and the std for the two highest-variance dims (yaw
    and speed) that motivated the normalizer in the first place.
    """
    n = getattr(agent, "state_normalizer", None)
    if n is None:
        return {}
    std = n.std  # (22,) float32
    return {
        "state_norm/n_samples":      float(n._count),
        "state_norm/std_mean":       float(std.mean()),
        "state_norm/std_min":        float(std.min()),
        "state_norm/std_max":        float(std.max()),
        "state_norm/std_vel_x":      float(std[_NORM_IDX_VEL_X]),
        "state_norm/std_avel_z":     float(std[_NORM_IDX_AVEL_Z]),
        "state_norm/std_speed":      float(std[_NORM_IDX_SPEED]),
        "state_norm/std_steer_ctrl": float(std[_NORM_IDX_STEER_CTRL]),
    }


FLAGS = flags.FLAGS

flags.DEFINE_string("simlingo_checkpoint", None, "Path to SimLingo epoch=013.ckpt directory.")
flags.DEFINE_enum("policy_mode", "single", ["single", "hierarchical"], "Policy mode: single SimLingo or HL+LL hierarchical SimLingo.")
flags.DEFINE_string("high_level_checkpoint", None, "Path to high-level SimLingo checkpoint for hierarchical mode.")
flags.DEFINE_string("low_level_checkpoint", None, "Path to low-level SimLingo checkpoint for hierarchical mode.")
flags.DEFINE_string("high_level_hydra_config", None, "Hydra config for high-level checkpoint if not stored beside checkpoint.")
flags.DEFINE_string("low_level_hydra_config", None, "Hydra config for low-level checkpoint if not stored beside checkpoint.")
flags.DEFINE_string("hierarchical_source_root", "", "Optional legacy source tree override for both hierarchical SimLingo models.")
flags.DEFINE_string("high_level_source_root", "/scratch/current/celinet/simlingo-steervla", "Source tree used to instantiate the high-level SimLingo model.")
flags.DEFINE_string("low_level_source_root", "/scratch/current/celinet/simlingo-tian", "Source tree used to instantiate the low-level SimLingo model.")
flags.DEFINE_string("route", None, "Bench2Drive route (scenario name, file basename, or route id).")
flags.DEFINE_bool("eval_only", False, "Run base policy only (no residual training).")
flags.DEFINE_integer("total_steps", 10_000, "Total environment steps for training.")
flags.DEFINE_integer("warmup_steps", 500, "Steps collecting data before SAC updates begin.")
flags.DEFINE_integer("learning_starts", 500, "Buffer size threshold before updates begin.")
flags.DEFINE_integer("updates_per_step", 10, "SAC gradient updates per env step / UTD ratio.")
flags.DEFINE_integer("batch_size", 256, "SAC mini-batch size.")
flags.DEFINE_integer("buffer_capacity", 10_000, "Replay buffer capacity.")
flags.DEFINE_float("res_scale", 0.1, "Legacy: sets both res_scale_accel and res_scale_steer when non-zero.")
flags.DEFINE_float("res_scale_accel", 2.0, "Residual scale for accel dimension.")
flags.DEFINE_float("res_scale_steer", 0.6, "Residual scale for steer dimension.")
flags.DEFINE_integer("residual_clip_schedule_steps", 0,
                     "Steps after warmup over which the residual clip limit ramps linearly from 0 to 1. "
                     "0 = no schedule (full [-1, 1] range immediately after warmup).")
flags.DEFINE_float("gamma", 0.97, "Discount factor.")
flags.DEFINE_float("tau", 0.01, "Target network soft-update coefficient.")
flags.DEFINE_float("actor_lr", 1e-4, "Actor learning rate.")
flags.DEFINE_float("critic_lr", 1e-4, "Critic learning rate.")
flags.DEFINE_float("actor_l2_reg", 0.0, "L2 regularization coefficient for the residual actor parameters.")
flags.DEFINE_string("save_dir", "./logs/simlingo_residual", "Directory for checkpoints and logs.")
flags.DEFINE_integer("save_interval", 2000, "Save residual SAC checkpoint every N steps.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("chunk_size", 10, "Waypoints to execute per VLM call (1–10). 1 = VLM every tick; 10 = full predicted chunk.")
flags.DEFINE_enum("obs_mode", "vlm_hidden", ["vlm_hidden", "encoder"],
                  "Observation for the residual SAC. 'vlm_hidden': 896-dim mean-pooled LLM last hidden states "
                  "(current default). 'encoder': 1792-dim = mean-pool(vision_encoder_tokens) ++ mean-pool(prompt_embeds).")
flags.DEFINE_bool("save_video", True, "Write per-episode mp4 videos of the simlingo camera feed.")
flags.DEFINE_string("carla_config", None, "Path to carla_config.yaml.")
flags.DEFINE_string("device", "cuda", "Torch device for SimLingo and residual SAC.")
flags.DEFINE_integer("gpu_rank", 0, "CARLA rendering GPU rank.")
# conda env name for the carla_env_server.py subprocess (must have carla 0.9.15 installed).
flags.DEFINE_string("server_conda_env", "simlingo", "conda env for the CARLA env server process.")
flags.DEFINE_string("carla_root", os.environ.get("CARLA_ROOT", "/home/celinet/fail2drive/f2d_carla"),
                    "CARLA root dir (forwarded to env server as CARLA_ROOT). "
                    "Defaults to $CARLA_ROOT env var, then ~/fail2drive/f2d_carla.")
flags.DEFINE_bool("include_ego_state", True,
                  "Include the 25-dim ego state vector as explicit input to the actor and critic.")
flags.DEFINE_bool("debug_neg_speed_reward", False,
                  "Debug: replace env reward with -speed (m/s). SAC should learn to slow the car.")
flags.DEFINE_float("debug_target_speed_reward", 0.0,
                   "Debug: replace env reward with -|speed - target| (m/s). 0 = disabled. "
                   "E.g. 5.0 → SAC should learn to drive at 5 m/s.")
flags.DEFINE_bool("expert_debug", False,
                  "Debug: drive with the CARLA expert action instead of base+residual (dagger_residual only).")
flags.DEFINE_string("expert_checkpoint", None,
                    "Path to a saved ResidualSACAgent .pt checkpoint to use as the expert when --expert_debug is set. "
                    "When provided, loads the full VLA model and uses the checkpoint's deterministic policy as the expert "
                    "instead of the PDM-Lite autopilot waypoint decoder.")
flags.DEFINE_string(
    "pretrained_critic", None,
    "Path to a pretrained ResidualSACAgent .pt checkpoint from pretrain_critic_simlingo.py. "
    "Injects only the critic and critic_target params into the freshly created agent. "
    "Actor, optimizer, and state_normalizer are NOT loaded, so a state_dim mismatch between "
    "pretraining (state_dim=0) and online (state_dim=EGO_STATE_DIM) is safe.",
)
flags.DEFINE_bool("terminate_on_infraction", False,
                  "Terminate the episode immediately when a collision, traffic violation, or off-route event occurs.")
flags.DEFINE_bool("expert_recover_debug", False,
                  "Debug: run SimLingo for a random [70,200] ticks per episode, then switch to CARLA expert.")
flags.DEFINE_bool("use_expert_in_critic", False,
                  "Feed the expert planner's (accel, steer) action as additional input to the SAC critic. "
                  "The expert action is read from obs['expert_action'] (provided by the CARLA env server).")
flags.DEFINE_bool("use_language_bow_critic", False,
                  "Feed a programmatic scene-grounded language BoW label as additional critic input. "
                  "Computed each step from the delta between the executed action and the expert planner "
                  "action, grounded in nearby scene actors visible in the ego vehicle's FoV. "
                  "Uses SCENE_DELTA_VOCAB (26 words + 1 validity flag = 27 dims).")
flags.DEFINE_bool("log_q_expert_diff", False,
                  "Log Q(expert action) - Q(buffer action) at each SAC update. "
                  "For use_expert_in_critic, uses the stored expert_actions. "
                  "For use_language_bow_critic / use_gemini_coach (delta), stores the expert action "
                  "in a separate log-only buffer field.")
flags.DEFINE_bool("use_noise_critic", False,
                  "Debug: feed i.i.d. Gaussian noise (same dim as language_bow: 27) as the critic's "
                  "coach-label input each step. Use as a sanity-check ablation against language_bow — "
                  "if the noise critic improves over none, the benefit is from extra capacity, not language.")
flags.DEFINE_string("run_group", "Debug", "W&B run group.")
flags.DEFINE_string("wandb_project", "OGBench-CARLA-SimLingo", "W&B project name.")
flags.DEFINE_string("wandb_run_name", None, "W&B run name. If None, auto-generates from route + seed.")
flags.DEFINE_enum("wandb_mode", "online", ["online", "offline", "disabled"], "W&B logging mode.")
flags.DEFINE_integer("log_interval", 1, "Log training metrics to W&B every N episodes.")
flags.DEFINE_integer("video_log_interval", 5, "Upload episode video to W&B every N episodes (0=never).")
flags.DEFINE_integer("eval_episodes", 2, "Number of episodes to run in eval-only mode.")
flags.DEFINE_integer("eval_step_limit", 4000, "Maximum CARLA ticks per eval episode.")
flags.DEFINE_enum("training_mode", "sac_residual", ["sac_residual", "dagger_residual"],
                  "Training mode: sac_residual (RL with env reward) or "
                  "dagger_residual (BC with expert planner labels).")

# ── Gemini coach flags ─────────────────────────────────────────────────────────
flags.DEFINE_bool("use_gemini_coach", False,
                  "Enable Gemini VLM coach: after each episode Gemini reviews the rollout "
                  "video and assigns per-step delta-commentary BoW labels that are "
                  "backfilled into the replay buffer to condition the critic.")
flags.DEFINE_string("gemini_api_key", "",
                    "Gemini API key. Falls back to GEMINI_API_KEY env var.")
flags.DEFINE_string("gemini_model", "gemini-2.0-flash",
                    "Gemini model name for VLM coaching.")
flags.DEFINE_integer("coach_action_chunk_steps", 10,
                     "Number of SAC global_steps per coach action chunk. "
                     "Each chunk gets one lateral/longitudinal feedback label. "
                     "Default: 10 steps ≈ 2.5 s of driving.")
flags.DEFINE_integer("coach_query_freq", 0,
                     "Query the Gemini coach every N episode steps mid-episode "
                     "(0 = episode end only). E.g. 50 = query after every 50 steps "
                     "as well as at episode end. Each query backfills only the new "
                     "transitions since the previous query.")
flags.DEFINE_enum("coach_label_mode", "bow", ["bow", "vlm_embed", "vlm_embed_raw"],
                  "How to encode Gemini coach feedback for the critic. "
                  "'bow': 17-dim delta-commentary bag-of-words (default). "
                  "'vlm_embed': 896-dim VLM embedding of the structured "
                  "lateral/longitudinal/detail phrase from Gemini call 2. "
                  "'vlm_embed_raw': 896-dim VLM embedding of the raw "
                  "description+correction text from Gemini call 1 (no schema compression).")
flags.DEFINE_bool("coach_embed_plot", False,
                  "After each episode backfill, run PCA on the coach label embeddings "
                  "and log a 2-D scatter plot to wandb (coach/embedding_pca). "
                  "Only meaningful when coach_label_mode is vlm_embed or vlm_embed_raw.")

# ── Observation histogram debug flags ─────────────────────────────────────────
flags.DEFINE_bool("debug_obs_hist", False,
                  "Collect obs samples for N steps then plot histograms of embedding "
                  "elements vs everything else (ego state, actions, expert). Exits after "
                  "the plot is saved.")
flags.DEFINE_integer("debug_obs_hist_steps", 2000,
                     "Number of steps to collect before plotting observation histograms. "
                     "Only used when --debug_obs_hist is set.")


# ── Video overlay ─────────────────────────────────────────────────────────────

_VIDEO_PANEL_H = 130


def _annotate_frame(
    image_rgb: np.ndarray,
    simlingo_base,
    target_points: Optional[np.ndarray],
    current_speed: float,
    base_action: np.ndarray,
    residual_action: Optional[np.ndarray] = None,
    *,
    reward_value: Optional[float] = None,
    env_reward_value: Optional[float] = None,
    info: Optional[Dict[str, Any]] = None,
    collision_events: int = 0,
    expert_waypoints: Optional[np.ndarray] = None,
    expert_action_2d: Optional[np.ndarray] = None,
    language_feedback: Optional[str] = None,
) -> np.ndarray:
    """Draw projected waypoints plus a black text panel like main_carla.py.

    Green dots   = speed waypoints (from speed head)
    Red dots     = route waypoints (from route head)
    Blue dots    = GPS target waypoints (from obs)
    Yellow dots  = expert route waypoints (dagger mode only)
    """
    frame = np.asarray(image_rgb)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    frame = np.ascontiguousarray(frame)
    H, W = frame.shape[:2]

    try:
        from PIL import Image as _PIL_Image, ImageDraw as _ImageDraw
        from team_code.simlingo_utils import project_points, get_camera_intrinsics  # type: ignore

        K = get_camera_intrinsics(W, H, 110).numpy()

        pil_img = _PIL_Image.fromarray(frame).convert("RGBA")
        draw = _ImageDraw.Draw(pil_img)

        def _draw_pts(waypoints_ego, color, r=4):
            if waypoints_ego is None or len(waypoints_ego) == 0:
                return
            pts = project_points(waypoints_ego, K)
            for p in pts:
                x, y = int(p[0]), int(p[1])
                if 0 <= x < W and 0 <= y < H:
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

        _draw_pts(simlingo_base._last_speed_wps, (0, 255, 0, 255), r=3)   # green
        _draw_pts(simlingo_base._last_route, (255, 0, 0, 255), r=2)        # red
        if target_points is not None:
            _draw_pts(target_points, (0, 0, 255, 255), r=5)                # blue
        if expert_waypoints is not None:
            _draw_pts(expert_waypoints, (255, 220, 0, 255), r=3)           # yellow
        frame = np.asarray(pil_img.convert("RGB"))
    except Exception:
        pass

    info = info or {}
    collision_count = int(info.get("collision_count", 0))
    collision_contact = bool(info.get("collision_contact_active", False))
    # Only show COLLISION banner when a new event fired this step or raw contact is active.
    # Do NOT use cumulative collision_count — that would make the banner persist for the
    # rest of the episode after any prior collision, masking steps where no penalty applies.
    collision_now = bool(collision_events > 0 or collision_contact)
    residual = residual_action if residual_action is not None else np.zeros(2, dtype=np.float32)
    _rs = np.array([FLAGS.res_scale_accel, FLAGS.res_scale_steer], dtype=np.float32)
    final_action = np.clip(base_action + _rs * residual, -1.0, 1.0)
    train_reward = "?" if reward_value is None else f"{reward_value:+.3f}"
    env_reward = "?" if env_reward_value is None else f"{env_reward_value:+.3f}"
    prompt = str(getattr(simlingo_base, "_last_prompt_text", "") or "")
    language = str(getattr(simlingo_base, "_last_language_text", "") or "")
    meta_action = str(getattr(simlingo_base, "_last_meta_action", "") or "")

    def _clip_text(txt: str, max_chars: int = 142) -> str:
        txt = " ".join(str(txt).split())
        return txt if len(txt) <= max_chars else (txt[: max_chars - 3] + "...")

    annotated = np.zeros((H + _VIDEO_PANEL_H, W, 3), dtype=np.uint8)
    annotated[:H, :, :] = frame
    try:
        import cv2  # type: ignore

        cv2.line(annotated, (0, H), (W - 1, H), (255, 255, 255), 1)
        if collision_now:
            collision_pen = float(info.get("penalty_collision", 0.0))
            label = f"COLLISION c={collision_count} Δ={collision_events} pen={collision_pen:+.0f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            x1 = W - 8
            x0 = max(8, x1 - tw - 12)
            y0 = 8
            y1 = y0 + th + baseline + 10
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (220, 0, 0), thickness=-1)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 255, 255), thickness=1)
            cv2.putText(annotated, label, (x0 + 6, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        pen_coll = float(info.get("penalty_collision", 0.0))
        pen_route = float(info.get("penalty_outside_route", 0.0))
        pen_traf = float(info.get("penalty_traffic_violation", 0.0))
        pen_crash = float(info.get("penalty_crash_stuck", 0.0))
        pen_term = float(info.get("reward_terminal", 0.0))
        contact = bool(info.get("collision_contact_active", False))
        traf_count = int(info.get("traffic_violation_count", 0))

        if expert_action_2d is not None:
            expert_str = f"expert=({expert_action_2d[0]:+.2f},{expert_action_2d[1]:+.3f})"
        else:
            expert_str = ""

        lang_fb_str = f"LangFB: {_clip_text(language_feedback)}" if language_feedback else ""
        lines = [
            f"Reward train={train_reward} env={env_reward} | speed={current_speed:.2f} m/s | collision={'YES' if collision_now else 'no'} c={collision_count} e={collision_events}",
            f"Action base=({base_action[0]:+.2f},{base_action[1]:+.3f}) residual=({residual[0]:+.2f},{residual[1]:+.3f}) final=({final_action[0]:+.2f},{final_action[1]:+.3f}){('  ' + expert_str) if expert_str else ''}",
            lang_fb_str if lang_fb_str else f"Meta-action: {_clip_text(meta_action) if meta_action else '(none)'}",
            f"Prompt: {_clip_text(prompt)}",
            f"Reasoning: {_clip_text(language) if language else '(no language output)'}",
            f"Pen: coll={pen_coll:+.1f}{'(bb)' if contact else ''}  route={pen_route:+.1f}  traf={pen_traf:+.1f}(n={traf_count})  crash={pen_crash:+.1f}  term={pen_term:+.1f}",
        ]
        y = H + 15
        for line in lines:
            cv2.putText(annotated, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
            y += 17
    except Exception:
        pass

    return annotated


# ── Environment proxy ─────────────────────────────────────────────────────────

class CarlaEnvProxy:
    """Communicates with carla_env_server.py over JSON/stdio subprocess."""

    def __init__(
        self,
        route: str,
        carla_config: Optional[str],
        gpu_rank: int,
        server_conda_env: str = "simlingo",
        carla_root: str = "/home/celinet/VLA_driving/software",
        terminate_on_infraction: bool = False,
    ):
        server_script = str(_IMPLS_ROOT / "carla_env_server.py")
        # Launch server in the same simlingo conda env (Python 3.8 + carla 0.9.15).
        # conda run sets up the full conda environment correctly.
        conda_root = os.environ.get("CONDA_ROOT",
                                    os.path.expanduser("~/miniconda3"))
        cmd = [
            "conda", "run", "-n", server_conda_env, "--no-capture-output",
            "python", server_script,
            f"--route={route}",
            f"--gpu_rank={gpu_rank}",
            f"--carla_root={carla_root}",
        ]
        if carla_config:
            cmd.append(f"--carla_config={carla_config}")
        if terminate_on_infraction:
            cmd.append("--terminate_on_infraction")

        print(f"[CarlaEnvProxy] Launching server: {' '.join(cmd)}", flush=True)
        child_env = os.environ.copy()
        child_env["CARLA_ROOT"] = carla_root
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(_REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            env=child_env,
        )
        # Wait for {"ready": true}
        ready_line = self._readline()
        ready = json.loads(ready_line)
        if not ready.get("ready"):
            raise RuntimeError(f"Server did not send ready signal: {ready_line!r}")
        print("[CarlaEnvProxy] Server ready.", flush=True)

    def _readline(self) -> str:
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("Server process closed stdout unexpectedly.")
            line = line.strip()
            if line:
                return line

    def _send(self, msg: dict) -> None:
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._send({"reset": True})
        resp = json.loads(self._readline())
        obs = self._wire_to_obs(resp["obs"])
        return obs, resp.get("info", {})

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._send({"action": action.tolist()})
        resp = json.loads(self._readline())
        obs = self._wire_to_obs(resp["obs"])
        return obs, resp["reward"], resp["terminated"], resp["truncated"], resp.get("info", {})

    def step_expert(self) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Step the env using PDM-Lite's direct control output (re-plans every tick)."""
        self._send({"expert_step": True})
        resp = json.loads(self._readline())
        obs = self._wire_to_obs(resp["obs"])
        return obs, resp["reward"], resp["terminated"], resp["truncated"], resp.get("info", {})

    @staticmethod
    def _wire_to_obs(wire: Dict) -> Dict[str, Any]:
        img_bytes = base64.b64decode(wire["simlingo_image_b64"])
        img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(wire["simlingo_image_shape"])
        tp_raw = wire.get("target_points", [[0.0, 0.0], [0.0, 0.0]])
        expert_raw = wire.get("expert_action")
        obs = {
            "state": np.array(wire["state"], dtype=np.float32),
            "simlingo_image": img,
            "routing_command": wire["routing_command"],
            "target_points": np.array(tp_raw, dtype=np.float32),  # (2, 2) ego-frame
        }
        if expert_raw is not None:
            obs["expert_action"] = np.array(expert_raw, dtype=np.float32)
        obs["scene_context"] = wire.get("scene_context") or {}
        return obs

    def close(self):
        try:
            self._send({"shutdown": True})
        except Exception:
            pass
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        self._proc.wait(timeout=10)


# ── Metric helpers ────────────────────────────────────────────────────────────

def _format_route_metrics(info: Dict[str, Any]) -> str:
    return (
        f"  collision_count={info.get('collision_count', 0)}"
        f"  outside_route={info.get('outside_route_value', 0.0):.3f}"
        f"  success={info.get('success', False)}"
    )


# ── Eval episode ──────────────────────────────────────────────────────────────

def run_eval_episode(env: CarlaEnvProxy, simlingo_base, step_limit: int = 4000) -> Dict[str, Any]:
    """Roll out base policy only; return episode stats."""
    obs, _ = env.reset()
    simlingo_base.reset_pid()

    episode_reward = 0.0
    steps = 0
    info: Dict[str, Any] = {}

    for _ in range(step_limit):
        base_action, _ = simlingo_base.get_action_and_features(
            simlingo_image=obs["simlingo_image"],
            ego_state=obs["state"],
            target_points=obs["target_points"],
            routing_command=obs.get("routing_command", ""),
        )
        obs, reward, terminated, truncated, info = env.step(base_action)
        episode_reward += reward
        steps += 1
        if terminated or truncated:
            break

    return {
        "episode_reward": episode_reward,
        "steps": steps,
        "success": info.get("success", False),
        "collision_count": info.get("collision_count", 0),
        "outside_route": info.get("outside_route_value", 0.0),
        "route": FLAGS.route,
    }


# ── DAgger expert action helpers ──────────────────────────────────────────────

# Each waypoint in the expert chunk covers 5 CARLA ticks at 20 Hz = 0.25 s.
_EXPERT_DT = 5.0 / 20.0


def _expert_action_to_accel_steer(
    expert_action_40d: np.ndarray,
    simlingo_base,
    current_speed: float,
) -> np.ndarray:
    """Convert a 40-D expert chunk to a single-tick (accel, steer) action.

    expert_action_40d: (40,) = 10 steps × [dx_speed, dy_speed, dx_route, dy_route].
    Each (dx, dy) is a displacement over dt = 0.25 s at the expert's target speed.
    Returns a (2,) float32 [accel, steer] in [-1, 1].

    Expert steer is computed from the expert's route waypoints via SimLingo's lateral
    PID. The PID window and cached route are saved before and restored after so this
    call has no side effects on simlingo_base state.
    """
    from vlas.simlingo_base import _interpolate_waypoints  # type: ignore

    chunk = np.asarray(expert_action_40d, dtype=np.float32).reshape(10, 4)

    # ── Expert accel from speed chunk first-step displacement ─────────────────
    expert_target_speed = float(np.linalg.norm(chunk[0, :2])) / max(_EXPERT_DT, 1e-6)
    expert_accel = simlingo_base.accel_for_desired_speed(expert_target_speed, current_speed)

    # ── Expert steer from route chunk, without mutating PID state ─────────────
    # _lateral_control and _turn_controller live on SimLingoBase, not on
    # HierarchicalSimLingoPolicy. Unwrap to the underlying base (`.low` for
    # hierarchical, self for single) before touching private attributes.
    _base = getattr(simlingo_base, "low", simlingo_base)
    tc = _base._turn_controller
    saved_route_interp = _base._last_route_interp
    tc.save_state()
    try:
        expert_route_wps = np.cumsum(chunk[:, 2:4], axis=0)  # (10, 2)
        expert_steer = _base._lateral_control(expert_route_wps, current_speed)
    except Exception:
        expert_steer = simlingo_base.steer_for_speed(current_speed)
    finally:
        tc.load_state()
        _base._last_route_interp = saved_route_interp

    return np.array([expert_accel, expert_steer], dtype=np.float32)


# ── Residual clip schedule ────────────────────────────────────────────────────

def _residual_clip_limit(global_step: int, warmup_steps: int, schedule_steps: int) -> float:
    """Linear ramp: 0 at end of warmup → 1 after schedule_steps post-warmup steps."""
    if schedule_steps <= 0:
        return 1.0
    post_warmup = global_step - warmup_steps
    if post_warmup <= 0:
        return 0.0
    return min(1.0, post_warmup / schedule_steps)


# ── Observation histogram ─────────────────────────────────────────────────────

def _plot_obs_histograms(
    embed_samples: List[np.ndarray],
    other_samples: Dict[str, List[np.ndarray]],
    save_dir: Path,
    step: int,
    obs_mode: str,
) -> str:
    """Plot per-component element histograms of obs and save as PNG.

    embed_samples: list of (D_embed,) arrays, one per step
    other_samples: ordered dict of {label: list of (D,) arrays}
    Returns path to the saved PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("VLM embedding", embed_samples, "steelblue")] + [
        (label, samples, color)
        for (label, samples), color in zip(
            other_samples.items(),
            ["darkorange", "seagreen", "mediumpurple", "crimson", "goldenrod"],
        )
        if samples
    ]
    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for i, (label, samples, color) in enumerate(panels):
        flat = np.concatenate([s.ravel() for s in samples])
        n_bins = min(200, max(50, int(np.sqrt(len(flat)))))
        axes_flat[i].hist(flat, bins=n_bins, color=color, alpha=0.85, edgecolor="none")
        dim_str = f"D={samples[0].shape[0]}" if samples else "D=?"
        extra = f"  ({obs_mode})" if i == 0 else ""
        axes_flat[i].set_title(
            f"{label}{extra}\n"
            f"n_steps={len(samples)}  {dim_str}\n"
            f"mean={flat.mean():.4f}  std={flat.std():.4f}  [{flat.min():.3f}, {flat.max():.3f}]",
            fontsize=9,
        )
        axes_flat[i].set_xlabel("element value")
        axes_flat[i].set_ylabel("count")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Observation element distributions  (global_step={step})", fontsize=11)
    fig.tight_layout()

    out_path = save_dir / f"obs_histograms_step{step:06d}.png"
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    return str(out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(_argv):
    np.random.seed(FLAGS.seed)

    _res_scale_vec = np.array([FLAGS.res_scale_accel, FLAGS.res_scale_steer], dtype=np.float32)

    if FLAGS.policy_mode == "single" and FLAGS.simlingo_checkpoint is None:
        raise ValueError("--simlingo_checkpoint is required.")
    if FLAGS.policy_mode == "hierarchical" and (FLAGS.high_level_checkpoint is None or FLAGS.low_level_checkpoint is None):
        raise ValueError("--high_level_checkpoint and --low_level_checkpoint are required for hierarchical mode.")
    if FLAGS.route is None:
        raise ValueError("--route is required.")

    # ── W&B setup ─────────────────────────────────────────────────────────────
    exp_name = get_exp_name(FLAGS.seed)
    save_dir_base = FLAGS.save_dir
    setup_wandb(
        project=FLAGS.wandb_project,
        group=FLAGS.run_group,
        name=FLAGS.wandb_run_name if FLAGS.wandb_run_name else f"{FLAGS.route}_{exp_name}",
        mode=FLAGS.wandb_mode,
    )
    FLAGS.save_dir = str(Path(save_dir_base) / wandb.run.project / FLAGS.run_group / exp_name)

    save_dir = Path(FLAGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load SimLingo base policy ─────────────────────────────────────────────
    # pid_only when expert_debug with no SAC checkpoint: skip the heavy VLA model load.
    _expert_debug_only = (FLAGS.expert_debug or FLAGS.expert_recover_debug) and not FLAGS.expert_checkpoint
    if _expert_debug_only:
        print("[main] expert_debug mode — skipping VLA model load, PID controllers only.", flush=True)
        from vlas.simlingo_base import SimLingoBase, VLM_FEATURE_DIM, VLM_ENCODER_FEATURE_DIM  # type: ignore
        simlingo_base = SimLingoBase("", pid_only=True)
    else:
        print(f"[main] Loading SimLingo policy (mode={FLAGS.policy_mode}) ...", flush=True)
        from vlas.simlingo_base import HierarchicalSimLingoPolicy, SimLingoBase, VLM_FEATURE_DIM, VLM_ENCODER_FEATURE_DIM  # type: ignore
        if FLAGS.policy_mode == "hierarchical":
            simlingo_base = HierarchicalSimLingoPolicy(
                high_checkpoint_path=FLAGS.high_level_checkpoint,
                low_checkpoint_path=FLAGS.low_level_checkpoint,
                device=FLAGS.device,
                high_hydra_config_path=FLAGS.high_level_hydra_config,
                low_hydra_config_path=FLAGS.low_level_hydra_config,
                source_root=FLAGS.hierarchical_source_root if FLAGS.hierarchical_source_root else None,
                high_source_root=FLAGS.high_level_source_root,
                low_source_root=FLAGS.low_level_source_root,
            )
        else:
            simlingo_base = SimLingoBase(FLAGS.simlingo_checkpoint, device=FLAGS.device)

    # ── Start CARLA env server ────────────────────────────────────────────────
    # Read the initial obs (server sends it right after ready signal)
    print("[main] Starting CARLA env server (simlingo conda, carla 0.9.15)...", flush=True)
    env = CarlaEnvProxy(
        route=FLAGS.route,
        carla_config=FLAGS.carla_config,
        gpu_rank=FLAGS.gpu_rank,
        server_conda_env=FLAGS.server_conda_env,
        carla_root=FLAGS.carla_root,
        terminate_on_infraction=FLAGS.terminate_on_infraction,
    )

    initial_obs, _ = env.reset()

    def _get_features(obs):
        """Return (desired_speeds, route_interp, obs_features) based on obs_mode flag."""
        if _expert_debug_only:
            # No VLA model in expert debug mode — return dummy features.
            _feat_dim = VLM_FEATURE_DIM if FLAGS.obs_mode == "vlm_hidden" else VLM_ENCODER_FEATURE_DIM
            return np.zeros(FLAGS.chunk_size, dtype=np.float32), None, np.zeros(_feat_dim, dtype=np.float32)
        desired_speeds, route_interp, vlm_feats = simlingo_base.get_chunk_and_features(
            simlingo_image=obs["simlingo_image"],
            ego_state=obs["state"],
            target_points=obs["target_points"],
            routing_command=obs.get("routing_command", ""),
        )
        if FLAGS.obs_mode == "encoder":
            obs_features = simlingo_base.get_encoder_features(
                simlingo_image=obs["simlingo_image"],
                ego_state=obs["state"],
                target_points=obs["target_points"],
                routing_command=obs.get("routing_command", ""),
            )
        else:
            obs_features = vlm_feats
        return desired_speeds, route_interp, obs_features

    # ── Video helper ──────────────────────────────────────────────────────────
    video_dir = save_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Per-episode frame buffer for wandb video upload
    _frame_buffer: List[np.ndarray] = []

    def _open_video(ep_idx: int):
        _frame_buffer.clear()
        if not FLAGS.save_video:
            return None
        import cv2
        path = str(video_dir / f"ep{ep_idx:04d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 20.0, (1024, 512 + _VIDEO_PANEL_H))
        print(f"[video] Writing {path}", flush=True)
        return writer

    def _write_frame(writer, image_rgb, annotated: Optional[np.ndarray] = None):
        frame = annotated if annotated is not None else image_rgb
        if frame.shape[:2] == (512, 1024):
            frame = _annotate_frame(
                frame,
                simlingo_base,
                None,
                0.0,
                np.zeros(2, dtype=np.float32),
            )
        _frame_buffer.append(frame)
        if writer is None:
            return
        import cv2
        writer.write(frame[:, :, ::-1])  # RGB → BGR

    def _close_video(writer) -> Optional[str]:
        if writer is None:
            return None
        import cv2
        writer.release()
        return None

    # ── Eval-only mode ────────────────────────────────────────────────────────
    if FLAGS.eval_only:
        print(f"[main] Eval-only mode: rolling out base policy on {FLAGS.eval_episodes} episode(s) ...", flush=True)
        results: List[Dict[str, Any]] = []

        # Eval always runs VLM every CARLA tick (1 tick per wp call) to match
        # the reference eval's 20Hz inference rate, regardless of chunk_size.
        ticks_per_wp = 1
        chunk_size = FLAGS.chunk_size  # only affects how many speed targets are pre-computed

        for ep_idx in range(FLAGS.eval_episodes):
            if ep_idx == 0:
                obs = initial_obs
            else:
                obs, _ = env.reset()
            simlingo_base.reset_pid()
            episode_reward = 0.0
            steps = 0
            info: Dict[str, Any] = {}
            video = _open_video(ep_idx)
            eval_prev_collision_count = 0

            print(f"\n[eval] Episode {ep_idx + 1} / {FLAGS.eval_episodes}", flush=True)
            done = False
            while not done and steps < FLAGS.eval_step_limit:
                # VLM call: get desired speeds + store route_interp for steer_for_speed()
                desired_speeds, _route_interp, _ = simlingo_base.get_chunk_and_features(
                    simlingo_image=obs["simlingo_image"],
                    ego_state=obs["state"],
                    target_points=obs["target_points"],
                    routing_command=obs.get("routing_command", ""),
                )
                for k in range(chunk_size):
                    for _tick in range(ticks_per_wp):
                        actual_speed = float(obs["state"][15])
                        base_accel = simlingo_base.accel_for_desired_speed(desired_speeds[k], actual_speed)
                        # Per-tick steer: lateral PID integrates error history each tick
                        # using the route from the current VLM call (matches control_pid)
                        base_steer = simlingo_base.steer_for_speed(actual_speed)
                        action = np.array([base_accel, base_steer], dtype=np.float32)
                        image_for_video = obs["simlingo_image"]
                        target_points_for_video = obs.get("target_points")
                        obs, reward, terminated, truncated, info = env.step(action)
                        eval_collision_delta = max(0, int(info.get("collision_count", 0)) - eval_prev_collision_count)
                        eval_prev_collision_count = int(info.get("collision_count", 0))
                        annotated = _annotate_frame(
                            image_for_video,
                            simlingo_base,
                            target_points_for_video,
                            actual_speed,
                            action,
                            reward_value=float(reward),
                            env_reward_value=float(reward),
                            info=info,
                            collision_events=eval_collision_delta,
                        )
                        _write_frame(video, image_for_video, annotated)
                        episode_reward += reward
                        steps += 1
                        done = terminated or truncated
                        if done:
                            break
                    if done:
                        break

            _close_video(video)
            driving_score = float(info.get("driving_score", 0.0))
            stats = {
                "episode_reward": episode_reward,
                "driving_score": driving_score,
                "steps": steps,
                "success": info.get("success", False),
                "collision_count": info.get("collision_count", 0),
                "outside_route": info.get("outside_route_value", 0.0),
                "traffic_violations": info.get("traffic_violation_count", 0),
                "route": FLAGS.route,
            }
            results.append(stats)
            print(
                f"[eval] ep={ep_idx+1}  reward={stats['episode_reward']:.2f}"
                f"  driving_score={driving_score:.2f}"
                f"  steps={stats['steps']}  success={stats['success']}"
                f"  collisions={stats['collision_count']}"
                f"  traffic_violations={stats['traffic_violations']}",
                flush=True,
            )
            wb_log = {
                "eval/episode_reward": episode_reward,
                "eval/driving_score": driving_score,
                "eval/steps": steps,
                "eval/success": float(stats["success"]),
                "eval/collision_count": float(stats["collision_count"]),
                "eval/outside_route": float(stats["outside_route"]),
            }
            if _frame_buffer:
                frames_np = np.stack(_frame_buffer)  # (T, H, W, 3)
                wb_log["eval/episode_video"] = wandb.Video(
                    frames_np.transpose(0, 3, 1, 2), fps=20, format="mp4"
                )
            wandb.log(wb_log, step=ep_idx)

        out_path = save_dir / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[eval] Results saved to {out_path}", flush=True)
        wandb.finish()
        env.close()
        return

    # ── DAgger residual training ──────────────────────────────────────────────
    if FLAGS.training_mode == "dagger_residual":
        from torch_agents.residual_sac import ResidualSACAgent, DaggerBuffer, EGO_STATE_DIM  # type: ignore
        import torch

        vlm_dim = VLM_FEATURE_DIM if FLAGS.obs_mode == "vlm_hidden" else VLM_ENCODER_FEATURE_DIM

        _state_dim = EGO_STATE_DIM if FLAGS.include_ego_state else 0
        agent = ResidualSACAgent(
            vlm_feature_dim=vlm_dim,
            action_dim=2,
            hidden_dims=(256, 256, 256),
            gamma=FLAGS.gamma,
            tau=FLAGS.tau,
            actor_lr=FLAGS.actor_lr,
            critic_lr=FLAGS.critic_lr,
            device=FLAGS.device,
            actor_l2_reg=FLAGS.actor_l2_reg,
            state_dim=_state_dim,
        )
        if FLAGS.pretrained_critic:
            _load_pretrained_critic_simlingo(agent, FLAGS.pretrained_critic)
        buffer: Any = DaggerBuffer(capacity=FLAGS.buffer_capacity, vlm_dim=vlm_dim, state_dim=_state_dim)

        _expert_sac_agent = None
        if FLAGS.expert_checkpoint:
            _expert_sac_agent = ResidualSACAgent(
                vlm_feature_dim=vlm_dim,
                action_dim=2,
                hidden_dims=(256, 256, 256),
                gamma=FLAGS.gamma,
                tau=FLAGS.tau,
                actor_lr=FLAGS.actor_lr,
                critic_lr=FLAGS.critic_lr,
                device=FLAGS.device,
                actor_l2_reg=FLAGS.actor_l2_reg,
                state_dim=_state_dim,
                coach_label_dim=0,
                expert_action_dim=0,
            )
            _expert_sac_agent.load(FLAGS.expert_checkpoint)
            print(f"[main] Loaded SAC expert from {FLAGS.expert_checkpoint}", flush=True)

        log_path = save_dir / "train_log.jsonl"
        log_file = open(log_path, "w")

        obs = initial_obs
        simlingo_base.reset_pid()

        episode_reward = 0.0
        episode_env_reward = 0.0
        episode_steps = 0
        episode_collision_count = 0
        episode_collision_events = 0
        prev_collision_count = 0
        num_episodes = 0
        total_updates = 0

        chunk_size = FLAGS.chunk_size
        ticks_per_wp = simlingo_base._WP_DILATION * simlingo_base._DATA_SAVE_FREQ

        desired_speeds, _route_interp, vlm_features = _get_features(obs)

        print(f"[train/dagger] Starting DAgger BC training for {FLAGS.total_steps} steps "
              f"(chunk_size={chunk_size}, ticks_per_wp={ticks_per_wp}) ...", flush=True)
        t0 = time.time()
        last_log_time = t0
        last_bc_metrics: Dict[str, float] = {}
        last_step_info: Dict[str, Any] = {}
        last_drive_metrics = _ego_drive_metrics_from_state_vec(obs["state"])
        last_env_reward = 0.0
        last_actual_speed = float(obs["state"][15])
        last_base_action = np.zeros(2, dtype=np.float32)
        last_final_action = np.zeros(2, dtype=np.float32)
        last_expert_action_2d = np.zeros(2, dtype=np.float32)
        last_expert_residual_target = np.zeros(2, dtype=np.float32)
        last_actor_output = np.zeros(2, dtype=np.float32)
        last_collision_delta = 0
        last_update_time = 0.0
        video = _open_video(num_episodes)

        _expert_recover_budget = int(np.random.randint(70, 201)) if FLAGS.expert_recover_debug else 0
        if FLAGS.expert_recover_debug:
            print(f"[expert_recover_debug] episode 0: SimLingo for {_expert_recover_budget} ticks then expert", flush=True)

        # Capture the expert action for the initial obs (expert action at state s,
        # used together with vlm_features which is also computed at state s).
        current_expert_action_40d: Optional[np.ndarray] = obs.get("expert_action")

        for global_step in range(FLAGS.total_steps):
            t_sample_start = time.time()
            in_warmup = global_step < FLAGS.warmup_steps
            dagger_clip_limit = _residual_clip_limit(global_step, FLAGS.warmup_steps, FLAGS.residual_clip_schedule_steps)
            # Capture speed and base action at the START of the chunk so that the
            # expert residual uses the same state as vlm_features.
            chunk_start_speed = float(obs["state"][15])
            chunk_start_state = obs["state"][6:].copy()
            if global_step < FLAGS.learning_starts:
                agent.update_state_normalizer(chunk_start_state)
            chunk_start_base_accel = simlingo_base.accel_for_desired_speed(desired_speeds[0], chunk_start_speed)
            chunk_start_base_steer = simlingo_base.steer_for_speed(chunk_start_speed)
            chunk_start_base_action = np.array([chunk_start_base_accel, chunk_start_base_steer], dtype=np.float32)
            if in_warmup:
                # Warmup: execute base policy only (no residual)
                residual_action = np.zeros(2, dtype=np.float32)
            else:
                # DAgger: execute deterministic actor mean (no exploration noise)
                residual_action = agent.get_eval_action(vlm_features, chunk_start_base_action, chunk_start_state)
                residual_action = np.clip(residual_action, -dagger_clip_limit, dagger_clip_limit)
            last_actor_output = residual_action.copy()
            t_sample_end = time.time()

            # Pre-compute expert route waypoints for video overlay (ego-frame cumsum of route deltas).
            # These are computed before the chunk to match vlm_features timing.
            _expert_route_wps_for_video: Optional[np.ndarray] = None
            if current_expert_action_40d is not None and not np.allclose(current_expert_action_40d, 0.0):
                try:
                    _ec = current_expert_action_40d.reshape(10, 4)
                    _expert_route_wps_for_video = np.cumsum(_ec[:, 2:4], axis=0)  # (10, 2) ego-frame
                except Exception:
                    pass

            # Expert intervention debug: precompute expert action for this chunk.
            _in_expert_recovery = FLAGS.expert_recover_debug and (episode_steps >= _expert_recover_budget)
            _expert_debug_action: Optional[np.ndarray] = None
            if (FLAGS.expert_debug or _in_expert_recovery) and current_expert_action_40d is not None:
                if not np.allclose(current_expert_action_40d, 0.0):
                    try:
                        _expert_debug_action = np.clip(
                            _expert_action_to_accel_steer(current_expert_action_40d, simlingo_base, chunk_start_speed),
                            -1.0, 1.0,
                        ).astype(np.float32)
                    except Exception as _exc:
                        print(f"[expert_debug] _expert_action_to_accel_steer failed: {_exc}", flush=True)
            if (FLAGS.expert_debug or _in_expert_recovery) and episode_steps % 20 == 0:
                ea_norm = float(np.linalg.norm(current_expert_action_40d)) if current_expert_action_40d is not None else -1.0
                print(f"[expert_debug] step={episode_steps} ea_norm={ea_norm:.4f} expert_action={'SET' if _expert_debug_action is not None else 'NONE'}", flush=True)

            chunk_reward = 0.0
            chunk_env_reward = 0.0
            done = False
            info: Dict[str, Any] = {}
            t_step_start = time.time()
            for k in range(chunk_size):
                for _tick in range(ticks_per_wp):
                    actual_speed = float(obs["state"][15])
                    base_accel = simlingo_base.accel_for_desired_speed(desired_speeds[k], actual_speed)
                    # Bug fix: reuse chunk_start_base_action[1] at tick 0 so the actor
                    # input steer (chunk_start_base_action) matches the executed steer.
                    # A second steer_for_speed() call here would advance the PID window
                    # again, producing a different value from what the actor saw.
                    if k == 0 and _tick == 0:
                        base_steer = chunk_start_base_action[1]
                    else:
                        base_steer = simlingo_base.steer_for_speed(actual_speed)
                    base_action = np.array([base_accel, base_steer], dtype=np.float32)
                    image_for_video = obs["simlingo_image"]
                    target_points_for_video = obs.get("target_points")
                    if FLAGS.expert_debug or _in_expert_recovery:
                        # Use PDM-Lite's direct control output — re-plans every tick,
                        # avoids the 5-tick stale-action delay of the 40D encode/decode.
                        next_obs, reward, terminated, truncated, info = env.step_expert()
                        final_action = _expert_debug_action if _expert_debug_action is not None else base_action
                    else:
                        final_action = np.clip(
                            base_action + _res_scale_vec * residual_action, -1.0, 1.0
                        ).astype(np.float32)
                        next_obs, reward, terminated, truncated, info = env.step(final_action)
                    env_reward = float(reward)
                    last_env_reward = env_reward
                    last_actual_speed = actual_speed
                    last_base_action = base_action
                    last_final_action = final_action
                    last_step_info = dict(info)
                    last_drive_metrics = _ego_drive_metrics_from_state_vec(next_obs["state"])
                    collision_count = int(info.get("collision_count", 0))
                    last_collision_delta = max(0, collision_count - prev_collision_count)
                    episode_collision_count = max(episode_collision_count, collision_count)
                    episode_collision_events += last_collision_delta
                    prev_collision_count = collision_count
                    annotated = _annotate_frame(
                        image_for_video,
                        simlingo_base,
                        target_points_for_video,
                        actual_speed,
                        base_action,
                        residual_action,
                        reward_value=float(reward),
                        env_reward_value=env_reward,
                        info=info,
                        collision_events=last_collision_delta,
                        expert_waypoints=_expert_route_wps_for_video,
                        expert_action_2d=last_expert_action_2d if not np.allclose(last_expert_action_2d, 0.0) else None,
                    )
                    _write_frame(video, image_for_video, annotated)
                    chunk_reward += reward
                    chunk_env_reward += env_reward
                    done = terminated or truncated
                    obs = next_obs
                    if done:
                        break
                if done:
                    break
            t_step_end = time.time()

            # ── Compute expert action for the current state ───────────────────
            # current_expert_action_40d was captured at the beginning of this step,
            # matching vlm_features (both from state s before the chunk).
            # chunk_start_speed and chunk_start_base_action align with that same state.
            expert_action_2d: Optional[np.ndarray] = None
            if _expert_sac_agent is not None:
                # SAC checkpoint as expert: call deterministic policy.
                try:
                    expert_residual = _expert_sac_agent.get_eval_action(vlm_features, chunk_start_base_action, chunk_start_state)
                    ea_2d = np.clip(
                        chunk_start_base_action + _res_scale_vec * expert_residual, -1.0, 1.0
                    ).astype(np.float32)
                    last_expert_action_2d = ea_2d
                    last_expert_residual_target = np.clip(
                        (ea_2d - chunk_start_base_action) / np.maximum(_res_scale_vec, 1e-6), -1.0, 1.0
                    ).astype(np.float32)
                    expert_action_2d = ea_2d
                except Exception as _ex:
                    print(f"[dagger] SAC expert action failed: {_ex}", flush=True)
            elif current_expert_action_40d is not None:
                ea_40d = current_expert_action_40d
                if not np.allclose(ea_40d, 0.0):
                    try:
                        ea_2d = _expert_action_to_accel_steer(ea_40d, simlingo_base, chunk_start_speed)
                        last_expert_action_2d = ea_2d
                        last_expert_residual_target = np.clip(
                            (ea_2d - chunk_start_base_action) / np.maximum(_res_scale_vec, 1e-6), -1.0, 1.0
                        ).astype(np.float32)
                        expert_action_2d = ea_2d
                    except Exception as _ex:
                        print(f"[dagger] expert action conversion failed: {_ex}", flush=True)

            # ── Next VLM call ─────────────────────────────────────────────────
            t_vlm_start = time.time()
            next_desired_speeds, _next_route_interp, next_vlm_features = _get_features(obs)
            t_vlm_end = time.time()

            # ── Buffer add ────────────────────────────────────────────────────
            if expert_action_2d is not None:
                buffer.add(vlm_features, chunk_start_base_action, chunk_start_state, expert_action_2d)

            episode_reward += chunk_reward
            episode_env_reward += chunk_env_reward
            episode_steps += 1

            # ── BC updates ────────────────────────────────────────────────────
            t_update_start = time.time()
            if len(buffer) >= FLAGS.learning_starts and not in_warmup:
                for _ in range(FLAGS.updates_per_step):
                    batch = buffer.sample(FLAGS.batch_size, torch.device(FLAGS.device))
                    last_bc_metrics = agent.bc_update(batch, res_scale=_res_scale_vec)
                    total_updates += 1
            t_update_end = time.time()
            last_update_time = t_update_end - t_update_start

            # ── Per-step W&B logging ──────────────────────────────────────────
            if global_step % FLAGS.log_interval == 0:
                step_log: Dict[str, Any] = {
                    "time/steps_per_sec": FLAGS.log_interval / max(time.time() - last_log_time, 1e-6),
                    "time/global_step": global_step,
                    "time/sample_time": t_sample_end - t_sample_start,
                    "time/step_time": t_step_end - t_step_start,
                    "time/vlm_time": t_vlm_end - t_vlm_start,
                    "time/update_time": last_update_time,
                    "training/in_warmup": float(in_warmup),
                    "training/buffer_size": len(buffer),
                    "training/total_updates": total_updates,
                    "reward/env_step": float(last_env_reward),
                    "reward/chunk_env": float(chunk_env_reward),
                    "rollout/current_episode_reward": float(episode_reward),
                    "rollout/current_episode_env_return": float(episode_env_reward),
                    "rollout/current_episode_steps": float(episode_steps),
                    "rollout/actual_speed": float(last_actual_speed),
                    "rollout/collision_count": float(last_step_info.get("collision_count", 0)),
                    "rollout/collision_events": float(last_collision_delta),
                    "rollout/episode_collision_events": float(episode_collision_events),
                    "action/base_accel": float(last_base_action[0]),
                    "action/base_steer": float(last_base_action[1]),
                    "action/actor_accel": float(last_actor_output[0]),
                    "action/actor_steer": float(last_actor_output[1]),
                    "action/final_accel": float(last_final_action[0]),
                    "action/final_steer": float(last_final_action[1]),
                    "action/res_scale_accel": float(FLAGS.res_scale_accel),
                    "action/res_scale_steer": float(FLAGS.res_scale_steer),
                    "action/residual_clip_limit": float(dagger_clip_limit),
                    "dagger/expert_accel": float(last_expert_action_2d[0]),
                    "dagger/expert_steer": float(last_expert_action_2d[1]),
                    "dagger/expert_residual_accel": float(last_expert_residual_target[0]),
                    "dagger/expert_residual_steer": float(last_expert_residual_target[1]),
                    "dagger/expert_valid": float(expert_action_2d is not None),
                    "dagger/buffer_size": len(buffer),
                    "dagger/total_updates": total_updates,
                    "dagger/bc_loss": float(last_bc_metrics.get("bc_loss", float("nan"))),
                    "dagger/base_mse": float(last_bc_metrics.get("base_mse", float("nan"))),
                    "dagger/residual_abs_mean": float(last_bc_metrics.get("residual_abs_mean", float("nan"))),
                    "dagger/residual_abs_max": float(last_bc_metrics.get("residual_abs_max", float("nan"))),
                    "simlingo/desired_speed_first": float(desired_speeds[0]),
                    "simlingo/desired_speed_mean": float(np.mean(desired_speeds)),
                    "simlingo/vlm_feature_norm": float(np.linalg.norm(vlm_features)),
                    "simlingo/obs_mode": FLAGS.obs_mode,
                    "simlingo/obs_dim": float(vlm_dim),
                }
                step_log.update({f"rollout/{k}": float(v) for k, v in last_drive_metrics.items()})
                if last_step_info.get("reward_total") is not None:
                    step_log.update({
                        "reward/env_total": float(last_step_info.get("reward_total", 0.0)),
                        "reward/progress": float(last_step_info.get("reward_progress", 0.0)),
                        "reward/penalty_collision": float(last_step_info.get("penalty_collision", 0.0)),
                        "reward/penalty_outside_route": float(last_step_info.get("penalty_outside_route", 0.0)),
                        "reward/penalty_traffic_violation": float(last_step_info.get("penalty_traffic_violation", 0.0)),
                        "rollout/traffic_violation_count": float(last_step_info.get("traffic_violation_count", 0)),
                        "rollout/lane_offset_m": float(last_step_info.get("lane_offset_m", 0.0)),
                        "rollout/speed_norm": float(last_step_info.get("speed_norm", 0.0)),
                    })
                step_log.update(_state_norm_wandb_log(agent))
                wandb.log(step_log, step=global_step)
                last_log_time = time.time()

            # ── Episode end ───────────────────────────────────────────────────
            if done:
                num_episodes += 1
                elapsed = time.time() - t0
                log_entry = {
                    "global_step": global_step,
                    "episode": num_episodes,
                    "episode_reward": episode_reward,
                    "episode_env_reward": episode_env_reward,
                    "episode_steps": episode_steps,
                    "success": info.get("success", False),
                    "collision_count": info.get("collision_count", 0),
                    "collision_events": episode_collision_events,
                    "termination_reason": info.get("termination_reason", "leaderboard"),
                    "elapsed_s": elapsed,
                }
                print(
                    f"[step {global_step}] ep={num_episodes}  "
                    f"R={episode_reward:.2f}  steps={episode_steps}  "
                    f"success={info.get('success', False)}  "
                    f"term={info.get('termination_reason', 'leaderboard')}  "
                    f"bc_updates={total_updates}",
                    flush=True,
                )
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()

                ep_log: Dict[str, Any] = {
                    "rollout/episode_reward": episode_reward,
                    "rollout/episode_env_reward": episode_env_reward,
                    "rollout/episode_return": episode_reward,
                    "rollout/episode_env_return": episode_env_reward,
                    "rollout/episode_steps": episode_steps,
                    "rollout/success": float(info.get("success", False)),
                    "rollout/collision_count": float(info.get("collision_count", 0)),
                    "rollout/episode_collision_events": float(episode_collision_events),
                    "rollout/outside_route": float(info.get("outside_route_value", 0.0)),
                    "rollout/num_episodes": num_episodes,
                }
                if _frame_buffer and FLAGS.video_log_interval > 0 and num_episodes % FLAGS.video_log_interval == 0:
                    frames_np = np.stack(_frame_buffer)
                    ep_log["rollout/episode_video"] = wandb.Video(
                        frames_np.transpose(0, 3, 1, 2), fps=20, format="mp4"
                    )
                wandb.log(ep_log, step=global_step)

                _close_video(video)
                obs, _ = env.reset()
                simlingo_base.reset_pid()
                episode_reward = 0.0
                episode_env_reward = 0.0
                episode_steps = 0
                episode_collision_count = 0
                episode_collision_events = 0
                prev_collision_count = 0
                if FLAGS.expert_recover_debug:
                    _expert_recover_budget = int(np.random.randint(70, 201))
                    print(f"[expert_recover_debug] episode {num_episodes}: SimLingo for {_expert_recover_budget} ticks then expert", flush=True)
                video = _open_video(num_episodes)
                desired_speeds, _route_interp, vlm_features = _get_features(obs)
                current_expert_action_40d = obs.get("expert_action")
            else:
                desired_speeds = next_desired_speeds
                vlm_features = next_vlm_features
                # Advance expert action: use the expert from the new obs (next state)
                current_expert_action_40d = obs.get("expert_action")

            if global_step > 0 and global_step % FLAGS.save_interval == 0:
                ckpt_path = str(save_dir / f"dagger_residual_{global_step}.pt")
                agent.save(ckpt_path)
                print(f"[step {global_step}] Saved checkpoint to {ckpt_path}", flush=True)

        _close_video(video)
        final_path = str(save_dir / "dagger_residual_final.pt")
        agent.save(final_path)
        log_file.close()
        wandb.finish()
        env.close()
        print(f"\n[train/dagger] Done. Final checkpoint at {final_path}", flush=True)
        return

    # ── Residual SAC training ─────────────────────────────────────────────────
    from torch_agents.residual_sac import ResidualSACAgent, ReplayBuffer, EGO_STATE_DIM  # type: ignore
    import torch

    vlm_dim = VLM_FEATURE_DIM if FLAGS.obs_mode == "vlm_hidden" else VLM_ENCODER_FEATURE_DIM

    _state_dim = EGO_STATE_DIM if FLAGS.include_ego_state else 0
    chunk_size = FLAGS.chunk_size  # waypoints per VLM call (1–10)
    # _WP_DILATION / _DATA_SAVE_FREQ are class-level constants; available even in pid_only mode.
    ticks_per_wp = simlingo_base._WP_DILATION * simlingo_base._DATA_SAVE_FREQ  # = 5

    # ── Gemini coach session (optional) ───────────────────────────────────────
    _coach_session = None
    _coach_label_dim = 0
    if FLAGS.use_gemini_coach:
        import os as _os
        from coaches.online_vlm_coach import OnlineVLMSession  # type: ignore
        from coaches.expert_label import NUM_DELTA_COMMENTARY_WORDS  # type: ignore
        # +1: the last label dim is a validity flag set by the replay buffer when
        # the coach backfills, so the critic can tell "no label yet" apart from a
        # genuine all-zero label (empty text / no feedback for this step).
        if FLAGS.coach_label_mode in ("vlm_embed", "vlm_embed_raw"):
            _coach_label_dim = vlm_dim + 1  # 896-dim frozen Qwen2 token embedding + validity
            _text_encoder = simlingo_base.encode_text
        else:
            _coach_label_dim = NUM_DELTA_COMMENTARY_WORDS + 1  # 17-dim delta-commentary BoW + validity
            _text_encoder = None
        if FLAGS.gemini_api_key:
            _os.environ["GEMINI_API_KEY"] = FLAGS.gemini_api_key
        _chunk_duration_sec = FLAGS.coach_action_chunk_steps * ticks_per_wp / 20.0
        _coach_session = OnlineVLMSession(
            {
                "provider": "gemini",
                "gemini_model": FLAGS.gemini_model,
                "query_every_n_episode_steps": FLAGS.coach_query_freq,
                "query_on_episode_end": True,
                "video_fps": 20.0,
                "video_frame_stride": ticks_per_wp,  # one trajectory record per global_step
                "action_chunk_steps": FLAGS.coach_action_chunk_steps,
                "action_chunk_duration_sec": _chunk_duration_sec,
                "bad_event_radius_chunks": 2,
                "annotate_video": False,
                "save_artifacts": True,
            },
            save_dir=save_dir,
            text_encoder=_text_encoder,
            use_raw_description=(FLAGS.coach_label_mode == "vlm_embed_raw"),
            plot_embeddings=FLAGS.coach_embed_plot,
        )
        _query_freq_str = f"every {FLAGS.coach_query_freq} steps" if FLAGS.coach_query_freq > 0 else "episode end only"
        print(
            f"[main] Gemini coach enabled: model={FLAGS.gemini_model}  "
            f"label_mode={FLAGS.coach_label_mode}  label_dim={_coach_label_dim}  "
            f"chunk_steps={FLAGS.coach_action_chunk_steps}  "
            f"chunk_dur={_chunk_duration_sec:.2f}s  query_freq={_query_freq_str}",
            flush=True,
        )

    # Programmatic language-bow critic: set coach_label_dim now (before buffer/agent init).
    if FLAGS.use_language_bow_critic and not FLAGS.use_gemini_coach:
        from coaches.expert_label import NUM_SCENE_DELTA_WORDS  # type: ignore
        # +1 validity flag (1.0 = label was available this step; 0.0 = no expert action).
        _coach_label_dim = NUM_SCENE_DELTA_WORDS + 1
        print(
            f"[main] Language-bow critic enabled: label_dim={_coach_label_dim} "
            f"(SCENE_DELTA_VOCAB={NUM_SCENE_DELTA_WORDS} words + 1 validity)",
            flush=True,
        )

    # Noise critic (ablation baseline): same dim as language_bow.
    if FLAGS.use_noise_critic and not FLAGS.use_gemini_coach and not FLAGS.use_language_bow_critic:
        from coaches.expert_label import NUM_SCENE_DELTA_WORDS  # type: ignore
        _coach_label_dim = NUM_SCENE_DELTA_WORDS + 1
        print(
            f"[main] Noise critic enabled (ablation): label_dim={_coach_label_dim} i.i.d. Gaussian noise",
            flush=True,
        )

    # [accel, steer, valid] — the trailing validity flag distinguishes "expert
    # unavailable this step" from a genuine zero (coast, straight) action.
    _expert_action_dim = 3 if FLAGS.use_expert_in_critic else 0
    # Separate 2D log-only buffer field for BoW/delta modes (not a critic input).
    # Not needed when use_expert_in_critic is on — update() reads expert_actions[:, :2] directly.
    _log_expert_dim = (
        2 if FLAGS.log_q_expert_diff
        and (FLAGS.use_language_bow_critic or FLAGS.use_gemini_coach)
        and not FLAGS.use_expert_in_critic
        else 0
    )

    if not _expert_debug_only:
        agent = ResidualSACAgent(
            vlm_feature_dim=vlm_dim,
            action_dim=2,
            hidden_dims=(256, 256, 256),
            gamma=FLAGS.gamma,
            tau=FLAGS.tau,
            actor_lr=FLAGS.actor_lr,
            critic_lr=FLAGS.critic_lr,
            device=FLAGS.device,
            actor_l2_reg=FLAGS.actor_l2_reg,
            res_scale=_res_scale_vec,
            state_dim=_state_dim,
            ticks_per_wp=ticks_per_wp,
            coach_label_dim=_coach_label_dim,
            expert_action_dim=_expert_action_dim,
            log_q_expert_diff=FLAGS.log_q_expert_diff,
        )
        buffer = ReplayBuffer(
            capacity=FLAGS.buffer_capacity,
            vlm_dim=vlm_dim,
            state_dim=_state_dim,
            coach_label_dim=_coach_label_dim,
            expert_action_dim=_expert_action_dim,
            log_expert_dim=_log_expert_dim,
        )
        if FLAGS.pretrained_critic:
            _load_pretrained_critic_simlingo(agent, FLAGS.pretrained_critic)
    else:
        agent = None
        buffer = None

    # Optional SAC checkpoint to use as expert instead of PDM-Lite autopilot.
    # Always instantiate with 0 for coach/expert dims — these must match the
    # checkpoint's own training config, not the current run's config.
    _expert_sac_agent = None
    if FLAGS.expert_checkpoint:
        _expert_sac_agent = ResidualSACAgent(
            vlm_feature_dim=vlm_dim,
            action_dim=2,
            hidden_dims=(256, 256, 256),
            gamma=FLAGS.gamma,
            tau=FLAGS.tau,
            actor_lr=FLAGS.actor_lr,
            critic_lr=FLAGS.critic_lr,
            device=FLAGS.device,
            actor_l2_reg=FLAGS.actor_l2_reg,
            res_scale=_res_scale_vec,
            state_dim=_state_dim,
            ticks_per_wp=ticks_per_wp,
            coach_label_dim=0,
            expert_action_dim=0,
        )
        _expert_sac_agent.load(FLAGS.expert_checkpoint)
        print(f"[main] Loaded SAC expert from {FLAGS.expert_checkpoint}", flush=True)

    log_path = save_dir / "train_log.jsonl"
    log_file = open(log_path, "w")

    obs = initial_obs
    simlingo_base.reset_pid()

    episode_reward = 0.0
    episode_env_reward = 0.0
    episode_steps = 0
    episode_collision_count = 0
    episode_collision_events = 0
    prev_collision_count = 0
    num_episodes = 0
    total_updates = 0

    # Initial VLM call so the loop starts with features ready.
    desired_speeds, _route_interp, vlm_features = _get_features(obs)

    # Coach: begin the first episode before the training loop starts.
    if _coach_session is not None:
        _coach_session.begin_episode(episode_count=0, route_name=FLAGS.route or "")

    if FLAGS.debug_neg_speed_reward:
        reward_mode = "neg_speed"
    elif FLAGS.debug_target_speed_reward > 0.0:
        reward_mode = f"target_speed_{FLAGS.debug_target_speed_reward:.1f}"
    else:
        reward_mode = "env"
    print(f"[train] Starting residual SAC training for {FLAGS.total_steps} steps "
          f"(chunk_size={chunk_size}, ticks_per_wp={ticks_per_wp}, "
          f"reward_mode={reward_mode}) ...", flush=True)
    t0 = time.time()
    last_log_time = t0
    last_sac_metrics: Dict[str, float] = {}
    last_step_info: Dict[str, Any] = {}
    last_drive_metrics = _ego_drive_metrics_from_state_vec(obs["state"])
    last_env_reward = 0.0
    last_train_reward = 0.0
    last_actual_speed = float(obs["state"][15])
    last_base_action = np.zeros(2, dtype=np.float32)
    last_final_action = np.zeros(2, dtype=np.float32)
    last_collision_delta = 0
    last_update_time = 0.0
    video = _open_video(num_episodes)
    last_expert_action_2d = np.zeros(2, dtype=np.float32)
    last_language_feedback: Optional[str] = None
    # For language_bow: track previous step's buffer ptr for next_coach_label backfill.
    _prev_lang_bow_ptr: Optional[int] = None
    # True when the previous buffer slot belongs to a finished episode — guards the
    # next-expert backfill below from writing the new episode's expert action into it.
    _prev_transition_done = True

    # ── Obs histogram accumulators (debug_obs_hist mode) ─────────────────────
    _hist_embed: List[np.ndarray] = []
    _hist_other: Dict[str, List[np.ndarray]] = {
        "ego state": [],
        "base action": [],
        "residual action": [],
        "expert action (critic)": [],
        "language BoW (critic)": [],
        "noise label (critic)": [],
    }

    for global_step in range(FLAGS.total_steps):
        t_sample_start = time.time()
        # ── Rollout: execute chunk_size waypoints, ticks_per_wp ticks each ───
        in_warmup = global_step < FLAGS.warmup_steps
        sac_clip_limit = _residual_clip_limit(global_step, FLAGS.warmup_steps, FLAGS.residual_clip_schedule_steps)
        current_speed = float(obs["state"][15])
        current_obs_state = obs["state"][6:].copy()
        # Refresh the expert chunk every step (and implicitly after env.reset) —
        # reading it once before the loop left the critic conditioned on the
        # expert plan from the very first frame of training.
        current_expert_action_40d: Optional[np.ndarray] = obs.get("expert_action")
        # Only update normalizer stats before learning starts (i.e. during the
        # random/zero-residual phase).  Freezing after that keeps the state
        # representation consistent for every (s, a, r, s') tuple in the buffer.
        if agent is not None and global_step < FLAGS.learning_starts:
            agent.update_state_normalizer(current_obs_state)
        current_base_action = np.zeros(2, dtype=np.float32) if _expert_debug_only else np.array([
            simlingo_base.accel_for_desired_speed(desired_speeds[0], current_speed),
            simlingo_base.steer_for_speed(current_speed),
        ], dtype=np.float32)
        if _expert_debug_only or in_warmup or agent is None:
            residual_action = np.zeros(2, dtype=np.float32)
        else:
            residual_action = agent.sample_actions(vlm_features, current_base_action, current_obs_state)
            residual_action = np.clip(residual_action, -sac_clip_limit, sac_clip_limit)
        t_sample_end = time.time()

        # Expert intervention debug: precompute expert action for this chunk.
        _expert_debug_action: Optional[np.ndarray] = None
        if FLAGS.expert_debug:
            if _expert_sac_agent is not None:
                # Use past SAC checkpoint as expert.
                try:
                    expert_residual = _expert_sac_agent.get_eval_action(vlm_features, current_base_action, current_obs_state)
                    _expert_debug_action = np.clip(
                        current_base_action + _res_scale_vec * expert_residual, -1.0, 1.0
                    ).astype(np.float32)
                except Exception as _exc:
                    print(f"[expert_debug] SAC expert action failed: {_exc}", flush=True)
            elif current_expert_action_40d is not None and not np.allclose(current_expert_action_40d, 0.0):
                try:
                    _expert_debug_action = np.clip(
                        _expert_action_to_accel_steer(current_expert_action_40d, simlingo_base, current_speed),
                        -1.0, 1.0,
                    ).astype(np.float32)
                except Exception as _exc:
                    print(f"[expert_debug] _expert_action_to_accel_steer failed: {_exc}", flush=True)
        if FLAGS.expert_debug and episode_steps % 20 == 0:
            if _expert_sac_agent is not None:
                act_str = f"accel={_expert_debug_action[0]:.3f} steer={_expert_debug_action[1]:.3f}" if _expert_debug_action is not None else "NONE"
                print(f"[expert_debug] step={episode_steps} SAC_expert {act_str}", flush=True)
            else:
                ea_norm = float(np.linalg.norm(current_expert_action_40d)) if current_expert_action_40d is not None else -1.0
                act_str = f"accel={_expert_debug_action[0]:.3f} steer={_expert_debug_action[1]:.3f}" if _expert_debug_action is not None else "NONE"
                print(f"[expert_debug] step={episode_steps} ea_norm={ea_norm:.4f} {act_str}", flush=True)

        chunk_reward = 0.0
        chunk_reward_discounted = 0.0  # gamma-weighted sum stored in the replay buffer
        chunk_env_reward = 0.0
        _tick_gamma = 1.0  # gamma^tick, for proper multi-step Bellman target
        done = False
        info: Dict[str, Any] = {}
        t_step_start = time.time()
        for k in range(chunk_size):
            for _tick in range(ticks_per_wp):
                actual_speed = float(obs["state"][15])
                base_accel = simlingo_base.accel_for_desired_speed(desired_speeds[k], actual_speed)
                # Bug fix: at the first tick reuse the steer that was already computed
                # above (PID call #1 = what the actor saw).  Calling steer_for_speed()
                # again here would be PID call #2, advancing _window a second time and
                # producing a different steer value — making the actor input steer ≠
                # the steer actually executed at tick 0.
                if k == 0 and _tick == 0:
                    base_steer = current_base_action[1]
                else:
                    base_steer = simlingo_base.steer_for_speed(actual_speed)
                base_action = np.array([base_accel, base_steer], dtype=np.float32)
                image_for_video = obs["simlingo_image"]
                target_points_for_video = obs.get("target_points")
                if FLAGS.expert_debug and _expert_sac_agent is None:
                    # Use PDM-Lite's direct control output — re-plans every tick.
                    next_obs, reward, terminated, truncated, info = env.step_expert()
                    final_action = _expert_debug_action if _expert_debug_action is not None else base_action
                elif _expert_debug_action is not None:
                    final_action = _expert_debug_action
                    next_obs, reward, terminated, truncated, info = env.step(final_action)
                else:
                    final_action = np.clip(
                        base_action + _res_scale_vec * residual_action, -1.0, 1.0
                    ).astype(np.float32)
                    next_obs, reward, terminated, truncated, info = env.step(final_action)
                env_reward = float(reward)
                if FLAGS.debug_neg_speed_reward:
                    reward = -float(next_obs["state"][15])
                elif FLAGS.debug_target_speed_reward > 0.0:
                    speed_error = abs(float(next_obs["state"][15]) - FLAGS.debug_target_speed_reward)
                    on_target_bonus = 100.0 if speed_error < 0.1 else 0.0
                    # steer_penalty = max(0.0, abs(float(final_action[1])) - 0.5)
                    steer_penalty = 0.0
                    reward = -speed_error + on_target_bonus - steer_penalty
                last_env_reward = env_reward
                last_train_reward = float(reward)
                last_actual_speed = actual_speed
                last_base_action = base_action
                last_final_action = final_action
                last_step_info = dict(info)
                last_drive_metrics = _ego_drive_metrics_from_state_vec(next_obs["state"])
                collision_count = int(info.get("collision_count", 0))
                last_collision_delta = max(0, collision_count - prev_collision_count)
                episode_collision_count = max(episode_collision_count, collision_count)
                episode_collision_events += last_collision_delta
                prev_collision_count = collision_count
                _show_expert = (
                    (FLAGS.use_expert_in_critic or FLAGS.use_language_bow_critic)
                    and not np.allclose(last_expert_action_2d, 0.0)
                )
                annotated = _annotate_frame(
                    image_for_video,
                    simlingo_base,
                    target_points_for_video,
                    actual_speed,
                    base_action,
                    residual_action,
                    reward_value=float(reward),
                    env_reward_value=env_reward,
                    info=info,
                    collision_events=last_collision_delta,
                    expert_action_2d=last_expert_action_2d if _show_expert else None,
                    language_feedback=last_language_feedback,
                )
                _write_frame(video, image_for_video, annotated)
                # Coach: record every CARLA tick as a video frame.
                if _coach_session is not None:
                    _coach_session.record_frame(annotated)
                chunk_reward += reward
                chunk_reward_discounted += _tick_gamma * reward
                _tick_gamma *= FLAGS.gamma
                chunk_env_reward += env_reward
                done = terminated or truncated
                obs = next_obs
                if done:
                    break
            if done:
                break
        t_step_end = time.time()

        # ── Next VLM call (for replay buffer and next iteration) ─────────────
        t_vlm_start = time.time()
        next_desired_speeds, _next_route_interp, next_vlm_features = _get_features(obs)
        t_vlm_end = time.time()
        next_actual_speed = float(obs["state"][15])
        next_base_action = np.array([
            simlingo_base.accel_for_desired_speed(next_desired_speeds[0], next_actual_speed),
            simlingo_base.steer_for_speed(next_actual_speed),
        ], dtype=np.float32)
        actor_chosen_final_action = np.clip(
            current_base_action + _res_scale_vec * residual_action, -1.0, 1.0
        ).astype(np.float32)
        _sac_expert_2d: Optional[np.ndarray] = None
        _sac_expert_3d: Optional[np.ndarray] = None
        if FLAGS.use_expert_in_critic and current_expert_action_40d is not None:
            if not np.allclose(current_expert_action_40d, 0.0):
                try:
                    _sac_expert_2d = _expert_action_to_accel_steer(
                        current_expert_action_40d, simlingo_base, current_speed
                    )
                    last_expert_action_2d = _sac_expert_2d
                except Exception as _ex:
                    print(f"[critic_mode] expert action conversion failed: {_ex}", flush=True)
        if FLAGS.use_expert_in_critic:
            # [accel, steer, valid] — valid=0 marks "expert unavailable" so the
            # critic can tell it apart from a genuine zero (coast, straight) action.
            _sac_expert_3d = (
                np.concatenate([_sac_expert_2d, [1.0]]).astype(np.float32)
                if _sac_expert_2d is not None
                else np.zeros(3, dtype=np.float32)
            )
            # Backfill: the expert action at s_t is the next-state expert action of
            # the *previous* transition, so its TD target conditions the target
            # critic on the matched pair. Skipped across episode resets (the
            # bootstrap is masked at terminals anyway).
            if buffer is not None and len(buffer) > 0 and not _prev_transition_done:
                buffer.update_next_expert_at(buffer.last_ptr, _sac_expert_3d)

        # ── Language-bow critic: compute expert vs agent delta with scene grounding ──
        _lang_bow_label: Optional[np.ndarray] = None
        _lang_expert_2d: Optional[np.ndarray] = None
        if FLAGS.use_language_bow_critic and not _expert_debug_only:
            # Resolve expert action for this step (prefer language_bow-specific path;
            # shares conversion with use_expert_in_critic if both are on).
            _lang_expert_2d = _sac_expert_2d
            if _lang_expert_2d is None and current_expert_action_40d is not None:
                if not np.allclose(current_expert_action_40d, 0.0):
                    try:
                        _lang_expert_2d = _expert_action_to_accel_steer(
                            current_expert_action_40d, simlingo_base, current_speed
                        )
                        last_expert_action_2d = _lang_expert_2d
                    except Exception:
                        pass
            if _lang_expert_2d is not None:
                try:
                    from coaches.expert_label import delta_commentary_accel_steer_grounded, NUM_SCENE_DELTA_WORDS  # type: ignore
                    _scene_ctx = obs.get("scene_context") or {}
                    _lang_text, _lang_bow = delta_commentary_accel_steer_grounded(
                        actor_chosen_final_action, _lang_expert_2d, _scene_ctx,
                    )
                    last_language_feedback = _lang_text
                    # Append validity flag (1.0 = label available).
                    _lang_bow_label = np.concatenate([_lang_bow, [1.0]]).astype(np.float32)
                except Exception as _lang_ex:
                    print(f"[language_bow] label computation failed: {_lang_ex}", flush=True)

        # Noise critic (ablation): i.i.d. Gaussian noise, same dim as language_bow.
        if FLAGS.use_noise_critic and buffer is not None and buffer.coach_label_dim > 0:
            _noise_label = np.random.randn(buffer.coach_label_dim).astype(np.float32)

        # For BoW/delta modes: expert action for Q comparison logging (log-only, not critic input).
        _log_expert_2d: Optional[np.ndarray] = None
        if FLAGS.log_q_expert_diff and buffer is not None and buffer.log_expert_dim > 0:
            # Reuse _lang_expert_2d if computed (use_language_bow_critic), else compute for gemini.
            _log_expert_2d = _lang_expert_2d
            if _log_expert_2d is None and current_expert_action_40d is not None:
                if not np.allclose(current_expert_action_40d, 0.0):
                    try:
                        _log_expert_2d = _expert_action_to_accel_steer(
                            current_expert_action_40d, simlingo_base, current_speed
                        )
                    except Exception:
                        pass

        if buffer is not None:
            buffer.add(
                vlm_features, next_vlm_features,
                current_base_action, next_base_action,
                actor_chosen_final_action,
                current_obs_state, obs["state"][6:],
                chunk_reward_discounted, done,  # discounted: r0 + γ·r1 + … + γ^(n-1)·r_{n-1}
                expert_action=_sac_expert_3d,
                log_expert_action=_log_expert_2d,
            )
            if FLAGS.use_language_bow_critic and _lang_bow_label is not None:
                _cur_ptr = buffer.last_ptr
                # Write coach label for s_t immediately; approximate next_coach = current.
                buffer.update_at(_cur_ptr, coach_label=_lang_bow_label, next_coach_label=_lang_bow_label)
                # Correct previous step's next_coach_label with the current step's label.
                if _prev_lang_bow_ptr is not None and not _prev_transition_done:
                    buffer.update_next_coach_at(_prev_lang_bow_ptr, _lang_bow_label)
                _prev_lang_bow_ptr = _cur_ptr
            elif FLAGS.use_language_bow_critic:
                # Expert unavailable this step — clear backfill chain to avoid stale carry-over.
                _prev_lang_bow_ptr = None
            if FLAGS.use_noise_critic:
                buffer.update_at(buffer.last_ptr, coach_label=_noise_label, next_coach_label=_noise_label)

        _prev_transition_done = done

        # ── Obs histogram collection ──────────────────────────────────────────
        # Runs after critic labels are computed so we only collect what actually
        # goes into the critic, not the raw 40D waypoint chunk.
        if FLAGS.debug_obs_hist:
            _hist_embed.append(vlm_features.copy())
            _hist_other["ego state"].append(current_obs_state.copy())
            _hist_other["base action"].append(current_base_action.copy())
            _hist_other["residual action"].append(residual_action.copy())
            if FLAGS.use_expert_in_critic and _sac_expert_3d is not None:
                _hist_other["expert action (critic)"].append(_sac_expert_3d.copy())
            if FLAGS.use_language_bow_critic and _lang_bow_label is not None:
                _hist_other["language BoW (critic)"].append(_lang_bow_label.copy())
            if FLAGS.use_noise_critic:
                _hist_other["noise label (critic)"].append(_noise_label.copy())
            if len(_hist_embed) >= FLAGS.debug_obs_hist_steps:
                print(f"[debug_obs_hist] {len(_hist_embed)} samples collected — plotting histograms...", flush=True)
                out_png = _plot_obs_histograms(_hist_embed, _hist_other, save_dir, global_step, FLAGS.obs_mode)
                print(f"[debug_obs_hist] Saved to {out_png}", flush=True)
                if FLAGS.wandb_mode != "disabled":
                    wandb.log({"debug/obs_histograms": wandb.Image(out_png)}, step=global_step)
                wandb.finish()
                env.close()
                return

        episode_reward += chunk_reward
        episode_env_reward += chunk_env_reward
        episode_steps += 1

        # Coach: record buffer index + trajectory step (1-indexed episode_steps).
        if _coach_session is not None:
            _coach_session.track_buffer_transition(
                buffer_index=buffer.last_ptr,
                episode_step=episode_steps,
            )
            _coach_session.record_trajectory_step({
                "step": global_step,
                "episode_step": episode_steps,
                "ego_speed_mps": float(last_actual_speed),
                "control_throttle": float(max(0.0, last_final_action[0])),
                "control_steer": float(last_final_action[1]),
                "control_brake": float(max(0.0, -last_final_action[0])),
                "collision": bool(last_collision_delta > 0),
                "collision_active": bool(last_step_info.get("collision_contact_active", False)),
                "route_progress_pct": float(last_step_info.get("route_completion_pct", 0.0)),
                "in_video": True,
                "video_frame_index": (episode_steps - 1) * ticks_per_wp,
                "video_timestamp_sec": (episode_steps - 1) * ticks_per_wp / 20.0,
            })
            # Mid-episode query: only when not done (episode end handles that separately).
            if not done and _coach_session.maybe_query(
                episode_step=episode_steps, done_info=None, force=False, global_step=global_step
            ):
                _coach_session.backfill_buffer(buffer, global_step=global_step)

        # ── SAC updates ───────────────────────────────────────────────────────
        t_update_start = time.time()
        if buffer is not None and agent is not None and len(buffer) >= FLAGS.learning_starts and not in_warmup:
            for _ in range(FLAGS.updates_per_step):
                batch = buffer.sample(FLAGS.batch_size, torch.device(FLAGS.device))
                last_sac_metrics = agent.update(batch)
                total_updates += 1
        t_update_end = time.time()
        last_update_time = t_update_end - t_update_start

        # ── Per-step wandb logging (SAC metrics + timing) ─────────────────────
        if global_step % FLAGS.log_interval == 0:
            step_log: Dict[str, Any] = {
                "time/steps_per_sec": FLAGS.log_interval / max(time.time() - last_log_time, 1e-6),
                "time/global_step": global_step,
                "time/sample_time": t_sample_end - t_sample_start,
                "time/step_time": t_step_end - t_step_start,
                "time/vlm_time": t_vlm_end - t_vlm_start,
                "time/update_time": last_update_time,
                "training/in_warmup": float(in_warmup),
                "training/buffer_size": len(buffer) if buffer is not None else 0,
                "training/total_updates": total_updates,
                "reward/mode": reward_mode,
                "reward/train": float(last_train_reward),
                "reward/env_step": float(last_env_reward),
                "reward/chunk_train": float(chunk_reward),
                "reward/chunk_env": float(chunk_env_reward),
                "rollout/chunk_reward": float(chunk_reward),
                "rollout/chunk_env_reward": float(chunk_env_reward),
                "rollout/env_reward": float(last_env_reward),
                "rollout/train_reward": float(last_train_reward),
                "rollout/debug_reward": float(last_train_reward),
                "rollout/collision_count": float(last_step_info.get("collision_count", 0)),
                "rollout/collision_events": float(last_collision_delta),
                "rollout/episode_collision_count": float(episode_collision_count),
                "rollout/episode_collision_events": float(episode_collision_events),
                "rollout/current_episode_reward": float(episode_reward),
                "rollout/current_episode_return": float(episode_reward),
                "rollout/current_episode_env_return": float(episode_env_reward),
                "rollout/current_episode_steps": float(episode_steps),
                "rollout/actual_speed": float(last_actual_speed),
                "action/base_accel": float(last_base_action[0]),
                "action/base_steer": float(last_base_action[1]),
                "action/residual_accel": float(residual_action[0]),
                "action/residual_steer": float(residual_action[1]),
                "action/residual_norm": float(np.linalg.norm(residual_action)),
                "action/final_accel": float(last_final_action[0]),
                "action/final_steer": float(last_final_action[1]),
                "action/res_scale_accel": float(FLAGS.res_scale_accel),
                "action/res_scale_steer": float(FLAGS.res_scale_steer),
                "action/residual_clip_limit": float(sac_clip_limit),
                "action/expert_accel": float(last_expert_action_2d[0]),
                "action/expert_steer": float(last_expert_action_2d[1]),
                "action/expert_valid": float(_sac_expert_2d is not None),
                "simlingo/desired_speed_first": float(desired_speeds[0]),
                "simlingo/desired_speed_mean": float(np.mean(desired_speeds)),
                "simlingo/desired_speed_min": float(np.min(desired_speeds)),
                "simlingo/desired_speed_max": float(np.max(desired_speeds)),
                "simlingo/vlm_feature_norm": float(np.linalg.norm(vlm_features)),
                "simlingo/obs_mode": FLAGS.obs_mode,
                "simlingo/obs_dim": float(vlm_dim),
            }
            step_log.update({f"rollout/{k}": float(v) for k, v in last_drive_metrics.items()})
            if last_step_info.get("reward_total") is not None:
                step_log.update({
                    # Keep reward/total aligned with the reward optimized by SAC.
                    # Raw CARLA reward remains available as reward/env_total.
                    "reward/total": float(last_train_reward) if reward_mode != "env" else float(last_step_info.get("reward_total", 0.0)),
                    "reward/env_total": float(last_step_info.get("reward_total", 0.0)),
                    "reward/progress": float(last_step_info.get("reward_progress", 0.0)),
                    "reward/centering": float(last_step_info.get("reward_centering", 0.0)),
                    "reward/heading": float(last_step_info.get("reward_heading", 0.0)),
                    "reward/terminal": float(last_step_info.get("reward_terminal", 0.0)),
                    "reward/penalty_collision": float(last_step_info.get("penalty_collision", 0.0)),
                    "reward/penalty_outside_route": float(last_step_info.get("penalty_outside_route", 0.0)),
                    "reward/penalty_traffic_violation": float(last_step_info.get("penalty_traffic_violation", 0.0)),
                    "rollout/traffic_violation_count": float(last_step_info.get("traffic_violation_count", 0)),
                    "reward/penalty_steer": float(last_step_info.get("penalty_steer", 0.0)),
                    "reward/penalty_brake": float(last_step_info.get("penalty_brake", 0.0)),
                    "reward/penalty_speed_limit": float(last_step_info.get("penalty_speed_limit", 0.0)),
                    "reward/penalty_crash_stuck": float(last_step_info.get("penalty_crash_stuck", 0.0)),
                    "reward/collision_penalty_active": float(bool(last_step_info.get("collision_penalty_active", False))),
                    "reward/collision_contact_active": float(bool(last_step_info.get("collision_contact_active", False))),
                    "rollout/lane_offset_m": float(last_step_info.get("lane_offset_m", 0.0)),
                    "rollout/heading_error_rad": float(last_step_info.get("heading_error_rad", 0.0)),
                    "rollout/speed_norm": float(last_step_info.get("speed_norm", 0.0)),
                    "rollout/centering_factor": float(last_step_info.get("centering_factor", 0.0)),
                    "rollout/heading_factor": float(last_step_info.get("heading_factor", 0.0)),
                })
            if last_sac_metrics:
                step_log.update({f"training/{k}": v for k, v in last_sac_metrics.items()})
            step_log.update(_state_norm_wandb_log(agent))
            wandb.log(step_log, step=global_step)
            last_log_time = time.time()

        # ── Episode end ───────────────────────────────────────────────────────
        if done:
            num_episodes += 1

            # Coach: query Gemini at episode end, backfill replay buffer labels,
            # then reset the session so the next episode starts clean.
            if _coach_session is not None:
                _coach_session.maybe_query(episode_step=episode_steps, done_info=info, force=True, global_step=global_step)
                _coach_session.backfill_buffer(buffer, global_step=global_step)
                _coach_session.reset_episode()

            elapsed = time.time() - t0
            log_entry = {
                "global_step": global_step,
                "episode": num_episodes,
                "episode_reward": episode_reward,
                "episode_env_reward": episode_env_reward,
                "episode_steps": episode_steps,
                "success": info.get("success", False),
                "collision_count": info.get("collision_count", 0),
                "collision_events": episode_collision_events,
                "termination_reason": info.get("termination_reason", "leaderboard"),
                "elapsed_s": elapsed,
            }
            print(
                f"[step {global_step}] ep={num_episodes}  "
                f"R={episode_reward:.2f}  steps={episode_steps}  "
                f"success={info.get('success', False)}  "
                f"term={info.get('termination_reason', 'leaderboard')}  "
                f"updates={total_updates}",
                flush=True,
            )
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            ep_log: Dict[str, Any] = {
                "rollout/episode_reward": episode_reward,
                "rollout/episode_env_reward": episode_env_reward,
                "rollout/episode_return": episode_reward,
                "rollout/episode_env_return": episode_env_reward,
                "rollout/episode_steps": episode_steps,
                "rollout/success": float(info.get("success", False)),
                "rollout/collision_count": float(info.get("collision_count", 0)),
                "rollout/episode_collision_count": float(episode_collision_count),
                "rollout/episode_collision_events": float(episode_collision_events),
                "rollout/collisions_over_episode": float(episode_collision_events) / max(float(episode_steps), 1.0),
                "rollout/outside_route": float(info.get("outside_route_value", 0.0)),
                "rollout/num_episodes": num_episodes,
                "rollout/episodes": num_episodes,
                "rollout/route": FLAGS.route or "?",
            }
            if info.get("reward_total") is not None:
                ep_log.update({
                    "reward/total": float(episode_reward) if reward_mode != "env" else float(info.get("reward_total", 0.0)),
                    "reward/env_total": float(info.get("reward_total", 0.0)),
                    "reward/progress": float(info.get("reward_progress", 0.0)),
                    "reward/centering": float(info.get("reward_centering", 0.0)),
                    "reward/heading": float(info.get("reward_heading", 0.0)),
                    "reward/terminal": float(info.get("reward_terminal", 0.0)),
                    "reward/penalty_collision": float(info.get("penalty_collision", 0.0)),
                    "reward/penalty_outside_route": float(info.get("penalty_outside_route", 0.0)),
                    "reward/penalty_steer": float(info.get("penalty_steer", 0.0)),
                    "reward/penalty_brake": float(info.get("penalty_brake", 0.0)),
                    "reward/penalty_speed_limit": float(info.get("penalty_speed_limit", 0.0)),
                    "reward/penalty_crash_stuck": float(info.get("penalty_crash_stuck", 0.0)),
                    # Backward-compatible names from early SimLingo SAC runs.
                    "reward/collision_penalty": float(info.get("penalty_collision", 0.0)),
                    "reward/outside_route_penalty": float(info.get("penalty_outside_route", 0.0)),
                    "rollout/final_step_reward": float(info.get("reward_total", 0.0)),
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
                })
            if _frame_buffer and FLAGS.video_log_interval > 0 and num_episodes % FLAGS.video_log_interval == 0:
                frames_np = np.stack(_frame_buffer)  # (T, H, W, 3)
                ep_log["rollout/episode_video"] = wandb.Video(
                    frames_np.transpose(0, 3, 1, 2), fps=20, format="mp4"
                )
            wandb.log(ep_log, step=global_step)

            _close_video(video)
            obs, _ = env.reset()
            simlingo_base.reset_pid()
            episode_reward = 0.0
            episode_env_reward = 0.0
            episode_steps = 0
            episode_collision_count = 0
            episode_collision_events = 0
            prev_collision_count = 0
            last_language_feedback = None
            _prev_lang_bow_ptr = None
            video = _open_video(num_episodes)
            desired_speeds, _route_interp, vlm_features = _get_features(obs)

            # Coach: begin the next episode after env reset.
            if _coach_session is not None:
                _coach_session.begin_episode(episode_count=num_episodes, route_name=FLAGS.route or "")
        else:
            desired_speeds = next_desired_speeds
            vlm_features = next_vlm_features
            # route_interp is already stored in simlingo_base._last_route_interp

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if agent is not None and global_step > 0 and global_step % FLAGS.save_interval == 0:
            ckpt_path = str(save_dir / f"residual_sac_{global_step}.pt")
            agent.save(ckpt_path)
            print(f"[step {global_step}] Saved checkpoint to {ckpt_path}", flush=True)

    # ── Final save ────────────────────────────────────────────────────────────
    _close_video(video)
    if agent is not None:
        final_path = str(save_dir / "residual_sac_final.pt")
        agent.save(final_path)
        print(f"\n[train] Done. Final checkpoint at {final_path}", flush=True)
    else:
        print("\n[train] Done (expert debug mode — no agent checkpoint).", flush=True)
    log_file.close()
    wandb.finish()
    env.close()


if __name__ == "__main__":
    app.run(main)
