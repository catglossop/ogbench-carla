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
flags.DEFINE_float("res_scale", 0.1, "Residual action scaling (final = base + scale*residual).")
flags.DEFINE_integer("residual_clip_schedule_steps", 0,
                     "Steps after warmup over which the residual clip limit ramps linearly from 0 to 1. "
                     "0 = no schedule (full [-1, 1] range immediately after warmup).")
flags.DEFINE_float("gamma", 0.97, "Discount factor.")
flags.DEFINE_float("tau", 0.01, "Target network soft-update coefficient.")
flags.DEFINE_float("actor_lr", 1e-4, "Actor learning rate.")
flags.DEFINE_float("critic_lr", 1e-4, "Critic learning rate.")
flags.DEFINE_string("save_dir", "./logs/simlingo_residual", "Directory for checkpoints and logs.")
flags.DEFINE_integer("save_interval", 2000, "Save residual SAC checkpoint every N steps.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("chunk_size", 10, "Waypoints to execute per VLM call (1–10). 1 = VLM every tick; 10 = full predicted chunk.")
flags.DEFINE_bool("save_video", True, "Write per-episode mp4 videos of the simlingo camera feed.")
flags.DEFINE_string("carla_config", None, "Path to carla_config.yaml.")
flags.DEFINE_string("device", "cuda", "Torch device for SimLingo and residual SAC.")
flags.DEFINE_integer("gpu_rank", 0, "CARLA rendering GPU rank.")
# conda env name for the carla_env_server.py subprocess (must have carla 0.9.15 installed).
flags.DEFINE_string("server_conda_env", "simlingo", "conda env for the CARLA env server process.")
flags.DEFINE_string("carla_root", "/home/celinet/VLA_driving/software",
                    "CARLA root dir (forwarded to env server as CARLA_ROOT). "
                    "Default is CARLA 0.9.15 which works with Town12.")
flags.DEFINE_bool("debug_neg_speed_reward", False,
                  "Debug: replace env reward with -speed (m/s). SAC should learn to slow the car.")
flags.DEFINE_bool("expert_debug", False,
                  "Debug: drive with the CARLA expert action instead of base+residual (dagger_residual only).")
flags.DEFINE_bool("expert_recover_debug", False,
                  "Debug: run SimLingo for a random [70,200] ticks per episode, then switch to CARLA expert.")
flags.DEFINE_string("run_group", "Debug", "W&B run group.")
flags.DEFINE_enum("wandb_mode", "online", ["online", "offline", "disabled"], "W&B logging mode.")
flags.DEFINE_integer("log_interval", 1, "Log training metrics to W&B every N episodes.")
flags.DEFINE_integer("video_log_interval", 5, "Upload episode video to W&B every N episodes (0=never).")
flags.DEFINE_integer("eval_episodes", 2, "Number of episodes to run in eval-only mode.")
flags.DEFINE_integer("eval_step_limit", 4000, "Maximum CARLA ticks per eval episode.")
flags.DEFINE_enum("training_mode", "sac_residual", ["sac_residual", "dagger_residual"],
                  "Training mode: sac_residual (RL with env reward) or "
                  "dagger_residual (BC with expert planner labels).")


# ── Video overlay ─────────────────────────────────────────────────────────────

_VIDEO_PANEL_H = 113


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
    collision_now = bool(collision_events > 0 or collision_count > 0)
    residual = residual_action if residual_action is not None else np.zeros(2, dtype=np.float32)
    final_action = np.clip(base_action + FLAGS.res_scale * residual, -1.0, 1.0)
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
            label = f"COLLISION c={collision_count} e={collision_events}"
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
        pen_crash = float(info.get("penalty_crash_stuck", 0.0))
        pen_term = float(info.get("reward_terminal", 0.0))
        contact = bool(info.get("collision_contact_active", False))

        if expert_action_2d is not None:
            expert_str = f"expert=({expert_action_2d[0]:+.2f},{expert_action_2d[1]:+.3f})"
        else:
            expert_str = ""

        lines = [
            f"Reward train={train_reward} env={env_reward} | speed={current_speed:.2f} m/s | collision={'YES' if collision_now else 'no'} c={collision_count} e={collision_events}",
            f"Action base=({base_action[0]:+.2f},{base_action[1]:+.3f}) residual=({residual[0]:+.2f},{residual[1]:+.3f}) final=({final_action[0]:+.2f},{final_action[1]:+.3f}){('  ' + expert_str) if expert_str else ''}",
            f"Meta-action: {_clip_text(meta_action) if meta_action else '(none)'}",
            f"Prompt: {_clip_text(prompt)}",
            f"Reasoning: {_clip_text(language) if language else '(no language output)'}",
            f"Pen: coll={pen_coll:+.1f}{'(bb)' if contact else ''}  route={pen_route:+.1f}  crash={pen_crash:+.1f}  term={pen_term:+.1f}",
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main(_argv):
    np.random.seed(FLAGS.seed)

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
        project="OGBench-CARLA-SimLingo",
        group=FLAGS.run_group,
        name=f"{FLAGS.route}_{exp_name}",
        mode=FLAGS.wandb_mode,
    )
    FLAGS.save_dir = str(Path(save_dir_base) / wandb.run.project / FLAGS.run_group / exp_name)

    save_dir = Path(FLAGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load SimLingo base policy ─────────────────────────────────────────────
    print(f"[main] Loading SimLingo policy (mode={FLAGS.policy_mode}) ...", flush=True)
    from vlas.simlingo_base import HierarchicalSimLingoPolicy, SimLingoBase, VLM_FEATURE_DIM  # type: ignore
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
    )

    # Read the initial obs that the server sends after startup
    # (the server sends ready + initial obs automatically)
    # We need to read that initial obs line:
    init_line = env._readline()
    init_resp = json.loads(init_line)
    initial_obs = CarlaEnvProxy._wire_to_obs(init_resp["obs"])

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
                        annotated = _annotate_frame(
                            image_for_video,
                            simlingo_base,
                            target_points_for_video,
                            actual_speed,
                            action,
                            reward_value=float(reward),
                            env_reward_value=float(reward),
                            info=info,
                            collision_events=int(info.get("collision_count", 0)),
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
            stats = {
                "episode_reward": episode_reward,
                "steps": steps,
                "success": info.get("success", False),
                "collision_count": info.get("collision_count", 0),
                "outside_route": info.get("outside_route_value", 0.0),
                "route": FLAGS.route,
            }
            results.append(stats)
            print(
                f"[eval] ep={ep_idx+1}  reward={stats['episode_reward']:.2f}"
                f"  steps={stats['steps']}  success={stats['success']}"
                f"  collisions={stats['collision_count']}",
                flush=True,
            )
            wb_log = {
                "eval/episode_reward": episode_reward,
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
        from torch_agents.residual_sac import ResidualSACAgent, DaggerBuffer  # type: ignore
        import torch

        vlm_dim = VLM_FEATURE_DIM

        agent = ResidualSACAgent(
            vlm_feature_dim=vlm_dim,
            action_dim=2,
            hidden_dims=(256, 256, 256),
            gamma=FLAGS.gamma,
            tau=FLAGS.tau,
            actor_lr=FLAGS.actor_lr,
            critic_lr=FLAGS.critic_lr,
            device=FLAGS.device,
        )
        buffer: Any = DaggerBuffer(capacity=FLAGS.buffer_capacity, vlm_dim=vlm_dim)

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

        desired_speeds, _route_interp, vlm_features = simlingo_base.get_chunk_and_features(
            simlingo_image=obs["simlingo_image"],
            ego_state=obs["state"],
            target_points=obs["target_points"],
            routing_command=obs.get("routing_command", ""),
        )

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
            if in_warmup:
                # Warmup: execute base policy only (no residual)
                residual_action = np.zeros(2, dtype=np.float32)
            else:
                # DAgger: execute deterministic actor mean (no exploration noise)
                residual_action = agent.get_eval_action(vlm_features)
                residual_action = np.clip(residual_action, -dagger_clip_limit, dagger_clip_limit)
            last_actor_output = residual_action.copy()
            t_sample_end = time.time()

            # Capture speed and base action at the START of the chunk so that the
            # expert residual uses the same state as vlm_features.
            chunk_start_speed = float(obs["state"][15])
            chunk_start_base_accel = simlingo_base.accel_for_desired_speed(desired_speeds[0], chunk_start_speed)
            chunk_start_base_steer = simlingo_base.steer_for_speed(chunk_start_speed)
            chunk_start_base_action = np.array([chunk_start_base_accel, chunk_start_base_steer], dtype=np.float32)

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
                    except Exception:
                        pass

            chunk_reward = 0.0
            chunk_env_reward = 0.0
            done = False
            info: Dict[str, Any] = {}
            t_step_start = time.time()
            for k in range(chunk_size):
                for _tick in range(ticks_per_wp):
                    actual_speed = float(obs["state"][15])
                    base_accel = simlingo_base.accel_for_desired_speed(desired_speeds[k], actual_speed)
                    base_steer = simlingo_base.steer_for_speed(actual_speed)
                    base_action = np.array([base_accel, base_steer], dtype=np.float32)
                    if _expert_debug_action is not None:
                        final_action = _expert_debug_action
                    else:
                        final_action = np.clip(
                            base_action + FLAGS.res_scale * residual_action, -1.0, 1.0
                        ).astype(np.float32)
                    image_for_video = obs["simlingo_image"]
                    target_points_for_video = obs.get("target_points")
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
            if current_expert_action_40d is not None:
                ea_40d = current_expert_action_40d
                if not np.allclose(ea_40d, 0.0):
                    try:
                        ea_2d = _expert_action_to_accel_steer(ea_40d, simlingo_base, chunk_start_speed)
                        last_expert_action_2d = ea_2d
                        last_expert_residual_target = np.clip(
                            (ea_2d - chunk_start_base_action) / max(FLAGS.res_scale, 1e-6), -1.0, 1.0
                        ).astype(np.float32)
                        expert_action_2d = ea_2d
                    except Exception as _ex:
                        print(f"[dagger] expert action conversion failed: {_ex}", flush=True)

            # ── Next VLM call ─────────────────────────────────────────────────
            t_vlm_start = time.time()
            next_desired_speeds, _next_route_interp, next_vlm_features = simlingo_base.get_chunk_and_features(
                simlingo_image=obs["simlingo_image"],
                ego_state=obs["state"],
                target_points=obs["target_points"],
                routing_command=obs.get("routing_command", ""),
            )
            t_vlm_end = time.time()

            # ── Buffer add ────────────────────────────────────────────────────
            if expert_action_2d is not None:
                buffer.add(vlm_features, chunk_start_base_action, expert_action_2d)

            episode_reward += chunk_reward
            episode_env_reward += chunk_env_reward
            episode_steps += 1

            # ── BC updates ────────────────────────────────────────────────────
            t_update_start = time.time()
            if len(buffer) >= FLAGS.learning_starts and not in_warmup:
                for _ in range(FLAGS.updates_per_step):
                    batch = buffer.sample(FLAGS.batch_size, torch.device(FLAGS.device))
                    last_bc_metrics = agent.bc_update(batch, res_scale=FLAGS.res_scale)
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
                    "action/res_scale": float(FLAGS.res_scale),
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
                }
                step_log.update({f"rollout/{k}": float(v) for k, v in last_drive_metrics.items()})
                if last_step_info.get("reward_total") is not None:
                    step_log.update({
                        "reward/env_total": float(last_step_info.get("reward_total", 0.0)),
                        "reward/progress": float(last_step_info.get("reward_progress", 0.0)),
                        "reward/penalty_collision": float(last_step_info.get("penalty_collision", 0.0)),
                        "reward/penalty_outside_route": float(last_step_info.get("penalty_outside_route", 0.0)),
                        "rollout/lane_offset_m": float(last_step_info.get("lane_offset_m", 0.0)),
                        "rollout/speed_norm": float(last_step_info.get("speed_norm", 0.0)),
                    })
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
                    "elapsed_s": elapsed,
                }
                print(
                    f"[step {global_step}] ep={num_episodes}  "
                    f"R={episode_reward:.2f}  steps={episode_steps}  "
                    f"success={info.get('success', False)}  "
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
                desired_speeds, _route_interp, vlm_features = simlingo_base.get_chunk_and_features(
                    simlingo_image=obs["simlingo_image"],
                    ego_state=obs["state"],
                    target_points=obs["target_points"],
                    routing_command=obs.get("routing_command", ""),
                )
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
    from torch_agents.residual_sac import ResidualSACAgent, ReplayBuffer  # type: ignore
    import torch

    vlm_dim = VLM_FEATURE_DIM  # 896

    agent = ResidualSACAgent(
        vlm_feature_dim=vlm_dim,
        action_dim=2,
        hidden_dims=(256, 256, 256),
        gamma=FLAGS.gamma,
        tau=FLAGS.tau,
        actor_lr=FLAGS.actor_lr,
        critic_lr=FLAGS.critic_lr,
        device=FLAGS.device,
    )
    buffer = ReplayBuffer(capacity=FLAGS.buffer_capacity, vlm_dim=vlm_dim)

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

    chunk_size = FLAGS.chunk_size  # waypoints per VLM call (1–10)
    # Each predicted waypoint covers WP_DILATION * DATA_SAVE_FREQ CARLA ticks.
    ticks_per_wp = simlingo_base._WP_DILATION * simlingo_base._DATA_SAVE_FREQ  # = 5

    # Initial VLM call so the loop starts with features ready.
    desired_speeds, _route_interp, vlm_features = simlingo_base.get_chunk_and_features(
        simlingo_image=obs["simlingo_image"],
        ego_state=obs["state"],
        target_points=obs["target_points"],
        routing_command=obs.get("routing_command", ""),
    )

    reward_mode = "neg_speed" if FLAGS.debug_neg_speed_reward else "env"
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

    for global_step in range(FLAGS.total_steps):
        t_sample_start = time.time()
        # ── Rollout: execute chunk_size waypoints, ticks_per_wp ticks each ───
        in_warmup = global_step < FLAGS.warmup_steps
        sac_clip_limit = _residual_clip_limit(global_step, FLAGS.warmup_steps, FLAGS.residual_clip_schedule_steps)
        if in_warmup:
            residual_action = np.zeros(2, dtype=np.float32)
        else:
            residual_action = agent.sample_actions(vlm_features)
            residual_action = np.clip(residual_action, -sac_clip_limit, sac_clip_limit)
        t_sample_end = time.time()

        chunk_reward = 0.0
        chunk_env_reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        t_step_start = time.time()
        for k in range(chunk_size):
            for _tick in range(ticks_per_wp):
                actual_speed = float(obs["state"][15])
                base_accel = simlingo_base.accel_for_desired_speed(desired_speeds[k], actual_speed)
                base_steer = simlingo_base.steer_for_speed(actual_speed)
                base_action = np.array([base_accel, base_steer], dtype=np.float32)
                final_action = np.clip(
                    base_action + FLAGS.res_scale * residual_action, -1.0, 1.0
                ).astype(np.float32)
                image_for_video = obs["simlingo_image"]
                target_points_for_video = obs.get("target_points")
                next_obs, reward, terminated, truncated, info = env.step(final_action)
                env_reward = float(reward)
                if FLAGS.debug_neg_speed_reward:
                    reward = -float(next_obs["state"][15])
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

        # ── Next VLM call (for replay buffer and next iteration) ─────────────
        t_vlm_start = time.time()
        next_desired_speeds, _next_route_interp, next_vlm_features = simlingo_base.get_chunk_and_features(
            simlingo_image=obs["simlingo_image"],
            ego_state=obs["state"],
            target_points=obs["target_points"],
            routing_command=obs.get("routing_command", ""),
        )
        t_vlm_end = time.time()
        buffer.add(vlm_features, next_vlm_features, residual_action, chunk_reward, done)

        episode_reward += chunk_reward
        episode_env_reward += chunk_env_reward
        episode_steps += 1

        # ── SAC updates ───────────────────────────────────────────────────────
        t_update_start = time.time()
        if len(buffer) >= FLAGS.learning_starts and not in_warmup:
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
                "training/buffer_size": len(buffer),
                "training/total_updates": total_updates,
                "reward/mode_is_neg_speed": float(FLAGS.debug_neg_speed_reward),
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
                "action/res_scale": float(FLAGS.res_scale),
                "action/residual_clip_limit": float(sac_clip_limit),
                "simlingo/desired_speed_first": float(desired_speeds[0]),
                "simlingo/desired_speed_mean": float(np.mean(desired_speeds)),
                "simlingo/desired_speed_min": float(np.min(desired_speeds)),
                "simlingo/desired_speed_max": float(np.max(desired_speeds)),
                "simlingo/vlm_feature_norm": float(np.linalg.norm(vlm_features)),
            }
            step_log.update({f"rollout/{k}": float(v) for k, v in last_drive_metrics.items()})
            if last_step_info.get("reward_total") is not None:
                step_log.update({
                    # Keep reward/total aligned with the reward optimized by SAC.
                    # Raw CARLA reward remains available as reward/env_total.
                    "reward/total": float(last_train_reward) if FLAGS.debug_neg_speed_reward else float(last_step_info.get("reward_total", 0.0)),
                    "reward/env_total": float(last_step_info.get("reward_total", 0.0)),
                    "reward/progress": float(last_step_info.get("reward_progress", 0.0)),
                    "reward/centering": float(last_step_info.get("reward_centering", 0.0)),
                    "reward/heading": float(last_step_info.get("reward_heading", 0.0)),
                    "reward/terminal": float(last_step_info.get("reward_terminal", 0.0)),
                    "reward/penalty_collision": float(last_step_info.get("penalty_collision", 0.0)),
                    "reward/penalty_outside_route": float(last_step_info.get("penalty_outside_route", 0.0)),
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
            wandb.log(step_log, step=global_step)
            last_log_time = time.time()

        # ── Episode end ───────────────────────────────────────────────────────
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
                "elapsed_s": elapsed,
            }
            print(
                f"[step {global_step}] ep={num_episodes}  "
                f"R={episode_reward:.2f}  steps={episode_steps}  "
                f"success={info.get('success', False)}  "
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
                    "reward/total": float(episode_reward) if FLAGS.debug_neg_speed_reward else float(info.get("reward_total", 0.0)),
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
            video = _open_video(num_episodes)
            desired_speeds, _route_interp, vlm_features = simlingo_base.get_chunk_and_features(
                simlingo_image=obs["simlingo_image"],
                ego_state=obs["state"],
                target_points=obs["target_points"],
                routing_command=obs.get("routing_command", ""),
            )
        else:
            desired_speeds = next_desired_speeds
            vlm_features = next_vlm_features
            # route_interp is already stored in simlingo_base._last_route_interp

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if global_step > 0 and global_step % FLAGS.save_interval == 0:
            ckpt_path = str(save_dir / f"residual_sac_{global_step}.pt")
            agent.save(ckpt_path)
            print(f"[step {global_step}] Saved checkpoint to {ckpt_path}", flush=True)

    # ── Final save ────────────────────────────────────────────────────────────
    _close_video(video)
    final_path = str(save_dir / "residual_sac_final.pt")
    agent.save(final_path)
    log_file.close()
    wandb.finish()
    env.close()
    print(f"\n[train] Done. Final checkpoint at {final_path}", flush=True)


if __name__ == "__main__":
    app.run(main)
