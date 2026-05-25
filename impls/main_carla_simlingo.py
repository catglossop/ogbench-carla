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
from ogbench.carla.carla_utils import ego_drive_metrics_from_state_vec
from utils.log_utils import get_exp_name, setup_wandb  # type: ignore

FLAGS = flags.FLAGS

flags.DEFINE_string("simlingo_checkpoint", None, "Path to SimLingo epoch=013.ckpt directory.")
flags.DEFINE_string("route", None, "Bench2Drive route (scenario name, file basename, or route id).")
flags.DEFINE_bool("eval_only", False, "Run base policy only (no residual training).")
flags.DEFINE_integer("total_steps", 10_000, "Total environment steps for training.")
flags.DEFINE_integer("warmup_steps", 500, "Steps collecting data before SAC updates begin.")
flags.DEFINE_integer("learning_starts", 500, "Buffer size threshold before updates begin.")
flags.DEFINE_integer("updates_per_step", 4, "SAC gradient updates per env step.")
flags.DEFINE_integer("batch_size", 256, "SAC mini-batch size.")
flags.DEFINE_integer("buffer_capacity", 10_000, "Replay buffer capacity.")
flags.DEFINE_float("res_scale", 0.1, "Residual action scaling (final = base + scale*residual).")
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
flags.DEFINE_string("run_group", "Debug", "W&B run group.")
flags.DEFINE_enum("wandb_mode", "online", ["online", "offline", "disabled"], "W&B logging mode.")
flags.DEFINE_integer("log_interval", 1, "Log training metrics to W&B every N episodes.")
flags.DEFINE_integer("video_log_interval", 5, "Upload episode video to W&B every N episodes (0=never).")


# ── Video overlay ─────────────────────────────────────────────────────────────

def _annotate_frame(
    image_rgb: np.ndarray,
    simlingo_base,
    target_points: Optional[np.ndarray],
    current_speed: float,
    base_action: np.ndarray,
    residual_action: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw projected waypoints + HUD onto a video frame.

    Green dots  = speed waypoints (from speed head)
    Red dots    = route waypoints (from route head)
    Blue dots   = GPS target waypoints (from obs)
    Yellow text = HUD (speed, accel, steer, residual)
    """
    try:
        from PIL import Image as _PIL_Image, ImageDraw as _ImageDraw, ImageFont as _ImageFont
        from team_code.simlingo_utils import project_points, get_camera_intrinsics  # type: ignore
    except ImportError:
        return image_rgb

    H, W = image_rgb.shape[:2]
    K = get_camera_intrinsics(W, H, 110).numpy()

    pil_img = _PIL_Image.fromarray(image_rgb).convert("RGBA")
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

    # HUD
    hud = [
        f"spd {current_speed:.1f} m/s",
        f"acc {base_action[0]:+.2f}  str {base_action[1]:+.3f}",
    ]
    if residual_action is not None:
        hud.append(f"res {residual_action[0]:+.2f} / {residual_action[1]:+.3f}")
    for i, line in enumerate(hud):
        draw.text((10, 10 + i * 18), line, fill=(255, 255, 0, 255))

    return np.array(pil_img.convert("RGB"))


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
        return {
            "state": np.array(wire["state"], dtype=np.float32),
            "simlingo_image": img,
            "routing_command": wire["routing_command"],
            "target_points": np.array(tp_raw, dtype=np.float32),  # (2, 2) ego-frame
        }

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main(_argv):
    np.random.seed(FLAGS.seed)

    if FLAGS.simlingo_checkpoint is None:
        raise ValueError("--simlingo_checkpoint is required.")
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
    print("[main] Loading SimLingo base policy ...", flush=True)
    from vlas.simlingo_base import SimLingoBase, VLM_FEATURE_DIM  # type: ignore
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
        writer = cv2.VideoWriter(path, fourcc, 20.0, (1024, 512))
        print(f"[video] Writing {path}", flush=True)
        return writer

    def _write_frame(writer, image_rgb, annotated: Optional[np.ndarray] = None):
        frame = annotated if annotated is not None else image_rgb
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
        print("[main] Eval-only mode: rolling out base policy on 2 episodes ...", flush=True)
        results: List[Dict[str, Any]] = []

        # Eval always runs VLM every CARLA tick (1 tick per wp call) to match
        # the reference eval's 20Hz inference rate, regardless of chunk_size.
        ticks_per_wp = 1
        chunk_size = FLAGS.chunk_size  # only affects how many speed targets are pre-computed

        for ep_idx in range(2):
            if ep_idx == 0:
                obs = initial_obs
            else:
                obs, _ = env.reset()
            simlingo_base.reset_pid()
            episode_reward = 0.0
            steps = 0
            info: Dict[str, Any] = {}
            video = _open_video(ep_idx)

            print(f"\n[eval] Episode {ep_idx + 1} / 2", flush=True)
            done = False
            while not done and steps < 4000:
                # VLM call: get desired speeds + store route_interp for steer_for_speed()
                desired_speeds, _route_interp, _ = simlingo_base.get_chunk_and_features(
                    simlingo_image=obs["simlingo_image"],
                    ego_state=obs["state"],
                    target_points=obs["target_points"],
                )
                for k in range(chunk_size):
                    for _tick in range(ticks_per_wp):
                        actual_speed = float(obs["state"][15])
                        base_accel = simlingo_base.accel_for_desired_speed(desired_speeds[k], actual_speed)
                        # Per-tick steer: lateral PID integrates error history each tick
                        # using the route from the current VLM call (matches control_pid)
                        base_steer = simlingo_base.steer_for_speed(actual_speed)
                        action = np.array([base_accel, base_steer], dtype=np.float32)
                        annotated = _annotate_frame(
                            obs["simlingo_image"], simlingo_base,
                            obs.get("target_points"), actual_speed, action,
                        )
                        _write_frame(video, obs["simlingo_image"], annotated)
                        obs, reward, terminated, truncated, info = env.step(action)
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
    )

    reward_mode = "neg_speed" if FLAGS.debug_neg_speed_reward else "env"
    print(f"[train] Starting residual SAC training for {FLAGS.total_steps} steps "
          f"(chunk_size={chunk_size}, ticks_per_wp={ticks_per_wp}, "
          f"reward_mode={reward_mode}) ...", flush=True)
    t0 = time.time()
    last_log_time = t0
    last_sac_metrics: Dict[str, float] = {}
    last_step_info: Dict[str, Any] = {}
    last_drive_metrics = ego_drive_metrics_from_state_vec(obs["state"])
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
        if in_warmup:
            residual_action = np.zeros(2, dtype=np.float32)
        else:
            residual_action = agent.sample_actions(vlm_features)
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
                annotated = _annotate_frame(
                    obs["simlingo_image"], simlingo_base,
                    obs.get("target_points"), actual_speed,
                    base_action, residual_action,
                )
                _write_frame(video, obs["simlingo_image"], annotated)
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
                last_drive_metrics = ego_drive_metrics_from_state_vec(next_obs["state"])
                collision_count = int(info.get("collision_count", 0))
                last_collision_delta = max(0, collision_count - prev_collision_count)
                episode_collision_count = max(episode_collision_count, collision_count)
                episode_collision_events += last_collision_delta
                prev_collision_count = collision_count
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
