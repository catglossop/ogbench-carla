#!/usr/bin/env python3
"""Collect "noisy" rollout data for critic pretraining by driving a route with an
*already-trained* checkpoint from the old torch-based SimLingo residual-SAC pipeline
(impls/main_carla_simlingo.py / impls/torch_agents/residual_sac.py), instead of the
autopilot. Useful for routes the autopilot can't complete (e.g. route 3936).

Writes data in the same layout impls/pretrain_critic.py's --noisy_root loader expects:

    <save-root>/ep_<NNN>/
        measurements/{t:04d}.json.gz
        rgb/{t:04d}.jpg

Unlike impls/collect_expert_data.py (which relies on AutoPilot.save() and only ever
produces the coarse non-noisy heuristic reward), this script logs the *exact* per-step
online reward breakdown -- the same reward_progress/penalty_*/collision/outside-route/
traffic-violation signals impls/main_carla.py's online CarlaEnv computes via
ogbench/carla/carla_utils.py's _compute_reward_and_info -- so that
impls/pretrain_critic.py's _rewards_noisy() path (which mirrors that function) trains
the critic on the exact same reward as online, not the cheap progress-only heuristic.

pretrain_critic.py's _waypoints_action() derives the action label purely from
consecutive ego_matrix positions, independent of what actually produced them -- so this
works regardless of the checkpoint's own action space (2D accel/steer here) vs. the
waypoints-based critic being trained.

Must be run the same way impls/main_carla_simlingo.py is: under the main .venv, which
spawns impls/carla_env_server.py in the `simlingo` conda env (CARLA 0.9.15) as a
subprocess -- no special invocation needed here.

Usage:
  uv run python impls/collect_checkpoint_rollout_data.py \\
      --route 3936 --save-root /scratch/current/celinet/critic_data_3936/noisy \\
      --checkpoint logs/simlingo_residual_port2020/OGBench-CARLA-SimLingo/Debug/sd000_20260527_071315/residual_sac_2000.pt \\
      --simlingo-checkpoint /home/celinet/simlingo_checkpoints/simlingo/checkpoints/epoch=013.ckpt \\
      --episodes 8 --max-steps 600
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

_IMPLS_ROOT = Path(__file__).resolve().parent
if str(_IMPLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPLS_ROOT))

import cv2


def _make_agent(vlm_feature_dim: int, state_dim: int, device: str, res_scale: float):
    from torch_agents.residual_sac import ResidualSACAgent

    return ResidualSACAgent(
        vlm_feature_dim=vlm_feature_dim,
        action_dim=2,
        hidden_dims=(256, 256, 256),
        gamma=0.97,
        tau=0.01,
        actor_lr=1e-4,
        critic_lr=1e-4,
        device=device,
        actor_l2_reg=0.0,
        res_scale=res_scale,
        state_dim=state_dim,
        coach_label_dim=0,
        expert_action_dim=0,
    )


def _measurement_from_info(ego_matrix, speed: float, info: dict) -> dict:
    return {
        "ego_matrix": ego_matrix,
        "speed": speed,
        "speed_limit": float(info.get("speed_limit_mps", 13.9)),
        "reward_component_progress": float(info.get("reward_progress", 0.0)),
        "reward_component_speed_limit_pen": float(info.get("penalty_speed_limit", 0.0)),
        "reward_component_crash_stuck_pen": float(info.get("penalty_crash_stuck", 0.0)),
        "reward_component_steer_pen": float(info.get("penalty_steer", 0.0)),
        "reward_component_brake_pen": float(info.get("penalty_brake", 0.0)),
        "reward_component_terminal": float(info.get("reward_terminal", 0.0)),
        "reward_collision_active": bool(info.get("collision_contact_active", False)),
        "reward_outside_road": bool(float(info.get("outside_route_value", 0.0)) > 0.0),
        "reward_traffic_light_violation": bool(float(info.get("traffic_violation_delta", 0.0)) > 0.0),
        "reward_stop_sign_violation": False,
        "reward_route_completion_delta": float(info.get("route_completion_delta", 0.0)),
    }


def collect_episode(env, simlingo_base, agent, res_scale_vec, *, episode_dir: Path, max_steps: int) -> int:
    meas_dir = episode_dir / "measurements"
    rgb_dir = episode_dir / "rgb"
    meas_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    obs, reset_info = env.reset()
    cur_ego_matrix = reset_info.get("ego_matrix")
    cur_speed = float(reset_info.get("speed", 0.0))

    t = 0
    terminated = truncated = False
    while t < max_steps:
        # NOTE: obs["image"] (rgb_front-derived) comes back all-black here -- the
        # server forces the SimLingo-only leaderboard agent (impls/carla_env_server.py
        # --leaderboard_agent default "simlingo"), which doesn't register the rgb_front
        # sensor. obs["simlingo_image"] (512x1024, SimLingo's own camera) is the only
        # populated frame available through this proxy path, so that's what gets saved
        # and SigLIP-encoded -- a different FOV/mount than the rgb_front frames the rest
        # of the critic-pretraining data (and the online BoN pipeline) use.
        image = obs.get("simlingo_image")
        if image is None:
            print(f"[collect_checkpoint_rollout] WARNING: no 'simlingo_image' in obs at t={t}; stopping episode.", flush=True)
            break
        rgb = np.asarray(image, dtype=np.uint8)
        cv2.imwrite(str(rgb_dir / f"{t:04d}.jpg"), rgb[:, :, ::-1])  # RGB -> BGR

        base_action, _vlm_feats = simlingo_base.get_action_and_features(
            simlingo_image=obs["simlingo_image"],
            ego_state=obs["state"],
            target_points=obs["target_points"],
            routing_command=obs.get("routing_command", ""),
        )
        obs_features = simlingo_base.get_encoder_features(
            simlingo_image=obs["simlingo_image"],
            ego_state=obs["state"],
            target_points=obs["target_points"],
            routing_command=obs.get("routing_command", ""),
        )
        state = obs["state"][6:].astype(np.float32)
        residual_action = agent.get_eval_action(obs_features, base_action, state)
        final_action = np.clip(base_action + res_scale_vec * residual_action, -1.0, 1.0)

        next_obs, _reward, terminated, truncated, info = env.step(final_action)

        m = _measurement_from_info(cur_ego_matrix, cur_speed, info)
        with gzip.open(meas_dir / f"{t:04d}.json.gz", "wt") as f:
            json.dump(m, f)

        cur_ego_matrix = info.get("ego_matrix")
        cur_speed = float(info.get("speed", 0.0))
        obs = next_obs
        t += 1
        if terminated or truncated:
            break

    print(
        f"[collect_checkpoint_rollout] {episode_dir} : {t} frames, "
        f"terminated={terminated} truncated={truncated}",
        flush=True,
    )
    return t


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--route", default="3936")
    p.add_argument(
        "--save-root", required=True,
        help="Writes <root>/ep_NNN/{measurements,rgb}/ (pass this <root> as "
        "pretrain_critic.py --noisy_root).",
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to a residual_sac_*.pt checkpoint from main_carla_simlingo.py "
        "(sac_residual training_mode).",
    )
    p.add_argument("--simlingo-checkpoint", required=True, help="SimLingo epoch=013.ckpt path.")
    p.add_argument("--carla-config", default=str(_IMPLS_ROOT / "configs" / "carla_config.yaml"))
    p.add_argument("--gpu-rank", type=int, default=None)
    p.add_argument("--server-conda-env", default="simlingo")
    p.add_argument("--carla-root", default="/home/celinet/VLA_driving/software")
    p.add_argument("--device", default="cuda")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--res-scale-accel", type=float, default=0.6,
                    help="Must match the res_scale the checkpoint was trained with "
                    "(0.6 for residual_sac_2000.pt's run, a single legacy res_scale "
                    "applied to both dims -- NOT this file's later res_scale_accel=2.0 default).")
    p.add_argument("--res-scale-steer", type=float, default=0.6)
    args = p.parse_args()

    from main_carla_simlingo import CarlaEnvProxy
    from torch_agents.residual_sac import EGO_STATE_DIM
    from vlas.simlingo_base import SimLingoBase, VLM_ENCODER_FEATURE_DIM

    print(f"[collect_checkpoint_rollout] Loading SimLingo base policy ...", flush=True)
    simlingo_base = SimLingoBase(args.simlingo_checkpoint, device=args.device)

    agent = _make_agent(
        vlm_feature_dim=VLM_ENCODER_FEATURE_DIM, state_dim=EGO_STATE_DIM,
        device=args.device, res_scale=args.res_scale_accel,
    )
    print(f"[collect_checkpoint_rollout] Loading residual SAC checkpoint {args.checkpoint} ...", flush=True)
    agent.load(args.checkpoint)
    agent.actor.eval()
    agent.critic.eval()

    res_scale_vec = np.array([args.res_scale_accel, args.res_scale_steer], dtype=np.float32)

    print("[collect_checkpoint_rollout] Starting CARLA env server (simlingo conda, carla 0.9.15)...", flush=True)
    env = CarlaEnvProxy(
        route=args.route,
        carla_config=args.carla_config,
        gpu_rank=args.gpu_rank,
        server_conda_env=args.server_conda_env,
        carla_root=args.carla_root,
    )

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    try:
        for i in range(args.episodes):
            total_frames += collect_episode(
                env, simlingo_base, agent, res_scale_vec,
                episode_dir=save_root / f"ep_{i:03d}", max_steps=args.max_steps,
            )
    finally:
        env.close()

    print(
        f"[collect_checkpoint_rollout] TOTAL frames: {total_frames} across "
        f"{args.episodes} episodes -> {save_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
