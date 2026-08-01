#!/usr/bin/env python3
"""Collect expert (+ noisy-expert) rollout data for one Bench2Drive route, laid
out the way impls/pretrain_critic.py's --non_noisy_root loader expects:

    <save-root>/database/simlingo/data/simlingo/<Town>_Rep<N>_<route>/
        measurements/{t:04d}.json.gz   (written automatically by AutoPilot.save())
        rgb/{t:04d}.jpg                (written here, from the env's own obs["image"])

Must run under the CARLA 0.9.15 Python (.venv-carla-0915) for Town12/13/Town06
routes, same as impls/carla_env_server.py -- this script mirrors that file's
_make_env() but drives the expert directly in-process (no JAX, no subprocess
wire protocol) since data collection needs neither a policy nor the JAX side.

Frame alignment: AutoPilot.save() (impls/coaches/simlingo/autopilot.py) writes
measurements/{frame:04d}.json.gz gated on its own internal self.step counter
(frame = self.step // config.data_save_freq). We force data_save_freq=1 so
frame == self.step, and save rgb/{t:04d}.jpg from our own external tick counter
t in lockstep with each step_expert() call, assuming AutoPilot.step increments
by exactly 1 per call starting at 0. Not independently verified frame-by-frame;
spot-check a few (rgb, measurements) pairs after collection before trusting it
for training.

Usage:
  .venv-carla-0915/bin/python impls/collect_expert_data.py \\
      --route 3936 --save-root /scratch/current/celinet/critic_data_3936 \\
      --clean-episodes 6 --noisy-episodes 6 --steer-noise 0.15 --max-steps 600
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_IMPLS_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _IMPLS_ROOT.parent
_REBUTTAL_ROOT = _REPO_ROOT / "simlingo-rebuttal"
_CARLA_ROOT = os.environ.get("CARLA_ROOT", "/home/celinet/VLA_driving/software")

# Same Bench2Drive/leaderboard path + env-var setup as impls/carla_env_server.py --
# ogbench.carla.carla_utils imports `leaderboard.autoagents...` etc. at module load,
# and RouteScenario discovers scenario classes by globbing SCENARIO_RUNNER_ROOT.
os.environ.setdefault("WORK_DIR", str(_REBUTTAL_ROOT))
os.environ.setdefault("CARLA_ROOT", _CARLA_ROOT)
os.environ["SCENARIO_RUNNER_ROOT"] = str(_REBUTTAL_ROOT / "Bench2Drive" / "scenario_runner")

for _p in [
    str(_REPO_ROOT),
    str(_IMPLS_ROOT),
    str(_REBUTTAL_ROOT),
    str(_REBUTTAL_ROOT / "leaderboard" / "leaderboard"),
    str(_REBUTTAL_ROOT / "leaderboard"),
    str(_REBUTTAL_ROOT / "scenario_runner"),
    str(_REBUTTAL_ROOT / "Bench2Drive" / "leaderboard" / "leaderboard"),
    str(_REBUTTAL_ROOT / "Bench2Drive" / "leaderboard"),
    str(_REBUTTAL_ROOT / "Bench2Drive" / "scenario_runner"),
    str(Path(_CARLA_ROOT) / "PythonAPI" / "carla"),
]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
import yaml


def _load_carla_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_env(carla_config: dict, route: str, gpu_rank: int | None, port: int | None):
    from ogbench.carla.carla_utils import CarlaBench2DriveWrapper

    cfg = dict(carla_config)
    cfg["expert_controller"] = "simlingo_autopilot"
    if gpu_rank is not None:
        cfg["gpu_rank"] = gpu_rank
    if port is not None:
        cfg["port"] = port
        cfg["traffic_manager_port"] = port + 6000
    env = CarlaBench2DriveWrapper(cfg, route=route)
    env.setup()
    return env


def _next_repetition(save_root: Path, town: str) -> int:
    """Resume-safe: pick the first REPETITION not already present under save_root."""
    if not save_root.exists():
        return 0
    used = set()
    for d in save_root.iterdir():
        m = re.match(rf"^{re.escape(town)}_Rep(\d+)_", d.name)
        if m:
            used.add(int(m.group(1)))
    rep = 0
    while rep in used:
        rep += 1
    return rep


def collect_episode(
    env,
    *,
    save_root: Path,
    town: str,
    repetition: int,
    steer_noise: float | None,
    max_steps: int,
    data_save_freq: int = 1,
) -> int:
    os.environ["DATAGEN"] = "1"
    os.environ["SAVE_PATH"] = str(save_root)
    os.environ["TOWN"] = town
    os.environ["REPETITION"] = str(repetition)

    obs_raw, _info = env.reset()
    agent = env._expert_agent
    if agent is None:
        raise RuntimeError(
            "env._expert_agent is None after reset() -- SimLingo autopilot failed to "
            "initialize (check stderr above for '[expert] SimLingo autopilot init failed')."
        )
    agent.config.data_save_freq = data_save_freq
    noisy = steer_noise is not None
    if noisy:
        agent.config.steer_noise = steer_noise

    episode_dir = agent.save_path
    rgb_dir = episode_dir / "rgb"
    rgb_dir.mkdir(exist_ok=True)

    n_frames = 0
    terminated = truncated = False
    for t in range(max_steps):
        image = obs_raw.get("image") if isinstance(obs_raw, dict) else None
        if image is not None:
            rgb = np.asarray(image, dtype=np.uint8)
            cv2.imwrite(str(rgb_dir / f"{t:04d}.jpg"), rgb[:, :, ::-1])  # RGB -> BGR
            n_frames += 1
        obs_raw, _reward, terminated, truncated, _info = env.step_expert(obs_raw)
        if terminated or truncated:
            break
    print(
        f"[collect] {episode_dir} (noisy={noisy}): {n_frames} frames, "
        f"terminated={terminated} truncated={truncated}",
        flush=True,
    )
    return n_frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--route", default="3936")
    p.add_argument("--carla-config", default=str(_IMPLS_ROOT / "configs" / "carla_config.yaml"))
    p.add_argument(
        "--save-root", required=True,
        help="Dataset root; writes under <root>/database/simlingo/data/simlingo/ "
        "(pass this <root> as pretrain_critic.py --non_noisy_root).",
    )
    p.add_argument("--gpu-rank", type=int, default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--clean-episodes", type=int, default=6)
    p.add_argument("--noisy-episodes", type=int, default=6)
    p.add_argument("--steer-noise", type=float, default=0.15)
    p.add_argument("--max-steps", type=int, default=600)
    args = p.parse_args()

    carla_config = _load_carla_config(args.carla_config)
    env = _make_env(carla_config, args.route, args.gpu_rank, args.port)

    save_root = Path(args.save_root) / "database" / "simlingo" / "data" / "simlingo"
    save_root.mkdir(parents=True, exist_ok=True)
    town = env.route_entry.town
    print(f"[collect] route={args.route} town={town} save_root={save_root}", flush=True)

    rep = _next_repetition(save_root, town)
    total_frames = 0
    try:
        for _ in range(args.clean_episodes):
            total_frames += collect_episode(
                env, save_root=save_root, town=town, repetition=rep,
                steer_noise=None, max_steps=args.max_steps,
            )
            rep += 1
        for _ in range(args.noisy_episodes):
            total_frames += collect_episode(
                env, save_root=save_root, town=town, repetition=rep,
                steer_noise=args.steer_noise, max_steps=args.max_steps,
            )
            rep += 1
    finally:
        env.close()

    print(
        f"[collect] TOTAL frames: {total_frames} across {rep} episodes -> {save_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
