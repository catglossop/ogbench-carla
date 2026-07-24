"""Standalone raw-control isolation test (route 3936, .venv-carla-0915).

Bypasses the VLA/RC-PID pipeline entirely: sends a hardcoded throttle=1.0,
steer=0.0, brake=0.0 VehicleControl for N consecutive ticks and prints ego
speed each tick. If the car accelerates normally here, the stall bug is in
the VLA/RC-PID desired-speed computation, not in control application /
physics. If it still doesn't move, the bug is downstream of control
application.

Run with the CARLA 0.9.15 client venv (matches carla_env_server.py's setup):
  CARLA_ROOT=/home/celinet/VLA_driving/software \
  .venv-carla-0915/bin/python impls/debug_raw_control.py --gpu_rank=3 --port=2500
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_IMPLS_ROOT = _REPO_ROOT / "impls"
_REBUTTAL_ROOT = _REPO_ROOT / "simlingo-rebuttal"

_CARLA_ROOT = os.environ.get("CARLA_ROOT", "/home/celinet/VLA_driving/software")
os.environ.setdefault("WORK_DIR", str(_REBUTTAL_ROOT))
os.environ.setdefault("CARLA_ROOT", _CARLA_ROOT)
os.environ["SCENARIO_RUNNER_ROOT"] = str(_REBUTTAL_ROOT / "Bench2Drive" / "scenario_runner")

for _p in [
    str(_REBUTTAL_ROOT / "leaderboard"),
    str(_REBUTTAL_ROOT / "Bench2Drive" / "leaderboard"),
    str(_REBUTTAL_ROOT / "Bench2Drive" / "scenario_runner"),
    str(_REPO_ROOT),
    str(Path(_CARLA_ROOT) / "PythonAPI" / "carla"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from absl import app, flags  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_string("route", "3936", "Route id.")
flags.DEFINE_integer("gpu_rank", 3, "CARLA render GPU.")
flags.DEFINE_integer("port", 2500, "CARLA RPC port.")
flags.DEFINE_integer("ticks", 15, "Number of raw-control ticks to run.")
flags.DEFINE_float("throttle", 1.0, "Hardcoded throttle.")


def main(_argv):
    import yaml
    from ogbench.carla.carla_utils import CarlaBench2DriveWrapper, EGO_STATE_IDX_SPEED

    cfg_path = _IMPLS_ROOT / "configs" / "carla_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["gpu_rank"] = FLAGS.gpu_rank
    cfg["port"] = FLAGS.port
    cfg["streaming_port"] = FLAGS.port + 1
    cfg["x_display_num"] = 0  # auto
    cfg["traffic_manager_port"] = 0  # auto

    print(f"[debug_raw_control] building env route={FLAGS.route} gpu_rank={FLAGS.gpu_rank} port={FLAGS.port}", flush=True)
    env = CarlaBench2DriveWrapper(cfg, route=FLAGS.route)
    env.setup()
    print("[debug_raw_control] env.setup() done, calling reset()...", flush=True)
    obs, info = env.reset()
    speed0 = float(obs["state"].reshape(-1)[EGO_STATE_IDX_SPEED])
    print(f"[debug_raw_control] reset done. initial speed={speed0:.4f}", flush=True)

    for i in range(FLAGS.ticks):
        running = env.step_raw_control(throttle=float(FLAGS.throttle), steer=0.0, brake=0.0)
        obs = env._obs_dict()
        speed = float(obs["state"].reshape(-1)[EGO_STATE_IDX_SPEED])
        print(f"[debug_raw_control] tick={i} throttle={FLAGS.throttle} speed={speed:.4f} running={running}", flush=True)
        if not running:
            print("[debug_raw_control] episode ended early.", flush=True)
            break

    print("[debug_raw_control] DONE.", flush=True)


if __name__ == "__main__":
    app.run(main)
