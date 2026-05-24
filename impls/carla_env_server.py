"""CARLA environment server — communicates JSON over stdin/stdout.

Run with: uv run python impls/carla_env_server.py [--route=...] [--gpu_rank=...]

Protocol (newline-delimited JSON):
  Startup:  server writes {"ready": true, "obs_space": {...}}
  Step:     controller writes {"action": [accel, steer]}
            server writes {"obs": {...}, "reward": float, "terminated": bool, "truncated": bool, "info": {...}}
  Reset:    controller writes {"reset": true}
            server writes {"obs": {...}, "info": {...}}
  Shutdown: controller closes stdin; server exits cleanly

Observation dict keys sent over wire:
  "state":          list[float] len=25
  "simlingo_image": base64-encoded bytes of C-contiguous uint8 (512, 1024, 3) array
  "routing_command": str
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# ── Wire I/O setup ─────────────────────────────────────────────────────────────
# Save the real stdout pipe fd, then point fd 1 at stderr so any C extension,
# child process, or print() that writes to fd 1 goes to stderr instead of the
# JSON wire.  All protocol writes use _wire_out (the saved fd).
import fcntl as _fcntl
_wire_out = os.fdopen(os.dup(1), "w", buffering=1)
# Mark _wire_out close-on-exec so child processes (Xvfb, CarlaUE4) don't
# inherit the pipe fd — UE4 is sensitive to unexpected open pipe descriptors.
_fcntl.fcntl(_wire_out.fileno(), _fcntl.F_SETFD, _fcntl.FD_CLOEXEC)
os.dup2(2, 1)            # fd 1 → stderr at OS level (inherited by children)
sys.stdout = sys.stderr  # Python-level redirect

_IMPLS_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _IMPLS_ROOT.parent
_REBUTTAL_ROOT = _REPO_ROOT / "simlingo-rebuttal"

# Default to CARLA 0.9.15 which loads Town12 without crashing.
# CARLA 0.9.16 has a known segfault on Town12 due to level-streaming init.
_CARLA_ROOT = os.environ.get("CARLA_ROOT", "/home/celinet/VLA_driving/software")

# leaderboard_evaluator.get_weather_id() reads ${WORK_DIR}/leaderboard/data/weather.xml.
# simlingo-rebuttal/leaderboard/data/weather.xml is the correct copy.
os.environ.setdefault("WORK_DIR", str(_REBUTTAL_ROOT))
os.environ.setdefault("CARLA_ROOT", _CARLA_ROOT)
# Bench2Drive RouteScenario discovers scenario classes by globbing
# ${SCENARIO_RUNNER_ROOT}/srunner/scenarios/*.py.  Do not inherit a generic
# ScenarioRunner root here: it may not contain Bench2Drive-specific scenarios
# such as MergerIntoSlowTrafficV2.
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
    # CARLA PythonAPI agents (global_route_planner etc.)
    str(Path(_CARLA_ROOT) / "PythonAPI" / "carla"),
]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("route", None, "Bench2Drive route name.")
flags.DEFINE_string("carla_config", None, "Path to carla_config.yaml.")
flags.DEFINE_integer("gpu_rank", 0, "CARLA rendering GPU rank.")
flags.DEFINE_string("carla_root", "/home/celinet/VLA_driving/software",
                    "Path to CARLA root dir (sets CARLA_ROOT env var if not already set).")


def _obs_to_wire(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy arrays to JSON-serializable form. Image is base64-encoded."""
    img = np.ascontiguousarray(obs["simlingo_image"])  # ensure C-contiguous uint8
    img_b64 = base64.b64encode(img.tobytes()).decode("ascii")
    tp = obs.get("target_points")
    return {
        "state": obs["state"].tolist(),
        "simlingo_image_b64": img_b64,
        "simlingo_image_shape": list(img.shape),
        "routing_command": obs["routing_command"],
        "target_points": tp.tolist() if tp is not None else [[0.0, 0.0], [0.0, 0.0]],
    }


def _load_carla_config(path):
    import yaml
    cfg_path = path or str(_IMPLS_ROOT / "configs" / "carla_config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _make_env(carla_config, route):
    from ogbench.carla.carla_utils import CarlaBench2DriveWrapper
    cfg = dict(carla_config)
    cfg["gpu_rank"] = FLAGS.gpu_rank
    simlingo_agent = str(
        _REPO_ROOT / "ogbench" / "carla" / "leaderboard_agents" / "simlingo_obs.py"
    )
    cfg["agent"] = simlingo_agent
    env = CarlaBench2DriveWrapper(cfg, route=route)
    env.setup()
    return env


def main(_argv):
    if FLAGS.route is None:
        sys.stderr.write("ERROR: --route is required\n")
        sys.exit(1)

    # Set CARLA_ROOT if not already in environment
    if not os.environ.get("CARLA_ROOT") and FLAGS.carla_root:
        os.environ["CARLA_ROOT"] = FLAGS.carla_root

    carla_config = _load_carla_config(FLAGS.carla_config)
    env = _make_env(carla_config, FLAGS.route)

    obs, info = env.reset()

    # Signal readiness
    startup = {"ready": True}
    _wire_out.write(json.dumps(startup) + "\n")
    _wire_out.flush()

    # Send initial obs
    obs_msg = {"obs": _obs_to_wire(obs), "info": {}}
    _wire_out.write(json.dumps(obs_msg) + "\n")
    _wire_out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            sys.stderr.write(f"[env_server] JSON decode error: {line!r}\n")
            continue

        if msg.get("reset"):
            obs, info = env.reset()
            resp = {"obs": _obs_to_wire(obs), "info": {}}
            _wire_out.write(json.dumps(resp) + "\n")
            _wire_out.flush()

        elif "action" in msg:
            action = np.array(msg["action"], dtype=np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Serialize info: keep only JSON-safe fields
            safe_info = {
                k: (float(v) if isinstance(v, (np.floating, float)) else
                    int(v) if isinstance(v, (np.integer, int)) else
                    bool(v) if isinstance(v, (np.bool_, bool)) else str(v))
                for k, v in info.items()
                if isinstance(v, (np.floating, np.integer, np.bool_, float, int, bool, str))
            }

            resp = {
                "obs": _obs_to_wire(next_obs),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": safe_info,
            }
            _wire_out.write(json.dumps(resp) + "\n")
            _wire_out.flush()

        elif msg.get("shutdown"):
            break

    env.close()


if __name__ == "__main__":
    app.run(main)
