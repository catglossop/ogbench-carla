"""CARLA environment server — communicates JSON over stdin/stdout.

Run with: uv run python impls/carla_env_server.py [--route=...] [--gpu_rank=...]

Protocol (newline-delimited JSON):
  Startup:  server writes {"ready": true, "obs_space": {...}}
  Step:     controller writes {"action": [accel, steer]}
            server writes {"obs": {...}, "reward": float, "terminated": bool, "truncated": bool,
                            "info": {..., "ego_matrix": [[...4x4...]] | null, "speed": float}}
  Reset:    controller writes {"reset": true}
            server writes {"obs": {...}, "info": {"ego_matrix": [[...4x4...]] | null, "speed": float}}
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
flags.DEFINE_integer("gpu_rank", None,
                     "CARLA rendering GPU rank. Default: use carla_config's gpu_rank.")
flags.DEFINE_enum("leaderboard_agent", "simlingo", ["simlingo", "config"],
                  "Leaderboard observation agent: 'simlingo' forces the SimLingo camera "
                  "(rgb_simlingo only, used by main_carla_simlingo.py); 'config' keeps the "
                  "agent from carla_config.yaml (default observation_only, which registers "
                  "rgb_front — required for SteerVLA policy images via main_carla.py).")
flags.DEFINE_string("carla_root", "/home/celinet/VLA_driving/software",
                    "Path to CARLA root dir (sets CARLA_ROOT env var if not already set).")
flags.DEFINE_bool("terminate_on_infraction", False,
                  "Terminate episode immediately on collision, traffic violation, or off-route event.")
flags.DEFINE_string("extra_config_json", None,
                    "JSON string of extra carla_config overrides (e.g. steervla_action_execution).")


def _obs_to_wire(obs: Dict[str, Any], include_simlingo: bool) -> Dict[str, Any]:
    """Convert numpy arrays to JSON-serializable form. Images are base64-encoded.

    ``include_simlingo=False`` (observation_only agent) sends the native rgb_front
    viz frame instead — the SimLingo camera isn't registered, so its decode would be
    all zeros and encoding it would just waste ~2 MB/step on the wire.
    """
    policy_img = np.ascontiguousarray(obs["image"])
    image_b64 = base64.b64encode(policy_img.tobytes()).decode("ascii")
    tp = obs.get("target_points")
    expert_action = obs.get("expert_action")
    scene_ctx = obs.get("scene_context") or {}
    out = {
        "state": obs["state"].tolist(),
        "image_b64": image_b64,
        "image_shape": list(policy_img.shape),
        "routing_command": obs["routing_command"],
        "target_points": tp.tolist() if tp is not None else [[0.0, 0.0], [0.0, 0.0]],
        "expert_action": expert_action.tolist() if expert_action is not None else None,
        "scene_context": {
            "vehicle_ahead": bool(scene_ctx.get("vehicle_ahead", False)),
            "vehicle_ahead_dist_m": float(scene_ctx.get("vehicle_ahead_dist_m", -1.0)),
            "pedestrian_in_fov": bool(scene_ctx.get("pedestrian_in_fov", False)),
            "pedestrian_dist_m": float(scene_ctx.get("pedestrian_dist_m", -1.0)),
            "traffic_light_state": str(scene_ctx.get("traffic_light_state", "none")),
            "stop_sign_ahead": bool(scene_ctx.get("stop_sign_ahead", False)),
        },
    }
    if include_simlingo:
        simlingo = np.ascontiguousarray(obs["simlingo_image"])
        out["simlingo_image_b64"] = base64.b64encode(simlingo.tobytes()).decode("ascii")
        out["simlingo_image_shape"] = list(simlingo.shape)
    elif obs.get("image_viz") is not None:
        viz = np.ascontiguousarray(obs["image_viz"])
        out["viz_image_b64"] = base64.b64encode(viz.tobytes()).decode("ascii")
        out["viz_image_shape"] = list(viz.shape)
    return out


def _add_ego_pose_to_info(env, info: Dict[str, Any]) -> None:
    """Add 4x4 ego transform matrix + raw speed (m/s) directly into an already-built
    `info` dict, for offline critic-pretraining data collection (pretrain_critic.py's
    _waypoints_action needs ego_matrix per frame). These aren't scalar, so they must be
    added *after* the safe_info scalar-only filter runs, not before -- CarlaEnvProxy's
    step()/reset()/step_expert() (main_carla_simlingo.py) pass `info` through to the
    client verbatim with no additional filtering, so anything placed here survives the
    wire round-trip without needing any client-side changes.
    """
    ego = env._ego_actor() if hasattr(env, "_ego_actor") else None
    if ego is None:
        info["ego_matrix"] = None
        info["speed"] = 0.0
        return
    v = ego.get_velocity()
    info["ego_matrix"] = ego.get_transform().get_matrix()
    info["speed"] = float((v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5)


def _load_carla_config(path):
    import yaml
    cfg_path = path or str(_IMPLS_ROOT / "configs" / "carla_config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _make_env(carla_config, route):
    from ogbench.carla.carla_utils import CarlaBench2DriveWrapper
    import json as _json
    cfg = dict(carla_config)
    if FLAGS.gpu_rank is not None:
        cfg["gpu_rank"] = FLAGS.gpu_rank
    cfg["terminate_on_infraction"] = FLAGS.terminate_on_infraction
    if FLAGS.extra_config_json:
        extra = _json.loads(FLAGS.extra_config_json)
        cfg.update(extra)
    if FLAGS.leaderboard_agent == "simlingo":
        cfg["agent"] = str(
            _REPO_ROOT / "ogbench" / "carla" / "leaderboard_agents" / "simlingo_obs.py"
        )
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
    include_simlingo = FLAGS.leaderboard_agent == "simlingo"

    # Signal readiness with action_space metadata.
    # Do NOT call env.reset() here — let the parent send the first "reset" message
    # so that episode setup happens fresh, with world settings correctly initialized.
    startup = {
        "ready": True,
        "action_space_shape": list(env.action_space.shape),
        "action_space_low": float(env.action_space.low.flat[0]),
        "action_space_high": float(env.action_space.high.flat[0]),
    }
    _wire_out.write(json.dumps(startup) + "\n")
    _wire_out.flush()

    # Single-slot checkpoint for main_carla_teleop.py's literal-rollback best-of-N mode:
    # env.checkpoint()/env.restore() live entirely in this process (carla.Transform /
    # carla.Vector3D objects aren't JSON-serializable), so the wire protocol just carries
    # an opaque "did it work" ack -- see CarlaEnvSubprocess.checkpoint()/restore().
    _current_checkpoint = None

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
            reset_info: Dict[str, Any] = {}
            _add_ego_pose_to_info(env, reset_info)
            resp = {"obs": _obs_to_wire(obs, include_simlingo), "info": reset_info}
            _wire_out.write(json.dumps(resp) + "\n")
            _wire_out.flush()

        elif msg.get("reinit_expert"):
            if hasattr(env, "reinit_expert"):
                env.reinit_expert()
            _wire_out.write(json.dumps({"ack": True}) + "\n")
            _wire_out.flush()

        elif msg.get("checkpoint"):
            _current_checkpoint = env.checkpoint()
            _wire_out.write(json.dumps({"ack": True}) + "\n")
            _wire_out.flush()

        elif msg.get("restore"):
            if _current_checkpoint is None:
                _wire_out.write(json.dumps({"ack": False, "error": "no checkpoint taken yet"}) + "\n")
            else:
                env.restore(_current_checkpoint)
                _wire_out.write(json.dumps({"ack": True}) + "\n")
            _wire_out.flush()

        elif msg.get("traffic_states"):
            states = env.traffic_actor_states() if hasattr(env, "traffic_actor_states") else []
            _wire_out.write(json.dumps({"ack": True, "states": states}) + "\n")
            _wire_out.flush()

        elif msg.get("teleport_to_obstacle"):
            offset_m = float(msg.get("offset_m", 1.0))
            ok = env.teleport_ego_to_obstacle(offset_m) if hasattr(env, "teleport_ego_to_obstacle") else False
            _wire_out.write(json.dumps({"ack": bool(ok)}) + "\n")
            _wire_out.flush()

        elif msg.get("drive_straight_until_close"):
            target_distance_m = float(msg.get("target_distance_m", 10.0))
            slowdown_distance_m = float(msg.get("slowdown_distance_m", 30.0))
            max_ticks = int(msg.get("max_ticks", 2000))
            throttle = float(msg.get("throttle", 0.4))
            slow_throttle = float(msg.get("slow_throttle", 0.12))
            ok = (
                env.drive_straight_until_close(
                    target_distance_m=target_distance_m,
                    slowdown_distance_m=slowdown_distance_m,
                    max_ticks=max_ticks,
                    throttle=throttle,
                    slow_throttle=slow_throttle,
                )
                if hasattr(env, "drive_straight_until_close") else False
            )
            resp: Dict[str, Any] = {"ack": bool(ok)}
            if ok:
                # Read the observation directly (no extra tick) -- some obstacle props
                # get removed from the world shortly after a collision resolves, so an
                # extra env.step() here (another tick) risks missing the window where
                # it's still actually present/visible.
                resp["obs"] = _obs_to_wire(env._obs_dict(), include_simlingo)
                dist = env.nearest_obstacle_distance_m() if hasattr(env, "nearest_obstacle_distance_m") else None
                resp["nearest_obstacle_distance_m"] = -1.0 if dist is None else float(dist)
                resp["collision_count"] = int(env._collision_count()) if hasattr(env, "_collision_count") else -1
            _wire_out.write(json.dumps(resp) + "\n")
            _wire_out.flush()

        elif msg.get("step_raw_control"):
            throttle = float(msg.get("throttle", 0.0))
            steer = float(msg.get("steer", 0.0))
            brake = float(msg.get("brake", 0.0))
            running = (
                env.step_raw_control(throttle, steer, brake)
                if hasattr(env, "step_raw_control") else False
            )
            resp = {
                "ack": True,
                "running": bool(running),
                "obs": _obs_to_wire(env._obs_dict(), include_simlingo),
            }
            dist = env.nearest_obstacle_distance_m() if hasattr(env, "nearest_obstacle_distance_m") else None
            resp["nearest_obstacle_distance_m"] = -1.0 if dist is None else float(dist)
            resp["collision_count"] = int(env._collision_count()) if hasattr(env, "_collision_count") else -1
            _wire_out.write(json.dumps(resp) + "\n")
            _wire_out.flush()

        elif msg.get("expert_step"):
            ea = msg.get("expert_action")
            obs_raw_in = {"expert_action": np.array(ea, dtype=np.float32)} if ea is not None else None
            next_obs, reward, terminated, truncated, info = env.step_expert(obs_raw_in)

            safe_info = {
                k: (float(v) if isinstance(v, (np.floating, float)) else
                    int(v) if isinstance(v, (np.integer, int)) else
                    bool(v) if isinstance(v, (np.bool_, bool)) else str(v))
                for k, v in info.items()
                if isinstance(v, (np.floating, np.integer, np.bool_, float, int, bool, str))
            }
            _add_ego_pose_to_info(env, safe_info)
            resp = {
                "obs": _obs_to_wire(next_obs, include_simlingo),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": safe_info,
            }
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
            _add_ego_pose_to_info(env, safe_info)

            resp = {
                "obs": _obs_to_wire(next_obs, include_simlingo),
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
