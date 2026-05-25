"""Bench2Drive / CARLA leaderboard helpers exposed as a Gymnasium env.

Public surface (intentionally small):

* :func:`make_env_and_datasets` -- constructs :class:`CarlaBench2DriveWrapper` for
  a single route, returning ``(env, None, None)`` so it slots into the same
  call-site as the OGBench MuJoCo factory.
* :class:`CarlaBench2DriveWrapper` -- gymnasium env. Takes one route (resolved by
  scenario name / file basename / route id via
  :mod:`ogbench.carla.route_registry`). ``step(action)`` drives the ego vehicle
  directly; the leaderboard's autoagent only registers sensors and stashes the
  latest sensor dict on ``info``.
* :func:`run_leaderboard` -- escape hatch for the standard multi-route benchmark.

Action space: by default ``Box(-1, 1, shape=(2,))`` as ``[accel, steer]`` (positive
``accel`` → throttle, negative → brake). When ``carla_config["steervla_action_execution"]``
is set (see ``impls/main_carla.py`` with SteerVLA DSRL), the space becomes the flattened
OpenPI trajectory chunk ``action_horizon * action_dim`` (normalized model outputs or policy
space); controls follow ``simlingo/team_code/agent_steervla.py`` cumsums + PID.

Observation:

* ``observation`` -- a :class:`gymnasium.spaces.Dict` with two keys:
    * ``"state"`` -- ``float32`` vector of shape ``(STATE_DIM,)`` (ego kinematics + routing command one-hot).
    * ``"image"`` -- ``uint8`` RGB array ``(*IMAGE_SHAPE_HWC,)`` downscaled for RL/VLA
    * ``"image_viz"`` -- ``uint8`` RGB at native CARLA camera resolution (logging only)
      (zeros until the first sensor frame is available).
* ``info["sensors"]`` -- raw leaderboard ``input_data`` dict when populated.

Constants :data:`IMAGE_SHAPE_HWC` and :data:`RGB_FRONT_CAMERA_TAG` match
:class:`~ogbench.carla.leaderboard_agents.observation_only.ObservationOnlyAgent`.

Before importing leaderboard / srunner, the CARLA Python API directory is
prepended to ``sys.path`` via :func:`ensure_carla_python_api_on_path`.

Set either:

  export CARLA_PYTHON_API_ROOT=/home/carla/carla-0-9-16/PythonAPI/carla

or:

  export CARLA_ROOT=/home/carla/carla-0-9-16

(the latter adds ``${CARLA_ROOT}/PythonAPI/carla``).
"""

from __future__ import annotations

import atexit
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium
import numpy as np
import yaml


def ensure_carla_python_api_on_path() -> None:
    """Prepend CARLA ``PythonAPI/carla`` so ``import agents`` and ``import carla`` match CARLA layouts."""
    root = os.environ.get("CARLA_PYTHON_API_ROOT")
    if not root and os.environ.get("CARLA_ROOT"):
        root = str(Path(os.environ["CARLA_ROOT"]).resolve() / "PythonAPI" / "carla")
    if not root:
        return
    p = str(Path(root).expanduser().resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


ensure_carla_python_api_on_path()

import carla
from leaderboard.autoagents.agent_wrapper import AgentError, TickRuntimeError, validate_sensor_configuration
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.utils.route_parser import RouteParser
from leaderboard.utils.statistics_manager import FAILURE_MESSAGES, StatisticsManager

import py_trees
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.leaderboard_evaluator import (
    LeaderboardEvaluator,
    get_weather_id,
    sensors_to_icons,
)

from ogbench.carla.route_registry import RouteEntry, find_route
from ogbench.carla.leaderboard_agents.observation_only import (
    IMAGE_SHAPE_HWC,
    RGB_FRONT_CAMERA_TAG,
    VIZ_IMAGE_SHAPE_HWC,
)


def _patch_speedometer_no_rpc() -> None:
    """Replace SpeedometerReader.__call__ with a cache-based implementation.

    The original SpeedometerReader.__call__ calls actor.get_velocity() and
    actor.get_transform() from a background daemon thread at 1/delta_time Hz,
    racing with the main thread's world.tick() and other CARLA RPC calls on
    the same TCP socket.  Concurrent socket writes corrupt msgpack framing,
    which CARLA's C++ rpclib I/O thread sees as type_error → terminate() →
    abort.

    CarlaDataProvider maintains _actor_velocity_map (a plain Python float per
    actor, updated on the main thread by on_carla_tick() after every tick).
    Reading it is a GIL-safe dict lookup — no CARLA RPC, no race.
    """
    from leaderboard.envs.sensor_interface import SpeedometerReader

    def _cached_call(self: SpeedometerReader):  # type: ignore[misc]
        try:
            speed = CarlaDataProvider.get_velocity(self._vehicle)
            return {"speed": float(speed) if speed is not None else 0.0}
        except Exception:
            return {"speed": 0.0}

    SpeedometerReader.__call__ = _cached_call  # type: ignore[method-assign]


_patch_speedometer_no_rpc()

# Flat ego-state vector layout (length = 25): 3 location, 3 rotation (rpy),
# 3 velocity, 3 angular velocity, 3 acceleration, 1 speed, 3 last-applied control,
# 6 routing command one-hot (RoadOption 1–6: LEFT, RIGHT, STRAIGHT, LANEFOLLOW,
# CHANGELANELEFT, CHANGELANERIGHT).
ROUTING_COMMAND_DIM = 6
STATE_DIM = 19 + ROUTING_COMMAND_DIM
ACTION_DIM = 2

# Indices into ``obs["state"]`` for :func:`_ego_state_vector` (length :data:`STATE_DIM`).
EGO_STATE_IDX_SPEED = 15
EGO_STATE_IDX_THROTTLE = 16
EGO_STATE_IDX_STEER = 17
EGO_STATE_IDX_BRAKE = 18
EGO_STATE_IDX_COMMAND_START = 19  # first of 6 one-hot routing-command dims

# Human-readable labels for the 6 RoadOption routing commands (values 1–6).
ROUTING_COMMAND_TEXT = {
    1: "go left at the next intersection",
    2: "go right at the next intersection",
    3: "go straight at the next intersection",
    4: "follow the road",
    5: "do a lane change to the left",
    6: "do a lane change to the right",
}

# Pre-computed per-town speed-limit lookup tables (simlingo, km/h, CARLA world coords).
_SPEED_LIMITS_DIR = Path(__file__).resolve().parent.parent.parent / "impls" / "coaches" / "simlingo" / "speed_limits"

DEFAULT_CRASH_STUCK_SPEED_THRESHOLD = 0.1
DEFAULT_CRASH_STUCK_STEPS = 20
DEFAULT_CRASH_STUCK_PENALTY = -1.0
DEFAULT_COLLISION_EVENT_PENALTY = -5.0 # catastrophic - should update to make sure any contact is given negative reward
DEFAULT_OUTSIDE_ROUTE_EVENT_PENALTY = -5.0 # should be pretty heavy
DEFAULT_PROGRESS_REWARD_WEIGHT = 1.0
# DEFAULT_CENTERING_REWARD_WEIGHT = 0.2
# DEFAULT_HEADING_REWARD_WEIGHT = 0.2
DEFAULT_STEER_PENALTY_WEIGHT = 0.05
DEFAULT_BRAKE_PENALTY_WEIGHT = 0.02
DEFAULT_SPEED_LIMIT_PENALTY_WEIGHT = 0.1
DEFAULT_MAX_EPISODE_STEPS = 250
SUCCESS_BONUS = 5.0
FAILURE_BONUS = -5.0


def _find_free_port(starting_port: int) -> int:
    """Find a free localhost TCP port starting from ``starting_port``."""
    port = max(int(starting_port), 1)
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            port += 1


def _display_lock_path(display_num: int) -> Path:
    return Path(f"/tmp/.X{int(display_num)}-lock")


def _clear_stale_display_lock(display_num: int) -> None:
    """Remove ``/tmp/.X<N>-lock`` when its PID no longer exists."""
    lock = _display_lock_path(display_num)
    if not lock.exists():
        return
    try:
        pid = int(lock.read_text().strip())
        os.kill(pid, 0)
        return
    except (ProcessLookupError, PermissionError, ValueError, OSError):
        try:
            lock.unlink()
        except OSError:
            pass


def _find_free_display_num(start: int = 10, end: int = 100) -> int:
    """Pick a free X display number, clearing stale locks along the way."""
    for display_num in range(start, end):
        _clear_stale_display_lock(display_num)
        if not _display_lock_path(display_num).exists():
            return display_num
    raise RuntimeError(f"No free X display numbers in :{start}..:{end - 1}")


class IsolatedLeaderboardEvaluator(LeaderboardEvaluator):
    """Leaderboard evaluator variant that honors explicit per-instance launch args."""

    def _setup_simulation(self, args):
        self.carla_path = os.environ["CARLA_ROOT"]

        rpc_port = int(getattr(args, "port", 0) or 0)
        if rpc_port <= 0:
            rpc_port = _find_free_port(2000)
        args.port = rpc_port

        display_num = int(getattr(args, "x_display_num", 0) or 0)
        if display_num > 0:
            _clear_stale_display_lock(display_num)
        else:
            display_num = _find_free_display_num()

        xvfb_cmd = [
            "Xvfb",
            f":{display_num}",
            "-screen",
            "0",
            "1280x1024x24",
            "-ac",
            "+extension",
            "GLX",
            "+render",
            "-noreset",
        ]
        self.xvfb = subprocess.Popen(xvfb_cmd, preexec_fn=os.setsid)
        atexit.register(os.killpg, self.xvfb.pid, signal.SIGKILL)
        time.sleep(2)

        carla_env = os.environ.copy()
        carla_env["DISPLAY"] = f":{display_num}"

        cmd = [
            os.path.join(self.carla_path, "CarlaUE4.sh"),
            "-RenderOffScreen",
            "-nosound",
            f"-carla-rpc-port={rpc_port}",
            f"-graphicsadapter={args.gpu_rank}",
        ]
        streaming_port = int(getattr(args, "streaming_port", 0) or 0)
        if streaming_port > 0:
            cmd.append(f"-carla-streaming-port={streaming_port}")
        self.server = subprocess.Popen(cmd, preexec_fn=os.setsid, env=carla_env)
        print(" ".join(cmd), self.server.returncode, flush=True)
        atexit.register(os.killpg, self.server.pid, signal.SIGKILL)
        time.sleep(30)

        attempts = 0
        num_max_restarts = 20
        while attempts < num_max_restarts:
            try:
                client = carla.Client(args.host, rpc_port)
                client_timeout = args.timeout if args.timeout else self.client_timeout
                client.set_timeout(client_timeout)

                settings = carla.WorldSettings(
                    synchronous_mode=True,
                    fixed_delta_seconds=1.0 / self.frame_rate,
                    deterministic_ragdolls=True,
                    spectator_as_ego=False,
                )
                client.get_world().apply_settings(settings)
                print(f"load_world success , attempts={attempts}", flush=True)
                break
            except Exception as e:
                print(f"load_world failed , attempts={attempts}", flush=True)
                print(e, flush=True)
                attempts += 1
                time.sleep(5)
        else:
            raise RuntimeError(f"CARLA server failed to come up on rpc port {rpc_port}")

        tm_port = int(getattr(args, "traffic_manager_port", 0) or 0)
        if tm_port <= 0:
            tm_port = _find_free_port(8000)
        args.traffic_manager_port = tm_port

        attempts = 0
        num_max_restarts = 40
        while attempts < num_max_restarts:
            try:
                traffic_manager = client.get_trafficmanager(tm_port)
                traffic_manager.set_synchronous_mode(True)
                traffic_manager.set_hybrid_physics_mode(True)
                print(f"traffic_manager init success, try_time={attempts}", flush=True)
                break
            except Exception as e:
                print(f"traffic_manager init fail, try_time={attempts}", flush=True)
                print(e, flush=True)
                attempts += 1
                time.sleep(5)
        else:
            raise RuntimeError(f"Traffic manager failed to come up on port {tm_port}")

        return client, client_timeout, traffic_manager


def ego_drive_metrics_from_state_vec(state: Any) -> Dict[str, float]:
    """Speed (m/s) and last-applied CARLA ``VehicleControl`` from gym ``obs['state']``."""
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    if s.size < STATE_DIM:
        raise ValueError(f"ego state vector expected length >= {STATE_DIM}, got {s.size}")
    return {
        "ego_speed": float(s[EGO_STATE_IDX_SPEED]),
        "control_throttle": float(s[EGO_STATE_IDX_THROTTLE]),
        "control_steer": float(s[EGO_STATE_IDX_STEER]),
        "control_brake": float(s[EGO_STATE_IDX_BRAKE]),
    }


def _zeros_rgb_image() -> np.ndarray:
    return np.zeros(IMAGE_SHAPE_HWC, dtype=np.uint8)


def _zeros_rgb_viz_image() -> np.ndarray:
    return np.zeros(VIZ_IMAGE_SHAPE_HWC, dtype=np.uint8)


def _bgra_to_rgb_hwc(arr: np.ndarray) -> np.ndarray:
    """CARLA leaderboard packs ``sensor.camera.rgb`` as H×W×4 BGRA uint8."""
    bgr = np.asarray(arr)[..., :3]
    return np.ascontiguousarray(bgr[..., ::-1], dtype=np.uint8)


def _decode_rgb_front_viz(sensor_dict: Dict[str, Any]) -> np.ndarray | None:
    """Decode ``rgb_front`` at native CARLA camera resolution, or ``None`` if missing."""
    if not sensor_dict or RGB_FRONT_CAMERA_TAG not in sensor_dict:
        return None
    tup = sensor_dict[RGB_FRONT_CAMERA_TAG]
    if not isinstance(tup, (tuple, list)) or len(tup) < 2:
        return None
    payload = tup[1]
    if payload is None:
        return None
    arr = np.asarray(payload)
    if arr.ndim != 3:
        return None
    if arr.shape[-1] == 4:
        rgb = _bgra_to_rgb_hwc(arr)
    elif arr.shape[-1] == 3:
        rgb = arr.astype(np.uint8, copy=False)
    else:
        return None
    if rgb.shape != VIZ_IMAGE_SHAPE_HWC:
        try:
            import cv2

            rgb = cv2.resize(
                rgb,
                (VIZ_IMAGE_SHAPE_HWC[1], VIZ_IMAGE_SHAPE_HWC[0]),
                interpolation=cv2.INTER_AREA,
            )
        except Exception:
            return None
    return rgb


def downscale_rgb_for_policy(rgb_viz: np.ndarray) -> np.ndarray:
    """Resize a viz-resolution RGB frame to policy/RL ``IMAGE_SHAPE_HWC``."""
    rgb_viz = np.asarray(rgb_viz, dtype=np.uint8)
    if rgb_viz.shape == IMAGE_SHAPE_HWC:
        return rgb_viz
    try:
        import cv2

        return cv2.resize(
            rgb_viz,
            (IMAGE_SHAPE_HWC[1], IMAGE_SHAPE_HWC[0]),
            interpolation=cv2.INTER_AREA,
        )
    except Exception:
        return _zeros_rgb_image()


def rgb_viz_from_leaderboard_dict(sensor_dict: Dict[str, Any]) -> np.ndarray:
    """Decode ``rgb_front`` at viz resolution for rollout video / W&B logging."""
    rgb = _decode_rgb_front_viz(sensor_dict)
    return rgb if rgb is not None else _zeros_rgb_viz_image()


def rgb_front_from_leaderboard_dict(sensor_dict: Dict[str, Any]) -> np.ndarray:
    """Decode ``rgb_front`` and downscale to policy resolution for RL/VLA/replay."""
    rgb = _decode_rgb_front_viz(sensor_dict)
    if rgb is None:
        return _zeros_rgb_image()
    return downscale_rgb_for_policy(rgb)


class SteppableScenarioManager(ScenarioManager):
    """ScenarioManager that runs one CARLA tick per call and applies an external control.

    ``pending_control`` is a ``carla.VehicleControl`` that the gym wrapper sets
    immediately before each ``step_once``. The agent's own ``run_step`` is still
    called (so its ``sensor_interface`` keeps draining and the gym wrapper can
    read fresh sensor data via ``agent.last_input_data``), but its returned
    control is discarded.
    """

    # Run build_scenarios / spawn_parked_vehicles every this many ticks from
    # the main thread instead of from a background thread.
    _BUILD_SCENARIOS_INTERVAL = 20

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pending_control: Optional[carla.VehicleControl] = None
        self.last_agent_input: Dict[str, Any] = {}
        self._build_scenarios_tick = 0

    def begin_scenario(self) -> None:
        if self._scenario_thread is not None:
            raise RuntimeError(
                "begin_scenario called while a scenario thread is still active; "
                "call stop_scenario first."
            )
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()
        self._watchdog = Watchdog(self._timeout)
        self._watchdog.start()
        self._agent_watchdog = Watchdog(self._timeout)
        self._agent_watchdog.start()
        self._running = True
        self._scenario_thread = threading.Thread(
            target=self.build_scenarios_loop,
            args=(self._debug_mode > 0,),
        )
        self._scenario_thread.start()

    def step_once(self) -> Tuple[bool, Any]:
        """Run one leaderboard tick. Returns ``(still_running, scenario_tree_status)``."""
        if not getattr(self, "_running", False):
            return False, None
        self._tick_scenario()
        status = self.scenario_tree.status if self.scenario_tree is not None else None
        return bool(self._running), status

    def build_scenarios_loop(self, debug: bool) -> None:
        """No-op idle thread.

        Scenario building is called synchronously every _BUILD_SCENARIOS_INTERVAL
        ticks from _tick_scenario_locked (main thread) to avoid a background thread
        sharing the CARLA RPC socket concurrently with world.tick().
        """
        while self._running:
            time.sleep(1)

    # The following is a near-verbatim copy of the upstream _tick_scenario, with two changes:
    #   1. We always apply ``self.pending_control`` (if set) instead of the agent's output.
    #   2. We stash the agent's last input_data on ``self.last_agent_input`` for the gym wrapper.
    def _tick_scenario(self) -> None:
        self._tick_scenario_locked()

    def _tick_scenario_locked(self) -> None:
        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            self.tick_count += 1
            self._watchdog.pause()

            if self.tick_count > 4000:
                raise TickRuntimeError("RuntimeError, tick_count > 4000")

            try:
                self._agent_watchdog.resume()
                self._agent_watchdog.update()
                agent_action = self._agent_wrapper()
                self._agent_watchdog.pause()
            except Exception as e:  # noqa: BLE001
                # Same exception filtering the upstream does.
                from leaderboard.envs.sensor_interface import SensorReceivedNoData

                if isinstance(e, SensorReceivedNoData):
                    raise RuntimeError(e)
                raise AgentError(e)

            agent_obj = getattr(self._agent_wrapper, "_agent", None)
            if agent_obj is not None:
                self.last_agent_input = getattr(agent_obj, "last_input_data", {}) or {}

            self._watchdog.resume()
            ego_action = self.pending_control if self.pending_control is not None else agent_action
            self.ego_vehicles[0].apply_control(ego_action)
            py_trees.blackboard.Blackboard().set("AV_control", ego_action, overwrite=True)
            self.scenario_tree.tick_once()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            ego_trans = self.ego_vehicles[0].get_transform()
            self._spectator.set_transform(
                carla.Transform(
                    ego_trans.location + carla.Location(z=70),
                    carla.Rotation(pitch=-90),
                )
            )

            # Build/spawn dynamic scenario actors periodically in the main thread
            # instead of via a background thread, to keep all CARLA RPC calls
            # single-threaded and avoid concurrent socket access.
            self._build_scenarios_tick += 1
            if self._build_scenarios_tick >= self._BUILD_SCENARIOS_INTERVAL:
                self._build_scenarios_tick = 0
                try:
                    self.scenario.build_scenarios(self.ego_vehicles[0], debug=self._debug_mode > 0)
                    self.scenario.spawn_parked_vehicles(self.ego_vehicles[0])
                except Exception:
                    pass


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "impls" / "configs" / "carla_config.yaml"


def load_carla_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else _default_config_path()
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_agent_module() -> str:
    """Path to the leaderboard agent that the gym wrapper uses by default."""
    return str((Path(__file__).resolve().parent / "leaderboard_agents" / "observation_only.py"))


def _resolve_route(carla_config: Dict[str, Any], route: Optional[str]) -> RouteEntry:
    """Pick the route: explicit ``route`` arg wins, else fall back to ``carla_config['route']``."""
    name = route or carla_config.get("route")
    if not name:
        raise ValueError(
            "Specify a route by passing route='<scenario-name>' (e.g. 'parking-cut-in-001') or "
            "setting carla_config['route']. See ogbench.carla.route_registry.list_routes()."
        )
    return find_route(str(name))


def carla_config_to_args(
    cfg: Dict[str, Any], route_entry: RouteEntry
) -> SimpleNamespace:
    """Build the ``args`` namespace expected by ``LeaderboardEvaluator``.

    ``route_entry`` overrides any ``routes`` / ``routes_subset`` in ``cfg``.
    """
    agent = cfg.get("agent") or _default_agent_module()
    return SimpleNamespace(
        host=str(cfg.get("host", "localhost")),
        port=int(cfg.get("port", 2000)),
        streaming_port=int(cfg.get("streaming_port", 0)),
        traffic_manager_port=int(cfg.get("traffic_manager_port", 8000)),
        traffic_manager_seed=int(cfg.get("traffic_manager_seed", 0)),
        debug=int(cfg.get("debug", 0)),
        record=str(cfg.get("record", "")),
        timeout=float(cfg.get("timeout", 600.0)),
        routes=str(route_entry.xml_path),
        routes_subset=route_entry.route_id,
        repetitions=int(cfg.get("repetitions", 1)),
        agent=str(agent),
        agent_config=str(cfg.get("agent_config", "")),
        track=str(cfg.get("track", "SENSORS")),
        resume=bool(cfg.get("resume", False)),
        checkpoint=str(cfg.get("checkpoint", "./simulation_results.json")),
        debug_checkpoint=str(cfg.get("debug_checkpoint", "./live_results.txt")),
        gpu_rank=int(cfg.get("gpu_rank", 0)),
        x_display_num=int(cfg.get("x_display_num", 0)),
    )


def run_leaderboard(args: SimpleNamespace) -> bool:
    """Run the full leaderboard loop (all routes selected by ``args``).

    Returns ``True`` if the run crashed fatally.
    """
    statistics_manager = StatisticsManager(args.checkpoint, args.debug_checkpoint)
    
    prev_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_rank)
    leaderboard_evaluator = IsolatedLeaderboardEvaluator(args, statistics_manager)
    if prev_cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible_devices
    else:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible_devices

    crashed = leaderboard_evaluator.run(args)
    del leaderboard_evaluator
    return crashed


def _action_to_control(action: np.ndarray) -> carla.VehicleControl:
    """Map ``[accel, steer]`` in ``[-1, 1]^2`` to a CARLA ``VehicleControl``."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] < 2:
        raise ValueError(f"Expected action of shape (2,), got {a.shape}")
    accel = float(np.clip(a[0], -1.0, 1.0))
    steer = float(np.clip(a[1], -1.0, 1.0))
    throttle = max(0.0, accel)
    brake = max(0.0, -accel)
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)


def _routing_command_to_onehot(command: int) -> np.ndarray:
    """Convert a RoadOption integer (1–6) to a :data:`ROUTING_COMMAND_DIM`-dim one-hot."""
    if command < 1 or command > 6:
        command = 4  # fallback to LANEFOLLOW
    out = np.zeros(ROUTING_COMMAND_DIM, dtype=np.float32)
    out[command - 1] = 1.0
    return out


def _ego_state_vector(
    ego: carla.Actor, last_control: carla.VehicleControl
) -> np.ndarray:
    """Build the first 19 dims of the ego-state vector (kinematics + last control)."""
    transform = ego.get_transform()
    loc = transform.location
    rot = transform.rotation
    vel = ego.get_velocity()
    avel = ego.get_angular_velocity()
    acc = ego.get_acceleration()
    speed = float(np.linalg.norm([vel.x, vel.y, vel.z]))
    return np.array(
        [
            loc.x, loc.y, loc.z,
            rot.roll, rot.pitch, rot.yaw,
            vel.x, vel.y, vel.z,
            avel.x, avel.y, avel.z,
            acc.x, acc.y, acc.z,
            speed,
            float(last_control.throttle),
            float(last_control.steer),
            float(last_control.brake),
        ],
        dtype=np.float32,
    )


class CarlaBench2DriveWrapper(gymnasium.Env):
    """Gymnasium env for a single Bench2Drive route. RL controls the ego vehicle.

    Pass either ``route="parking-cut-in-001"`` (scenario name), ``"bench2drive_007"``
    (file basename), or a numeric ``"1711"`` (route id). Sensors come from a tiny
    leaderboard agent (``ObservationOnlyAgent``); replace ``carla_config['agent']``
    if you need a different sensor suite.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        carla_config: Dict[str, Any],
        route: Optional[str] = None,
    ):
        super().__init__()
        self.carla_config = dict(carla_config)
        self.route_entry: RouteEntry = _resolve_route(self.carla_config, route)

        self._evaluator: Optional[LeaderboardEvaluator] = None
        self._args: Optional[SimpleNamespace] = None
        self._base_agent_config: str = ""
        self._scenario_active = False
        self._last_control = carla.VehicleControl()
        self._crash_stuck_speed_threshold = float(
            self.carla_config.get("crash_stuck_speed_threshold", DEFAULT_CRASH_STUCK_SPEED_THRESHOLD)
        )
        self._crash_stuck_steps = int(
            self.carla_config.get("crash_stuck_steps", DEFAULT_CRASH_STUCK_STEPS)
        )
        self._crash_stuck_penalty = float(
            self.carla_config.get("crash_stuck_penalty", DEFAULT_CRASH_STUCK_PENALTY)
        )
        self._crash_stuck_ticks = 0
        self._collision_event_penalty = float(
            self.carla_config.get("collision_event_penalty", DEFAULT_COLLISION_EVENT_PENALTY)
        )
        self._outside_route_event_penalty = float(
            self.carla_config.get("outside_route_event_penalty", DEFAULT_OUTSIDE_ROUTE_EVENT_PENALTY)
        )
        self._progress_reward_weight = float(
            self.carla_config.get("progress_reward_weight", DEFAULT_PROGRESS_REWARD_WEIGHT)
        )
        self._steer_penalty_weight = float(
            self.carla_config.get("steer_penalty_weight", DEFAULT_STEER_PENALTY_WEIGHT)
        )
        self._brake_penalty_weight = float(
            self.carla_config.get("brake_penalty_weight", DEFAULT_BRAKE_PENALTY_WEIGHT)
        )
        self._speed_limit_penalty_weight = float(
            self.carla_config.get("speed_limit_penalty_weight", DEFAULT_SPEED_LIMIT_PENALTY_WEIGHT)
        )
        self._prev_collision_count = 0
        self._prev_outside_route_value = 0.0
        self._max_episode_steps = int(
            self.carla_config.get("max_episode_steps", DEFAULT_MAX_EPISODE_STEPS)
        )
        self._episode_step_count = 0

        self._expert_controller_kind = str(self.carla_config.get("expert_controller", "") or "").strip().lower()
        self._expert_agent: Any | None = None
        self._cached_world_map: Any | None = None
        self._route_planner: Any | None = None
        self._current_routing_command: int = 4  # LANEFOLLOW until route is loaded
        self._routing_last_command_tmp: int = -1  # simlingo carryover state
        self._routing_last_command: int = -1
        self._speed_limit_tree: Any | None = None
        self._speed_limit_values: Any | None = None
        self._speed_limit_map_name: str = ""
        try:
            from impls.coaches.expert_label import ExpertLabelComputer
            self._label_computer = ExpertLabelComputer()
        except Exception:
            self._label_computer = None

        self.observation_space = gymnasium.spaces.Dict(
            {
                "state": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
                ),
                "image": gymnasium.spaces.Box(
                    low=0, high=255, shape=IMAGE_SHAPE_HWC, dtype=np.uint8
                ),
            }
        )
        self._steervla_exec_cfg: Optional[Dict[str, Any]] = None
        self._steervla_decoder: Any | None = None
        exec_raw = carla_config.get("steervla_action_execution")
        if isinstance(exec_raw, dict):
            need = ("output_action_format", "action_horizon", "action_dim", "action_input_space")
            missing = [k for k in need if k not in exec_raw]
            if missing:
                raise ValueError(
                    "carla_config['steervla_action_execution'] missing keys "
                    f"{missing}; expected {need}"
                )
            ah = int(exec_raw["action_horizon"])
            ad = int(exec_raw["action_dim"])
            self._steervla_exec_cfg = exec_raw
            try:
                from ogbench.carla.steervla_simlingo_control import SimlingoStyleWaypointDecoder

                self._steervla_decoder = SimlingoStyleWaypointDecoder()
            except ImportError as e:
                raise ImportError(
                    "SteerVLA waypoint decoding failed to load dependencies "
                    "(needs scipy, carla, openpi for normalized chunks). "
                    f"Original error: {e}"
                ) from e
            self.action_space = gymnasium.spaces.Box(
                low=-1.0, high=1.0, shape=(ah * ad,), dtype=np.float32
            )
        else:
            self.action_space = gymnasium.spaces.Box(
                low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
            )

    @property
    def evaluator(self) -> LeaderboardEvaluator:
        if self._evaluator is None:
            raise RuntimeError("Call setup() before reset() or step().")
        return self._evaluator

    @staticmethod
    def _kill_stale_carla_processes(
        rpc_port: Optional[int] = None,
        x_display_num: Optional[int] = None,
    ) -> None:
        """Kill only the CARLA/Xvfb processes for this instance's launch args.

        When the training process dies via C++ abort() (SIGABRT from msgpack
        type_error → terminate()), Python's atexit handlers never fire, so the
        LeaderboardEvaluator's registered os.killpg calls are skipped.  Orphan
        CarlaUE4.sh and Xvfb processes then consume GPU memory and hold Xvfb
        display locks, destabilizing the next training run's CARLA server.

        Also removes stale /tmp/.X{N}-lock files for dead Xvfb instances so the
        leaderboard's display-number picker (which scans :10–:99) doesn't exhaust
        all display numbers after repeated crashes.
        """
        import subprocess as _sp
        import glob as _glob
        if rpc_port is not None:
            _sp.run(
                ["pkill", "-9", "-f", f"CarlaUE4.*-carla-rpc-port={int(rpc_port)}"],
                capture_output=True,
            )
            time.sleep(1)
        if x_display_num is not None:
            _sp.run(["pkill", "-9", "-f", f"Xvfb :{int(x_display_num)}"], capture_output=True)
            time.sleep(1)
            _clear_stale_display_lock(int(x_display_num))
        # Remove lock files for dead Xvfb instances.
        for lock in _glob.glob("/tmp/.X*-lock"):
            try:
                with open(lock) as f:
                    pid = int(f.read().strip())
                # Check if the process is still alive.
                os.kill(pid, 0)
            except (ProcessLookupError, PermissionError, ValueError, OSError):
                try:
                    os.remove(lock)
                except OSError:
                    pass
        time.sleep(1)

    def setup(self) -> None:
        """Instantiate ``LeaderboardEvaluator`` and swap in :class:`SteppableScenarioManager`."""
        self._kill_stale_carla_processes(
            rpc_port=int(self.carla_config.get("port", 0) or 0),
            x_display_num=int(self.carla_config.get("x_display_num", 0) or 0),
        )
        self._args = carla_config_to_args(self.carla_config, self.route_entry)
        self._base_agent_config = self._args.agent_config
        statistics_manager = StatisticsManager(
            self._args.checkpoint, self._args.debug_checkpoint
        )
        prev_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._args.gpu_rank)
        # Force NVIDIA-only Vulkan ICD so -graphicsadapter=N maps to physical GPU N.
        # Without this, llvmpipe and other ICDs shift the Vulkan device indices, causing
        # UE4's render thread to select the wrong GPU or fail to initialize.
        _NVIDIA_VK_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"
        prev_vk_icd = os.environ.get("VK_ICD_FILENAMES")
        os.environ["VK_ICD_FILENAMES"] = _NVIDIA_VK_ICD
        self._evaluator = IsolatedLeaderboardEvaluator(self._args, statistics_manager)
        if prev_cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible_devices
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        if prev_vk_icd is not None:
            os.environ["VK_ICD_FILENAMES"] = prev_vk_icd
        else:
            os.environ.pop("VK_ICD_FILENAMES", None)
        self._evaluator.manager = SteppableScenarioManager(
            self._args.timeout, statistics_manager, self._args.debug
        )
        # Register SIGTERM handler so Ctrl+C / kill signals flush the leaderboard's
        # atexit callbacks (which kill the CARLA subprocess) before exiting.
        # SIGABRT (from C++ terminate()) cannot be intercepted this way.
        import signal as _signal
        _orig_sigterm = _signal.getsignal(_signal.SIGTERM)

        def _sigterm_handler(signum, frame):
            import atexit as _atexit
            _atexit._run_exitfuncs()  # type: ignore[attr-defined]
            if callable(_orig_sigterm):
                _orig_sigterm(signum, frame)
            else:
                raise SystemExit(0)

        _signal.signal(_signal.SIGTERM, _sigterm_handler)

    def run_leaderboard(self) -> bool:
        """Full benchmark across all routes in the configured XML file (escape hatch)."""
        if self._evaluator is not None:
            raise RuntimeError("run_leaderboard() expects a fresh process; do not call setup() first.")
        args = carla_config_to_args(self.carla_config, self.route_entry)
        return run_leaderboard(args)

    def _get_single_route_config(self):
        configs = RouteParser.parse_routes_file(self._args.routes, self._args.routes_subset)
        if not configs:
            raise ValueError(
                f"No routes for file={self._args.routes!r} routes_subset={self._args.routes_subset!r}"
            )
        if len(configs) > 1:
            raise ValueError(
                "Stepping mode requires a unique route; route_registry returned multiple entries "
                f"for {self.route_entry.scenario_name!r}."
            )
        config = configs[0]
        config.index = 0
        config.repetition_index = int(self.carla_config.get("repetition_index", 0))
        return config

    def _drain_pseudo_sensors(self) -> None:
        """Stop SpeedometerReader/OpenDriveMapReader threads before any CARLA cleanup.

        These threads call get_velocity() / get_transform() / to_opendrive() over the
        CARLA RPC socket on a timer. If the main thread makes any CARLA call while one
        of these threads is mid-RPC, the two concurrent requests corrupt the msgpack
        framing → clmdep_msgpack::type_error → C++ terminate() → abort.

        We set _run_ps=False on every pseudo-sensor and sleep briefly to let any
        in-progress CARLA call on the sensor thread finish before we proceed.
        """
        try:
            wrapper = self._evaluator.manager._agent_wrapper
            if wrapper is None:
                return
            for sensor in list(wrapper._sensors_list):
                if sensor is not None and hasattr(sensor, "_run_ps"):
                    sensor._run_ps = False
            time.sleep(0.3)
        except Exception:
            pass

    def _stop_active_scenario(self) -> None:
        if not self._scenario_active or self._evaluator is None:
            return
        try:
            print("\033[1m> Stopping the route (wrapper)\033[0m", flush=True)
            self._evaluator.manager.stop_scenario()
            # Reset world/TM modes before tearing down route actors. This helps ensure
            # background traffic can be respawned correctly on the next route load.
            self._evaluator._reset_world_settings()
        except Exception:
            print("\033[91mFailed to stop scenario in wrapper:\033[0m", flush=True)
            print(traceback.format_exc(), flush=True)
        self._drain_pseudo_sensors()
        self._evaluator._cleanup()
        self._scenario_active = False
        # Let Traffic Manager finish dropping route actors before the next load.
        time.sleep(0.3)
        # Do not set _needs_setup_on_reset here: we reuse the existing CARLA
        # client/world across episodes to avoid spawning a new server (subprocess)
        # while JAX threads are running, which triggers fork() and crashes.

    def _load_route_and_begin_stepping(self, config) -> None:
        from datetime import datetime

        ev = self._evaluator
        args = self._args
        args.agent_config = self._base_agent_config

        route_name = f"{config.name}_rep{config.repetition_index}"
        scenario_name = config.scenario_configs[0].name
        town_name = str(config.town)
        weather_id = get_weather_id(config.weather[0][1])
        current_time = datetime.now().strftime("%m_%d_%H_%M_%S")
        save_name = f"{route_name}_{town_name}_{scenario_name}_{weather_id}_{current_time}"
        ev.statistics_manager.create_route_data(
            route_name, scenario_name, weather_id, save_name, town_name, config.index
        )

        # Force a clean scenario-init mode for each route reload.
        CarlaDataProvider.set_runtime_init_mode(False)
        ev._load_and_wait_for_world(args, config.town)
        ev.route_scenario = RouteScenario(world=ev.world, config=config, debug_mode=args.debug)
        ev.statistics_manager.set_scenario(ev.route_scenario)

        ev._agent_watchdog = Watchdog(args.timeout)
        ev._agent_watchdog.start()
        agent_class_name = getattr(ev.module_agent, "get_entry_point")()
        agent_class_obj = getattr(ev.module_agent, agent_class_name)

        if getattr(agent_class_obj, "get_ros_version")() == 1 and ev._ros1_server is None:
            from leaderboard.autoagents.ros1_agent import ROS1Server

            ev._ros1_server = ROS1Server()
            ev._ros1_server.start()

        ev.agent_instance = agent_class_obj(args.host, args.port, args.debug)
        ev.agent_instance.set_global_plan(ev.route_scenario.gps_route, ev.route_scenario.route)
        args.agent_config = args.agent_config + "+" + save_name
        ev.agent_instance.setup(args.agent_config)

        # Sensors were destroyed in _cleanup(); always re-register them.
        ev.sensors = None
        ev.sensors_initialized = False
        if not ev.sensors:
            ev.sensors = ev.agent_instance.sensors()
            track = ev.agent_instance.track
            validate_sensor_configuration(ev.sensors, track, args.track)
            ev.sensor_icons = [sensors_to_icons[sensor["type"]] for sensor in ev.sensors]
            ev.statistics_manager.save_sensors(ev.sensor_icons)
            ev.statistics_manager.write_statistics()
            ev.sensors_initialized = True

        ev._agent_watchdog.stop()
        ev._agent_watchdog = None

        if args.record:
            ev.client.start_recorder(
                "{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index)
            )

        ev.manager.load_scenario(
            ev.route_scenario, ev.agent_instance, config.index, config.repetition_index
        )
        ev.manager.tick_count = 0
        ev.manager.pending_control = None
        ev.manager.last_agent_input = {}
        ev.manager.begin_scenario()
        self._expert_agent = None
        if self._expert_controller_kind == "simlingo_autopilot":
            self._expert_agent = self._build_simlingo_autopilot()
        self._cached_world_map = None
        self._current_routing_command = 4
        self._routing_last_command_tmp = -1
        self._routing_last_command = -1
        self._route_planner = None
        try:
            from coaches.simlingo.nav_planner import RoutePlanner as _RoutePlanner
            planner = _RoutePlanner(min_distance=7.5, max_distance=50.0)
            planner.set_route(ev.route_scenario.route, gps=False)
            self._route_planner = planner
        except Exception as _rp_exc:
            print(f"[routing_command] RoutePlanner init failed: {_rp_exc}", flush=True)
        self._scenario_active = True
        self._last_control = carla.VehicleControl()
        self._crash_stuck_ticks = 0
        self._prev_collision_count = 0
        self._prev_outside_route_value = 0.0

    def _ego_actor(self) -> Optional[carla.Actor]:
        ev = self._evaluator
        if ev is None or not getattr(ev, "manager", None):
            return None
        ego_list = getattr(ev.manager, "ego_vehicles", None) or []
        return ego_list[0] if ego_list else None

    def _update_routing_command(self) -> None:
        """Advance the route planner by one step and cache the current routing command."""
        if self._route_planner is None:
            return
        ego = self._ego_actor()
        if ego is None:
            return
        loc = ego.get_transform().location
        ego_pos = np.array([loc.x, loc.y, loc.z], dtype=np.float64)
        waypoint_route = self._route_planner.run_step(ego_pos)
        if len(waypoint_route) > 1:
            _, cmd = waypoint_route[1]
        elif len(waypoint_route) > 0:
            _, cmd = waypoint_route[0]
        else:
            return
        far_cmd_int = int(getattr(cmd, "value", cmd))
        if not (1 <= far_cmd_int <= 6):
            return
        # Simlingo carryover: when transitioning back to LANEFOLLOW after a turn,
        # keep showing the turn command (last_command) until the next route event.
        if self._routing_last_command_tmp != far_cmd_int:
            self._routing_last_command = self._routing_last_command_tmp
        self._routing_last_command_tmp = far_cmd_int
        if self._routing_last_command in (1, 2, 3) and far_cmd_int == 4:
            self._current_routing_command = self._routing_last_command
        else:
            self._current_routing_command = far_cmd_int

    def _load_speed_limit_map(self, map_name: str) -> bool:
        """Load the precomputed speed-limit cKDTree for ``map_name``; return True on success."""
        if map_name == self._speed_limit_map_name and self._speed_limit_tree is not None:
            return True
        npy_path = _SPEED_LIMITS_DIR / f"{map_name}_speed_limits.npy"
        if not npy_path.exists():
            return False
        try:
            from scipy.spatial import cKDTree
            data = np.load(str(npy_path), allow_pickle=True).item()
            self._speed_limit_tree = cKDTree(data["locations"])
            self._speed_limit_values = data["speed_limits"]  # km/h
            self._speed_limit_map_name = map_name
            return True
        except Exception as exc:
            print(f"[speed_limit] failed to load {npy_path}: {exc}", flush=True)
            return False

    def _lookup_speed_limit(self, location) -> Optional[float]:
        """Return speed limit in m/s for the given CARLA location, or None if unavailable."""
        if self._speed_limit_tree is None:
            return None
        try:
            pos = np.array([location.x, location.y, location.z], dtype=np.float64)
            _, idx = self._speed_limit_tree.query(pos, k=1)
            return float(self._speed_limit_values[idx]) / 3.6  # km/h → m/s
        except Exception:
            return None

    def _get_state_vector(self) -> np.ndarray:
        ego = self._ego_actor()
        if ego is None:
            return np.zeros(STATE_DIM, dtype=np.float32)
        ego_vec = _ego_state_vector(ego, self._last_control)
        cmd_onehot = _routing_command_to_onehot(self._current_routing_command)
        return np.concatenate([ego_vec, cmd_onehot])

    def _build_simlingo_autopilot(self):
        """Instantiate SimLingo's privileged autopilot as the expert controller."""
        try:
            from coaches.simlingo import AutoPilot
            route_index = f"{self.route_entry.scenario_name}_{self.route_entry.route_id}"
            ev = self._evaluator
            world = ev.world
            client = ev.client
            ego = ev.manager.ego_vehicles[0]
            CarlaDataProvider.set_client(client)
            CarlaDataProvider.set_world(world)
            CarlaDataProvider._carla_actor_pool[ego.id] = ego
            if not hasattr(CarlaDataProvider, "active_scenarios"):
                CarlaDataProvider.active_scenarios = []
            try:
                CarlaDataProvider.register_actor(ego, ego.get_transform())
            except Exception:
                pass
            agent = AutoPilot("ogbench-simlingo-autopilot", route_index=route_index)
            agent.setup("ogbench-simlingo-autopilot", route_index=route_index, traffic_manager=None)
            route_scenario = getattr(self._evaluator, "route_scenario", None)
            if route_scenario is None:
                raise RuntimeError("route_scenario missing during SimLingo autopilot setup")
            agent.set_global_plan(route_scenario.gps_route, route_scenario.route)
            print("[expert] SimLingo autopilot initialized.", flush=True)
            return agent
        except Exception as exc:
            print(f"[expert] SimLingo autopilot init failed: {exc}", flush=True)
            return None

    def _compute_language_label(self, expert_action=None):
        """Query expert commentary from live CARLA state."""
        try:
            from coaches.expert_label import NUM_COMMENTARY_WORDS
            _zero = np.zeros(NUM_COMMENTARY_WORDS, dtype=np.float32)
        except Exception:
            return "", np.zeros(0, dtype=np.float32)
        if self._label_computer is None:
            return "", _zero
        ego = self._ego_actor()
        if ego is None:
            return "", _zero
        try:
            ev = self._evaluator
            world = ev.world
            if self._cached_world_map is None:
                self._cached_world_map = world.get_map()
            route = getattr(getattr(ev, "agent_instance", None), "_global_plan_world_coord", []) or []
            scenario_type = self.route_entry.scenario_type
            _, _, text, bow = self._label_computer.compute_full(
                ego, world, self._cached_world_map, route,
                scenario_type=scenario_type, expert_action=expert_action,
            )
            return text, bow
        except Exception:
            return "", _zero

    def _compute_expert_action(self, action_horizon: int = 10, action_dim: int = 4) -> np.ndarray:
        """Expert action for DAgger / critic feedback.

        Prefer the live SimLingo expert's synchronized planner state when
        available; fall back to a route-based approximation otherwise.
        """
        out = np.zeros(action_horizon * action_dim, dtype=np.float32)
        if action_dim != 4:
            return out
        ego = self._ego_actor()
        if ego is None:
            return out
        try:
            ev = self._evaluator
            world = ev.world
            if self._cached_world_map is None:
                self._cached_world_map = world.get_map()

            live_expert = self._expert_agent
            live_data = getattr(live_expert, "last_driving_data", None) if live_expert is not None else None
            if isinstance(live_data, dict):
                route_local = np.asarray(live_data.get("route", []), dtype=np.float32)
                target_speed_live = float(live_data.get("target_speed", 0.0))
                if route_local.ndim == 2 and route_local.shape[1] >= 2 and route_local.shape[0] >= 2:
                    route_xy = np.asarray(route_local[:, :2], dtype=np.float32)
                    chunk = np.zeros((action_horizon, 4), dtype=np.float32)
                    dt = 5.0 / 20.0
                    route_dist = np.linalg.norm(np.diff(route_xy, axis=0), axis=1)
                    cum_d = np.concatenate(
                        [np.zeros(1, dtype=np.float32), np.cumsum(route_dist, dtype=np.float32)],
                        axis=0,
                    )
                    if float(cum_d[-1]) > 1e-3:
                        prev_xy = np.zeros(2, dtype=np.float32)
                        for i in range(action_horizon):
                            s = min(target_speed_live * dt * (i + 1), float(cum_d[-1]))
                            x_wp = float(np.interp(s, cum_d, route_xy[:, 0]))
                            y_wp = float(np.interp(s, cum_d, route_xy[:, 1]))
                            delta_xy = np.array([x_wp, y_wp], dtype=np.float32) - prev_xy
                            chunk[i, :2] = delta_xy
                            chunk[i, 2:] = delta_xy
                            prev_xy = np.array([x_wp, y_wp], dtype=np.float32)
                        return chunk.flatten()

            agent_instance = getattr(ev, "agent_instance", None)
            scenario_route = getattr(getattr(ev, "route_scenario", None), "route", None) or []
            route_dense = (
                getattr(agent_instance, "org_dense_route_world_coord", None)
                or scenario_route
                or []
            )
            route_sparse = getattr(agent_instance, "_global_plan_world_coord", None) or []
            route = route_dense or route_sparse

            scenario_type = str(getattr(self.route_entry, "scenario_type", "") or "")
            if "parking" in scenario_type.lower():
                try:
                    speed_limit = max(ego.get_speed_limit() / 3.6, 1.0)
                except Exception:
                    speed_limit = 8.33
                target_speed = float(np.clip(speed_limit * 0.5, 3.0, 5.0))
            else:
                try:
                    if self._label_computer is not None:
                        _, speed_ctx = self._label_computer._speed_info(ego, world, self._cached_world_map)
                        target_speed = float(speed_ctx.get("target_speed", 0.0))
                    else:
                        target_speed = 0.0
                except Exception as _se:
                    print(f"[expert_action] _speed_info failed: {_se}", flush=True)
                    target_speed = 0.0

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            fwd = ego_tf.get_forward_vector()
            right = ego_tf.get_right_vector()

            def _to_ego(loc):
                rel = loc - ego_loc
                x_e = rel.x * fwd.x + rel.y * fwd.y
                y_e = -(rel.x * right.x + rel.y * right.y)
                return x_e, y_e

            dt = 5.0 / 20.0
            chunk = np.zeros((action_horizon, 4), dtype=np.float32)

            def _build_chunk_from_route(route_seq):
                route_locs = [tf.location for tf, _ in route_seq if tf is not None]
                if len(route_locs) < 2:
                    return np.zeros((action_horizon, 4), dtype=np.float32), 0
                nearest_idx = min(range(len(route_locs)), key=lambda i: route_locs[i].distance(ego_loc))
                future_locs = route_locs[nearest_idx:]
                if len(future_locs) < 2:
                    return np.zeros((action_horizon, 4), dtype=np.float32), 0
                cum_d = [0.0]
                xy_ego = []
                prev_loc = ego_loc
                for loc in future_locs:
                    cum_d.append(cum_d[-1] + float(loc.distance(prev_loc)))
                    xy_ego.append(_to_ego(loc))
                    prev_loc = loc
                if cum_d[-1] < 1e-3:
                    return np.zeros((action_horizon, 4), dtype=np.float32), 0
                dists_r = np.asarray(cum_d, dtype=np.float64)
                xs_r = np.asarray([0.0] + [xy[0] for xy in xy_ego], dtype=np.float64)
                ys_r = np.asarray([0.0] + [xy[1] for xy in xy_ego], dtype=np.float64)
                route_chunk = np.zeros((action_horizon, 4), dtype=np.float32)
                prev_x = prev_y = 0.0
                for i in range(action_horizon):
                    s = min(target_speed * dt * (i + 1), float(dists_r[-1]))
                    x_wp = float(np.interp(s, dists_r, xs_r))
                    y_wp = float(np.interp(s, dists_r, ys_r))
                    route_chunk[i, 0] = x_wp - prev_x
                    route_chunk[i, 1] = y_wp - prev_y
                    route_chunk[i, 2] = x_wp - prev_x
                    route_chunk[i, 3] = y_wp - prev_y
                    prev_x, prev_y = x_wp, y_wp
                return route_chunk, len(future_locs)

            if route:
                chunk, route_pts = _build_chunk_from_route(route)
            else:
                route_pts = 0

            if route_pts < 2:
                curr_wp = self._cached_world_map.get_waypoint(ego_loc, project_to_road=True)
                fallback_route = []
                if curr_wp is not None:
                    fallback_route.append((curr_wp.transform, None))
                    for _ in range(action_horizon + 20):
                        nxt = curr_wp.next(1.0)
                        if not nxt:
                            break
                        curr_wp = nxt[0]
                        fallback_route.append((curr_wp.transform, None))
                chunk, route_pts = _build_chunk_from_route(fallback_route)

            if route_pts < 2:
                dx = target_speed * dt
                chunk[:, 0] = dx
                chunk[:, 2] = dx

            return chunk.flatten()
        except Exception as _e:
            print(f"[expert_action] exception: {_e}", flush=True)
            return out

    def _get_expert_input_data(self) -> Dict[str, Any]:
        """Build a minimal leaderboard-style sensor dict for the expert agent."""
        sensors = dict(getattr(self.evaluator.manager, "last_agent_input", None) or {})
        if "imu" in sensors:
            return sensors
        try:
            ego = self.evaluator.manager.ego_vehicles[0]
        except Exception:
            return sensors
        ts = float(GameTime.get_time())
        transform = ego.get_transform()
        velocity = ego.get_velocity()
        accel = ego.get_acceleration()
        ang_vel = ego.get_angular_velocity()
        speed = float(math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))
        gps = np.array([float(transform.location.x), float(transform.location.y), float(transform.location.z)], dtype=np.float32)
        compass = float(np.deg2rad(transform.rotation.yaw + 90.0))
        imu = np.array([float(accel.x), float(accel.y), float(accel.z), float(ang_vel.x), float(ang_vel.y), float(ang_vel.z), compass], dtype=np.float32)
        sensors.setdefault("imu", (ts, imu))
        sensors.setdefault("gps", (ts, gps))
        sensors.setdefault("speed", (ts, {"speed": speed}))
        return sensors

    def tick_expert(self) -> None:
        """Run the expert agent's planning step without applying the control output.

        Call this every env step during VLA rollout when expert_recover_debug is
        active so SimLingo's GameTime delta stays bounded. Silently skips if no
        expert agent is loaded.
        """
        if self._expert_agent is None:
            return
        try:
            sensors = self._get_expert_input_data()
            self._expert_agent.run_step(sensors, GameTime.get_time())
        except Exception:
            pass

    def _reset_simlingo_autopilot_state(self, agent: Any) -> None:
        """Clear controller history without discarding the expert's live route/planner state."""
        speed_controller = getattr(agent, "speed_controller", None)
        if hasattr(speed_controller, "reset_error_integral"):
            speed_controller.reset_error_integral()

        turn_controller = getattr(agent, "turn_controller", None)
        if turn_controller is not None:
            if hasattr(turn_controller, "_window"):
                turn_controller._window = []
            if hasattr(turn_controller, "error_history"):
                turn_controller.error_history = []

        if hasattr(agent, "ego_blocked_for_ticks"):
            agent.ego_blocked_for_ticks = 0

    def reinit_expert(self) -> None:
        """Prepare the expert for takeover without discarding its synchronized route state.

        During expert_recover_debug the expert is ticked in the background while
        VLA controls the car. At handoff we want fresh controller history, but
        rebuilding the SimLingo autopilot from scratch loses its live planner
        progress and can immediately produce a bad takeover action.
        """
        if self._expert_controller_kind != "simlingo_autopilot":
            return
        try:
            if self._expert_agent is None:
                self._expert_agent = self._build_simlingo_autopilot()
            elif getattr(self._expert_agent, "initialized", False):
                self._reset_simlingo_autopilot_state(self._expert_agent)
        except Exception as exc:
            print(f"[reinit_expert] failed: {exc}", flush=True)

    def step_expert(self, obs_raw: dict):
        """Step the env driven by the PDM-Lite expert (SimLingo autopilot)."""
        if self._expert_agent is not None:
            sensors = self._get_expert_input_data()
            try:
                control = self._expert_agent.run_step(sensors, GameTime.get_time())
                return self._step_with_control(control)
            except Exception as _ee:
                print(f"[step_expert] SimLingo autopilot failed: {_ee}", flush=True)
        expert_action = obs_raw.get("expert_action")
        if expert_action is None:
            expert_action = np.zeros(self.action_space.shape, dtype=np.float32)
        if self._steervla_exec_cfg is not None:
            orig_space = self._steervla_exec_cfg.get("action_input_space", "normalized")
            self._steervla_exec_cfg["action_input_space"] = "policy_output"
            try:
                return self.step(np.asarray(expert_action, dtype=np.float32))
            finally:
                self._steervla_exec_cfg["action_input_space"] = orig_space
        return self.step(np.asarray(expert_action, dtype=np.float32))

    def _compute_reward_and_info(
        self,
        *,
        tree_status: Any,
        terminated: bool,
    ) -> tuple[float, bool, Dict[str, Any]]:
        """Shared reward/termination bookkeeping for both policy and expert stepping."""
        ego = self._ego_actor()
        speed = 0.0
        if ego is not None:
            v = ego.get_velocity()
            speed = float(np.linalg.norm([v.x, v.y, v.z]))
        lane_metrics = self._lane_alignment_metrics()
        lane_offset_m = float(lane_metrics["lane_offset_m"])
        heading_error_rad = float(lane_metrics["heading_error_rad"])
        lane_width_m = float(lane_metrics["lane_width_m"])
        speed_limit_mps = float(lane_metrics["speed_limit_mps"])
        lane_half_width = max(0.5 * lane_width_m, 1e-3)
        centering_factor = float(np.clip(1.0 - abs(lane_offset_m) / lane_half_width, 0.0, 1.0))
        heading_factor = float(np.clip(math.cos(heading_error_rad), 0.0, 1.0))
        speed_norm = float(np.clip(speed / max(speed_limit_mps, 1e-3), 0.0, 1.0))
        overspeed_frac = max(0.0, speed / max(speed_limit_mps, 1e-3) - 1.0)
        steer_pen = self._steer_penalty_weight * abs(float(getattr(self._last_control, "steer", 0.0)))
        brake_pen = self._brake_penalty_weight * float(getattr(self._last_control, "brake", 0.0))
        speed_limit_pen = self._speed_limit_penalty_weight * overspeed_frac
        progress_reward = self._progress_reward_weight * speed_norm * centering_factor * heading_factor
        # centering_reward = self._centering_reward_weight * centering_factor
        # heading_reward = self._heading_reward_weight * heading_factor

        reward = (
            progress_reward
            # + centering_reward
            # + heading_reward
            - steer_pen
            - brake_pen
            - speed_limit_pen
        )
        info = self._info_with_sensors(
            {"scenario_tree_status": getattr(tree_status, "name", str(tree_status))}
        )
        crash_stuck, collision_count = self._update_crash_stuck_state(speed)
        outside_route_value, _min_speed_value = self._route_infraction_values()

        collision_delta = max(0, collision_count - self._prev_collision_count)
        outside_route_delta = max(0.0, outside_route_value - self._prev_outside_route_value)
        self._prev_collision_count = collision_count
        self._prev_outside_route_value = outside_route_value

        collision_pen = self._collision_event_penalty * float(collision_delta)
        outside_route_pen = self._outside_route_event_penalty * float(outside_route_delta)
        reward += collision_pen + outside_route_pen

        terminal_bonus = 0.0
        info["collision_count"] = collision_count
        info["route_progress_pct"] = self._route_completion_pct()
        info["crash_stuck_ticks"] = self._crash_stuck_ticks
        info["outside_route_value"] = outside_route_value
        info["collision_delta"] = float(collision_delta)
        info["outside_route_delta"] = float(outside_route_delta)
        info["lane_offset_m"] = lane_offset_m
        info["heading_error_rad"] = heading_error_rad
        info["lane_width_m"] = lane_width_m
        info["speed_limit_mps"] = speed_limit_mps
        info["speed_norm"] = speed_norm
        info["overspeed_frac"] = overspeed_frac
        info["centering_factor"] = centering_factor
        info["heading_factor"] = heading_factor
        info["penalty_collision"] = collision_pen
        info["penalty_outside_route"] = outside_route_pen
        info["penalty_steer"] = -steer_pen
        info["penalty_brake"] = -brake_pen
        info["penalty_speed_limit"] = -speed_limit_pen
        info["reward_progress"] = progress_reward
        # info["reward_centering"] = centering_reward
        # info["reward_heading"] = heading_reward

        if crash_stuck:
            terminated = True
            reward += self._crash_stuck_penalty
            info["success"] = False
            info["termination_reason"] = "crash_stuck"
            info["penalty_crash_stuck"] = self._crash_stuck_penalty
        else:
            info["penalty_crash_stuck"] = 0.0

        if terminated:
            if crash_stuck:
                self._finalize_route("Finished", "Agent crashed and got stuck")
            else:
                success = info["scenario_tree_status"] == "SUCCESS"
                terminal_bonus = SUCCESS_BONUS if success else FAILURE_BONUS
                reward += terminal_bonus
                info["success"] = success
                self._finalize_route("Finished", "")

        info["reward_terminal"] = terminal_bonus
        info["reward_total"] = float(reward)
        return float(reward), bool(terminated), info

    def _apply_episode_max_steps(
        self,
        reward: float,
        terminated: bool,
        info: Dict[str, Any],
    ) -> tuple[float, bool, Dict[str, Any]]:
        """Force route termination in the simulator once the step cap is reached."""
        self._episode_step_count += 1
        info = dict(info)
        info["episode_step_count"] = self._episode_step_count

        if (
            self._max_episode_steps > 0
            and self._episode_step_count >= self._max_episode_steps
            and not terminated
        ):
            info["termination_reason"] = "episode_max_steps"
            info["success"] = False
            info["scenario_tree_status"] = "TIMEOUT"
            info["reward_terminal"] = 0.0
            info["reward_total"] = float(reward)
            self._finalize_route("Finished", "Episode max steps")
            terminated = True
        return float(reward), bool(terminated), info

    def _step_with_control(self, control):
        """Apply a pre-computed VehicleControl and run one leaderboard tick."""
        self._last_control = control
        self.evaluator.manager.pending_control = control
        try:
            running, tree_status = self.evaluator.manager.step_once()
        except Exception as e:
            return self._obs_dict(), -1.0, True, False, {"error": str(e)}
        terminated = not running
        self._update_routing_command()
        reward, terminated, info = self._compute_reward_and_info(
            tree_status=tree_status,
            terminated=terminated,
        )
        reward, terminated, info = self._apply_episode_max_steps(reward, terminated, info)
        if self._expert_agent is not None:
            self.tick_expert()
        return self._obs_dict(), float(reward), terminated, False, info

    def _obs_dict(self) -> Dict[str, np.ndarray]:
        sensors = getattr(self.evaluator.manager, "last_agent_input", None) or {}
        expert_action = self._compute_expert_action()
        commentary_text, language_label = self._compute_language_label(expert_action=expert_action)
        rgb_viz = rgb_viz_from_leaderboard_dict(sensors)
        return {
            "state": self._get_state_vector(),
            "image": downscale_rgb_for_policy(rgb_viz),
            "image_viz": rgb_viz,
            "language_label": language_label,
            "commentary_text": commentary_text,
            "expert_action": expert_action,
            "routing_command": ROUTING_COMMAND_TEXT.get(self._current_routing_command, "follow the road"),
        }

    def _info_with_sensors(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "route": self.route_entry.scenario_name,
            "scenario_type": self.route_entry.scenario_type,
            "town": self.route_entry.town,
            "tick": getattr(self.evaluator.manager, "tick_count", 0),
        }
        sensors = getattr(self.evaluator.manager, "last_agent_input", None)
        if sensors:
            info["sensors"] = sensors
        if extra:
            info.update(extra)
        return info

    def _collision_count(self) -> int:
        scenario = getattr(self.evaluator, "route_scenario", None)
        if scenario is None:
            return 0
        for criterion in scenario.get_criteria():
            if getattr(criterion, "name", "") == "CollisionTest":
                return int(getattr(criterion, "actual_value", 0))
        return 0

    def _route_completion_pct(self) -> float:
        """Route completion percentage from leaderboard ``RouteCompletionTest`` (0–100)."""
        scenario = getattr(self.evaluator, "route_scenario", None)
        if scenario is None:
            return 0.0
        for criterion in scenario.get_criteria():
            if getattr(criterion, "name", "") == "RouteCompletionTest":
                try:
                    return float(getattr(criterion, "actual_value", 0.0))
                except Exception:
                    return 0.0
        return 0.0

    def _route_infraction_values(self) -> Tuple[float, float]:
        """Return cumulative infraction values for outside-route and minimum-speed criteria."""
        scenario = getattr(self.evaluator, "route_scenario", None)
        if scenario is None:
            return 0.0, 0.0

        outside_route_val = 0.0
        min_speed_val = 0.0
        for criterion in scenario.get_criteria():
            name = str(getattr(criterion, "name", "")).lower()
            try:
                value = float(getattr(criterion, "actual_value", 0.0))
            except Exception:
                value = 0.0

            if ("outside" in name and ("route" in name or "lane" in name)) or ("off" in name and "route" in name):
                outside_route_val = max(outside_route_val, value)
            if "minspeed" in name or ("minimum" in name and "speed" in name):
                min_speed_val = max(min_speed_val, value)

        return outside_route_val, min_speed_val

    @staticmethod
    def _wrap_angle_rad(angle: float) -> float:
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)

    def _lane_alignment_metrics(self) -> Dict[str, float]:
        ego = self._ego_actor()
        if ego is None:
            return {
                "lane_offset_m": 0.0,
                "heading_error_rad": 0.0,
                "lane_width_m": 3.5,
                "speed_limit_mps": 8.0,
            }
        try:
            if self._cached_world_map is None:
                self._cached_world_map = self.evaluator.world.get_map()
            ego_tf = ego.get_transform()
            wp = self._cached_world_map.get_waypoint(
                ego_tf.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if wp is None:
                raise RuntimeError("no waypoint available")
            lane_tf = wp.transform
            lane_yaw = math.radians(lane_tf.rotation.yaw)
            dx = float(ego_tf.location.x - lane_tf.location.x)
            dy = float(ego_tf.location.y - lane_tf.location.y)
            right_x = -math.sin(lane_yaw)
            right_y = math.cos(lane_yaw)
            lane_offset_m = dx * right_x + dy * right_y
            heading_error_rad = self._wrap_angle_rad(
                math.radians(float(ego_tf.rotation.yaw - lane_tf.rotation.yaw))
            )
            map_name = self._cached_world_map.name.split("/")[-1]
            self._load_speed_limit_map(map_name)
            speed_limit_mps = self._lookup_speed_limit(ego_tf.location)
            if speed_limit_mps is None:
                speed_limit_mps = max(float(ego.get_speed_limit()) / 3.6, 1.0)
            else:
                speed_limit_mps = max(speed_limit_mps, 1.0)
            lane_width_m = max(float(getattr(wp, "lane_width", 3.5)), 1.0)
            return {
                "lane_offset_m": float(lane_offset_m),
                "heading_error_rad": float(heading_error_rad),
                "lane_width_m": float(lane_width_m),
                "speed_limit_mps": float(speed_limit_mps),
            }
        except Exception:
            return {
                "lane_offset_m": 0.0,
                "heading_error_rad": 0.0,
                "lane_width_m": 3.5,
                "speed_limit_mps": 8.0,
            }

    def _update_crash_stuck_state(self, speed: float) -> Tuple[bool, int]:
        collision_count = self._collision_count()
        if collision_count > 0 and speed < self._crash_stuck_speed_threshold:
            self._crash_stuck_ticks += 1
        else:
            self._crash_stuck_ticks = 0
        return self._crash_stuck_ticks >= self._crash_stuck_steps, collision_count

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if options:
            seed = options.get("seed", seed)
        super().reset(seed=seed)
        if self._evaluator is None:
            self.setup()

        self._episode_step_count = 0
        self._stop_active_scenario()

        config = self._get_single_route_config()
        try:
            self._load_route_and_begin_stepping(config)
        except SensorConfigurationInvalid as e:
            entry_status, crash_message = FAILURE_MESSAGES["Sensors"]
            self.evaluator.statistics_manager.save_entry_status(entry_status)
            self.evaluator.statistics_manager.compute_route_statistics(
                config.index,
                self.evaluator.manager.scenario_duration_system,
                self.evaluator.manager.scenario_duration_game,
                crash_message,
            )
            self._drain_pseudo_sensors()
            self.evaluator._cleanup()
            raise RuntimeError(f"Invalid sensors: {e}") from e
        except Exception:
            self._drain_pseudo_sensors()
            self.evaluator._cleanup()
            raise

        return self._obs_dict(), self._info_with_sensors()

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        flat = np.asarray(action, dtype=np.float32).reshape(-1)
        control = None
        self._update_routing_command()
        if self._steervla_exec_cfg is not None and self._steervla_decoder is not None:
            ah = int(self._steervla_exec_cfg["action_horizon"])
            ad = int(self._steervla_exec_cfg["action_dim"])
            expected = ah * ad
            if flat.size == expected:
                from ogbench.carla.steervla_simlingo_control import maybe_steervla_vehicle_control

                control = maybe_steervla_vehicle_control(
                    action,
                    state_vec=self._get_state_vector(),
                    exec_cfg=self._steervla_exec_cfg,
                    decoder=self._steervla_decoder,
                )
            elif flat.size == ACTION_DIM:
                control = _action_to_control(action)
            else:
                raise ValueError(
                    f"SteerVLA chunk env expects action length {expected} or legacy {ACTION_DIM}, "
                    f"got {flat.size}"
                )
        if control is None:
            control = _action_to_control(action)
        self._last_control = control
        self.evaluator.manager.pending_control = control

        try:
            running, tree_status = self.evaluator.manager.step_once()
        except AgentError as e:
            self._finalize_route(*FAILURE_MESSAGES["Agent_runtime"])
            return (
                self._obs_dict(),
                -1.0, True, False,
                self._info_with_sensors({"error": "agent_runtime", "exception": str(e)}),
            )
        except TickRuntimeError as e:
            self._finalize_route("Started", "TickRuntime")
            return (
                self._obs_dict(),
                -1.0, True, False,
                self._info_with_sensors({"error": "tick_runtime", "exception": str(e)}),
            )
        except Exception as e:
            self._finalize_route(*FAILURE_MESSAGES["Simulation"])
            return (
                self._obs_dict(),
                -1.0, True, False,
                self._info_with_sensors({"error": "simulation", "exception": str(e)}),
            )

        terminated = not running
        reward, terminated, info = self._compute_reward_and_info(
            tree_status=tree_status,
            terminated=terminated,
        )
        reward, terminated, info = self._apply_episode_max_steps(reward, terminated, info)
        return self._obs_dict(), float(reward), terminated, False, info

    def _finalize_route(self, entry_status: str, crash_message: str) -> None:
        if not self._scenario_active:
            return
        config_index = self.evaluator.manager.route_index
        try:
            print("\033[1m> Stopping the route (wrapper)\033[0m", flush=True)
            self.evaluator.manager.stop_scenario()
            print("\033[1m> Registering the route statistics (wrapper)\033[0m", flush=True)
            self.evaluator.statistics_manager.save_entry_status(entry_status)
            self.evaluator.statistics_manager.compute_route_statistics(
                config_index,
                self.evaluator.manager.scenario_duration_system,
                self.evaluator.manager.scenario_duration_game,
                crash_message,
            )
            self.evaluator.statistics_manager.write_statistics()
            if self._args.record:
                self.evaluator.client.stop_recorder()
        finally:
            self._drain_pseudo_sensors()
            try:
                self.evaluator._cleanup()
            except Exception:
                pass
            self._scenario_active = False
            # Reuse the existing evaluator/CARLA client on next reset rather than
            # spawning a new server.  Calling setup() after JAX is initialized
            # triggers subprocess.Popen (Xvfb + CarlaUE4.sh), which forks while
            # JAX threads are live and can corrupt the msgpack RPC connection.

    def render(self):
        """Placeholder frame for evaluation video paths (avoid NotImplementedError)."""
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self) -> None:
        if self._evaluator is not None:
            try:
                self._stop_active_scenario()
            finally:
                try:
                    self._evaluator._reset_world_settings()
                except Exception:
                    pass
                self._evaluator = None
                self._args = None
        super().close()


def make_env_and_datasets(
    env_name: str,
    env_only: bool = False,
    carla_config_path: Optional[Union[str, Path]] = None,
    route: Optional[str] = None,
) -> Union[
    CarlaBench2DriveWrapper,
    Tuple[CarlaBench2DriveWrapper, None, None],
]:
    """Construct the CARLA Bench2Drive gym wrapper.

    ``env_name`` is accepted for symmetry with the OGBench MuJoCo factory but is
    only used to resolve the route when ``route`` is not given (it is treated as
    a fallback registry key, e.g. ``"parking-cut-in-001"``).
    """
    carla_config = load_carla_config(carla_config_path)
    if route is None and env_name and env_name not in {"carla", "bench2drive"}:
        route = env_name
    env = CarlaBench2DriveWrapper(carla_config, route=route)
    if env_only:
        return env, None, None
    raise NotImplementedError(
        "CARLA offline datasets are not implemented; call make_env_and_datasets(..., env_only=True)."
    )
