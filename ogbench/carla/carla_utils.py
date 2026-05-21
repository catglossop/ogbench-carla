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
    * ``"state"`` -- ``float32`` vector of shape ``(STATE_DIM,)`` (ego kinematics).
    * ``"image"`` -- ``uint8`` RGB array ``(*IMAGE_SHAPE_HWC,)`` from the front camera
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

# Flat ego-state vector layout (length = 19): 3 location, 3 rotation (rpy),
# 3 velocity, 3 angular velocity, 3 acceleration, 1 speed, 3 last-applied control.
STATE_DIM = 19
ACTION_DIM = 2

# Indices into ``obs["state"]`` for :func:`_ego_state_vector` (length :data:`STATE_DIM`).
EGO_STATE_IDX_SPEED = 15
EGO_STATE_IDX_THROTTLE = 16
EGO_STATE_IDX_STEER = 17
EGO_STATE_IDX_BRAKE = 18

DEFAULT_CRASH_STUCK_SPEED_THRESHOLD = 0.1
DEFAULT_CRASH_STUCK_STEPS = 20
DEFAULT_CRASH_STUCK_PENALTY = -1.0
DEFAULT_COLLISION_EVENT_PENALTY = -5.0
DEFAULT_OUTSIDE_ROUTE_EVENT_PENALTY = -5.0
SUCCESS_BONUS = 5.0
FAILURE_BONUS = -5.0
CARLA_FPS = 20.0
REWARD_TTC_PERSIST_STEPS = 500
REWARD_COMFORT_PERSIST_STEPS = 500
REWARD_BLOCKED_SPEED_THRESHOLD_MPS = 0.1
REWARD_BLOCKED_TIME_SECONDS = 90.0
REWARD_TTC_FORECAST_SECONDS = 1.0
REWARD_TTC_INTERVAL_SECONDS = 0.2
REWARD_SPEEDING_MARGIN_KMH = 8.0
COMFORT_THRESHOLDS = {
    "longitudinal_acceleration_min": -20.0,
    "longitudinal_acceleration_max": 10.0,
    "lateral_acceleration_abs": 9.0,
    "absolute_jerk_abs": 30.0,
    "longitudinal_jerk_abs": 30.0,
    "yaw_rate_abs": 1.0,
    "yaw_acceleration_abs": 3.0,
}


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


def _bgra_to_rgb_hwc(arr: np.ndarray) -> np.ndarray:
    """CARLA leaderboard packs ``sensor.camera.rgb`` as H×W×4 BGRA uint8."""
    bgr = np.asarray(arr)[..., :3]
    return np.ascontiguousarray(bgr[..., ::-1], dtype=np.uint8)


def rgb_front_from_leaderboard_dict(sensor_dict: Dict[str, Any]) -> np.ndarray:
    """Decode ``rgb_front`` leaderboard sensor payload to RGB ``uint8`` H×W×3."""
    if not sensor_dict or RGB_FRONT_CAMERA_TAG not in sensor_dict:
        return _zeros_rgb_image()
    tup = sensor_dict[RGB_FRONT_CAMERA_TAG]
    if not isinstance(tup, (tuple, list)) or len(tup) < 2:
        return _zeros_rgb_image()
    payload = tup[1]
    if payload is None:
        return _zeros_rgb_image()
    arr = np.asarray(payload)
    if arr.ndim != 3:
        return _zeros_rgb_image()
    if arr.shape[-1] == 4:
        rgb = _bgra_to_rgb_hwc(arr)
    elif arr.shape[-1] == 3:
        rgb = arr.astype(np.uint8, copy=False)
    else:
        return _zeros_rgb_image()
    if rgb.shape != IMAGE_SHAPE_HWC:
        # Resize rarely needed if sensor definition matches IMAGE_SHAPE_HWC.
        try:
            import cv2

            rgb = cv2.resize(rgb, (IMAGE_SHAPE_HWC[1], IMAGE_SHAPE_HWC[0]), interpolation=cv2.INTER_AREA)
        except Exception:
            return _zeros_rgb_image()
    return rgb


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


def _ego_state_vector(
    ego: carla.Actor, last_control: carla.VehicleControl
) -> np.ndarray:
    """Build a length-:data:`STATE_DIM` flat float32 vector describing the ego car."""
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
        self._prev_collision_count = 0
        self._prev_outside_route_value = 0.0
        self._blocked_ticks = 0
        self._blocked_steps = int(
            self.carla_config.get(
                "reward_blocked_steps",
                round(REWARD_BLOCKED_TIME_SECONDS * CARLA_FPS),
            )
        )
        self._use_leave_route_done = bool(self.carla_config.get("use_leave_route_done", False))
        self._min_thresh_lat_dist = float(self.carla_config.get("min_thresh_lat_dist", 2.0))
        self._terminal_reward = float(self.carla_config.get("terminal_reward", 0.0))
        self._use_perc_progress = bool(self.carla_config.get("use_perc_progress", True))
        self._speeding_infraction = bool(self.carla_config.get("speeding_infraction", False))
        self._comfort_infraction = bool(self.carla_config.get("comfort_infraction", False))
        self._route_progress_xyz: Optional[np.ndarray] = None
        self._route_progress_s: Optional[np.ndarray] = None
        self._route_total_distance_m = 0.0
        self._route_progress_index = 0
        self._route_transforms: list[Any] = []
        self._route_completion_accum_perc: list[float] = []
        self._route_completion_index = 0
        self._last_route_completion = 0.0
        self._in_route_current_index = 0
        self._in_route_out_route_distance = 0.0
        self._in_route_safe = True
        self._in_route_accum_meters: list[float] = []
        self._ttc_penalty_ticks = 0
        self._comfort_penalty_ticks = 0
        self._comfort_penalty_factor = 1.0
        self._prev_ego_accel_world: Optional[np.ndarray] = None
        self._prev_ego_yaw_rate = 0.0
        self._last_expert_action_source: str | None = None

        self._expert_controller_kind = str(self.carla_config.get("expert_controller", "") or "").strip().lower()
        self._expert_agent: Any | None = None
        self._cached_world_map: Any | None = None
        try:
            from coaches.expert_label import ExpertLabelComputer
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
        self._scenario_active = True
        self._last_control = carla.VehicleControl()
        self._crash_stuck_ticks = 0
        self._prev_collision_count = 0
        self._prev_outside_route_value = 0.0
        self._blocked_ticks = 0
        self._ttc_penalty_ticks = 0
        self._comfort_penalty_ticks = 0
        self._comfort_penalty_factor = 1.0
        self._prev_ego_accel_world = None
        self._prev_ego_yaw_rate = 0.0
        self._init_route_progress_cache()
        self._last_expert_action_source = None

    def _ego_actor(self) -> Optional[carla.Actor]:
        ev = self._evaluator
        if ev is None or not getattr(ev, "manager", None):
            return None
        ego_list = getattr(ev.manager, "ego_vehicles", None) or []
        return ego_list[0] if ego_list else None

    def _get_state_vector(self) -> np.ndarray:
        ego = self._ego_actor()
        if ego is None:
            return np.zeros(STATE_DIM, dtype=np.float32)
        return _ego_state_vector(ego, self._last_control)

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
        def _log_source(source: str) -> None:
            if self._last_expert_action_source != source:
                print(f"[expert_action] source={source}", flush=True)
                self._last_expert_action_source = source

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
                        _log_source("live_expert")
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

            source = "route_fallback"
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
                source = "lane_waypoint_fallback"

            if route_pts < 2:
                dx = target_speed * dt
                chunk[:, 0] = dx
                chunk[:, 2] = dx
                source = "straight_line_fallback"

            _log_source(source)
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
        criteria = self._criteria_snapshot()
        route_completion = self._route_completion_percent(criteria)
        route_completion_delta = max(0.0, route_completion - self._last_route_completion)
        self._last_route_completion = route_completion
        reward = 0.01 * speed
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

        info["collision_count"] = collision_count
        info["crash_stuck_ticks"] = self._crash_stuck_ticks
        info["outside_route_value"] = outside_route_value
        info["route_completion"] = float(route_completion)
        info["route_completion_delta"] = float(route_completion_delta)
        info["collision_delta"] = float(collision_delta)
        info["outside_route_delta"] = float(outside_route_delta)
        info["penalty_collision"] = collision_pen
        info["penalty_outside_route"] = outside_route_pen
        info["reward_terminal"] = 0.0
        info["reward_total"] = float(reward)
        if crash_stuck:
            terminated = True
            reward += self._crash_stuck_penalty
            info["success"] = False
            info["termination_reason"] = "crash_stuck"
            info["reward_total"] = float(reward)
        if terminated:
            if crash_stuck:
                self._finalize_route("Finished", "Agent crashed and got stuck")
            else:
                success = info["scenario_tree_status"] == "SUCCESS"
                reward += SUCCESS_BONUS if success else FAILURE_BONUS
                info["success"] = success
                self._finalize_route("Finished", "")
                info["reward_total"] = float(reward)
        return float(reward), bool(terminated), info

    def _step_with_control(self, control, *, tick_expert_after: bool = False):
        """Apply a pre-computed VehicleControl and run one leaderboard tick."""
        self._last_control = control
        self.evaluator.manager.pending_control = control
        try:
            running, tree_status = self.evaluator.manager.step_once()
        except Exception as e:
            return self._obs_dict(), -1.0, True, False, {"error": str(e)}
        terminated = not running
        reward, terminated, info = self._compute_reward_and_info(
            tree_status=tree_status,
            terminated=terminated,
        )
        if tick_expert_after and self._expert_agent is not None:
            self.tick_expert()
        return self._obs_dict(), float(reward), terminated, False, info

    def _obs_dict(self) -> Dict[str, np.ndarray]:
        sensors = getattr(self.evaluator.manager, "last_agent_input", None) or {}
        expert_action = self._compute_expert_action()
        commentary_text, language_label = self._compute_language_label(expert_action=expert_action)
        return {
            "state": self._get_state_vector(),
            "image": rgb_front_from_leaderboard_dict(sensors),
            "language_label": language_label,
            "commentary_text": commentary_text,
            "expert_action": expert_action,
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

    def _criteria_snapshot(self) -> list[dict[str, Any]]:
        scenario = getattr(self.evaluator, "route_scenario", None)
        if scenario is None:
            return []
        out: list[dict[str, Any]] = []
        for criterion in scenario.get_criteria():
            name = str(getattr(criterion, "name", "") or "")
            try:
                value = float(getattr(criterion, "actual_value", 0.0))
            except Exception:
                value = 0.0
            status = str(
                getattr(
                    criterion,
                    "test_status",
                    getattr(criterion, "status", ""),
                )
                or ""
            )
            out.append(
                {
                    "name": name,
                    "name_lower": name.lower(),
                    "value": value,
                    "status": status.lower(),
                }
            )
        return out

    @staticmethod
    def _criterion_matches(name_lower: str, token_groups: tuple[tuple[str, ...], ...]) -> bool:
        return any(all(token in name_lower for token in group) for group in token_groups)

    def _criterion_value(
        self,
        criteria: list[dict[str, Any]],
        token_groups: tuple[tuple[str, ...], ...],
    ) -> float:
        value = 0.0
        for criterion in criteria:
            if self._criterion_matches(criterion["name_lower"], token_groups):
                value = max(value, float(criterion["value"]))
        return value

    def _criterion_triggered(
        self,
        criteria: list[dict[str, Any]],
        token_groups: tuple[tuple[str, ...], ...],
    ) -> bool:
        for criterion in criteria:
            if not self._criterion_matches(criterion["name_lower"], token_groups):
                continue
            if criterion["value"] > 0.0:
                return True
            if any(tag in criterion["status"] for tag in ("fail", "failure", "invalid")):
                return True
        return False

    def _collision_count(self) -> int:
        scenario = getattr(self.evaluator, "route_scenario", None)
        if scenario is None:
            return 0
        for criterion in scenario.get_criteria():
            if getattr(criterion, "name", "") == "CollisionTest":
                return int(getattr(criterion, "actual_value", 0))
        return 0

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

    def _world_map(self):
        if self._cached_world_map is None:
            self._cached_world_map = self.evaluator.world.get_map()
        return self._cached_world_map

    def _waypoint_at_location(
        self,
        location: carla.Location,
        *,
        project_to_road: bool,
        lane_type: carla.LaneType,
    ):
        try:
            return self._world_map().get_waypoint(
                location,
                project_to_road=project_to_road,
                lane_type=lane_type,
            )
        except Exception:
            return None

    def _lane_alignment_metrics(self) -> Dict[str, Any]:
        ego = self._ego_actor()
        if ego is None:
            return {
                "lane_offset_m": 0.0,
                "heading_error_rad": 0.0,
                "lane_width_m": 3.5,
                "speed_limit_mps": 8.0,
                "is_junction": False,
                "driving_waypoint": None,
                "any_waypoint": None,
            }
        try:
            ego_tf = ego.get_transform()
            driving_wp = self._waypoint_at_location(
                ego_tf.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            any_wp = self._waypoint_at_location(
                ego_tf.location,
                project_to_road=False,
                lane_type=carla.LaneType.Any,
            )
            if driving_wp is None:
                raise RuntimeError("no waypoint available")
            lane_tf = driving_wp.transform
            lane_yaw = math.radians(lane_tf.rotation.yaw)
            dx = float(ego_tf.location.x - lane_tf.location.x)
            dy = float(ego_tf.location.y - lane_tf.location.y)
            right_x = -math.sin(lane_yaw)
            right_y = math.cos(lane_yaw)
            lane_offset_m = dx * right_x + dy * right_y
            heading_error_rad = self._wrap_angle_rad(
                math.radians(float(ego_tf.rotation.yaw - lane_tf.rotation.yaw))
            )
            speed_limit_mps = max(float(ego.get_speed_limit()) / 3.6, 1.0)
            lane_width_m = max(float(getattr(driving_wp, "lane_width", 3.5)), 1.0)
            return {
                "lane_offset_m": float(lane_offset_m),
                "heading_error_rad": float(heading_error_rad),
                "lane_width_m": float(lane_width_m),
                "speed_limit_mps": float(speed_limit_mps),
                "is_junction": bool(getattr(driving_wp, "is_junction", False)),
                "driving_waypoint": driving_wp,
                "any_waypoint": any_wp,
            }
        except Exception:
            return {
                "lane_offset_m": 0.0,
                "heading_error_rad": 0.0,
                "lane_width_m": 3.5,
                "speed_limit_mps": 8.0,
                "is_junction": False,
                "driving_waypoint": None,
                "any_waypoint": None,
            }

    def _init_route_progress_cache(self) -> None:
        ev = self._evaluator
        route = getattr(getattr(ev, "route_scenario", None), "route", None) or []
        self._route_transforms = [
            route_item[0] if isinstance(route_item, (tuple, list)) and route_item else route_item
            for route_item in route
        ]
        xyz: list[list[float]] = []
        for transform in self._route_transforms:
            loc = getattr(transform, "location", None)
            if loc is None:
                continue
            xyz.append([float(loc.x), float(loc.y), float(loc.z)])
        if len(xyz) < 2:
            self._route_progress_xyz = None
            self._route_progress_s = None
            self._route_total_distance_m = 0.0
            self._route_progress_index = 0
            self._route_completion_accum_perc = []
            self._route_completion_index = 0
            self._last_route_completion = 0.0
            self._in_route_current_index = 0
            self._in_route_out_route_distance = 0.0
            self._in_route_safe = True
            self._in_route_accum_meters = []
            return

        route_xyz = np.asarray(xyz, dtype=np.float32)
        seg = np.linalg.norm(route_xyz[1:] - route_xyz[:-1], axis=1)
        route_s = np.concatenate(
            [np.zeros(1, dtype=np.float32), np.cumsum(seg, dtype=np.float32)],
            axis=0,
        )
        self._route_progress_xyz = route_xyz
        self._route_progress_s = route_s
        self._route_total_distance_m = float(max(route_s[-1], 1e-6))
        self._route_progress_index = 0
        self._route_completion_accum_perc = (
            (100.0 * route_s / self._route_total_distance_m).astype(np.float32).tolist()
        )
        self._route_completion_index = 0
        self._last_route_completion = 0.0
        self._in_route_current_index = 0
        self._in_route_out_route_distance = 0.0
        self._in_route_safe = True
        self._in_route_accum_meters = route_s.astype(np.float32).tolist()

    def _route_completion_percent(self, criteria: list[dict[str, Any]]) -> float:
        criterion_value = self._criterion_value(
            criteria,
            (("route", "completion"), ("route", "completed")),
        )
        if criterion_value > 0.0:
            return float(np.clip(criterion_value, 0.0, 100.0))

        ego = self._ego_actor()
        if ego is None or not self._route_transforms or not self._route_completion_accum_perc:
            return self._last_route_completion

        location = ego.get_transform().location
        route_length = len(self._route_transforms)
        for index in range(
            self._route_completion_index,
            min(self._route_completion_index + 3, route_length),
        ):
            route_transform = self._route_transforms[index]
            route_location = route_transform.location
            wp_dir = route_transform.get_forward_vector()
            wp_veh = location - route_location
            if wp_veh.dot(wp_dir) > 0:
                self._route_completion_index = index
        return float(round(self._route_completion_accum_perc[self._route_completion_index], 2))

    def _route_deviation_distance(self) -> float:
        ego = self._ego_actor()
        if ego is None or self._route_progress_xyz is None:
            return 0.0
        loc = ego.get_location()
        pos = np.array([float(loc.x), float(loc.y), float(loc.z)], dtype=np.float32)
        search_start = max(0, self._route_progress_index - 20)
        search_end = min(self._route_progress_xyz.shape[0], self._route_progress_index + 200)
        route_xyz = self._route_progress_xyz[search_start:search_end]
        if route_xyz.shape[0] == 0:
            return 0.0
        return float(np.min(np.linalg.norm(route_xyz - pos[None, :], axis=1)))

    def _closest_route_lateral_distance(self) -> float:
        ego = self._ego_actor()
        if ego is None or not self._route_transforms:
            return 0.0
        ego_tf = ego.get_transform()
        pos = np.array([float(ego_tf.location.x), float(ego_tf.location.y)], dtype=np.float32)
        if len(self._route_transforms) > 1:
            close_point_global = np.array(
                [
                    self._route_transforms[0].location.x,
                    self._route_transforms[0].location.y,
                ],
                dtype=np.float32,
            )
            next_point_global = np.array(
                [
                    self._route_transforms[1].location.x,
                    self._route_transforms[1].location.y,
                ],
                dtype=np.float32,
            )
            distance = next_point_global - close_point_global
            if float(np.linalg.norm(distance)) < 0.1:
                yaw_route = self._route_transforms[0].rotation.yaw
            else:
                yaw_route = np.rad2deg(np.arctan2(distance[1], distance[0]))
        else:
            close_point_global = np.array(
                [
                    self._route_transforms[0].location.x,
                    self._route_transforms[0].location.y,
                ],
                dtype=np.float32,
            )
            yaw_route = self._route_transforms[0].rotation.yaw

        dx = pos[0] - close_point_global[0]
        dy = pos[1] - close_point_global[1]
        yaw_rad = math.radians(float(yaw_route))
        lat = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
        return float(abs(lat))

    def _in_route_ok(self) -> bool:
        ego = self._ego_actor()
        if ego is None or not self._route_transforms or not self._in_route_accum_meters:
            return True

        location = ego.get_location()
        shortest_distance = float("inf")
        closest_index = -1
        route_length = len(self._route_transforms)

        for index in range(
            self._in_route_current_index,
            min(self._in_route_current_index + 6, route_length),
        ):
            ref_location = self._route_transforms[index].location
            distance = math.sqrt((location.x - ref_location.x) ** 2 + (location.y - ref_location.y) ** 2)
            if distance <= shortest_distance:
                closest_index = index
                shortest_distance = distance

        if closest_index == -1 or shortest_distance == float("inf"):
            return True

        off_route = True
        if shortest_distance < 30.0:
            off_route = False
            self._in_route_safe = bool(shortest_distance < 15.0)

        if self._in_route_current_index != closest_index:
            new_dist = self._in_route_accum_meters[closest_index] - self._in_route_accum_meters[self._in_route_current_index]
            if not self._in_route_safe:
                self._in_route_out_route_distance += new_dist
                out_route_percentage = 100.0 * self._in_route_out_route_distance / max(self._in_route_accum_meters[-1], 1e-6)
                if out_route_percentage > 30.0:
                    off_route = True
            self._in_route_current_index = closest_index

        return not off_route

    @staticmethod
    def _vehicle_frame_components(
        vector_world: np.ndarray,
        transform: carla.Transform,
    ) -> tuple[float, float]:
        forward = transform.get_forward_vector()
        right = transform.get_right_vector()
        long_comp = (
            float(vector_world[0]) * forward.x
            + float(vector_world[1]) * forward.y
            + float(vector_world[2]) * forward.z
        )
        lat_comp = (
            float(vector_world[0]) * right.x
            + float(vector_world[1]) * right.y
            + float(vector_world[2]) * right.z
        )
        return long_comp, lat_comp

    def _compute_comfort_metrics(self) -> Dict[str, Any]:
        ego = self._ego_actor()
        if ego is None:
            return {
                "violations": [],
                "factor": 1.0,
                "longitudinal_acceleration": 0.0,
                "lateral_acceleration": 0.0,
                "absolute_jerk": 0.0,
                "longitudinal_jerk": 0.0,
                "yaw_rate": 0.0,
                "yaw_acceleration": 0.0,
            }

        transform = ego.get_transform()
        accel = ego.get_acceleration()
        accel_world = np.array([float(accel.x), float(accel.y), float(accel.z)], dtype=np.float32)
        long_acc, lat_acc = self._vehicle_frame_components(accel_world, transform)

        yaw_rate = math.radians(float(ego.get_angular_velocity().z))
        dt = 1.0 / CARLA_FPS
        if self._prev_ego_accel_world is None:
            jerk_world = np.zeros(3, dtype=np.float32)
            yaw_accel = 0.0
        else:
            jerk_world = (accel_world - self._prev_ego_accel_world) / dt
            yaw_accel = (yaw_rate - self._prev_ego_yaw_rate) / dt
        long_jerk, _ = self._vehicle_frame_components(jerk_world, transform)
        abs_jerk = float(np.linalg.norm(jerk_world))

        self._prev_ego_accel_world = accel_world
        self._prev_ego_yaw_rate = yaw_rate

        violations: list[str] = []
        if long_acc < COMFORT_THRESHOLDS["longitudinal_acceleration_min"] or long_acc > COMFORT_THRESHOLDS["longitudinal_acceleration_max"]:
            violations.append("longitudinal_acceleration")
        if abs(lat_acc) > COMFORT_THRESHOLDS["lateral_acceleration_abs"]:
            violations.append("lateral_acceleration")
        if abs(abs_jerk) > COMFORT_THRESHOLDS["absolute_jerk_abs"]:
            violations.append("absolute_jerk")
        if abs(long_jerk) > COMFORT_THRESHOLDS["longitudinal_jerk_abs"]:
            violations.append("longitudinal_jerk")
        if abs(yaw_rate) > COMFORT_THRESHOLDS["yaw_rate_abs"]:
            violations.append("yaw_rate")
        if abs(yaw_accel) > COMFORT_THRESHOLDS["yaw_acceleration_abs"]:
            violations.append("yaw_acceleration")

        factor = 1.0 - 0.5 * (len(violations) / 6.0)
        return {
            "violations": violations,
            "factor": float(np.clip(factor, 0.0, 1.0)),
            "longitudinal_acceleration": float(long_acc),
            "lateral_acceleration": float(lat_acc),
            "absolute_jerk": float(abs_jerk),
            "longitudinal_jerk": float(long_jerk),
            "yaw_rate": float(yaw_rate),
            "yaw_acceleration": float(yaw_accel),
        }

    @staticmethod
    def _bbox_at_prediction_step(actor: carla.Actor, step_seconds: float) -> Optional[carla.BoundingBox]:
        try:
            transform = actor.get_transform()
            bbox = actor.bounding_box
            velocity = actor.get_velocity()
            angular_velocity = actor.get_angular_velocity()
            transform.location = carla.Location(
                x=float(transform.location.x + velocity.x * step_seconds),
                y=float(transform.location.y + velocity.y * step_seconds),
                z=float(transform.location.z + velocity.z * step_seconds),
            )
            transform.rotation = carla.Rotation(
                pitch=float(transform.rotation.pitch + angular_velocity.x * step_seconds),
                yaw=float(transform.rotation.yaw + angular_velocity.z * step_seconds),
                roll=float(transform.rotation.roll + angular_velocity.y * step_seconds),
            )
            world_bbox = carla.BoundingBox(transform.transform(bbox.location), bbox.extent)
            world_bbox.rotation = transform.rotation
            return world_bbox
        except Exception:
            return None

    @staticmethod
    def _dot_product(vec1, vec2) -> float:
        return float(vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z)

    @staticmethod
    def _cross_product(vec1, vec2):
        return carla.Vector3D(
            x=vec1.y * vec2.z - vec1.z * vec2.y,
            y=vec1.z * vec2.x - vec1.x * vec2.z,
            z=vec1.x * vec2.y - vec1.y * vec2.x,
        )

    @classmethod
    def _has_separating_plane(cls, rel_pos, plane_normal, obb1, obb2) -> bool:
        projection_distance = abs(cls._dot_product(rel_pos, plane_normal))
        obb1_projection = (
            abs(cls._dot_product(obb1.rotation.get_forward_vector() * obb1.extent.x, plane_normal))
            + abs(cls._dot_product(obb1.rotation.get_right_vector() * obb1.extent.y, plane_normal))
            + abs(cls._dot_product(obb1.rotation.get_up_vector() * obb1.extent.z, plane_normal))
        )
        obb2_projection = (
            abs(cls._dot_product(obb2.rotation.get_forward_vector() * obb2.extent.x, plane_normal))
            + abs(cls._dot_product(obb2.rotation.get_right_vector() * obb2.extent.y, plane_normal))
            + abs(cls._dot_product(obb2.rotation.get_up_vector() * obb2.extent.z, plane_normal))
        )
        return projection_distance > obb1_projection + obb2_projection

    @classmethod
    def _obb_intersects(cls, obb1, obb2) -> bool:
        rel_pos = obb2.location - obb1.location
        axes = [
            obb1.rotation.get_forward_vector(),
            obb1.rotation.get_right_vector(),
            obb1.rotation.get_up_vector(),
            obb2.rotation.get_forward_vector(),
            obb2.rotation.get_right_vector(),
            obb2.rotation.get_up_vector(),
        ]
        axes.extend(
            [
                cls._cross_product(a1, a2)
                for a1 in axes[:3]
                for a2 in axes[3:]
            ]
        )
        for axis in axes:
            if abs(axis.x) < 1e-6 and abs(axis.y) < 1e-6 and abs(axis.z) < 1e-6:
                continue
            if cls._has_separating_plane(rel_pos, axis, obb1, obb2):
                return False
        return True

    def _ttc_violation(self) -> bool:
        ego = self._ego_actor()
        if ego is None:
            return False
        try:
            actors = self.evaluator.world.get_actors()
        except Exception:
            return False
        horizon_steps = int(round(REWARD_TTC_FORECAST_SECONDS / REWARD_TTC_INTERVAL_SECONDS))
        relevant_actors = [
            actor
            for actor in actors
            if actor.id != ego.id and actor.is_alive and (
                "vehicle" in actor.type_id or "walker" in actor.type_id
            )
        ]
        for step_idx in range(1, horizon_steps + 1):
            t = step_idx * REWARD_TTC_INTERVAL_SECONDS
            ego_bbox = self._bbox_at_prediction_step(ego, t)
            if ego_bbox is None:
                return False
            for actor in relevant_actors:
                actor_bbox = self._bbox_at_prediction_step(actor, t)
                if actor_bbox is None:
                    continue
                if self._obb_intersects(ego_bbox, actor_bbox):
                    return True
        return False

    def _outside_lane_soft_violation(self, lane_metrics: Dict[str, Any]) -> bool:
        any_wp = lane_metrics.get("any_waypoint")
        driving_wp = lane_metrics.get("driving_waypoint")
        if any_wp is not None and getattr(any_wp, "lane_type", None) == carla.LaneType.Sidewalk:
            return True
        if driving_wp is None:
            return False
        if bool(getattr(driving_wp, "is_junction", False)):
            return False
        return abs(float(lane_metrics["heading_error_rad"])) > (0.5 * math.pi)

    def _offroad_terminal_violation(self, lane_metrics: Dict[str, Any], criteria: list[dict[str, Any]]) -> bool:
        if self._criterion_triggered(
            criteria,
            (("off", "road"), ("outside", "road"), ("outside", "drivable")),
        ):
            return True
        any_wp = lane_metrics.get("any_waypoint")
        if any_wp is None:
            return True
        lane_type = getattr(any_wp, "lane_type", None)
        return lane_type == carla.LaneType.NONE

    def _soft_penalty_state(
        self,
        *,
        lane_metrics: Dict[str, Any],
        speed_mps: float,
    ) -> Dict[str, Any]:
        lane_width_m = float(lane_metrics["lane_width_m"])
        lane_half_width = max(0.5 * lane_width_m, 1e-3)
        lane_center_factor = 1.0
        if not bool(lane_metrics.get("is_junction", False)):
            lane_center_factor = float(
                np.clip(1.0 - abs(float(lane_metrics["lane_offset_m"])) / lane_half_width, 0.0, 1.0)
            )

        outside_lanes = self._outside_lane_soft_violation(lane_metrics)
        outside_lanes_factor = 0.0 if outside_lanes else 1.0

        speed_limit_mps = max(float(lane_metrics["speed_limit_mps"]), 1e-3)
        if self._speeding_infraction:
            overspeed_kmh = max(0.0, (speed_mps - speed_limit_mps) * 3.6)
            speeding_factor = float(np.clip(1.0 - overspeed_kmh / REWARD_SPEEDING_MARGIN_KMH, 0.0, 1.0))
        else:
            overspeed_kmh = 0.0
            speeding_factor = 1.0

        ttc_violated_now = self._ttc_violation()
        if ttc_violated_now:
            self._ttc_penalty_ticks = REWARD_TTC_PERSIST_STEPS
        ttc_active = self._ttc_penalty_ticks > 0
        ttc_factor = 0.5 if ttc_active else 1.0
        if self._ttc_penalty_ticks > 0:
            self._ttc_penalty_ticks -= 1

        comfort_metrics = self._compute_comfort_metrics()
        if self._comfort_infraction:
            comfort_violated_now = len(comfort_metrics["violations"]) > 0
            if comfort_violated_now:
                self._comfort_penalty_ticks = REWARD_COMFORT_PERSIST_STEPS
                self._comfort_penalty_factor = float(comfort_metrics["factor"])
            comfort_active = self._comfort_penalty_ticks > 0
            comfort_factor = self._comfort_penalty_factor if comfort_active else 1.0
            if self._comfort_penalty_ticks > 0:
                self._comfort_penalty_ticks -= 1
        else:
            comfort_violated_now = False
            comfort_factor = 1.0

        penalty_product = (
            outside_lanes_factor
            * lane_center_factor
            * speeding_factor
            * ttc_factor
            * comfort_factor
        )
        return {
            "outside_lanes": outside_lanes,
            "outside_lanes_factor": float(outside_lanes_factor),
            "lane_center_factor": float(lane_center_factor),
            "speeding_factor": float(speeding_factor),
            "overspeed_kmh": float(overspeed_kmh),
            "ttc_violated_now": bool(ttc_violated_now),
            "ttc_factor": float(ttc_factor),
            "comfort_violated_now": bool(comfort_violated_now),
            "comfort_factor": float(comfort_factor),
            "penalty_product": float(penalty_product),
            "comfort_metrics": comfort_metrics,
        }

    def _terminal_state(
        self,
        *,
        criteria: list[dict[str, Any]],
        lane_metrics: Dict[str, Any],
        speed_mps: float,
        route_completion: float,
        base_terminated: bool,
    ) -> Dict[str, Any]:
        collision = self._collision_count() > 0
        offroad = self._offroad_terminal_violation(lane_metrics, criteria)
        run_red_light = self._criterion_triggered(
            criteria,
            (("red", "light"), ("running", "red"), ("run", "red"), ("traffic", "light")),
        )
        run_stop_sign = self._criterion_triggered(
            criteria,
            (("stop", "sign"), ("running", "stop"), ("run", "stop")),
        )
        route_deviation_distance = self._route_deviation_distance()
        left_route = self._use_leave_route_done and (self._closest_route_lateral_distance() > self._min_thresh_lat_dist)
        in_route_ok = self._in_route_ok()
        route_deviation = bool(left_route or (not in_route_ok))
        if speed_mps < REWARD_BLOCKED_SPEED_THRESHOLD_MPS:
            self._blocked_ticks += 1
        else:
            self._blocked_ticks = 0
        blocked = self._blocked_ticks >= self._blocked_steps

        termination_reason = ""
        route_completed = route_completion >= 99.9
        for name, triggered in (
            ("collision", collision),
            ("off_road", offroad),
            ("run_red_light", run_red_light),
            ("run_stop_sign", run_stop_sign),
            ("route_deviation", route_deviation),
            ("blocked", blocked),
        ):
            if triggered:
                termination_reason = name
                break

        hard_infraction = bool(termination_reason)
        success = bool(route_completed and not hard_infraction)
        terminated = bool(base_terminated or hard_infraction or route_completed)
        if terminated and not termination_reason and success:
            termination_reason = "success"
        elif terminated and not termination_reason:
            termination_reason = "scenario_end"

        return {
            "terminated": terminated,
            "success": bool(success),
            "termination_reason": termination_reason,
            "terminal_reward": float(self._terminal_reward if termination_reason and termination_reason != "success" else 0.0),
            "collision": bool(collision),
            "off_road": bool(offroad),
            "run_red_light": bool(run_red_light),
            "run_stop_sign": bool(run_stop_sign),
            "route_deviation": bool(route_deviation),
            "route_deviation_distance_m": float(route_deviation_distance),
            "blocked": bool(blocked),
            "left_route": bool(left_route),
            "in_route_ok": bool(in_route_ok),
            "route_completed": bool(route_completed),
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
