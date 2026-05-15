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

import os
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
DEFAULT_MIN_SPEED_EVENT_PENALTY = -2.0
SUCCESS_BONUS = 5.0
FAILURE_BONUS = -5.0


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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pending_control: Optional[carla.VehicleControl] = None
        self.last_agent_input: Dict[str, Any] = {}

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

    # The following is a near-verbatim copy of the upstream _tick_scenario, with two changes:
    #   1. We always apply ``self.pending_control`` (if set) instead of the agent's output.
    #   2. We stash the agent's last input_data on ``self.last_agent_input`` for the gym wrapper.
    def _tick_scenario(self) -> None:
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
    )


def run_leaderboard(args: SimpleNamespace) -> bool:
    """Run the full leaderboard loop (all routes selected by ``args``).

    Returns ``True`` if the run crashed fatally.
    """
    statistics_manager = StatisticsManager(args.checkpoint, args.debug_checkpoint)
    
    prev_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_rank)
    leaderboard_evaluator = LeaderboardEvaluator(args, statistics_manager)
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
        # Rebuild evaluator after any `_cleanup()` to avoid partially-cleaned
        # leaderboard state carrying across episode resets (can drop traffic actors).
        self._needs_setup_on_reset = False
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
        self._min_speed_event_penalty = float(
            self.carla_config.get("min_speed_event_penalty", DEFAULT_MIN_SPEED_EVENT_PENALTY)
        )
        self._prev_collision_count = 0
        self._prev_outside_route_value = 0.0
        self._prev_min_speed_value = 0.0

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

    def setup(self) -> None:
        """Instantiate ``LeaderboardEvaluator`` and swap in :class:`SteppableScenarioManager`."""
        self._args = carla_config_to_args(self.carla_config, self.route_entry)
        self._base_agent_config = self._args.agent_config
        statistics_manager = StatisticsManager(
            self._args.checkpoint, self._args.debug_checkpoint
        )
        prev_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._args.gpu_rank)
        self._evaluator = LeaderboardEvaluator(self._args, statistics_manager)
        if prev_cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible_devices
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        self._evaluator.manager = SteppableScenarioManager(
            self._args.timeout, statistics_manager, self._args.debug
        )

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

    def _stop_active_scenario(self) -> None:
        if not self._scenario_active or self._evaluator is None:
            return
        try:
            print("\033[1m> Stopping the route (wrapper)\033[0m", flush=True)
            self._evaluator.manager.stop_scenario()
        except Exception:
            print("\033[91mFailed to stop scenario in wrapper:\033[0m", flush=True)
            print(traceback.format_exc(), flush=True)
        self._evaluator._cleanup()
        self._scenario_active = False
        self._needs_setup_on_reset = True

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
        self._scenario_active = True
        self._last_control = carla.VehicleControl()
        self._crash_stuck_ticks = 0
        self._prev_collision_count = 0
        self._prev_outside_route_value = 0.0
        self._prev_min_speed_value = 0.0

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

    def _obs_dict(self) -> Dict[str, np.ndarray]:
        sensors = getattr(self.evaluator.manager, "last_agent_input", None) or {}
        return {
            "state": self._get_state_vector(),
            "image": rgb_front_from_leaderboard_dict(sensors),
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
        if self._evaluator is None or self._needs_setup_on_reset:
            self.setup()
            self._needs_setup_on_reset = False

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
            self.evaluator._cleanup()
            raise RuntimeError(f"Invalid sensors: {e}") from e
        except Exception:
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
        # Reward: speed shaping + explicit infraction penalties.
        ego = self._ego_actor()
        speed = 0.0
        if ego is not None:
            v = ego.get_velocity()
            speed = float(np.linalg.norm([v.x, v.y, v.z]))
        reward = 0.01 * speed
        info = self._info_with_sensors(
            {"scenario_tree_status": getattr(tree_status, "name", str(tree_status))}
        )
        crash_stuck, collision_count = self._update_crash_stuck_state(speed)
        outside_route_value, min_speed_value = self._route_infraction_values()

        collision_delta = max(0, collision_count - self._prev_collision_count)
        outside_route_delta = max(0.0, outside_route_value - self._prev_outside_route_value)
        min_speed_delta = max(0.0, min_speed_value - self._prev_min_speed_value)
        self._prev_collision_count = collision_count
        self._prev_outside_route_value = outside_route_value
        self._prev_min_speed_value = min_speed_value

        collision_pen = self._collision_event_penalty * float(collision_delta)
        outside_route_pen = self._outside_route_event_penalty * float(outside_route_delta)
        min_speed_pen = self._min_speed_event_penalty * float(min_speed_delta)
        reward += collision_pen + outside_route_pen + min_speed_pen

        info["collision_count"] = collision_count
        info["crash_stuck_ticks"] = self._crash_stuck_ticks
        info["outside_route_value"] = outside_route_value
        info["min_speed_value"] = min_speed_value
        info["collision_delta"] = float(collision_delta)
        info["outside_route_delta"] = float(outside_route_delta)
        info["min_speed_delta"] = float(min_speed_delta)
        info["penalty_collision"] = collision_pen
        info["penalty_outside_route"] = outside_route_pen
        info["penalty_min_speed"] = min_speed_pen
        if crash_stuck:
            terminated = True
            reward += self._crash_stuck_penalty
            info["success"] = False
            info["termination_reason"] = "crash_stuck"
        if terminated:
            if crash_stuck:
                self._finalize_route("Finished", "Agent crashed and got stuck")
            else:
                success = info["scenario_tree_status"] == "SUCCESS"
                reward += SUCCESS_BONUS if success else FAILURE_BONUS
                info["success"] = success
                self._finalize_route("Finished", "")
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
            try:
                self.evaluator._cleanup()
            except Exception:
                pass
            self._scenario_active = False
            self._needs_setup_on_reset = True

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
