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
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import math
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

from srunner.scenariomanager.scenarioatomics.atomic_criteria import MinimumSpeedRouteTest
from srunner.scenariomanager.traffic_events import TrafficEvent, TrafficEventType

from ogbench.carla.route_registry import RouteEntry, find_route
from ogbench.carla.leaderboard_agents.observation_only import (
    IMAGE_SHAPE_HWC,
    RGB_FRONT_CAMERA_TAG,
)
from ogbench.carla.expert_label import ExpertLabelComputer, NUM_COMMENTARY_WORDS, commentary_to_bow


def _patch_minimum_speed_route_test() -> None:
    """Fix Bench2Drive's min-speed criterion so >100% does not count as an infraction."""
    if getattr(MinimumSpeedRouteTest, "_ogbench_patched", False):
        return

    def _patched_set_traffic_event(self):
        if self._speed_points > 0 and self._mean_speed:
            self._mean_speed /= self._speed_points
            self._actor_speed /= self._speed_points
            checkpoint_value = round(self._actor_speed / (self.RATIO * self._mean_speed) * 100, 2)
        else:
            checkpoint_value = 100

        self._checkpoint_values.append(checkpoint_value)
        if checkpoint_value >= self.success_value:
            if self.test_status in ("INIT", "RUNNING"):
                self.test_status = "SUCCESS"
            return

        self.test_status = "FAILURE"
        self._traffic_event = TrafficEvent(TrafficEventType.MIN_SPEED_INFRACTION, GameTime.get_frame())
        self._traffic_event.set_dict({"percentage": checkpoint_value})
        self._traffic_event.set_message(
            f"Average speed is {checkpoint_value}% of the surrounding traffic's one"
        )
        self.events.append(self._traffic_event)

    MinimumSpeedRouteTest._set_traffic_event = _patched_set_traffic_event
    MinimumSpeedRouteTest._ogbench_patched = True


_patch_minimum_speed_route_test()


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
DEFAULT_OUTSIDE_ROUTE_PROGRESS_PENALTY = -1.0
SUCCESS_BONUS = 5.0
FAILURE_BONUS = -5.0
DEFAULT_SPEED_REWARD_SCALE = 0.01
DEFAULT_ROUTE_PROGRESS_REWARD_SCALE = 0.1


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
        streaming_port=int(cfg.get("streaming_port", 0)),
        x_display_num=int(cfg.get("x_display_num", 10 + (int(cfg.get("port", 2000)) % 90))),
        use_cuda_visible_devices=bool(cfg.get("use_cuda_visible_devices", False)),
        server_boot_wait_seconds=float(cfg.get("server_boot_wait_seconds", 60.0)),
    )


def _find_free_port(starting_port: int) -> int:
    """Return the first free localhost TCP port at or above ``starting_port``."""
    port = int(starting_port)
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            port += 1


def run_leaderboard(args: SimpleNamespace) -> bool:
    """Run the full leaderboard loop (all routes selected by ``args``).

    Returns ``True`` if the run crashed fatally.
    """
    statistics_manager = StatisticsManager(args.checkpoint, args.debug_checkpoint)
    
    prev_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if getattr(args, "use_cuda_visible_devices", False):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_rank)
    leaderboard_evaluator = LeaderboardEvaluator(args, statistics_manager)
    if getattr(args, "use_cuda_visible_devices", False):
        if prev_cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible_devices
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

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
        self._outside_route_progress_penalty = float(
            self.carla_config.get("outside_route_progress_penalty", DEFAULT_OUTSIDE_ROUTE_PROGRESS_PENALTY)
        )
        self._speed_reward_scale = float(
            self.carla_config.get("speed_reward_scale", DEFAULT_SPEED_REWARD_SCALE)
        )
        self._route_progress_reward_scale = float(
            self.carla_config.get("route_progress_reward_scale", DEFAULT_ROUTE_PROGRESS_REWARD_SCALE)
        )
        self._prev_collision_count = 0
        self._prev_outside_route_value = 0.0
        self._prev_min_speed_value = 0.0
        self._prev_route_completion_value = 0.0

        self.observation_space = gymnasium.spaces.Dict(
            {
                "state": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
                ),
                "image": gymnasium.spaces.Box(
                    low=0, high=255, shape=IMAGE_SHAPE_HWC, dtype=np.uint8
                ),
                "language_label": gymnasium.spaces.Box(
                    low=0.0, high=1.0, shape=(NUM_COMMENTARY_WORDS,), dtype=np.float32
                ),
            }
        )
        self._label_computer = ExpertLabelComputer()
        self._cached_world_map = None  # populated after first CARLA tick
        self._expert_controller_kind = str(self.carla_config.get("expert_controller", "") or "").strip().lower()
        self._expert_agent: Any | None = None
        self._steervla_exec_cfg: Optional[Dict[str, Any]] = None
        self._steervla_decoder: Any | None = None
        self._xvfb_proc: subprocess.Popen[str] | None = None
        self._x_display_num: int | None = None
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
    def _cleanup_stale_x_display_lock(display_num: int) -> None:
        lock_path = f"/tmp/.X{display_num}-lock"
        if not os.path.exists(lock_path):
            return
        try:
            with open(lock_path) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError, ValueError, OSError):
            try:
                os.remove(lock_path)
            except OSError:
                pass

    def _ensure_xvfb(self, display_num: int) -> None:
        self._cleanup_stale_x_display_lock(display_num)
        lock_path = f"/tmp/.X{display_num}-lock"
        if os.path.exists(lock_path):
            raise RuntimeError(
                f"X display :{display_num} is already in use. Pick a different "
                "carla_config.x_display_num / --x-display-num for concurrent runs."
            )
        cmd = [
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
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        time.sleep(1.0)
        if proc.poll() is not None:
            raise RuntimeError(f"Failed to start Xvfb on :{display_num}; exit code {proc.returncode}")
        self._xvfb_proc = proc
        self._x_display_num = display_num

    def _stop_xvfb(self) -> None:
        proc = self._xvfb_proc
        self._xvfb_proc = None
        display_num = self._x_display_num
        self._x_display_num = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        if display_num is not None:
            self._cleanup_stale_x_display_lock(display_num)

    @staticmethod
    def _patch_leaderboard_launch() -> None:
        if getattr(LeaderboardEvaluator, "_ogbench_isolated_launch_patch", False):
            return

        def _setup_simulation_isolated(ev_self, args):
            ev_self.carla_path = os.environ["CARLA_ROOT"]
            args.port = _find_free_port(args.port)
            streaming_port = int(getattr(args, "streaming_port", 0) or 0)
            if streaming_port <= 0:
                streaming_port = _find_free_port(args.port + 1)

            display_num = int(getattr(args, "x_display_num", 10 + (args.port % 90)))
            env = os.environ.copy()
            env["DISPLAY"] = f":{display_num}"

            cmd_parts = [
                os.path.join(ev_self.carla_path, "CarlaUE4.sh"),
                "-RenderOffScreen",
                "-nosound",
                f"-carla-rpc-port={args.port}",
                f"-carla-streaming-port={streaming_port}",
                f"-graphicsadapter={args.gpu_rank}",
            ]
            cmd1 = " ".join(cmd_parts)
            ev_self.server = subprocess.Popen(
                cmd1,
                shell=True,
                preexec_fn=os.setsid,
                env=env,
            )
            print(f"{cmd1} DISPLAY={env['DISPLAY']}", ev_self.server.returncode, flush=True)
            import atexit

            atexit.register(os.killpg, ev_self.server.pid, signal.SIGKILL)
            time.sleep(float(getattr(args, "server_boot_wait_seconds", 60.0)))

            attempts = 0
            num_max_restarts = 20
            while attempts < num_max_restarts:
                try:
                    client = carla.Client(args.host, args.port)
                    client_timeout = args.timeout if args.timeout else 600.0
                    client.set_timeout(client_timeout)

                    settings = carla.WorldSettings(
                        synchronous_mode=True,
                        fixed_delta_seconds=1.0 / ev_self.frame_rate,
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
            attempts = 0
            num_max_restarts = 40
            while attempts < num_max_restarts:
                try:
                    args.traffic_manager_port = _find_free_port(args.traffic_manager_port)
                    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
                    traffic_manager.set_synchronous_mode(True)
                    traffic_manager.set_hybrid_physics_mode(True)
                    print(f"traffic_manager init success, try_time={attempts}", flush=True)
                    break
                except Exception as e:
                    print(f"traffic_manager init fail, try_time={attempts}", flush=True)
                    print(e, flush=True)
                    attempts += 1
                    time.sleep(5)
            return client, client_timeout, traffic_manager

        LeaderboardEvaluator._setup_simulation = _setup_simulation_isolated
        LeaderboardEvaluator._ogbench_isolated_launch_patch = True

    def setup(self) -> None:
        """Instantiate ``LeaderboardEvaluator`` and swap in :class:`SteppableScenarioManager`."""
        self._patch_leaderboard_launch()
        self._args = carla_config_to_args(self.carla_config, self.route_entry)
        self._ensure_xvfb(int(self._args.x_display_num))
        self._base_agent_config = self._args.agent_config
        statistics_manager = StatisticsManager(
            self._args.checkpoint, self._args.debug_checkpoint
        )
        prev_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if getattr(self._args, "use_cuda_visible_devices", False):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._args.gpu_rank)
        # Force NVIDIA-only Vulkan ICD so -graphicsadapter=N maps to physical GPU N.
        # Without this, llvmpipe and other ICDs shift the Vulkan device indices, causing
        # UE4's render thread to select the wrong GPU or fail to initialize.
        _NVIDIA_VK_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"
        prev_vk_icd = os.environ.get("VK_ICD_FILENAMES")
        os.environ["VK_ICD_FILENAMES"] = _NVIDIA_VK_ICD
        try:
            self._evaluator = LeaderboardEvaluator(self._args, statistics_manager)
        except Exception:
            self._stop_xvfb()
            raise
        finally:
            if getattr(self._args, "use_cuda_visible_devices", False):
                if prev_cuda_visible_devices is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible_devices
                else:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
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
        except Exception:
            print("\033[91mFailed to stop scenario in wrapper:\033[0m", flush=True)
            print(traceback.format_exc(), flush=True)
        self._drain_pseudo_sensors()
        self._evaluator._cleanup()
        self._scenario_active = False
        # Do not set _needs_setup_on_reset here: we reuse the existing CARLA
        # client/world across episodes to avoid spawning a new server (subprocess)
        # while JAX threads are running, which triggers fork() and crashes.

    def _build_simlingo_autopilot(self):
        """Instantiate SimLingo's privileged autopilot as the expert controller."""
        try:
            from impls.coaches.simlingo import AutoPilot

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
        self._scenario_active = True
        self._last_control = carla.VehicleControl()
        self._crash_stuck_ticks = 0
        self._prev_collision_count = 0
        self._prev_outside_route_value = 0.0
        self._prev_min_speed_value = 0.0
        self._prev_route_completion_value = 0.0
        self._cached_world_map = None  # refresh map cache each episode

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

    def _compute_language_label(self, expert_action: Optional[np.ndarray] = None) -> tuple[str, np.ndarray]:
        """Query expert commentary from live CARLA state.

        Returns ``(commentary_text, bow_vector)``. Falls back to empty string
        and zero vector on any error.
        """
        _zero = np.zeros(NUM_COMMENTARY_WORDS, dtype=np.float32)
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
                ego,
                world,
                self._cached_world_map,
                route,
                scenario_type=scenario_type,
                expert_action=expert_action,
            )
            return text, bow
        except Exception:
            return "", _zero

    def _compute_expert_action(self, action_horizon: int = 10, action_dim: int = 4) -> np.ndarray:
        """PDM-Lite expert action approximated from route waypoints + target speed.

        Matches the DELTA_XY_T_DELTA_XY_SPACE layout used by SteerVLA:
        per step ``[dx_speed, dy_speed, dx_route, dy_route]`` where x is
        forward and y is left in ego frame (simlingo/transfuser convention).

        Returns a flat float32 array of shape ``(action_horizon * action_dim,)``.
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

            agent_instance = getattr(ev, "agent_instance", None)
            scenario_route = getattr(getattr(ev, "route_scenario", None), "route", None) or []
            route_dense = (
                getattr(agent_instance, "org_dense_route_world_coord", None)
                or scenario_route
                or []
            )
            route_sparse = getattr(agent_instance, "_global_plan_world_coord", None) or []
            route = route_dense or route_sparse

            # Target speed from expert label computer. Parking-style routes are a special case:
            # the generic "vehicle ahead" heuristic often latches onto the cut-in obstacle and
            # freezes the expert at 0 m/s even though the intended behavior is to keep exiting
            # the parking area along the planned route.
            scenario_type = str(getattr(self.route_entry, "scenario_type", "") or "")
            if "parking" in scenario_type.lower():
                try:
                    speed_limit = max(ego.get_speed_limit() / 3.6, 1.0)
                except Exception:
                    speed_limit = 8.33
                target_speed = float(np.clip(speed_limit * 0.5, 3.0, 5.0))
            else:
                try:
                    _, speed_ctx = self._label_computer._speed_info(ego, world, self._cached_world_map)
                    target_speed = float(speed_ctx.get("target_speed", 0.0))
                except Exception as _se:
                    print(f"[expert_action] _speed_info failed: {_se}", flush=True)
                    target_speed = 0.0

            # Ego frame basis: x = forward, y = left (negative right)
            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            fwd = ego_tf.get_forward_vector()
            right = ego_tf.get_right_vector()

            def _to_ego(loc):
                rel = loc - ego_loc
                x_e = rel.x * fwd.x + rel.y * fwd.y
                y_e = -(rel.x * right.x + rel.y * right.y)
                return x_e, y_e

            # Match SimLingo / SteerVLA control decoding timebase:
            # action rows are interpreted at policy_fps = carla_fps / data_save_freq = 20 / 5 = 4 Hz.
            dt = 5.0 / 20.0
            chunk = np.zeros((action_horizon, 4), dtype=np.float32)
            route_debug_source = "dense" if route_dense else ("sparse" if route_sparse else "fallback")

            def _build_chunk_from_route(route_seq) -> tuple[np.ndarray, int]:
                route_locs = [tf.location for tf, _ in route_seq if tf is not None]
                if len(route_locs) < 2:
                    return np.zeros((action_horizon, 4), dtype=np.float32), 0

                nearest_idx = min(
                    range(len(route_locs)),
                    key=lambda i: route_locs[i].distance(ego_loc),
                )
                future_locs = route_locs[nearest_idx:]
                if len(future_locs) < 2:
                    return np.zeros((action_horizon, 4), dtype=np.float32), 0

                cum_d = [0.0]
                xy_ego: list[tuple[float, float]] = []
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
                fallback_route: list[tuple[Any, Any]] = []
                if curr_wp is not None:
                    fallback_route.append((curr_wp.transform, None))
                    for _ in range(action_horizon + 20):
                        nxt = curr_wp.next(1.0)
                        if not nxt:
                            break
                        curr_wp = nxt[0]
                        fallback_route.append((curr_wp.transform, None))
                chunk, route_pts = _build_chunk_from_route(fallback_route)
                route_debug_source = "road_fallback"

            print(
                f"[expert_action] route_src={route_debug_source} route_pts={route_pts} "
                f"route_total={len(route)} target_speed={target_speed:.2f}",
                flush=True,
            )

            if route_pts < 2:
                dx = target_speed * dt
                chunk[:, 0] = dx
                chunk[:, 2] = dx

            return chunk.flatten()

        except Exception as _e:
            print(f"[expert_action] exception: {_e}", flush=True)
            return out

    def step_expert(self, obs_raw: dict):
        """Step the env using the PDM-Lite expert action from ``obs_raw``.

        The ``expert_action`` array is in physical units (metres/step) and is
        passed directly to the waypoint decoder as ``policy_output`` space,
        bypassing the OpenPI normalisation layer used for VLA actions.
        """
        if self._expert_agent is not None:
            sensors = self._get_expert_input_data()
            try:
                control = self._expert_agent.run_step(sensors, GameTime.get_time())
                return self._step_with_control(control)
            except Exception as _ee:
                print(f"[step_expert] SimLingo autopilot failed, falling back to expert_action: {_ee}", flush=True)

        expert_action = obs_raw.get("expert_action")
        if expert_action is None:
            expert_action = np.zeros(self.action_space.shape, dtype=np.float32)
        if self._steervla_exec_cfg is not None:
            control = None
            if self._steervla_decoder is not None:
                try:
                    from ogbench.carla.steervla_simlingo_control import maybe_steervla_vehicle_control

                    control = maybe_steervla_vehicle_control(
                        np.asarray(expert_action, dtype=np.float32),
                        state_vec=self._get_state_vector(),
                        exec_cfg=self._steervla_exec_cfg,
                        decoder=self._steervla_decoder,
                    )
                except Exception as _ce:
                    print(f"[step_expert] control decode failed: {_ce}", flush=True)
            if control is not None:
                print(
                    f"[step_expert] steer={control.steer:.3f} throttle={control.throttle:.3f} brake={float(control.brake):.3f}",
                    flush=True,
                )
            orig_space = self._steervla_exec_cfg.get("action_input_space", "normalized")
            self._steervla_exec_cfg["action_input_space"] = "policy_output"
            try:
                return self.step(np.asarray(expert_action, dtype=np.float32))
            finally:
                self._steervla_exec_cfg["action_input_space"] = orig_space
        return self.step(np.asarray(expert_action, dtype=np.float32))

    def _get_expert_input_data(self) -> Dict[str, Any]:
        """Build a minimal leaderboard-style sensor dict for local expert controllers."""
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
        gps = np.array(
            [
                float(transform.location.x),
                float(transform.location.y),
                float(transform.location.z),
            ],
            dtype=np.float32,
        )
        # SimLingo expects the raw IMU compass value; its helper converts this to
        # CARLA yaw by subtracting 90 degrees.
        compass = float(np.deg2rad(transform.rotation.yaw + 90.0))
        imu = np.array(
            [
                float(accel.x),
                float(accel.y),
                float(accel.z),
                float(ang_vel.x),
                float(ang_vel.y),
                float(ang_vel.z),
                compass,
            ],
            dtype=np.float32,
        )
        sensors.setdefault("imu", (ts, imu))
        sensors.setdefault("gps", (ts, gps))
        sensors.setdefault("speed", (ts, {"speed": speed}))
        return sensors

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

    def _route_completion_value(self) -> float:
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
        return self._step_with_control(control)

    def _step_with_control(
        self, control: carla.VehicleControl
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
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
        # Reward: route progress + speed shaping + bounded/event-style infractions.
        ego = self._ego_actor()
        speed = 0.0
        if ego is not None:
            v = ego.get_velocity()
            speed = float(np.linalg.norm([v.x, v.y, v.z]))
        route_completion_value = self._route_completion_value()
        route_completion_delta = max(0.0, route_completion_value - self._prev_route_completion_value)
        reward_speed = self._speed_reward_scale * speed
        reward_route_progress = self._route_progress_reward_scale * route_completion_delta
        reward = reward_speed + reward_route_progress
        info = self._info_with_sensors(
            {"scenario_tree_status": getattr(tree_status, "name", str(tree_status))}
        )
        crash_stuck, collision_count = self._update_crash_stuck_state(speed)
        outside_route_value, min_speed_value = self._route_infraction_values()

        collision_delta = max(0, collision_count - self._prev_collision_count)
        outside_route_delta = max(0.0, outside_route_value - self._prev_outside_route_value)
        outside_route_onset = float(self._prev_outside_route_value <= 0.0 and outside_route_value > 0.0)
        self._prev_collision_count = collision_count
        self._prev_outside_route_value = outside_route_value
        self._prev_min_speed_value = min_speed_value
        self._prev_route_completion_value = route_completion_value

        collision_pen = self._collision_event_penalty * float(collision_delta)
        outside_route_pen = self._outside_route_event_penalty * outside_route_onset
        outside_route_progress_pen = self._outside_route_progress_penalty * float(outside_route_delta > 0.0)
        reward += collision_pen + outside_route_pen + outside_route_progress_pen

        info["collision_count"] = collision_count
        info["crash_stuck_ticks"] = self._crash_stuck_ticks
        info["route_completion_value"] = route_completion_value
        info["route_completion_delta"] = float(route_completion_delta)
        info["outside_route_value"] = outside_route_value
        info["min_speed_value"] = min_speed_value
        info["collision_delta"] = float(collision_delta)
        info["outside_route_delta"] = float(outside_route_delta)
        info["outside_route_onset"] = outside_route_onset
        info["penalty_collision"] = collision_pen
        info["penalty_outside_route"] = outside_route_pen
        info["penalty_outside_route_progress"] = outside_route_progress_pen
        info["reward_speed"] = reward_speed
        info["reward_route_progress"] = reward_route_progress
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
        self._stop_xvfb()
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
