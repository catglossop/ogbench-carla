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
import weakref
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


def warn_if_carla_root_mismatched(route_source: str, cfg: Dict[str, Any]) -> None:
    """Log a clear warning if the active CARLA install can't serve this route's assets.

    Fail2Drive routes reference static props (``brickwall``, ``walkingkid``,
    ``ampel`` etc.) that vanilla CARLA 0.9.16 doesn't ship — those scenarios
    silently fail to spawn the intended obstacle. ``run_simlingo_fail2drive.sh``
    switches ``CARLA_ROOT`` automatically; this catches the case where someone
    invoked the script directly without doing that.
    """
    if route_source != "fail2drive":
        return
    f2d_root = cfg.get("fail2drive_carla_root")
    if not f2d_root:
        return
    current_root = os.environ.get("CARLA_ROOT", "")
    current_api = os.environ.get("CARLA_PYTHON_API_ROOT", "")
    f2d_root_resolved = str(Path(str(f2d_root)).expanduser().resolve())
    if (
        current_root
        and Path(current_root).resolve() != Path(f2d_root_resolved)
        and not current_api.startswith(f2d_root_resolved)
    ):
        print(
            f"\033[93m[fail2drive] WARNING: route is from Fail2Drive but CARLA_ROOT="
            f"{current_root!r} is not the Fail2Drive install ({f2d_root_resolved!r}). "
            f"Assets like static.prop.brickwall / walkingkid may be missing — set "
            f"CARLA_ROOT={f2d_root_resolved} (and relaunch the CARLA server from there).\033[0m"
        )



def warn_if_carla_root_mismatched(route_source: str, cfg: Dict[str, Any]) -> None:
    """Log a clear warning if the active CARLA install can't serve this route's assets.

    Fail2Drive routes reference static props (``brickwall``, ``walkingkid``,
    ``ampel`` etc.) that vanilla CARLA 0.9.16 doesn't ship — those scenarios
    silently fail to spawn the intended obstacle. ``run_simlingo_fail2drive.sh``
    switches ``CARLA_ROOT`` automatically; this catches the case where someone
    invoked the script directly without doing that.
    """
    if route_source != "fail2drive":
        return
    f2d_root = cfg.get("fail2drive_carla_root")
    if not f2d_root:
        return
    current_root = os.environ.get("CARLA_ROOT", "")
    current_api = os.environ.get("CARLA_PYTHON_API_ROOT", "")
    f2d_root_resolved = str(Path(str(f2d_root)).expanduser().resolve())
    if (
        current_root
        and Path(current_root).resolve() != Path(f2d_root_resolved)
        and not current_api.startswith(f2d_root_resolved)
    ):
        print(
            f"\033[93m[fail2drive] WARNING: route is from Fail2Drive but CARLA_ROOT="
            f"{current_root!r} is not the Fail2Drive install ({f2d_root_resolved!r}). "
            f"Assets like static.prop.brickwall / walkingkid may be missing — set "
            f"CARLA_ROOT={f2d_root_resolved} (and relaunch the CARLA server from there).\033[0m"
        )

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
from leaderboard.envs.sensor_interface import GenericMeasurement

from leaderboard.leaderboard_evaluator import (
    LeaderboardEvaluator,
    get_weather_id,
    sensors_to_icons,
)

# Register fail2drive scenario classes (ImageOnObject / ObscuredStopSign /
# RoadBlocked / etc.) with the leaderboard's discovery now that srunner +
# leaderboard are importable. Faithful to the fail2drive source — see
# ``ogbench/carla/fail2drive_compat.py``. No-op if the fail2drive package
# isn't installed.
from ogbench.carla.fail2drive_compat import apply as _apply_fail2drive_compat

_apply_fail2drive_compat()

from ogbench.carla.route_registry import RouteEntry, find_route
from ogbench.carla.leaderboard_agents.observation_only import (
    IMAGE_SHAPE_HWC,
    RGB_FRONT_CAMERA_TAG,
    VIZ_IMAGE_SHAPE_HWC,
)
from ogbench.carla.leaderboard_agents.simlingo_obs import (
    SIMLINGO_CAMERA_TAG,
    SIMLINGO_IMAGE_SHAPE_HWC,
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


def format_routing_command(
    command_id: int,
    next_command_id: int,
    *,
    dist_m: int,
    include_distance: bool,
) -> str:
    """Render the routing instruction in SimLingo's training grammar.

    Ported verbatim from simlingo's ``eval_route_as == 'command'`` branch (kept in this repo at
    ``ogbench/carla/carla.py``), because that is what produced the ``routing_command`` strings the
    SteerVLA checkpoint was fine-tuned on. Two forms, and the trailing clause is the part this env
    used to drop:

        follow the road.
        follow the road then go right at the next intersection.
        go right at the next intersection in 20 meter then follow the road.

    The follow-on clause is omitted when it would repeat the current command, which is why plain
    ``follow the road.`` dominates -- 888 of 1017 sampled training steps, against 128 that carry a
    ``then`` clause and 1 manoeuvre with a distance but no continuation.
    """
    command = ROUTING_COMMAND_TEXT.get(int(command_id), "follow the road")
    next_command = ROUTING_COMMAND_TEXT.get(int(next_command_id), "follow the road")
    suffix = "" if next_command == command else f" then {next_command}"
    if include_distance:
        return f"{command} in {int(dist_m)} meter{suffix}."
    return f"{command}{suffix}."


# Pre-computed per-town speed-limit lookup tables (simlingo, km/h, CARLA world coords).
_SPEED_LIMITS_DIR = Path(__file__).resolve().parent.parent.parent / "impls" / "coaches" / "simlingo" / "speed_limits"

DEFAULT_CRASH_STUCK_SPEED_THRESHOLD = 0.1
DEFAULT_CRASH_STUCK_STEPS = 20
DEFAULT_CRASH_STUCK_PENALTY = -20.0
DEFAULT_COLLISION_EVENT_PENALTY = -20.0  # legacy: applied while contact is active unless split below
DEFAULT_COLLISION_CONTACT_PENALTY = None  # float | None; if None, uses legacy collision_event_penalty
DEFAULT_OUTSIDE_ROUTE_EVENT_PENALTY = -20.0
DEFAULT_TRAFFIC_VIOLATION_PENALTY = -20.0  # per new RunningStop or RunningRedLight event
# Dense progress reward = weight * route_progress_delta / 100 (see _compute_reward_and_info).
# NOTE: routing-commands carried 10.0 while master rewrote the formula to divide by 100;
# those two changes merged without a conflict. Pinned back to 5.0 -- see ogbench/carla/README.md
# ("Reward settings") before changing, and re-check any critic checkpoint trained at another scale.
DEFAULT_PROGRESS_REWARD_WEIGHT = 5.0
DEFAULT_TERMINATE_ON_INFRACTION = False
DEFAULT_MAX_EPISODE_STEPS = 4000
# DEFAULT_CENTERING_REWARD_WEIGHT = 0.2
# DEFAULT_HEADING_REWARD_WEIGHT = 0.2
DEFAULT_STEER_PENALTY_WEIGHT = 0.0
DEFAULT_BRAKE_PENALTY_WEIGHT = 0.0
DEFAULT_SPEED_LIMIT_PENALTY_WEIGHT = 0.1
SUCCESS_BONUS = 50.0
FAILURE_BONUS = -20.0


def _find_free_port(starting_port: int, span: int = 1) -> int:
    """Find the start of ``span`` consecutive free localhost TCP ports from ``starting_port``.

    A CARLA server bound with ``-carla-rpc-port=N`` also listens on the streaming ports ``N+1``
    and ``N+2`` (there is no separate free-port search for those). Pass ``span=3`` for the rpc
    port so the whole block is reserved up front; otherwise two concurrent servers can pass the
    single-port check yet collide on streaming, crashing the second one at boot with
    ``bind: Address already in use``.
    """
    port = max(int(starting_port), 1)
    span = max(int(span), 1)
    while True:
        socks: list[socket.socket] = []
        try:
            for offset in range(span):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("localhost", port + offset))
                socks.append(s)
            return port
        except OSError:
            port += 1
        finally:
            for s in socks:
                s.close()


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


def _gpu_index_has_err(gpu_index: int) -> bool:
    """True when ``nvidia-smi -i <index>`` reports driver error state."""
    try:
        import subprocess as _sp

        out = _sp.run(
            ["nvidia-smi", "-i", str(int(gpu_index))],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
        return "ERR!" in out
    except Exception:
        return False


def _pick_healthy_sim_gpu(requested: int) -> int:
    """Return ``requested`` unless that GPU is wedged, then fall back to a healthy index."""
    import subprocess as _sp

    requested = int(requested)
    if not _gpu_index_has_err(requested):
        return requested
    for idx in range(8):
        if idx == requested:
            continue
        try:
            probe = _sp.run(
                ["nvidia-smi", "-i", str(idx)],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            combined = probe.stdout + probe.stderr
            if "ERR!" not in combined and "not found" not in combined.lower():
                print(
                    f"\033[93m[carla] GPU {requested} is unhealthy (ERR!); "
                    f"falling back to GPU {idx} for CARLA until you reboot.\033[0m",
                    flush=True,
                )
                return idx
        except Exception:
            continue
    return requested


def _warn_unhealthy_gpus() -> None:
    """Print a warning when ``nvidia-smi`` reports GPUs in an error state.

    A prior CARLA abort can leave a GPU wedged (``ERR!`` in ``nvidia-smi``). UE4
    may hang during Vulkan init while enumerating the broken device even when
    ``-graphicsadapter=0`` targets a healthy card.
    """
    try:
        import subprocess as _sp

        out = _sp.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
        if "ERR!" not in out:
            return
        print(
            "\033[93m[carla] WARNING: nvidia-smi reports unhealthy GPU(s) (ERR!).\033[0m",
            flush=True,
        )
        for line in out.splitlines():
            if "ERR!" in line or "GeForce" in line or "NVIDIA" in line:
                print(f"  {line.strip()}", flush=True)
        print(
            "\033[93m[carla] If the wedged GPU is the primary/display GPU, "
            "``nvidia-smi --gpu-reset`` will fail — reboot the machine to clear it.\033[0m\n"
            "  bash ~/ogbench-carla/reset_carla.sh && sudo reboot\n"
            "\033[93m[carla] Until reboot, CARLA will auto-fallback to a healthy GPU "
            "(see message above if fallback occurs).\033[0m",
            flush=True,
        )
    except Exception:
        pass


# Candidate locations for the NVIDIA Vulkan ICD manifest. Distros disagree on
# this path (Debian/Ubuntu driver packages install under /etc, the .run installer
# and some images use /usr/share), so we probe instead of hardcoding one.
_NVIDIA_VK_ICD_CANDIDATES = (
    "/etc/vulkan/icd.d/nvidia_icd.json",
    "/etc/vulkan/icd.d/nvidia_icd.x86_64.json",
    "/usr/share/vulkan/icd.d/nvidia_icd.json",
    "/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json",
)


def _resolve_nvidia_vk_icd() -> Optional[str]:
    """Return a path to an existing NVIDIA Vulkan ICD, or ``None`` if none found.

    Pointing ``VK_ICD_FILENAMES`` at a non-existent file makes the Vulkan loader
    skip default discovery and find *no* driver, so CARLA's ``-RenderOffScreen``
    (Vulkan RHI) exits immediately with an empty log. We therefore (1) honor a
    caller-set ``VK_ICD_FILENAMES`` only if every listed file exists, else (2)
    probe known NVIDIA ICD locations, else (3) return ``None`` so the caller can
    leave the var unset and let the loader auto-discover from its default dirs.
    """
    explicit = os.environ.get("VK_ICD_FILENAMES")
    if explicit:
        paths = [p for p in explicit.split(os.pathsep) if p]
        if paths and all(os.path.exists(p) for p in paths):
            return explicit
    for candidate in _NVIDIA_VK_ICD_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def _describe_exit(returncode: Optional[int]) -> str:
    """Human-readable description of a subprocess return code (decodes signals)."""
    if returncode is None:
        return "still running"
    if returncode < 0:
        sig = -returncode
        try:
            name = signal.Signals(sig).name
        except (ValueError, AttributeError):
            name = f"signal {sig}"
        return f"killed by {name} ({sig})"
    return f"exit code {returncode}"


def _read_log_tail(path: str, n_bytes: int = 4000) -> str:
    """Read the last ``n_bytes`` of a log, flagging empty/unreadable logs explicitly."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = f.read()
    except Exception as exc:
        return f"(log unreadable: {exc})"
    if not data.strip():
        return (
            "(log is empty -- CARLA wrote nothing before exiting. This usually "
            "means UE4 died before logging, e.g. Vulkan/GPU init failure, a missing "
            "NVIDIA Vulkan ICD, or the process being killed by a signal.)"
        )
    return data[-n_bytes:]


def _carla_subprocess_env(display_num: int, sim_gpu_rank: int) -> Dict[str, str]:
    """Minimal UE4 environment with GPU/Vulkan vars CARLA needs to boot off-screen."""
    _sys_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    carla_env: Dict[str, str] = {
        "HOME": os.environ.get("HOME", "/root"),
        "USER": os.environ.get("USER", "root"),
        "LOGNAME": os.environ.get("LOGNAME", os.environ.get("USER", "root")),
        "PATH": _sys_path,
        "DISPLAY": f":{display_num}",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        # Hide other GPUs from UE4/Vulkan so a wedged card cannot hang enumeration.
        "NVIDIA_VISIBLE_DEVICES": str(int(sim_gpu_rank)),
        "CUDA_VISIBLE_DEVICES": "0",
        "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    }
    vk_icd = _resolve_nvidia_vk_icd()
    if vk_icd is not None:
        carla_env["VK_ICD_FILENAMES"] = vk_icd
    else:
        # No NVIDIA ICD found at any known path: leave VK_ICD_FILENAMES unset so
        # the Vulkan loader scans its default dirs instead of a bogus file.
        print(
            "\033[33m[carla] WARNING: no NVIDIA Vulkan ICD found at any of "
            f"{_NVIDIA_VK_ICD_CANDIDATES}; leaving VK_ICD_FILENAMES unset and "
            "relying on Vulkan default discovery. If CARLA fails to boot, install "
            "the NVIDIA Vulkan ICD or set VK_ICD_FILENAMES to its manifest.\033[0m",
            flush=True,
        )
    for _k in (
        "CUDA_HOME",
        "CUDA_ROOT",
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_DATA_DIRS",
        "DBUS_SESSION_BUS_ADDRESS",
        "XAUTHORITY",
    ):
        if _k in os.environ:
            carla_env[_k] = os.environ[_k]
    # CARLA 0.9.16+ requires XDG_RUNTIME_DIR; synthesize one if the parent env lacks it
    # (headless/systemd-less boxes), rather than only forwarding it when already set.
    _xdg = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    carla_env["XDG_RUNTIME_DIR"] = _xdg if os.path.isdir(_xdg) else "/tmp"
    return carla_env


def _child_process_setup() -> None:
    """Session leader + kill child sim processes if the training parent dies abruptly."""
    os.setsid()
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:
        pass


_child_reaper_installed = False
_emergency_cleanup_installed = False
_ACTIVE_CARLA_ENV: Optional[weakref.ReferenceType["CarlaBench2DriveWrapper"]] = None


def _install_child_process_reaper() -> None:
    """No-op: do not set ``SIGCHLD`` to ``SIG_IGN``.

    Ignoring ``SIGCHLD`` makes CARLA exit immediately during boot (``poll()==0``,
    empty log). Zombie prevention is handled via ``PR_SET_PDEATHSIG`` on sim
    children and explicit ``wait()`` in :meth:`kill_subprocesses`.
    """
    global _child_reaper_installed
    _child_reaper_installed = True


def _register_carla_env_for_emergency_cleanup(env: "CarlaBench2DriveWrapper") -> None:
    global _ACTIVE_CARLA_ENV
    _ACTIVE_CARLA_ENV = weakref.ref(env)


def _emergency_carla_shutdown(signum: int, _frame: Any) -> None:
    """Best-effort CARLA teardown when Python dies via native abort (SIGABRT/SIGSEGV).

    ``atexit`` handlers do not run on C++ ``abort()``. Without this, Xvfb/CARLA
    children are orphaned and the parent can remain a zombie holding GPU memory.
    """
    ref = _ACTIVE_CARLA_ENV
    env = ref() if ref is not None else None
    try:
        if env is not None:
            env._kill_carla_subprocesses()
            CarlaBench2DriveWrapper._kill_stale_carla_processes(
                rpc_port=int(env.carla_config.get("port", 0) or 0),
                x_display_num=int(env.carla_config.get("x_display_num", 0) or 0),
            )
    except Exception:
        pass
    # Hard exit: do not wait for JAX/CUDA threads during normal interpreter shutdown.
    os._exit(128 + (signum if 0 < signum < 128 else 0))


def _install_emergency_carla_cleanup() -> None:
    global _emergency_cleanup_installed
    if _emergency_cleanup_installed:
        return
    _install_child_process_reaper()
    for sig in (signal.SIGABRT, signal.SIGSEGV):
        try:
            signal.signal(sig, _emergency_carla_shutdown)
        except Exception:
            pass
    _emergency_cleanup_installed = True


class IsolatedLeaderboardEvaluator(LeaderboardEvaluator):
    """Leaderboard evaluator variant that honors explicit per-instance launch args."""

    def _setup_simulation(self, args):
        self.carla_path = os.environ["CARLA_ROOT"]

        rpc_port = int(getattr(args, "port", 0) or 0)
        if rpc_port <= 0:
            # Reserve rpc + the two streaming ports (rpc+1, rpc+2) as one free block.
            rpc_port = _find_free_port(2000, span=3)
        args.port = rpc_port

        display_num = int(getattr(args, "x_display_num", 0) or 0)
        if display_num > 0:
            _clear_stale_display_lock(display_num)
        else:
            display_num = _find_free_display_num()

        _warn_unhealthy_gpus()
        sim_gpu_rank = _pick_healthy_sim_gpu(int(getattr(args, "gpu_rank", 0) or 0))

        xvfb_cmd = [
            "Xvfb", f":{display_num}",
            "-screen", "0", "1280x1024x24",
            "-ac", "+extension", "GLX", "+render", "-noreset",
        ]
        # stdin=DEVNULL so Xvfb/CARLA don't inherit any pipe fds from our process.
        self._launch_rpc_port = rpc_port
        self._launch_display_num = display_num
        self._carla_log_file = None

        self.xvfb = subprocess.Popen(
            xvfb_cmd,
            preexec_fn=_child_process_setup,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(self.kill_subprocesses)
        time.sleep(2)

        carla_env = _carla_subprocess_env(display_num, sim_gpu_rank)

        cmd = [
            os.path.join(self.carla_path, "CarlaUE4.sh"),
            "-RenderOffScreen",
            "-nosound",
            # Busy shared hosts can stall UE4's render thread for more than the
            # Linux default of 60 s during initial world/shader setup.  Let that
            # startup finish instead of crashing the simulator watchdog.
            "-g.TimeoutForBlockOnRenderFence=300000",
            f"-carla-rpc-port={rpc_port}",
            f"-graphicsadapter={sim_gpu_rank}",
        ]
        streaming_port = int(getattr(args, "streaming_port", 0) or 0)
        if streaming_port > 0:
            cmd.append(f"-carla-streaming-port={streaming_port}")
        _carla_log_path = f"/tmp/carla_rpc{rpc_port}.log"
        self._carla_log_file = open(_carla_log_path, "w", buffering=1)
        self.server = subprocess.Popen(
            cmd,
            preexec_fn=_child_process_setup,
            env=carla_env,
            stdin=subprocess.DEVNULL,
            stdout=self._carla_log_file,
            stderr=self._carla_log_file,
        )
        # NOTE: returncode is None immediately after Popen; log pid + paths instead.
        print(
            f"[carla] launched UE4 (pid={self.server.pid}) on display :{display_num} "
            f"sim_gpu_rank={sim_gpu_rank}; VK_ICD_FILENAMES={carla_env.get('VK_ICD_FILENAMES', '<unset>')}; "
            f"log={_carla_log_path}\n[carla] cmd: {' '.join(cmd)}",
            flush=True,
        )

        max_boot_s = max(60, int(os.environ.get("CARLA_BOOT_TIMEOUT", "180")))
        print(f"[carla] waiting up to {max_boot_s}s for UE4 RPC on port {rpc_port}...", flush=True)
        boot_start = time.time()
        client = None
        client_timeout = args.timeout if args.timeout else self.client_timeout
        while time.time() - boot_start < max_boot_s:
            elapsed = int(time.time() - boot_start)
            if self.server.poll() is not None:
                tail = _read_log_tail(_carla_log_path)
                raise RuntimeError(
                    f"CARLA server exited during boot ({_describe_exit(self.server.returncode)}) "
                    f"after {elapsed}s on display :{display_num} sim_gpu_rank={sim_gpu_rank}; "
                    f"see {_carla_log_path}\n{tail}"
                )
            try:
                probe = carla.Client(args.host, rpc_port)
                probe.set_timeout(2.0)
                probe.get_server_version()
                client = probe
                print(f"[carla] RPC ready after {elapsed}s", flush=True)
                break
            except Exception:
                if elapsed > 0 and elapsed % 10 == 0:
                    print(f"[carla] still booting... ({elapsed}s)", flush=True)
                time.sleep(5)
        else:
            tail = _read_log_tail(_carla_log_path)
            still_alive = self.server.poll() is None
            status = "still running but unresponsive" if still_alive else _describe_exit(self.server.returncode)
            raise RuntimeError(
                f"CARLA server did not open RPC port {rpc_port} within {max_boot_s}s "
                f"(server {status}); see {_carla_log_path}. If nvidia-smi shows ERR! on a "
                f"GPU, reset it or reboot.\n{tail}"
            )

        attempts = 0
        num_max_restarts = 20
        client_timeout = args.timeout if args.timeout else self.client_timeout
        # World/shader setup can exceed a minute on a busy shared host. Match
        # the extended render-fence watchdog so we do not abandon the first
        # request and stack repeated setup RPCs while UE4 is still rendering.
        _setup_attempt_timeout = 300.0
        while attempts < num_max_restarts:
            try:
                client = carla.Client(args.host, rpc_port)
                client.set_timeout(_setup_attempt_timeout)

                settings = carla.WorldSettings(
                    synchronous_mode=True,
                    fixed_delta_seconds=1.0 / self.frame_rate,
                    deterministic_ragdolls=True,
                    spectator_as_ego=False,
                )
                client.get_world().apply_settings(settings)
                client.set_timeout(client_timeout)
                print(f"load_world success , attempts={attempts}", flush=True)
                break
            except Exception as e:
                print(f"load_world failed , attempts={attempts}", flush=True)
                print(e, flush=True)
                # A UE4 render-fence segfault ("GameThread timed out waiting for RenderThread")
                # kills the server outright, and every remaining attempt then burns its full
                # 300 s client timeout against a process that is gone -- up to ~100 minutes of
                # silent stall before this loop gives up. The server is never relaunched from
                # here (only run_carla.sh's outer retry does that), so once it is dead the only
                # useful move is to fail fast and let that outer retry bring up a fresh one.
                if self.server is not None and self.server.poll() is not None:
                    raise RuntimeError(
                        f"CARLA server (pid {self.server.pid}) exited with "
                        f"{self.server.returncode} during world setup on rpc port {rpc_port}; "
                        f"see the server log for the UE4 crash. Not retrying a dead server."
                    ) from e
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

    def kill_subprocesses(self) -> None:
        """Terminate Xvfb + CARLA server process groups started by this evaluator."""
        for attr in ("server", "xvfb"):
            proc = getattr(self, attr, None)
            if proc is None:
                continue
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                proc.wait(timeout=3)
            except Exception:
                pass
            setattr(self, attr, None)
        log_f = getattr(self, "_carla_log_file", None)
        if log_f is not None:
            try:
                log_f.close()
            except Exception:
                pass
            self._carla_log_file = None


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


def _decode_simlingo_image(sensor_dict: Dict[str, Any]) -> np.ndarray:
    """Decode ``rgb_simlingo`` at native 1024×512 resolution for SimLingo inference."""
    if not sensor_dict or SIMLINGO_CAMERA_TAG not in sensor_dict:
        return np.zeros(SIMLINGO_IMAGE_SHAPE_HWC, dtype=np.uint8)
    tup = sensor_dict[SIMLINGO_CAMERA_TAG]
    if not isinstance(tup, (tuple, list)) or len(tup) < 2:
        return np.zeros(SIMLINGO_IMAGE_SHAPE_HWC, dtype=np.uint8)
    payload = tup[1]
    if payload is None:
        return np.zeros(SIMLINGO_IMAGE_SHAPE_HWC, dtype=np.uint8)
    arr = np.asarray(payload)
    if arr.ndim != 3:
        return np.zeros(SIMLINGO_IMAGE_SHAPE_HWC, dtype=np.uint8)
    rgb = _bgra_to_rgb_hwc(arr) if arr.shape[-1] == 4 else arr.astype(np.uint8, copy=False)
    if rgb.shape != SIMLINGO_IMAGE_SHAPE_HWC:
        try:
            import cv2
            rgb = cv2.resize(rgb, (SIMLINGO_IMAGE_SHAPE_HWC[1], SIMLINGO_IMAGE_SHAPE_HWC[0]), interpolation=cv2.INTER_AREA)
        except Exception:
            return np.zeros(SIMLINGO_IMAGE_SHAPE_HWC, dtype=np.uint8)
    return rgb


def _compute_target_point_ego(ego_actor, route_planner) -> np.ndarray:
    """Return the next route waypoint in the ego vehicle's local 2-D frame.

    Matches the SimLingo convention: x is forward, y is left (right-hand frame).
    Returns zeros if no route info is available.
    """
    if route_planner is None or ego_actor is None:
        return np.zeros(2, dtype=np.float32)
    try:
        import math as _math
        tf = ego_actor.get_transform()
        ego_pos = np.array([tf.location.x, tf.location.y, tf.location.z], dtype=np.float64)
        waypoint_route = route_planner.run_step(ego_pos)
        if len(waypoint_route) > 1:
            far_wp, _ = waypoint_route[1]
        elif len(waypoint_route) > 0:
            far_wp, _ = waypoint_route[0]
        else:
            return np.zeros(2, dtype=np.float32)
        yaw = _math.radians(tf.rotation.yaw)
        rel_x = float(far_wp[0]) - tf.location.x
        rel_y = float(far_wp[1]) - tf.location.y
        # Rotate from CARLA world to ego frame: x=forward, y=left
        x_e = rel_x * _math.cos(yaw) + rel_y * _math.sin(yaw)
        y_e = -rel_x * _math.sin(yaw) + rel_y * _math.cos(yaw)
        return np.array([x_e, y_e], dtype=np.float32)
    except Exception:
        return np.zeros(2, dtype=np.float32)


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


# Framing crop applied before the squeeze to square, mirroring
# ``openpi.training.steervla_rlds_dataset.SIMLINGO_FRAMING_CROP``. Keep the two in sync.
#
# Every SimLingo-derived training corpus reaches the model through this box: the
# ``simlingo_dataset_*_img512_1116`` builds baked it in at dataset-creation time, and
# ``simplified_reasoning_dataset`` -- which stored the full frame -- is re-cropped to it at decode
# time by ``DATASET_IMAGE_CROPS``. The live CARLA camera renders the same 1024x512 view those
# corpora were built from (see ``observation_only.py``: fov/x/z now match ``simlingo_obs.py``), so
# without this crop the policy would see a wider FOV with the ego hood at the bottom -- the one
# framing no longer present anywhere in training.
#
# Normalized so it lands correctly whatever the incoming resolution: on the native 1024x512 frame it
# is the box (170, 0, 852, 359); on a stored 512x512 frame, (85, 0, 426, 359).
SIMLINGO_FRAMING_CROP = (170 / 1024, 0.0, 852 / 1024, 359 / 512)


def crop_to_simlingo_framing(rgb: np.ndarray) -> np.ndarray:
    """Crop ``rgb`` to :data:`SIMLINGO_FRAMING_CROP`. Aspect is *not* preserved by the caller's resize."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = SIMLINGO_FRAMING_CROP
    left, top = int(round(x0 * w)), int(round(y0 * h))
    right, bottom = int(round(x1 * w)), int(round(y1 * h))
    cropped = rgb[top:bottom, left:right]
    # A degenerate box (absurdly small input) would produce an empty array that cv2.resize rejects.
    return cropped if cropped.size else rgb


def downscale_rgb_for_policy(rgb_viz: np.ndarray, *, already_processed: bool = False) -> np.ndarray:
    """Crop to the SimLingo training framing, then squeeze to policy/RL ``IMAGE_SHAPE_HWC``.

    **Crop-then-resize is the only supported preprocessing for SteerVLA — there is no flag to
    skip the crop.** It is what makes the live frame match what every SteerVLA checkpoint was
    trained on; the resize is a plain distorting squeeze to square, matching the dataset
    preprocessing (a pad would introduce black bars the backbone has never seen).

    ``already_processed`` is the *only* way to get a frame through uncropped, and it exists
    solely for frames that have already been through this function (a replayed or stored
    frame), where cropping again would zoom in past the training framing. It is stated by the
    caller rather than inferred from the shape: an uncropped frame that merely happens to be
    ``IMAGE_SHAPE_HWC`` would otherwise skip the crop silently, which is exactly the bug this
    signature prevents. No caller in this repo passes it -- ``_decode_rgb_front_viz``
    normalises every live frame to ``VIZ_IMAGE_SHAPE_HWC`` first.

    Raises rather than returning a black frame on failure: a silent ``_zeros_rgb_image()``
    would put the policy on blank input and score as bad driving, indistinguishable from a
    genuine failure.
    """
    rgb_viz = np.asarray(rgb_viz, dtype=np.uint8)
    if already_processed:
        if rgb_viz.shape != IMAGE_SHAPE_HWC:
            raise ValueError(
                f"downscale_rgb_for_policy(already_processed=True) expects a frame already at "
                f"{IMAGE_SHAPE_HWC}, got {rgb_viz.shape}."
            )
        return rgb_viz
    if rgb_viz.shape == IMAGE_SHAPE_HWC:
        raise ValueError(
            f"downscale_rgb_for_policy got a frame already at policy resolution "
            f"{IMAGE_SHAPE_HWC} without already_processed=True. Refusing to guess whether it "
            f"has been cropped: passing it through would feed SteerVLA the uncropped framing "
            f"no checkpoint was trained on. Live frames arrive at {VIZ_IMAGE_SHAPE_HWC}."
        )

    import cv2

    return cv2.resize(
        crop_to_simlingo_framing(rgb_viz),
        (IMAGE_SHAPE_HWC[1], IMAGE_SHAPE_HWC[0]),
        interpolation=cv2.INTER_AREA,
    )


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


def _sync_pseudo_sensors_for_tick(agent_wrapper: Any) -> None:
    """Push pseudo-sensor readings for the current tick frame on the main thread.

    ``SpeedometerReader`` runs in a background thread and is paused during long
    VLA / best-of-N inference (``_run_ps=False``). Physical sensors enqueue on
    ``world.tick()``, but a paused speedometer may never publish before
    ``AutonomousAgent.__call__`` → ``sensor_interface.get_data()``, which blocks
    for up to 300s waiting for all sensors.
    """
    if agent_wrapper is None:
        return
    frame = GameTime.get_frame()
    for sensor in list(getattr(agent_wrapper, "_sensors_list", []) or []):
        if sensor is None or not hasattr(sensor, "__call__"):
            continue
        cb = getattr(sensor, "_callback", None)
        if cb is None:
            continue
        # Only pseudo-sensors (SpeedometerReader / OpenDriveMapReader) have _run_ps.
        if not hasattr(sensor, "_run_ps"):
            continue
        try:
            if not getattr(sensor, "_run_ps", True):
                sensor._run_ps = True
            cb(GenericMeasurement(sensor(), frame))
        except Exception:
            pass


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
                _sync_pseudo_sensors_for_tick(self._agent_wrapper)
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
        timeout=float(cfg.get("timeout", 2000.0)),
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


def _summarize_route_commands(route: Any) -> list[dict]:
    """Collapse a full route's per-waypoint RoadOptions into an ordered maneuver list.

    ``route`` is the leaderboard ``route_scenario.route`` — a list of
    ``(carla.Transform, RoadOption)`` covering the whole episode. The interpolated route
    carries one RoadOption per densely-sampled waypoint (mostly LANEFOLLOW with turns /
    lane changes near intersections); we collapse each contiguous run of the same command
    into a single maneuver so the result reads as the high-level "task" for the episode,
    e.g. ``follow the road`` → ``go right at the next intersection`` → ``follow the road``.

    Returns a list of ``{"command_id", "command", "start_distance_m"}`` dicts in route order
    (``start_distance_m`` is the cumulative metres travelled along the route before the
    maneuver begins).
    """
    plan: list[dict] = []
    prev_cmd: Optional[int] = None
    cum_dist = 0.0
    prev_xy: Optional[tuple[float, float]] = None
    for pos, cmd in route:
        loc = getattr(pos, "location", pos)
        x, y = float(loc.x), float(loc.y)
        if prev_xy is not None:
            cum_dist += ((x - prev_xy[0]) ** 2 + (y - prev_xy[1]) ** 2) ** 0.5
        prev_xy = (x, y)
        cmd_int = int(getattr(cmd, "value", cmd))
        if not (1 <= cmd_int <= 6):
            continue
        if cmd_int != prev_cmd:
            plan.append(
                {
                    "command_id": cmd_int,
                    "command": ROUTING_COMMAND_TEXT.get(cmd_int, "follow the road"),
                    "start_distance_m": round(cum_dist, 1),
                }
            )
            prev_cmd = cmd_int
    return plan


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


def _carla_actor_alive(actor) -> bool:
    """True if ``actor``'s server-side counterpart still exists and is alive.

    Valid only **after a tick has run**. Measured against a live 0.9.16 server, in synchronous mode,
    immediately after ``try_spawn_actor`` and before any ``world.tick()``: ``world.get_actor(id)``
    returns the actor but with ``is_alive == False``, and the client-side ``actor.is_alive`` flag
    also reads False. One ``world.tick()`` later both read True. The result was 0/4 then 4/4,
    identical via raw ``try_spawn_actor`` and via ``CarlaDataProvider.request_new_actor(tick=False)``.

    So there is no liveness signal to consult before the first tick -- which is why the caller skips
    this check entirely for ``tick=False`` spawns rather than picking a different flag.
    """
    try:
        if actor is None:
            return False
        world = CarlaDataProvider._world
        if world is None:
            return bool(getattr(actor, "is_alive", False))
        live = world.get_actor(actor.id)
        return live is not None and live.is_alive
    except Exception:
        return False


def _destroy_leaked_actor(actor) -> None:
    """Drop a spawned-but-rejected actor from the pool *and* the world.

    Popping it from ``_carla_actor_pool`` alone leaves the vehicle physically parked on the
    spawn point, so every subsequent attempt at that transform fails with
    ``WARNING: Cannot spawn actor`` -- a self-inflicted cascade.
    """
    try:
        CarlaDataProvider._carla_actor_pool.pop(actor.id, None)
    except Exception:
        pass
    try:
        actor.destroy()
    except Exception:
        pass


def _install_carla_actor_spawn_guard() -> None:
    """Make ``CarlaDataProvider.request_new_actor`` never hand back a stale actor handle.

    Bench2Drive scenarios (e.g. ``srunner/scenarios/cut_in.py``) call
    ``actor.set_simulate_physics(...)`` immediately after spawning a scenario actor. On the
    large Bench2Drive maps the just-spawned actor can be torn down by tile streaming during
    the spawn tick, leaving a dead actor id. ``set_simulate_physics`` on a dead id throws a
    C++ ``std::runtime_error`` that escapes into ``std::terminate()`` and aborts the whole
    process ("Actor could not be found in the registry ... Fatal Python error: Aborted") —
    not a catchable Python exception. ``carla.Actor`` is an extension type so the method
    itself cannot be wrapped; instead we wrap the spawn so the returned actor is verified
    alive (with a bounded retry + extra tick to let streaming settle). If it still can't be
    spawned alive we return ``None`` so the caller raises an ordinary, catchable Python
    error (a clean episode failure) instead of aborting the process.

    Only the *stale handle* case is retried. When the underlying ``try_spawn_actor`` refuses the
    transform it returns ``None`` and srunner logs ``WARNING: Cannot spawn actor ...``; that means
    the spawn point is occupied, and retrying inside the same tick cannot help because nothing has
    moved. ``BackgroundBehavior`` calls this every tick for each traffic source, so retrying a
    blocked point tripled both the log spam and the blocking RPC round-trips on the sim's critical
    path (~112k wasted calls in a single observed episode).

    The liveness check runs **only when the caller asked ``request_new_actor`` to tick**. With
    ``tick=False`` nothing has advanced the simulation between the spawn and the check, and no
    liveness signal is valid yet -- both ``world.get_actor(id).is_alive`` and ``actor.is_alive``
    read False on a vehicle that is physically in the world (see :func:`_carla_actor_alive`).
    Checking anyway rejected ~100% of them, and since ``BackgroundBehavior._spawn_source_actor`` is
    the *only* way road traffic is replenished as the ego drives, that silently disabled continuous
    background traffic: only the initial batch population (``request_new_batch_actors``, which this
    guard does not wrap) ever survived, and it thinned out as the ego moved away from it.

    Idempotent; safe to call on every env construction. Retries via
    ``CARLA_SPAWN_GUARD_RETRIES`` (default 3); set to 0/1 to disable retrying.
    """
    if getattr(CarlaDataProvider, "_spawn_guard_installed", False):
        return
    _orig_request_new_actor = CarlaDataProvider.request_new_actor
    # ``tick`` is the 9th positional parameter of CarlaDataProvider.request_new_actor.
    _TICK_ARG_INDEX = 8

    def request_new_actor_guarded(*args, **kwargs):
        attempts = max(1, int(os.environ.get("CARLA_SPAWN_GUARD_RETRIES", "3")))
        if "tick" in kwargs:
            ticked = bool(kwargs["tick"])
        elif len(args) > _TICK_ARG_INDEX:
            ticked = bool(args[_TICK_ARG_INDEX])
        else:
            ticked = True
        for i in range(attempts):
            # When ``tick`` is set, request_new_actor ticks once internally before returning, so
            # the actor is registered server-side by then. Do NOT add another tick here: the
            # scenario freezes physics (set_simulate_physics(False)) right after spawn,
            # and an extra physics tick could itself collide/destroy the fresh actor.
            actor = _orig_request_new_actor(*args, **kwargs)
            if actor is None:
                # Occupied spawn point, not a stale handle -- let the caller deal with it.
                return None
            if not ticked:
                # Un-checkable (see docstring). set_simulate_physics on a dead id is still
                # covered later by _install_carla_physics_guard.
                return actor
            if _carla_actor_alive(actor):
                return actor
            # Stale handle: destroy it so it cannot keep blocking the spawn point, then retry.
            _destroy_leaked_actor(actor)
            print(
                f"[carla spawn guard] spawned actor {getattr(actor, 'id', '?')} was not "
                f"alive after tick (attempt {i + 1}/{attempts}); retrying",
                flush=True,
            )
        print(
            "[carla spawn guard] could not spawn a live actor after "
            f"{attempts} attempts; returning None (episode will fail cleanly)",
            flush=True,
        )
        return None

    CarlaDataProvider.request_new_actor = staticmethod(request_new_actor_guarded)
    CarlaDataProvider._spawn_guard_installed = True
    print("[carla spawn guard] installed request_new_actor liveness guard", flush=True)


def _install_carla_physics_guard() -> None:
    """Skip ``set_simulate_physics`` on actors that are no longer in the registry.

    Scenario spawn/teardown (e.g. ``BatchActorTransformSetter``, cut-in actors) calls
    ``actor.set_simulate_physics(...)``. If streaming or cleanup already destroyed the
    actor, CARLA's C++ API throws ``std::runtime_error`` ("Actor could not be found
    in the registry") → ``std::terminate()`` → uncatchable process abort.

    Use the client-side ``is_alive`` flag only — do **not** call ``world.get_actor``
    here. An extra RPC during spawn/teardown races with scenario setup and can falsely
    skip physics on live actors, breaking scenarios immediately after load.
    """
    if getattr(carla.Actor, "_physics_guard_installed", False):
        return
    _orig_set_simulate_physics = carla.Actor.set_simulate_physics

    def guarded_set_simulate_physics(self, enabled=True):
        try:
            if hasattr(self, "is_alive") and not self.is_alive:
                return
        except Exception:
            return
        return _orig_set_simulate_physics(self, enabled)

    carla.Actor.set_simulate_physics = guarded_set_simulate_physics
    carla.Actor._physics_guard_installed = True
    print("[carla physics guard] installed set_simulate_physics liveness guard", flush=True)


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
        # Guard scenario actor spawns so a stale actor id can't abort the process via
        # an uncatchable C++ throw in set_simulate_physics (see fn docstring).
        _install_carla_actor_spawn_guard()
        _install_carla_physics_guard()
        self.carla_config = dict(carla_config)
        self.route_entry: RouteEntry = _resolve_route(self.carla_config, route)
        warn_if_carla_root_mismatched(self.route_entry.source, self.carla_config)

        self._evaluator: Optional[LeaderboardEvaluator] = None
        self._args: Optional[SimpleNamespace] = None
        self._base_agent_config: str = ""
        self._scenario_active = False
        self._last_driving_score = 0.0
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
        collision_contact_penalty = self.carla_config.get(
            "collision_contact_penalty", DEFAULT_COLLISION_CONTACT_PENALTY
        )
        self._collision_contact_penalty = (
            None if collision_contact_penalty is None else float(collision_contact_penalty)
        )
        self._outside_route_event_penalty = float(
            self.carla_config.get("outside_route_event_penalty", DEFAULT_OUTSIDE_ROUTE_EVENT_PENALTY)
        )
        self._traffic_violation_penalty = float(
            self.carla_config.get("traffic_violation_penalty", DEFAULT_TRAFFIC_VIOLATION_PENALTY)
        )
        self._terminate_on_infraction = bool(
            self.carla_config.get("terminate_on_infraction", DEFAULT_TERMINATE_ON_INFRACTION)
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
        self._success_bonus = float(self.carla_config.get("success_bonus", SUCCESS_BONUS))
        self._failure_bonus = float(self.carla_config.get("failure_bonus", FAILURE_BONUS))
        self._prev_collision_count = 0
        self._prev_outside_route_value = 0.0
        self._prev_route_progress_pct = 0.0
        self._max_episode_steps = int(
            self.carla_config.get("max_episode_steps", DEFAULT_MAX_EPISODE_STEPS)
        )
        self._episode_step_count = 0
        self._prev_traffic_violation_count = 0
        self._raw_collision_sensor: Any | None = None
        self._raw_collision_active: bool = False
        self._collision_recently_active: bool = False
        self._last_route_completion = 0.0
        # Set by checkpoint() the first time it's called in an episode: once the world has
        # been rolled back once, the leaderboard's RouteCompletionTest criterion (a monotonic
        # "furthest ever reached" value) can no longer be trusted for reward -- it doesn't
        # reset on teleport, so a candidate's own scoring trial silently inflates the baseline
        # the real committed run is then measured against, undercounting its progress reward
        # (see restore()). Once True, _route_completion_percent always uses the geometric
        # fallback (recomputed fresh from ego position each call) instead of the criterion.
        self._geometric_route_completion_only = False
        self._route_progress_xyz: Optional[np.ndarray] = None
        self._route_progress_s: Optional[np.ndarray] = None
        self._route_total_distance_m = 0.0
        self._route_progress_index = 0
        self._route_transforms: list[Any] = []
        self._route_completion_accum_perc: list[float] = []
        self._route_completion_index = 0

        self._expert_controller_kind = str(self.carla_config.get("expert_controller", "") or "").strip().lower()
        self._expert_agent: Any | None = None
        self._cached_world_map: Any | None = None
        self._route_planner: Any | None = None
        self._route_command_plan: list[dict] = []  # full ordered maneuver plan (set at reset)
        self._current_routing_command: int = 4  # LANEFOLLOW until route is loaded
        self._routing_last_command_tmp: int = -1  # simlingo carryover state
        self._routing_last_command: int = -1
        self._routing_dist_to_waypoint: int = 0  # metres; only shown when far_cmd != LANEFOLLOW
        self._routing_include_distance: bool = False
        self._routing_next_command: int = 4  # the manoeuvre after the current one ("then ...")
        self._target_points_ego: np.ndarray = np.zeros((2, 2), dtype=np.float32)  # (2, 2) ego-frame
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

                _decoder_kwargs = {}
                for _k in ("brake_speed", "brake_ratio", "stuck_threshold", "creep_throttle", "creep_duration"):
                    if _k in exec_raw:
                        _decoder_kwargs[_k] = exec_raw[_k]
                self._steervla_decoder = SimlingoStyleWaypointDecoder(**_decoder_kwargs)
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
        tm_port: Optional[int] = None,
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

        ``tm_port``: if set, any process listening on this TCP port (typically
        an orphaned carla_env_server holding the Traffic Manager port open) is
        killed before a new CARLA server is started.  Without this cleanup, the
        new CARLA's TM fails to bind the port and retries for minutes before
        raising RuntimeError.
        """
        import subprocess as _sp
        import glob as _glob
        if rpc_port is not None:
            _sp.run(
                ["pkill", "-9", "-f", f"CarlaUE4.*-carla-rpc-port={int(rpc_port)}"],
                capture_output=True,
            )
            time.sleep(1)
        if tm_port is not None:
            # Kill any process (typically an old carla_env_server) that is still
            # listening on the TM port — fuser sends SIGKILL and ignores errors.
            _sp.run(["fuser", "-k", f"{int(tm_port)}/tcp"], capture_output=True)
            time.sleep(1)
        if x_display_num is not None and int(x_display_num) > 0:
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
        _install_emergency_carla_cleanup()
        _register_carla_env_for_emergency_cleanup(self)
        self._kill_stale_carla_processes(
            rpc_port=int(self.carla_config.get("port", 0) or 0),
            x_display_num=int(self.carla_config.get("x_display_num", 0) or 0),
            tm_port=int(self.carla_config.get("traffic_manager_port", 0) or 0) or None,
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
        # Portable: probes the known distro locations rather than hardcoding one
        # (/usr/share/vulkan/... vs /etc/vulkan/...). See ogbench/carla/README.md.
        _NVIDIA_VK_ICD = _resolve_nvidia_vk_icd()
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
            try:
                if self._evaluator is not None:
                    self._evaluator.kill_subprocesses()
            except Exception:
                pass
            import atexit as _atexit
            _atexit._run_exitfuncs()  # type: ignore[attr-defined]
            if callable(_orig_sigterm):
                _orig_sigterm(signum, frame)
            else:
                raise SystemExit(0)

        _signal.signal(_signal.SIGTERM, _sigterm_handler)

    def _kill_carla_subprocesses(self) -> None:
        """Stop CARLA/Xvfb launched for this env instance."""
        ev = self._evaluator
        if ev is not None:
            try:
                ev.kill_subprocesses()
            except Exception:
                pass
        self._kill_stale_carla_processes(
            rpc_port=int(self.carla_config.get("port", 0) or 0),
            x_display_num=int(self.carla_config.get("x_display_num", 0) or 0),
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

    def _set_pseudo_sensors_running(self, running: bool, *, settle_s: float = 0.0) -> None:
        """Pause/resume SpeedometerReader background threads (avoids CARLA RPC during VLA)."""
        try:
            wrapper = self._evaluator.manager._agent_wrapper
            if wrapper is None:
                return
            for sensor in list(wrapper._sensors_list):
                if sensor is not None and hasattr(sensor, "_run_ps"):
                    sensor._run_ps = bool(running)
            if settle_s > 0:
                time.sleep(settle_s)
        except Exception:
            pass

    def pause_leaderboard_sensors(self) -> None:
        """Pause pseudo-sensor RPC threads during long off-tick work (SteerVLA / best-of-N)."""
        if not self._scenario_active or self._evaluator is None:
            return
        self._set_pseudo_sensors_running(False, settle_s=0.05)

    def resume_leaderboard_sensors(self) -> None:
        """Resume pseudo-sensor threads before the next env tick."""
        if not self._scenario_active or self._evaluator is None:
            return
        self._set_pseudo_sensors_running(True)

    def _pause_leaderboard_watchdogs(self) -> None:
        """Pause scenario watchdogs during long VLA inference (no ``world.tick()`` yet)."""
        try:
            mgr = self._evaluator.manager
            for wd in (getattr(mgr, "_watchdog", None), getattr(mgr, "_agent_watchdog", None)):
                if wd is not None:
                    wd.pause()
        except Exception:
            pass

    def _resume_leaderboard_watchdogs(self) -> None:
        """Resume watchdogs and reset their timers before the next env tick."""
        try:
            mgr = self._evaluator.manager
            for wd in (getattr(mgr, "_watchdog", None), getattr(mgr, "_agent_watchdog", None)):
                if wd is not None:
                    wd.resume()
                    wd.update()
        except Exception:
            pass

    def pause_for_vla_inference(self) -> None:
        """Hold CARLA watchdogs + pseudo-sensors while SteerVLA/best-of-N runs off-tick."""
        self.pause_leaderboard_sensors()
        self._pause_leaderboard_watchdogs()

    def resume_after_vla_inference(self) -> None:
        """Undo :meth:`pause_for_vla_inference` immediately before ``env.step``."""
        self.resume_leaderboard_sensors()
        self._resume_leaderboard_watchdogs()

    def _drain_pseudo_sensors(self) -> None:
        """Stop SpeedometerReader/OpenDriveMapReader threads before any CARLA cleanup.

        These threads call get_velocity() / get_transform() / to_opendrive() over the
        CARLA RPC socket on a timer. If the main thread makes any CARLA call while one
        of these threads is mid-RPC, the two concurrent requests corrupt the msgpack
        framing → clmdep_msgpack::type_error → C++ terminate() → abort.

        We set _run_ps=False on every pseudo-sensor and sleep briefly to let any
        in-progress CARLA call on the sensor thread finish before we proceed.
        """
        if not self._scenario_active or self._evaluator is None:
            return
        self._set_pseudo_sensors_running(False, settle_s=0.3)

    def _stop_active_scenario(self) -> None:
        if not self._scenario_active or self._evaluator is None:
            return
        self._destroy_raw_collision_sensor()
        self._drain_pseudo_sensors()
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
        scenario_name = config.scenario_configs[0].name if config.scenario_configs else "NoScenario"
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
        self._routing_dist_to_waypoint = 0
        self._routing_include_distance = False
        self._routing_next_command = 4
        self._target_points_ego = np.zeros((2, 2), dtype=np.float32)
        self._route_planner = None
        try:
            self._route_planner = self._create_route_planner(ev.route_scenario.route)
        except Exception as _rp_exc:
            print(f"[routing_command] RoutePlanner init failed: {_rp_exc}", flush=True)
        # Precompute the full ordered maneuver plan for the episode so downstream consumers
        # (e.g. the VLM coach) can present the overall task, not just the current command.
        self._route_command_plan = []
        try:
            self._route_command_plan = _summarize_route_commands(ev.route_scenario.route)
        except Exception as _rc_exc:
            print(f"[routing_command] route summary failed: {_rc_exc}", flush=True)
        self._scenario_active = True
        self._last_control = carla.VehicleControl()
        self._crash_stuck_ticks = 0
        self._prev_collision_count = 0
        self._prev_outside_route_value = 0.0
        self._prev_route_progress_pct = 0.0
        self._prev_traffic_violation_count = 0
        self._raw_collision_active = False
        self._collision_recently_active = False
        self._last_route_completion = 0.0
        self._geometric_route_completion_only = False
        self._init_route_progress_cache()
        self._spawn_raw_collision_sensor()

    def _ego_actor(self) -> Optional[carla.Actor]:
        ev = self._evaluator
        if ev is None or not getattr(ev, "manager", None):
            return None
        ego_list = getattr(ev.manager, "ego_vehicles", None) or []
        return ego_list[0] if ego_list else None

    def traffic_actor_states(self) -> list[dict[str, Any]]:
        """Ground-truth state (id/type/speed/location) of every non-ego vehicle/walker
        actor, read via the same live client/world reference the ego uses (so this
        doesn't hit the actor-visibility issues a freshly-opened separate client
        connection can run into). Debug-only -- see --debug_log_traffic."""
        if self._evaluator is None:
            return []
        world = self._evaluator.world
        ego = self._ego_actor()
        ego_id = ego.id if ego is not None else None
        out: list[dict[str, Any]] = []
        for actor in world.get_actors():
            if not actor.type_id.startswith(("vehicle.", "walker.pedestrian.")):
                continue
            if actor.id == ego_id:
                continue
            v = actor.get_velocity()
            speed = float((v.x**2 + v.y**2 + v.z**2) ** 0.5)
            loc = actor.get_location()
            out.append({
                "id": actor.id,
                "type": actor.type_id,
                "speed": speed,
                "loc": (float(loc.x), float(loc.y), float(loc.z)),
            })
        return out

    # Keyword match on actor type_id for scenario obstacle props (e.g. Fail2Drive's
    # "brickwall" roadblock asset pack -- see run_simlingo_fail2drive.sh's doc comment).
    _OBSTACLE_PROP_KEYWORDS = ("wall", "brick", "barrier", "roadblock")

    def nearest_obstacle_distance_m(self) -> Optional[float]:
        """Ground-truth XY distance from ego to the nearest scenario obstacle prop.

        Ground truth (queries the actual CARLA actor list + transform), not a proxy
        like speed -- more reliable for e.g. main_carla_teleop.py's auto_then_manual
        mode deciding when to hand off from autonomous rollout to a human. Returns
        None if the ego or no matching prop is currently in the world.

        XY-only (ignores Z): Bench2Drive/Fail2Drive scenario obstacle props (e.g. the
        brickwall roadblock) are staged at a large, unrelated Z offset from the road
        surface until the leaderboard's own route-progress trigger fires and moves
        them into place -- a naive 3D distance is dominated by that staging offset
        (observed: ~200m of pure Z mismatch, well before the wall is anywhere near
        the ego) and would never cross a close-range threshold like
        --obstacle_trigger_distance_m.
        """
        ego = self._ego_actor()
        if ego is None:
            return None
        world = self.evaluator.world
        ego_loc = ego.get_transform().location
        best: Optional[float] = None
        for actor in world.get_actors():
            if actor.type_id.startswith(("vehicle.", "walker.", "sensor.", "traffic.", "controller.")):
                continue
            tid = actor.type_id.lower()
            if not any(kw in tid for kw in self._OBSTACLE_PROP_KEYWORDS):
                continue
            actor_loc = actor.get_transform().location
            d = ((ego_loc.x - actor_loc.x) ** 2 + (ego_loc.y - actor_loc.y) ** 2) ** 0.5
            if best is None or d < best:
                best = d
        return best

    def teleport_ego_to_obstacle(self, offset_m: float = 1.0) -> bool:
        """Teleport the ego to sit ``offset_m`` in front of the nearest obstacle prop.

        Debug-only helper (main_carla_teleop.py's wall_snapshot mode) for visually
        inspecting candidate-action diversity right at a scenario obstacle without
        driving the whole route there first. "Front" is measured surface-to-surface
        (wall face to ego front bumper) using both actors' bounding-box extents, and
        the ego is placed along the line from its current position to the obstacle so
        it naturally faces the obstacle. Returns False if no obstacle prop or ego is
        currently in the world.

        CAVEAT: the obstacle prop itself (e.g. Fail2Drive's brickwall) is not
        necessarily *visible* after this call. Bench2Drive stages some scenario props
        at an unrelated Z offset until the leaderboard's own route-progress trigger
        fires and moves them onto the road; a raw teleport places the ego at the
        correct on-road XY next to the prop's (possibly still-staged) transform, but
        doesn't itself fire that trigger. Only useful for the ego's own position/
        candidate-diversity inspection, not for guaranteeing the obstacle renders.
        """
        ego = self._ego_actor()
        if ego is None:
            return False
        world = self.evaluator.world
        ego_loc = ego.get_transform().location
        best_actor = None
        best_dist = None
        for actor in world.get_actors():
            if actor.type_id.startswith(("vehicle.", "walker.", "sensor.", "traffic.", "controller.")):
                continue
            tid = actor.type_id.lower()
            if not any(kw in tid for kw in self._OBSTACLE_PROP_KEYWORDS):
                continue
            actor_loc = actor.get_transform().location
            # XY-only: obstacle props may be staged at an unrelated Z until the
            # leaderboard's own route-progress trigger moves them onto the road.
            d = ((ego_loc.x - actor_loc.x) ** 2 + (ego_loc.y - actor_loc.y) ** 2) ** 0.5
            if best_dist is None or d < best_dist:
                best_dist = d
                best_actor = actor
        if best_actor is None:
            return False
        wall_loc = best_actor.get_transform().location
        dx, dy = wall_loc.x - ego_loc.x, wall_loc.y - ego_loc.y
        dist_xy = max((dx**2 + dy**2) ** 0.5, 1e-3)
        ux, uy = dx / dist_xy, dy / dist_xy
        ego_half_length = 2.3
        try:
            ego_half_length = float(ego.bounding_box.extent.x)
        except Exception:
            pass
        wall_half_depth = 0.5
        try:
            wall_half_depth = float(max(best_actor.bounding_box.extent.x, best_actor.bounding_box.extent.y))
        except Exception:
            pass
        gap = offset_m + ego_half_length + wall_half_depth
        new_loc = carla.Location(x=wall_loc.x - ux * gap, y=wall_loc.y - uy * gap, z=ego_loc.z)
        yaw = math.degrees(math.atan2(uy, ux))
        ego.set_transform(carla.Transform(new_loc, carla.Rotation(pitch=0.0, roll=0.0, yaw=yaw)))
        ego.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        ego.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        world.tick()
        return True

    def drive_straight_until_close(
        self,
        *,
        target_distance_m: float = 10.0,
        slowdown_distance_m: float = 30.0,
        max_ticks: int = 2000,
        throttle: float = 0.4,
        slow_throttle: float = 0.12,
    ) -> bool:
        """Debug-only: drive straight (holding the ego's initial heading, bypassing
        the policy) until within ``target_distance_m`` (XY) of the nearest scenario
        obstacle prop, then brake to a stop right there -- or stop immediately if a
        collision happens first. For main_carla_teleop.py's wall_snapshot mode:
        driving via the policy is cautious around traffic/junctions and can take
        hundreds of decisions (or get stuck entirely) before reaching a distant
        obstacle; a raw straight-line drive is much faster when the obstacle sits
        ahead on a straight stretch of the route.

        Ticks via ``self.evaluator.manager.step_once()`` (the same path env.step()
        uses), not a bare ``world.tick()`` -- this matters because some scenario
        obstacle props (e.g. Fail2Drive's brickwall) are only moved onto the road /
        made collidable once the leaderboard's own scenario tree registers real
        route progress; a bare world tick never fires that trigger no matter how
        close the ego's raw position gets (verified empirically: zero collisions
        driving all the way to point-blank range via bare ticks).

        A small proportional steer correction holds the ego's spawn-time heading
        (assumes the stretch toward the obstacle is straight, per the caller) --
        zero steer alone measurably drifted off the road over ~200+m in testing.

        Throttle drops to ``slow_throttle`` once within ``slowdown_distance_m``, so
        a single tick's movement doesn't overshoot straight through the
        ``target_distance_m`` stop window into a collision (observed at full
        throttle: distance can jump from "far" to "collided" in one tick). Distance
        is checked *before* each tick's movement, and braked to a stop the moment
        it's within range, rather than after -- the intent is to end up close but
        not touching. Still stops immediately (without braking room) on the first
        tick a collision is registered regardless of distance, as a fallback: some
        obstacle props only become queryable in essentially the same tick the ego
        reaches them, so a collision can in principle still occur before the
        distance check ever sees a close reading. Some props are also removed from
        the world shortly after a collision resolves, so stopping immediately and
        letting the caller read the observation right away (no further ticks)
        maximizes the chance the obstacle is still actually visible/present.

        Returns whether the obstacle was reached, either by distance or by collision
        (False if max_ticks elapsed first, e.g. blocked by traffic, stuck against
        something off-route, or the episode ended).
        """
        ego = self._ego_actor()
        if ego is None:
            return False
        manager = self.evaluator.manager
        initial_yaw = ego.get_transform().rotation.yaw
        prev_collision_count = self._collision_count()
        for _ in range(max(1, int(max_ticks))):
            current_yaw = ego.get_transform().rotation.yaw
            yaw_error = ((current_yaw - initial_yaw + 180.0) % 360.0) - 180.0
            steer = float(np.clip(-0.02 * yaw_error, -0.3, 0.3))
            d = self.nearest_obstacle_distance_m()
            if d is not None and d <= target_distance_m:
                manager.pending_control = carla.VehicleControl(throttle=0.0, steer=steer, brake=1.0)
                try:
                    manager.step_once()
                except Exception:
                    pass
                return True
            cur_throttle = slow_throttle if (d is not None and d <= slowdown_distance_m) else throttle
            manager.pending_control = carla.VehicleControl(
                throttle=float(cur_throttle), steer=steer, brake=0.0
            )
            try:
                running, _tree_status = manager.step_once()
            except Exception:
                return False
            if not running:
                return False
            if self._collision_count() > prev_collision_count:
                return True
        return False

    def step_raw_control(self, throttle: float, steer: float, brake: float = 0.0) -> bool:
        """One ``manager.step_once()`` tick with a raw ``VehicleControl``, bypassing
        the policy/action-format decoding entirely. Debug-only (main_carla_teleop.py
        wall_snapshot video recording) -- lets the caller interleave driving ticks
        with its own policy queries (the policy itself lives client-side, not in
        this env subprocess), unlike drive_straight_until_close()'s single blocking
        call. Returns whether the episode is still running (False if it just
        terminated this tick).
        """
        manager = self.evaluator.manager
        manager.pending_control = carla.VehicleControl(
            throttle=float(throttle), steer=float(steer), brake=float(brake)
        )
        try:
            running, _tree_status = manager.step_once()
        except Exception:
            return False
        return bool(running)
    def _create_route_planner(self, route) -> Any | None:
        """Build a SimLingo-style command planner for the active route."""
        route_planner_cls = None
        try:
            from team_code.nav_planner import RoutePlanner as route_planner_cls  # type: ignore
        except Exception:
            try:
                from impls.coaches.simlingo.nav_planner import RoutePlanner as route_planner_cls
            except Exception as exc:
                print(f"[routing_command] RoutePlanner import failed: {exc}", flush=True)
                return None
        try:
            planner = route_planner_cls(min_distance=7.5, max_distance=50.0)
            planner.set_route(route, gps=False)
            return planner
        except Exception as exc:
            print(f"[routing_command] RoutePlanner set_route failed: {exc}", flush=True)
            return None

    def _update_routing_command(self) -> None:
        """Advance the route planner by one step and cache the current routing command."""
        if self._route_planner is None:
            return
        ego = self._ego_actor()
        if ego is None:
            return
        import math as _math
        tf = ego.get_transform()
        loc = tf.location
        ego_pos = np.array([loc.x, loc.y, loc.z], dtype=np.float64)
        waypoint_route = self._route_planner.run_step(ego_pos)
        if not waypoint_route:
            return

        if len(waypoint_route) > 2:
            far_wp, cmd = waypoint_route[1]
            next_far_wp, next_cmd = waypoint_route[2]
        elif len(waypoint_route) > 1:
            far_wp, cmd = waypoint_route[1]
            next_far_wp, next_cmd = waypoint_route[1]
        else:
            far_wp, cmd = waypoint_route[0]
            next_far_wp, next_cmd = waypoint_route[0]

        far_cmd_int = int(getattr(cmd, "value", cmd))
        if not (1 <= far_cmd_int <= 6):
            return
        next_cmd_int = int(getattr(next_cmd, "value", next_cmd))
        if not (1 <= next_cmd_int <= 6):
            next_cmd_int = far_cmd_int

        yaw = _math.radians(tf.rotation.yaw)
        cos_y, sin_y = _math.cos(yaw), _math.sin(yaw)

        def _to_ego(wp_world):
            dx = float(wp_world[0]) - loc.x
            dy = float(wp_world[1]) - loc.y
            return np.array([dx * cos_y + dy * sin_y, -dx * sin_y + dy * cos_y], dtype=np.float32)

        ego_tp = _to_ego(far_wp)
        ego_next_tp = _to_ego(next_far_wp)
        self._target_points_ego = np.array([ego_tp, ego_next_tp], dtype=np.float32)
        dist = int(np.linalg.norm(ego_tp))

        prev_tmp = self._routing_last_command_tmp
        if prev_tmp != far_cmd_int:
            self._routing_last_command = prev_tmp
        self._routing_last_command_tmp = far_cmd_int
        if self._routing_last_command in (1, 2, 3) and far_cmd_int == 4:
            # Just cleared a manoeuvre: keep naming it (without distance) and let LANEFOLLOW become
            # the follow-on, exactly as simlingo's ``eval_route_as == 'command'`` branch does
            # (mirrored in ``ogbench/carla/carla.py``).
            self._current_routing_command = self._routing_last_command
            self._routing_include_distance = False
            self._routing_next_command = far_cmd_int
        else:
            self._current_routing_command = far_cmd_int
            self._routing_include_distance = far_cmd_int != 4
            self._routing_dist_to_waypoint = dist
            self._routing_next_command = next_cmd_int

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

    def _compute_scene_context(self) -> dict:
        """Return a dict describing nearby actors visible in the ego vehicle's FoV.

        FoV is approximated as ±55° of the forward direction (matching the 110°
        SimLingo camera) within 30 m.  Used by the language-feedback critic mode
        to ground corrective commentary in visible scene objects.
        """
        ctx: dict = {
            "vehicle_ahead": False,
            "vehicle_ahead_dist_m": -1.0,
            "pedestrian_in_fov": False,
            "pedestrian_dist_m": -1.0,
            "traffic_light_state": "none",
            "stop_sign_ahead": False,
        }
        try:
            ego = self._ego_actor()
            if ego is None:
                return ctx
            ev = self._evaluator
            world = ev.world
            ego_loc = ego.get_location()
            ego_tf = ego.get_transform()
            ego_fwd = ego_tf.get_forward_vector()

            fov_cos = math.cos(math.radians(55.0))  # ±55° half-angle
            look_ahead_m = 30.0

            actors = world.get_actors()

            # Vehicles in FoV
            min_veh_dist = float("inf")
            for v in actors.filter("*vehicle*"):
                if v.id == ego.id:
                    continue
                rel = v.get_location() - ego_loc
                dist = math.sqrt(rel.x ** 2 + rel.y ** 2)
                if dist > look_ahead_m or dist < 0.5:
                    continue
                cos_a = (rel.x * ego_fwd.x + rel.y * ego_fwd.y) / dist
                if cos_a < fov_cos:
                    continue
                ctx["vehicle_ahead"] = True
                if dist < min_veh_dist:
                    min_veh_dist = dist
                    ctx["vehicle_ahead_dist_m"] = float(dist)

            # Walkers / pedestrians in FoV
            min_ped_dist = float("inf")
            for w in actors.filter("*walker*"):
                rel = w.get_location() - ego_loc
                dist = math.sqrt(rel.x ** 2 + rel.y ** 2)
                if dist > look_ahead_m or dist < 0.5:
                    continue
                cos_a = (rel.x * ego_fwd.x + rel.y * ego_fwd.y) / dist
                if cos_a < fov_cos:
                    continue
                ctx["pedestrian_in_fov"] = True
                if dist < min_ped_dist:
                    min_ped_dist = dist
                    ctx["pedestrian_dist_m"] = float(dist)

            # Traffic light affecting ego
            try:
                tl = ego.get_traffic_light()
                if tl is not None:
                    state_str = str(tl.get_state())
                    if "Red" in state_str:
                        ctx["traffic_light_state"] = "red"
                    elif "Yellow" in state_str:
                        ctx["traffic_light_state"] = "yellow"
                    elif "Green" in state_str:
                        ctx["traffic_light_state"] = "green"
            except Exception:
                pass

            # Stop signs in FoV (within 15 m)
            for sign in actors.filter("*traffic.stop*"):
                rel = sign.get_location() - ego_loc
                dist = math.sqrt(rel.x ** 2 + rel.y ** 2)
                if dist > 15.0 or dist < 0.5:
                    continue
                cos_a = (rel.x * ego_fwd.x + rel.y * ego_fwd.y) / dist
                if cos_a < fov_cos:
                    continue
                ctx["stop_sign_ahead"] = True
                break
        except Exception:
            pass
        return ctx

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
                    # Prepend ego-frame origin so cum_d[0]=0 maps to the current
                    # position (0,0), not to the first route point (~2 m ahead).
                    # Without this, _expert_action_to_accel_steer extracts a speed
                    # of (dist_to_first_wp + v*dt)/dt instead of v, causing the
                    # expert to appear to drive far too fast and crash into leading
                    # vehicles even when the PDM-Lite planner has slowed down.
                    route_xy_from_ego = np.vstack(
                        [np.zeros((1, 2), dtype=np.float32), route_xy]
                    )
                    route_dist = np.linalg.norm(np.diff(route_xy_from_ego, axis=0), axis=1)
                    cum_d = np.concatenate(
                        [np.zeros(1, dtype=np.float32), np.cumsum(route_dist, dtype=np.float32)],
                        axis=0,
                    )
                    if float(cum_d[-1]) > 1e-3:
                        prev_xy = np.zeros(2, dtype=np.float32)
                        for i in range(action_horizon):
                            s = min(target_speed_live * dt * (i + 1), float(cum_d[-1]))
                            x_wp = float(np.interp(s, cum_d, route_xy_from_ego[:, 0]))
                            y_wp = float(np.interp(s, cum_d, route_xy_from_ego[:, 1]))
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
                return self._step_with_control(control, skip_expert_tick=True)
            except Exception as _ee:
                print(f"[step_expert] SimLingo autopilot failed: {_ee}", flush=True)
        expert_action = (obs_raw or {}).get("expert_action")
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
        route_progress_pct = self._route_completion_pct()
        route_progress_delta = max(0.0, route_progress_pct - self._prev_route_progress_pct)
        self._prev_route_progress_pct = route_progress_pct
        
        criteria = self._criteria_snapshot()
        progress_reward = self._progress_reward_weight * route_progress_delta / 100.0
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
        traffic_violation_count = self._traffic_violation_count()

        collision_delta = max(0, collision_count - self._prev_collision_count)
        outside_route_delta = max(0.0, outside_route_value - self._prev_outside_route_value)
        traffic_violation_delta = max(0, traffic_violation_count - self._prev_traffic_violation_count)
        self._prev_collision_count = collision_count
        self._prev_outside_route_value = outside_route_value
        self._prev_traffic_violation_count = traffic_violation_count

        collision_contact_active = self._raw_collision_active
        collision_penalty_active = collision_contact_active or (collision_delta > 0)
        if self._collision_contact_penalty is None:
            # Legacy mode: one coefficient applies whenever a new collision fires
            # or bounding boxes remain in contact.
            collision_pen = self._collision_event_penalty if collision_penalty_active else 0.0
        else:
            collision_pen = (
                self._collision_event_penalty * float(collision_delta > 0)
                + self._collision_contact_penalty * float(collision_contact_active)
            )
        outside_route_pen = self._outside_route_event_penalty * float(outside_route_delta)
        traffic_violation_pen = self._traffic_violation_penalty * float(traffic_violation_delta)
        reward += collision_pen + outside_route_pen + traffic_violation_pen

        terminal_bonus = 0.0
        info["collision_count"] = collision_count
        # Use the same route_progress_pct/_delta the progress reward above was computed
        # from, so the logged value always matches the reward term.
        info["route_progress_pct"] = route_progress_pct
        info["route_progress_delta"] = route_progress_delta
        _wall_dist = self.nearest_obstacle_distance_m()
        info["nearest_obstacle_distance_m"] = -1.0 if _wall_dist is None else float(_wall_dist)
        info["crash_stuck_ticks"] = self._crash_stuck_ticks
        info["collision_penalty_active"] = bool(collision_penalty_active)
        info["collision_contact_active"] = bool(collision_contact_active)
        info["outside_route_value"] = outside_route_value
        info["collision_delta"] = float(collision_delta)
        info["outside_route_delta"] = float(outside_route_delta)
        info["traffic_violation_count"] = traffic_violation_count
        info["traffic_violation_delta"] = float(traffic_violation_delta)
        info["lane_offset_m"] = lane_offset_m
        info["heading_error_rad"] = heading_error_rad
        info["lane_width_m"] = lane_width_m
        info["speed_limit_mps"] = speed_limit_mps
        info["route_progress_pct"] = float(route_progress_pct)
        info["route_progress_delta"] = float(route_progress_delta)
        # Absolute route geometry, so consumers can express progress in METRES rather than only
        # as a percentage. The routing-command plan (``_summarize_route_commands``) keys each
        # maneuver by ``start_distance_m``, so a percentage alone cannot be lined up against the
        # plan -- the VLM review needs "we are 84 m along a 210 m route, and the plan says the
        # left turn starts at 78 m" to tell whether the commanded maneuver was actually executed.
        info["route_total_distance_m"] = float(self._route_total_distance_m)
        info["route_distance_m"] = float(
            route_progress_pct / 100.0 * self._route_total_distance_m
        )
        # The routing command in force at this step, as the integer id and the human-readable
        # text used everywhere else (ROUTING_COMMAND_TEXT). Recorded per step so a divergence
        # from the plan -- going straight through a junction the plan says to turn left at --
        # can be attributed to the exact moment it happened.
        info["routing_command_id"] = int(self._current_routing_command)
        info["routing_command_text"] = ROUTING_COMMAND_TEXT.get(
            int(self._current_routing_command), "follow the road"
        )
        info["speed_norm"] = speed_norm
        info["overspeed_frac"] = overspeed_frac
        info["centering_factor"] = centering_factor
        info["heading_factor"] = heading_factor
        info["penalty_collision"] = collision_pen
        info["penalty_outside_route"] = outside_route_pen
        info["penalty_traffic_violation"] = traffic_violation_pen
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

        if not terminated and self._terminate_on_infraction:
            if collision_delta > 0 or outside_route_delta > 0 or traffic_violation_delta > 0:
                terminated = True
                info["termination_reason"] = "infraction"

        if terminated:
            if crash_stuck:
                self._finalize_route("Finished", "Agent crashed and got stuck")
            else:
                success = info["scenario_tree_status"] == "SUCCESS"
                terminal_bonus = self._success_bonus if success else self._failure_bonus
                reward += terminal_bonus
                info["success"] = success
                self._finalize_route("Finished", "")
            info["driving_score"] = self._last_driving_score

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
            # _finalize_route computes the leaderboard statistics; surface the composed driving
            # score here too so a step-capped episode reports it like any other termination.
            info["driving_score"] = self._last_driving_score
            terminated = True
        return float(reward), bool(terminated), info

    def _step_with_control(self, control, *, skip_expert_tick: bool = False):
        """Apply a pre-computed VehicleControl and run one leaderboard tick.

        ``skip_expert_tick``: set by step_expert(), whose caller already ran
        self._expert_agent.run_step() this tick to produce ``control`` -- ticking it
        again here would be a second, wasted (result-discarded) full planning pass.
        The background tick_expert() call otherwise keeps the expert's route/planner
        state warm while some other action source (the policy) is actually driving,
        for later expert takeover (e.g. auto_then_manual, expert_recover_debug).
        """
        self._last_control = control
        self.evaluator.manager.pending_control = control
        self._raw_collision_active = False
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
        if self._expert_agent is not None and not skip_expert_tick:
            self.tick_expert()
        return self._obs_dict(), float(reward), terminated, False, info

    def _obs_dict(self) -> Dict[str, np.ndarray]:
        sensors = getattr(self.evaluator.manager, "last_agent_input", None) or {}
        expert_action = self._compute_expert_action()
        commentary_text, language_label = self._compute_language_label(expert_action=expert_action)
        scene_context = self._compute_scene_context()
        rgb_viz = rgb_viz_from_leaderboard_dict(sensors)
        return {
            "state": self._get_state_vector(),
            "image": downscale_rgb_for_policy(rgb_viz),
            "image_viz": rgb_viz,
            "simlingo_image": _decode_simlingo_image(sensors),
            "language_label": language_label,
            "commentary_text": commentary_text,
            "expert_action": expert_action,
            "scene_context": scene_context,
            "routing_command": format_routing_command(
                self._current_routing_command,
                self._routing_next_command,
                dist_m=self._routing_dist_to_waypoint,
                include_distance=self._routing_include_distance,
            ),
            # Constant per episode: the full ordered list of routing commands over the route
            # (see :func:`_summarize_route_commands`). Lets downstream consumers show the
            # overall task, not just the current command.
            "route_command_plan": self._route_command_plan,
            "target_points": self._target_points_ego,
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

    def _traffic_violation_count(self) -> int:
        """Total count of RunningStopTest + RunningRedLightTest infractions so far."""
        scenario = getattr(self.evaluator, "route_scenario", None)
        if scenario is None:
            return 0
        count = 0
        for criterion in scenario.get_criteria():
            name = str(getattr(criterion, "name", ""))
            if name in ("RunningStopTest", "RunningRedLightTest"):
                try:
                    count += int(getattr(criterion, "actual_value", 0))
                except Exception:
                    pass
        return count

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

    def _traffic_violation_count(self) -> int:
        """Return total count of RunningStopTest + RunningRedLightTest infractions so far."""
        scenario = getattr(self.evaluator, "route_scenario", None)
        if scenario is None:
            return 0
        count = 0
        for criterion in scenario.get_criteria():
            name = str(getattr(criterion, "name", ""))
            if name in ("RunningStopTest", "RunningRedLightTest"):
                try:
                    count += int(getattr(criterion, "actual_value", 0))
                except Exception:
                    pass
        return count

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
                getattr(criterion, "test_status", getattr(criterion, "status", "")) or ""
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

    def _route_completion_percent(self, criteria: list[dict[str, Any]]) -> float:
        if not self._geometric_route_completion_only:
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
        # Latch: set on any new collision event, cleared only when the car gets back up
        # to speed (i.e. it has physically freed itself).  This avoids two failure modes:
        # (a) using raw _raw_collision_active alone: CARLA collision sensor fires on new
        #     contact events, not continuously, so a wedged car stops generating events
        #     and stuck_ticks never accumulates.
        # (b) using cumulative collision_count: a stop sign/traffic light after any prior
        #     collision would spuriously trigger crash_stuck.
        if self._raw_collision_active:
            self._collision_recently_active = True
        if speed >= self._crash_stuck_speed_threshold:
            self._collision_recently_active = False
        if self._collision_recently_active and speed < self._crash_stuck_speed_threshold:
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
            self._destroy_raw_collision_sensor()
            self._drain_pseudo_sensors()
            self.evaluator._cleanup()
            raise RuntimeError(f"Invalid sensors: {e}") from e
        except Exception:
            self._destroy_raw_collision_sensor()
            self._drain_pseudo_sensors()
            self.evaluator._cleanup()
            raise

        self._update_routing_command()
        return self._obs_dict(), self._info_with_sensors()

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        flat = np.asarray(action, dtype=np.float32).reshape(-1)
        control = None
        if self._steervla_decoder is not None:
            # Cleared so ``info["pid_debug"]`` below can only ever report *this* tick's PID call --
            # the legacy 2-D control path leaves the decoder untouched.
            self._steervla_decoder.last_debug = {}
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
        self._raw_collision_active = False

        try:
            running, tree_status = self.evaluator.manager.step_once()
        except AgentError as e:
            self._update_routing_command()
            self._finalize_route(*FAILURE_MESSAGES["Agent_runtime"])
            return (
                self._obs_dict(),
                -1.0, True, False,
                self._info_with_sensors({"error": "agent_runtime", "exception": str(e)}),
            )
        except TickRuntimeError as e:
            self._update_routing_command()
            self._finalize_route("Started", "TickRuntime")
            return (
                self._obs_dict(),
                -1.0, True, False,
                self._info_with_sensors({"error": "tick_runtime", "exception": str(e)}),
            )
        except Exception as e:
            self._update_routing_command()
            self._finalize_route(*FAILURE_MESSAGES["Simulation"])
            return (
                self._obs_dict(),
                -1.0, True, False,
                self._info_with_sensors({"error": "simulation", "exception": str(e)}),
            )

        self._update_routing_command()
        terminated = not running
        reward, terminated, info = self._compute_reward_and_info(
            tree_status=tree_status,
            terminated=terminated,
        )
        reward, terminated, info = self._apply_episode_max_steps(reward, terminated, info)
        if self._expert_agent is not None:
            self.tick_expert()
        if self._steervla_decoder is not None and self._steervla_decoder.last_debug:
            # PID internals for this tick (steer / desired vs actual speed / heading error /
            # lookahead distance). main_carla logs these as ``pid/*`` -- the lookahead and
            # heading-error traces are what distinguish a healthy chunk replay from a stale one.
            info["pid_debug"] = dict(self._steervla_decoder.last_debug)
        return self._obs_dict(), float(reward), terminated, False, info

    def checkpoint(self) -> Dict[str, Any]:
        """Snapshot every actor's kinematic state + reward bookkeeping for a cheap rollback point.

        Unlike ``reset()``, this does not touch the leaderboard's scenario tree /
        statistics manager -- it only records what's needed to teleport the physical
        world back and re-baseline reward deltas via :meth:`restore`. Cost is
        O(num actors), independent of how far into the episode this is called.

        Permanently switches this episode's route-completion reward to the geometric
        fallback (see ``_geometric_route_completion_only``) -- once any rollback happens,
        the leaderboard's monotonic completion criterion is no longer a trustworthy
        reward source for the rest of the episode.
        """
        self._geometric_route_completion_only = True
        world = self.evaluator.world
        actors: Dict[int, Tuple[Any, Any, Any]] = {}
        for actor in world.get_actors():
            if actor.type_id.startswith(("vehicle.", "walker.pedestrian.")):
                actors[actor.id] = (
                    actor.get_transform(), actor.get_velocity(), actor.get_angular_velocity(),
                )
        return {
            "actors": actors,
            "last_control": self._last_control,
            "route_completion_index": self._route_completion_index,
            # True snapshot (not re-baselined on restore, unlike the reward-delta
            # trackers below) -- crash_stuck termination needs to see genuine elapsed
            # stuck time across decision boundaries, or it can never reach
            # _crash_stuck_steps and the episode never ends. See restore().
            "crash_stuck_ticks": self._crash_stuck_ticks,
            "collision_recently_active": self._collision_recently_active,
        }

    def restore(self, ckpt: Dict[str, Any]) -> None:
        """Teleport actors back to a checkpoint and re-baseline reward deltas.

        Infraction/route-completion CRITERIA state inside the leaderboard's scenario
        tree is not (and can't cheaply be) rolled back -- only the physical world is.
        Delta-tracking attributes (collision/violation/route-completion counters used
        for the *reward*) are re-baselined to *current* criterion readings (taken
        immediately after the teleport) instead of restored to their pre-checkpoint
        values, so a candidate is only ever scored on what changes during its own
        trial, never charged for a sibling candidate's rollout.

        ``crash_stuck_ticks``/``collision_recently_active`` are the one exception:
        they drive *termination*, not reward, and restoring them to "fresh" (0/False)
        every call -- instead of back to the checkpoint's true saved values -- would
        mean a genuinely stuck episode can never accumulate enough consecutive stuck
        ticks to actually terminate, since every decision's checkpoint()/restore()
        would wipe its progress before it reaches ``_crash_stuck_steps``. Restoring
        them exactly still avoids cross-candidate contamination (every candidate at
        this decision point restores from the same saved snapshot), while correctly
        carrying real elapsed stuck-time forward across decisions.
        """
        world = self.evaluator.world
        for actor in world.get_actors():
            state = ckpt["actors"].get(actor.id)
            if state is None:
                continue
            transform, velocity, angular_velocity = state
            actor.set_transform(transform)
            actor.set_target_velocity(velocity)
            actor.set_target_angular_velocity(angular_velocity)
        world.tick()
        self._last_control = ckpt["last_control"]
        self._route_completion_index = ckpt["route_completion_index"]
        self._raw_collision_active = False
        self._collision_recently_active = ckpt["collision_recently_active"]
        self._crash_stuck_ticks = ckpt["crash_stuck_ticks"]
        self._prev_collision_count = self._collision_count()
        self._prev_outside_route_value, _ = self._route_infraction_values()
        self._prev_traffic_violation_count = self._traffic_violation_count()
        self._last_route_completion = self._route_completion_percent(self._criteria_snapshot())

    def _finalize_route(self, entry_status: str, crash_message: str) -> None:
        if not self._scenario_active:
            return
        self._destroy_raw_collision_sensor()
        config_index = self.evaluator.manager.route_index
        self._drain_pseudo_sensors()
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
            try:
                records = self.evaluator.statistics_manager._results.checkpoint.records
                self._last_driving_score = float(records[config_index].scores["score_composed"])
            except Exception:
                self._last_driving_score = 0.0
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

    def _spawn_raw_collision_sensor(self) -> None:
        """Attach a raw sensor.other.collision to the ego; sets _raw_collision_active each tick."""
        ego = self._ego_actor()
        if ego is None:
            return
        try:
            world = self.evaluator.world
            bp = world.get_blueprint_library().find("sensor.other.collision")
            sensor = world.spawn_actor(bp, carla.Transform(), attach_to=ego)
            sensor.listen(lambda _event: setattr(self, "_raw_collision_active", True))
            self._raw_collision_sensor = sensor
        except Exception as exc:
            print(f"[raw_collision_sensor] spawn failed: {exc}", flush=True)
            self._raw_collision_sensor = None

    def _destroy_raw_collision_sensor(self) -> None:
        """Stop and destroy the raw collision sensor before scenario teardown."""
        sensor = self._raw_collision_sensor
        self._raw_collision_sensor = None
        if sensor is None:
            return
        try:
            sensor.stop()
        except Exception:
            pass
        try:
            sensor.destroy()
        except Exception:
            pass

    def render(self):
        """Placeholder frame for evaluation video paths (avoid NotImplementedError)."""
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self) -> None:
        try:
            if self._evaluator is not None:
                # A UE4 crash leaves the CARLA Python client configured with the
                # normal (very large) RPC timeout. Calling the usual scenario
                # teardown in that state attempts to destroy each sensor over a
                # dead connection and can block for hours (e.g. 7,200,000 ms).
                # The evaluator owns the server subprocess, so its exit status is
                # a cheap, local liveness check. When it is known dead, skip all
                # CARLA RPC teardown and let _kill_carla_subprocesses() below reap
                # the remaining local processes. This lets the outer launcher
                # observe the crash and apply its normal retry/resume policy.
                server = getattr(self._evaluator, "server", None)
                server_dead = server is not None and server.poll() is not None
                if server_dead:
                    print(
                        "[carla] UE4 already exited; skipping RPC teardown to avoid a long client timeout.",
                        flush=True,
                    )
                else:
                    try:
                        self._stop_active_scenario()
                    finally:
                        try:
                            self._evaluator._reset_world_settings()
                        except Exception:
                            pass
        finally:
            self._kill_carla_subprocesses()
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
