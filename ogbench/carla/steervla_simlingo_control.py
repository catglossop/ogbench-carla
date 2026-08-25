"""Decode flattened SteerVLA trajectory chunks into CARLA ``VehicleControl``.

Mirrors ``simlingo/team_code/agent_steervla.py`` (run_step / control_pid /
interpolate_waypoints): cumulative deltas for speed waypoints (first two columns)
and route waypoints (last two columns), then longitudinal + lateral PID.

Supports:

* **normalized** chunks — apply OpenPI ``denormalize_actions`` (same scaling as
  :mod:`openpi.visualizing.steervla_visualization`) before cumsums.
* **policy_output** chunks — values already in physical units (meters / degrees) after
  OpenPI ``Unnormalize`` and fixed ``denormalize_actions`` scaling in the VLA actor.
* **normalized** chunks — apply only fixed ``denormalize_actions`` (legacy path when the
  actor returns raw model outputs without OpenPI ``Unnormalize``).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.interpolate import PchipInterpolator

import carla

from ogbench.carla.steervla_pid_utils import LateralPIDController, PIDController

# Must match :data:`ogbench.carla.carla_utils.EGO_STATE_IDX_SPEED`.
EGO_STATE_IDX_SPEED = 15

ActionInputSpace = Literal["normalized", "policy_output"]


def _denormalize_actions_local(
    actions: np.ndarray,
    action_dim: int,
    output_action_format: str | None = None,
) -> np.ndarray:
    """Mirror of ``openpi.visualizing.steervla_visualization.denormalize_actions``.

    openpi requires Python >= 3.11, but when CARLA runs in the 0.9.15 subprocess
    (``CARLA_0915_ROOT``) this module is imported under Python 3.10, where openpi
    cannot be installed. The scaling is pure numpy with no norm stats, so it is
    mirrored here. openpi stays the source of truth and is preferred when
    importable — keep the two in sync if the RLDS normalization changes.
    """
    actions = actions[..., :action_dim]

    if output_action_format in (
        "delta_speed_t_delta_course_t_delta_course_space",
        "DELTA_SPEED_T_DELTA_COURSE_T_DELTA_COURSE_SPACE",
    ):
        out = np.empty_like(actions)
        out[..., 0] = actions[..., 0] * 10.0
        out[..., 1] = actions[..., 1] * 180.0
        out[..., 2] = actions[..., 2] * 180.0
        return out

    if output_action_format in (
        "delta_xy_t_delta_xy_space",
        "DELTA_XY_T_DELTA_XY_SPACE",
    ):
        out = np.empty_like(actions)
        out[..., :2] = actions[..., :2] * 7.0
        out[..., 2:] = actions[..., 2:]
        return out

    if output_action_format in (
        "delta_xy_t_delta_course_space",
        "DELTA_XY_T_DELTA_COURSE_SPACE",
    ):
        out = np.empty_like(actions)
        out[..., :2] = actions[..., :2] * 7.0
        out[..., 2] = actions[..., 2] * 180.0
        return out

    # Default nuScenes format: [delta_speed/10, course/180, ...]
    out = np.empty_like(actions)
    out[..., 0] = actions[..., 0] * 10.0
    out[..., 1] = actions[..., 1] * 180.0
    if action_dim > 2:
        out[..., 2:] = actions[..., 2:] * 15.0
    return out


def _denormalize_action_chunk(
    chunks: np.ndarray,
    *,
    output_action_format: str,
    action_dim: int,
    space: ActionInputSpace,
) -> np.ndarray:
    if space == "policy_output":
        return np.asarray(chunks[..., :action_dim], dtype=np.float64)
    try:
        from openpi.visualizing.steervla_visualization import denormalize_actions
    except ImportError:
        denormalize_actions = _denormalize_actions_local
    ad = min(action_dim, int(np.asarray(chunks).shape[-1]))
    return np.asarray(
        denormalize_actions(np.asarray(chunks, dtype=np.float32), ad, output_action_format),
        dtype=np.float64,
    )


def _chunks_to_speed_and_route_waypoints(pred_action_chunks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same layout as ``agent_steervla.run_step`` after receiving VLA actions."""
    pred_speed_wps = np.cumsum(pred_action_chunks[:, :2], axis=0)
    pred_route = np.cumsum(
        np.concatenate([np.zeros((1, 2), dtype=pred_action_chunks.dtype), pred_action_chunks[:, 2:]], axis=0),
        axis=0,
    )
    return pred_speed_wps, pred_route


def interpolate_waypoints(waypoints: np.ndarray) -> np.ndarray:
    """Copied from ``simlingo/team_code/agent_steervla.py`` — spacing ~0.1 m."""
    waypoints = waypoints.copy()
    waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
    shift = np.roll(waypoints, 1, axis=0)
    shift[0] = shift[1]

    dists = np.linalg.norm(waypoints - shift, axis=1)
    dists = np.cumsum(dists)
    dists += np.arange(0, len(dists)) * 1e-4

    interp = PchipInterpolator(dists, waypoints, axis=0)

    x = np.arange(0.1, float(dists[-1]), 0.1)

    interp_points = interp(x)

    if interp_points.shape[0] == 0:
        interp_points = waypoints[None, -1]

    return interp_points


class SimlingoStyleWaypointDecoder:
    """Stateful PID controllers (matches leaderboard-style SteerVLA evaluation)."""

    def __init__(
        self,
        *,
        carla_fps: float = 20.0,
        wp_dilation: int = 1,
        data_save_freq: int = 5,
        brake_speed: float = 0.1,
        brake_ratio: float = 1.1,
        clip_delta: float = 1.0,
        clip_throttle: float = 1.0,
        speed_kp: float = 1.75,
        speed_ki: float = 1.0,
        speed_kd: float = 2.0,
        speed_n: int = 20,
        stuck_threshold: int = 800,
        creep_duration: int = 15,
        creep_throttle: float = 0.4,
    ) -> None:
        self.carla_fps = float(carla_fps)
        self.wp_dilation = int(wp_dilation)
        self.data_save_freq = int(data_save_freq)
        self.brake_speed = float(brake_speed)
        self.brake_ratio = float(brake_ratio)
        self.clip_delta = float(clip_delta)
        self.clip_throttle = float(clip_throttle)
        self.stuck_threshold = int(stuck_threshold)
        self.creep_duration = int(creep_duration)
        self.creep_throttle = float(creep_throttle)

        self.speed_controller = PIDController(
            k_p=speed_kp, k_i=speed_ki, k_d=speed_kd, n=speed_n
        )
        self.turn_controller = LateralPIDController(inference_mode=False)

        # Matches simlingo-rebuttal/team_code/agent_simlingo.py's stuck-detector: if the
        # PID has commanded the car to stay near-stopped for `stuck_threshold` consecutive
        # calls (regardless of *why* -- brake-ratio death-spiral, a degenerate near-zero
        # desired_speed prediction at standstill, etc.), force a `creep_throttle` burst for
        # `creep_duration` calls to physically break static friction. Deterministic safety
        # net underneath the stochastic flow-noise escape (see main_carla.py's
        # `vla_noise_scale`), which is not reliable on its own.
        self._stuck_counter = 0
        self._force_move = 0

        # Scalars from the most recent ``control_pid`` call, surfaced through the env's ``info``
        # dict as ``pid_debug`` so a run can be diagnosed without parsing the ``[RC-PID]`` prints.
        # ``heading_error`` is the one that exposes a stale (un-re-anchored) cached chunk: it pins
        # to a constant for the whole hold, because this decoder is handed only the ego *speed*
        # besides the waypoints and so cannot see the ego drifting off the plan.
        self.last_debug: dict[str, float] = {}

    def control_pid(
        self,
        route_waypoints: np.ndarray,
        velocity_scalar: float,
        speed_waypoints: np.ndarray,
    ) -> tuple[float, float, bool]:
        """``route_waypoints`` / ``speed_waypoints``: shape ``(N, 2)`` (ego-frame deltas cumulated)."""
        route_waypoints = np.asarray(route_waypoints, dtype=np.float64)
        speed_waypoints = np.asarray(speed_waypoints, dtype=np.float64)

        assert route_waypoints.ndim == 2 and route_waypoints.shape[1] == 2
        assert speed_waypoints.ndim == 2 and speed_waypoints.shape[1] == 2

        speed = float(velocity_scalar)

        one_second = int(self.carla_fps // (self.wp_dilation * self.data_save_freq))
        half_second = max(one_second // 2, 1)
        idx_hi = min(one_second - 2, speed_waypoints.shape[0] - 1)
        idx_lo = min(half_second - 2, speed_waypoints.shape[0] - 1)
        idx_hi = max(idx_hi, 0)
        idx_lo = max(idx_lo, 0)

        desired_speed = (
            np.linalg.norm(speed_waypoints[idx_hi] - speed_waypoints[idx_lo]) * 2.0
        )
        print(f"[RC-PID] Desired speed: {desired_speed:.4f}  Current speed: {speed:.4f}", flush=True)

        brake = (desired_speed < self.brake_speed) or (
            (speed / max(desired_speed, 1e-6)) > self.brake_ratio
        )

        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = float(np.clip(throttle, 0.0, self.clip_throttle))
        throttle = throttle if not brake else 0.0

        if speed < 0.1:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
        if self._stuck_counter > self.stuck_threshold:
            self._force_move = self.creep_duration
        forcing_move = self._force_move > 0
        if forcing_move:
            throttle = max(self.creep_throttle, throttle)
            brake = False
            self._force_move -= 1
            print(f"[RC-PID] force_move: {self._force_move}", flush=True)

        route_interp = interpolate_waypoints(route_waypoints.squeeze())
        steer = float(self.turn_controller.step(route_interp, speed))
        steer = float(np.clip(round(steer, 3), -1.0, 1.0))
        if forcing_move:
            # Best-of-N re-selects among diverse candidates every tick, even during
            # recovery, so route_waypoints (and thus steer) can swing tick to tick --
            # unlike impls/debug_raw_control.py's fixed steer=0, which reliably broke
            # the post-reset stiction in ~11 ticks. A whipping steer command at
            # near-zero speed can burn the forced throttle on lateral motion instead of
            # building forward momentum, so clamp steer small while forcing through.
            steer = float(np.clip(steer, -0.05, 0.05))
        print(f"[RC-PID] Steer: {steer:.4f}  Throttle: {throttle:.4f}  Brake: {brake}", flush=True)

        self.last_debug = {
            "steer": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
            "desired_speed": float(desired_speed),
            "speed": float(speed),
            "speed_error": float(desired_speed - speed),
            "heading_error": float(self.turn_controller.last_heading_error),
            "lookahead_m": float(self.turn_controller.last_lookahead_m),
            "route_len_m": float(0.1 * route_interp.shape[0]),
            "forcing_move": float(forcing_move),
        }

        return steer, throttle, brake

    def _flat_action_to_pid(
        self,
        action_flat: np.ndarray,
        *,
        state_vec: np.ndarray,
        output_action_format: str,
        action_horizon: int,
        action_dim: int,
        action_input_space: ActionInputSpace,
    ) -> tuple[float, float, bool]:
        """Shared decode: flat chunk -> waypoints -> PID ``(steer, throttle, brake)``."""
        flat = np.asarray(action_flat, dtype=np.float32).reshape(-1)
        expected = action_horizon * action_dim
        if flat.size != expected:
            raise ValueError(
                f"Expected flat action length {expected} (horizon={action_horizon} * dim={action_dim}), "
                f"got {flat.size}"
            )
        chunks = flat.reshape(action_horizon, action_dim)
        denorm = _denormalize_action_chunk(
            chunks,
            output_action_format=output_action_format,
            action_dim=action_dim,
            space=action_input_space,
        )
        pred_speed_wps, pred_route = _chunks_to_speed_and_route_waypoints(np.asarray(denorm, dtype=np.float64))
        print(
            f"[RC-PID] Model chunk raw: {np.array2string(chunks, precision=4, suppress_small=False)}",
            flush=True,
        )
        print(
            f"[RC-PID] Denorm chunk: {np.array2string(np.asarray(denorm), precision=4, suppress_small=False)}",
            flush=True,
        )
        print(
            f"[RC-PID] Speed wps: {np.array2string(np.asarray(pred_speed_wps), precision=4)}  "
            f"Route wps: {np.array2string(np.asarray(pred_route), precision=4)}",
            flush=True,
        )

        s = np.asarray(state_vec, dtype=np.float32).reshape(-1)
        gt_velocity = float(s[EGO_STATE_IDX_SPEED]) if s.size > EGO_STATE_IDX_SPEED else 0.0

        return self.control_pid(pred_route, gt_velocity, pred_speed_wps)

    def flat_action_to_vehicle_control(
        self,
        action_flat: np.ndarray,
        *,
        state_vec: np.ndarray,
        output_action_format: str,
        action_horizon: int,
        action_dim: int,
        action_input_space: ActionInputSpace,
    ) -> carla.VehicleControl:
        steer, throttle, brake = self._flat_action_to_pid(
            action_flat,
            state_vec=state_vec,
            output_action_format=output_action_format,
            action_horizon=action_horizon,
            action_dim=action_dim,
            action_input_space=action_input_space,
        )
        return carla.VehicleControl(steer=steer, throttle=throttle, brake=float(brake))

    def flat_action_to_accel_steer(
        self,
        action_flat: np.ndarray,
        *,
        state_vec: np.ndarray,
        output_action_format: str,
        action_horizon: int,
        action_dim: int,
        action_input_space: ActionInputSpace,
    ) -> np.ndarray:
        """Decode a flat chunk to a 2-D ``[accel, steer]`` action in ``[-1, 1]``.

        Exact inverse of :func:`ogbench.carla.carla_utils._action_to_control`:
        ``accel = throttle`` when not braking (throttle is 0 when braking), else
        ``accel = -1``. Lets a residual policy act in the same bounded 2-D control
        space the env executes, so the residual never perturbs the waypoint chunk.
        """
        steer, throttle, brake = self._flat_action_to_pid(
            action_flat,
            state_vec=state_vec,
            output_action_format=output_action_format,
            action_horizon=action_horizon,
            action_dim=action_dim,
            action_input_space=action_input_space,
        )
        accel = -1.0 if brake else float(throttle)
        return np.array([accel, steer], dtype=np.float32)


def maybe_steervla_vehicle_control(
    action: np.ndarray,
    *,
    state_vec: np.ndarray,
    exec_cfg: dict[str, Any] | None,
    decoder: SimlingoStyleWaypointDecoder | None,
) -> carla.VehicleControl | None:
    """If ``exec_cfg`` is set and action length matches chunk layout, return PID control; else ``None``."""
    if exec_cfg is None or decoder is None:
        return None
    fmt = str(exec_cfg.get("output_action_format", "DELTA_XY_T_DELTA_XY_SPACE"))
    ah = int(exec_cfg["action_horizon"])
    ad = int(exec_cfg["action_dim"])
    expected = ah * ad
    flat = np.asarray(action, dtype=np.float32).reshape(-1)
    space = str(exec_cfg.get("action_input_space", "normalized"))
    if flat.size == expected and space in ("normalized", "policy_output"):
        return decoder.flat_action_to_vehicle_control(
            flat,
            state_vec=state_vec,
            output_action_format=fmt,
            action_horizon=ah,
            action_dim=ad,
            action_input_space=space,  # type: ignore[arg-type]
        )
    return None
