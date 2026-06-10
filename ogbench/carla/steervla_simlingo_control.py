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
    except ImportError as exc:
        raise ImportError(
            "action_input_space='normalized' requires openpi with "
            "openpi.visualizing.steervla_visualization.denormalize_actions"
        ) from exc
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
        brake_speed: float = 0.4,
        brake_ratio: float = 1.1,
        clip_delta: float = 1.0,
        clip_throttle: float = 1.0,
        speed_kp: float = 1.75,
        speed_ki: float = 1.0,
        speed_kd: float = 2.0,
        speed_n: int = 20,
    ) -> None:
        self.carla_fps = float(carla_fps)
        self.wp_dilation = int(wp_dilation)
        self.data_save_freq = int(data_save_freq)
        self.brake_speed = float(brake_speed)
        self.brake_ratio = float(brake_ratio)
        self.clip_delta = float(clip_delta)
        self.clip_throttle = float(clip_throttle)

        self.speed_controller = PIDController(
            k_p=speed_kp, k_i=speed_ki, k_d=speed_kd, n=speed_n
        )
        self.turn_controller = LateralPIDController(inference_mode=False)

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

        brake = (desired_speed < self.brake_speed) or (
            (speed / max(desired_speed, 1e-6)) > self.brake_ratio
        )

        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = float(np.clip(throttle, 0.0, self.clip_throttle))
        throttle = throttle if not brake else 0.0

        route_interp = interpolate_waypoints(route_waypoints.squeeze())
        steer = float(self.turn_controller.step(route_interp, speed))
        steer = float(np.clip(round(steer, 3), -1.0, 1.0))

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
        """Decode a flat chunk and run the stateful PIDs; returns ``(steer, throttle, brake)``."""
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
        ``accel = -1``. Lets a residual policy act in the same bounded 2-D space
        as the torch SimLingo residual SAC (``main_carla_simlingo.py``).
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
