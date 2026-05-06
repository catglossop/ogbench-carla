"""Self-contained longitudinal / lateral PID helpers for SteerVLA waypoint control.

Logic matches ``simlingo/team_code/transfuser_utils.py`` (speed PID) and
``simlingo/team_code/nav_planner.py`` (:class:`LateralPIDController`), adapted from
CarLA Garage–style leaderboard code so Bench2Drive does **not** need ``team_code``
on ``PYTHONPATH``.

Original CARLA Garage code referenced above is MIT-licensed; Simlingo builds on it.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class PIDController:
    """PID controller (same behavior as ``transfuser_utils.PIDController``)."""

    def __init__(self, k_p: float = 1.0, k_i: float = 0.0, k_d: float = 0.0, n: int = 20):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.window: deque[float] = deque([0.0 for _ in range(n)], maxlen=n)

    def step(self, error: float) -> float:
        self.window.append(float(error))
        if len(self.window) >= 2:
            integral = float(np.mean(self.window))
            derivative = float(self.window[-1] - self.window[-2])
        else:
            integral = 0.0
            derivative = 0.0
        return self.k_p * error + self.k_i * integral + self.k_d * derivative


class LateralPIDController:
    """Lateral steering PID (same behavior as ``nav_planner.LateralPIDController``)."""

    def __init__(
        self,
        k_p: float = 3.118357247806046,
        k_d: float = 1.3782508892109167,
        k_i: float = 0.6406067986034124,
        speed_scale: float = 0.9755321901954155,
        speed_offset: float = 1.9152884533402488,
        default_lookahead: int = 24,
        speed_threshold: float = 23.150102938235136,
        n: int = 6,
        inference_mode: bool = False,
    ):
        self.k_p = k_p
        self.k_d = k_d
        self.k_i = k_i
        self.speed_scale = speed_scale
        self.speed_offset = speed_offset
        self.default_lookahead = default_lookahead
        self.speed_threshold = speed_threshold
        self.n = n
        self.inference_mode = inference_mode

        self._saved_window: list[float] = []
        self._window: list[float] = []

    def step(self, route_np: np.ndarray, current_speed: float) -> float:
        current_speed = current_speed * 3.6
        if self.inference_mode:
            n_lookahead = np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 105) / 10
            n_lookahead = n_lookahead - 2
            n_lookahead = int(min(n_lookahead, route_np.shape[0] - 1))
        else:
            n_lookahead = int(
                min(np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 105), route_np.shape[0] - 1)
            )

        n_lookahead = min(n_lookahead, len(route_np) - 1)
        desired_heading_vec = route_np[n_lookahead]

        yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
        heading_error = yaw_path % (2 * np.pi)
        heading_error = heading_error if heading_error < np.pi else heading_error - 2 * np.pi

        heading_error = heading_error * 180.0 / np.pi / 90.0

        self._window.append(float(heading_error))
        self._window = self._window[-self.n :]

        derivative = 0.0 if len(self._window) == 1 else self._window[-1] - self._window[-2]
        integral = float(np.mean(self._window))

        steering = float(np.clip(self.k_p * heading_error + self.k_d * derivative + self.k_i * integral, -1.0, 1.0))
        return steering

    def save(self) -> None:
        self._saved_window = self._window.copy()

    def load(self) -> None:
        self._window = self._saved_window.copy()
