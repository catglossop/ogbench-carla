"""Project ego-frame route/speed waypoints onto rollout camera frames (SimLingo-style)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ogbench.carla.steervla_simlingo_control import (
    _chunks_to_speed_and_route_waypoints,
    _denormalize_action_chunk,
    ActionInputSpace,
)


def get_camera_intrinsics(width: int, height: int, fov_deg: float = 110.0) -> np.ndarray:
    """Pinhole intrinsics for a CARLA RGB camera (matches SimLingo ``simlingo_utils``)."""
    w, h = int(width), int(height)
    focal = w / (2.0 * np.tan(float(fov_deg) * np.pi / 360.0))
    k = np.identity(3, dtype=np.float64)
    k[0, 0] = k[1, 1] = focal
    k[0, 2] = w / 2.0
    k[1, 2] = h / 2.0
    return k


def project_points(
    points_xy: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    tvec: np.ndarray | None = None,
    rvec: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    """Project ego-frame (forward, lateral) waypoints to image pixels."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if rvec is None:
        rvec_cv = np.zeros((3, 1), dtype=np.float32)
    else:
        rvec_cv = np.array([[-rvec[1], rvec[2], rvec[0]]], dtype=np.float32)
    if tvec is None:
        tvec_cv = np.array([[0.0, 2.0, 1.5]], dtype=np.float32)
    else:
        tvec_cv = np.asarray(tvec, dtype=np.float32).reshape(1, 3)

    dist = np.zeros((5, 1), dtype=np.float32)
    out: list[tuple[float, float]] = []
    for point in pts:
        pos_3d = np.array([[point[1], 0.0, point[0] + float(tvec_cv[0, 2])]], dtype=np.float64)
        pix, _ = cv2.projectPoints(
            pos_3d,
            rvec=rvec_cv,
            tvec=tvec_cv,
            cameraMatrix=np.asarray(camera_matrix, dtype=np.float64),
            distCoeffs=dist,
        )
        out.append((float(pix[0, 0, 0]), float(pix[0, 0, 1])))
    return out


def flat_action_to_waypoints(
    action_flat: np.ndarray,
    *,
    action_horizon: int,
    action_dim: int,
    output_action_format: str,
    action_input_space: ActionInputSpace,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pred_speed_wps, pred_route)`` each ``(H, 2)`` in ego frame."""
    flat = np.asarray(action_flat, dtype=np.float32).reshape(-1)
    expected = int(action_horizon) * int(action_dim)
    if flat.size != expected:
        raise ValueError(f"Expected action length {expected}, got {flat.size}")
    chunks = flat.reshape(int(action_horizon), int(action_dim))
    denorm = _denormalize_action_chunk(
        chunks,
        output_action_format=output_action_format,
        action_dim=int(action_dim),
        space=action_input_space,
    )
    return _chunks_to_speed_and_route_waypoints(np.asarray(denorm, dtype=np.float64))


def annotate_waypoints_on_frame(
    frame: np.ndarray,
    *,
    action_flat: np.ndarray | None,
    exec_cfg: dict[str, Any] | None,
    target_points: np.ndarray | None = None,
    camera_fov_deg: float = 110.0,
) -> np.ndarray:
    """Draw target (blue), route (red), and speed (green) waypoints on an RGB ``uint8`` frame."""
    out = np.array(frame, copy=True)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)

    h, w = out.shape[:2]
    k = get_camera_intrinsics(w, h, camera_fov_deg)

    def _draw_points(points: np.ndarray | None, color_rgb: tuple[int, int, int], radius: int) -> None:
        if points is None:
            return
        arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if arr.size == 0:
            return
        for u, v in project_points(arr, k):
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < w and 0 <= vi < h:
                cv2.circle(out, (ui, vi), radius, color_rgb, thickness=-1, lineType=cv2.LINE_AA)

    if target_points is not None:
        _draw_points(np.asarray(target_points, dtype=np.float64).reshape(-1, 2), (0, 0, 255), 4)

    if action_flat is not None and exec_cfg is not None:
        try:
            pred_speed, pred_route = flat_action_to_waypoints(
                action_flat,
                action_horizon=int(exec_cfg["action_horizon"]),
                action_dim=int(exec_cfg["action_dim"]),
                output_action_format=str(exec_cfg.get("output_action_format", "DELTA_XY_T_DELTA_XY_SPACE")),
                action_input_space=str(exec_cfg.get("action_input_space", "normalized")),  # type: ignore[arg-type]
            )
            _draw_points(pred_route, (255, 0, 0), 3)
            _draw_points(pred_speed, (0, 255, 0), 2)
        except Exception:
            pass

    return out
