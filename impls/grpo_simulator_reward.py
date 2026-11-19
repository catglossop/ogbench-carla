"""Read-only CARLA scores for candidate SimLingo waypoint chunks.

Each component is normalized to roughly [-1, 1]; progress, speed efficiency, and heading alignment
are positive, while deviation, risk, and roughness components are non-positive. Weights therefore
remain comparable and are directly configurable in ``steervla_grpo_config.py``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ogbench.carla.steervla_simlingo_control import (
    _chunks_to_speed_and_route_waypoints,
    _denormalize_action_chunk,
)


_DEFAULT_WEIGHTS = {
    "route_progress": 1.0,
    "route_deviation": 0.75,
    "lane_deviation": 0.25,
    "heading_alignment": 0.25,
    "speed_efficiency": 0.1,
    "speed_limit": 0.5,
    "vehicle_collision": 3.0,
    "pedestrian_collision": 6.0,
    "static_obstacle": 3.0,
    "red_light": 2.0,
    "stop_sign": 1.5,
    "acceleration_smoothness": 0.1,
    "jerk_smoothness": 0.05,
    "path_smoothness": 0.1,
}


def _project_path(points: np.ndarray, route_xy: np.ndarray, route_s: np.ndarray):
    segments = route_xy[1:] - route_xy[:-1]
    lengths_sq = np.maximum(np.sum(segments * segments, axis=1), 1e-8)
    lengths = np.sqrt(lengths_sq)
    distances, progress, indices = [], [], []
    for point in points:
        rel = point[None, :] - route_xy[:-1]
        frac = np.clip(np.sum(rel * segments, axis=1) / lengths_sq, 0.0, 1.0)
        projection = route_xy[:-1] + frac[:, None] * segments
        dist_sq = np.sum((point[None, :] - projection) ** 2, axis=1)
        index = int(np.argmin(dist_sq))
        distances.append(float(np.sqrt(dist_sq[index])))
        progress.append(float(route_s[index] + frac[index] * lengths[index]))
        indices.append(index)
    return np.asarray(distances), np.asarray(progress), np.asarray(indices), segments, lengths


def _collision_risk(points, times, actors, margin: float) -> float:
    risk = 0.0
    for point, time_s in zip(points, times):
        for actor_xy, actor_v, radius, _category in actors:
            distance = float(np.linalg.norm(point - (actor_xy + actor_v * time_s)))
            clearance = distance - radius - 1.25  # ego footprint approximation
            risk = max(risk, float(np.clip((margin - clearance) / margin, 0.0, 1.0)))
    return risk


def score_candidate_chunks(
    env: Any,
    chunks: np.ndarray,
    exec_cfg: dict,
    reward_cfg: dict,
) -> dict[str, np.ndarray]:
    """Score waypoint plans from one read-only CARLA scene snapshot; never ticks or steps CARLA."""
    import carla

    chunks = np.asarray(chunks, dtype=np.float32)
    if chunks.ndim != 2 or chunks.shape[0] == 0 or chunks.shape[1] == 0 or chunks.shape[1] % 4:
        raise ValueError(f"Expected candidate chunks shaped (N, H*4), got {chunks.shape}.")
    collision_margin = float(reward_cfg.get("collision_margin_m", 3.0))
    if not math.isfinite(collision_margin) or collision_margin <= 0:
        raise ValueError("simulator_reward.collision_margin_m must be finite and greater than zero.")
    ego = env._ego_actor()
    route_xyz, route_s = env._route_progress_xyz, env._route_progress_s
    if ego is None or route_xyz is None or route_s is None or len(route_xyz) < 2:
        raise RuntimeError("Simulator GRPO scoring requires a live ego and route geometry.")

    horizon = chunks.shape[1] // 4
    dt = max(float(reward_cfg.get("waypoint_dt_s", 0.25)), 1e-3)
    physical = _denormalize_action_chunk(
        chunks.reshape(len(chunks), horizon, 4),
        output_action_format=str(exec_cfg.get("output_action_format", "DELTA_XY_T_DELTA_XY_SPACE")),
        action_dim=4,
        space=str(exec_cfg.get("action_input_space", "normalized")),
    )
    ego_tf = ego.get_transform()
    yaw = math.radians(float(ego_tf.rotation.yaw))
    c, s = math.cos(yaw), math.sin(yaw)
    origin = np.asarray([ego_tf.location.x, ego_tf.location.y], dtype=np.float64)
    current_idx = min(max(int(env._route_progress_index), 0), len(route_xyz) - 2)
    current_s = float(route_s[current_idx])
    lookahead_m = max(float(reward_cfg.get("route_lookahead_m", 60.0)), 1.0)
    end = min(len(route_xyz) - 1, int(np.searchsorted(route_s, current_s + lookahead_m, side="right")))
    end = max(end, current_idx + 1)
    start = max(0, current_idx - 1)
    route_xy = np.asarray(route_xyz[start : end + 1, :2], dtype=np.float64)
    route_distances = np.asarray(route_s[start : end + 1], dtype=np.float64)
    map_obj = env.evaluator.world.get_map()
    speed_limit_now = float(env._lane_alignment_metrics()["speed_limit_mps"])
    speed_tree_available = env._speed_limit_tree is not None
    world = env.evaluator.world

    # Snapshot nearby actors once. Static props use XY distance because some route obstacles
    # are staged at a different Z until their scenario trigger moves them onto the road.
    actor_states = []
    stop_signs = []
    actor_radius_m = max(float(reward_cfg.get("actor_query_radius_m", 60.0)), 1.0)
    for actor in world.get_actors():
        type_id = str(actor.type_id).lower()
        if actor.id == ego.id or type_id.startswith(("sensor.", "controller.")):
            continue
        is_vehicle = type_id.startswith("vehicle.")
        is_pedestrian = type_id.startswith(("walker.", "pedestrian."))
        is_static = any(tag in type_id for tag in ("static.prop", "wall", "barrier", "roadblock", "obstacle"))
        is_stop_sign = "stop" in type_id and "traffic" in type_id
        if not (is_vehicle or is_pedestrian or is_static or is_stop_sign):
            continue
        try:
            transform = actor.get_transform()
            actor_xy = np.asarray([transform.location.x, transform.location.y], dtype=np.float64)
            if np.linalg.norm(actor_xy - origin) > actor_radius_m:
                continue
            ext = actor.bounding_box.extent
            radius = max(0.4, math.hypot(float(ext.x), float(ext.y)))
            if is_stop_sign:
                stop_signs.append(actor_xy)
            if is_vehicle:
                velocity = actor.get_velocity()
                actor_states.append((actor_xy, np.asarray([velocity.x, velocity.y]), radius, "vehicle"))
            elif is_pedestrian:
                velocity = actor.get_velocity()
                actor_states.append((actor_xy, np.asarray([velocity.x, velocity.y]), radius, "pedestrian"))
            elif is_static:
                actor_states.append((actor_xy, np.zeros(2, dtype=np.float64), radius, "static"))
        except Exception:
            continue

    components: dict[str, list[float]] = {key: [] for key in _DEFAULT_WEIGHTS}
    times = dt * np.arange(1, horizon + 1, dtype=np.float64)
    horizon_m = max(speed_limit_now * dt * horizon, 1.0)
    red_light_trigger = None
    try:
        light = ego.get_traffic_light()
        if light is not None and "red" in str(ego.get_traffic_light_state()).lower():
            trigger = light.trigger_volume.location
            light.get_transform().transform(trigger)
            red_light_trigger = np.asarray([trigger.x, trigger.y], dtype=np.float64)
    except Exception:
        pass

    for candidate in physical:
        speed_wps, route_wps = _chunks_to_speed_and_route_waypoints(candidate)
        local = route_wps[1:, :2]
        points = np.empty_like(local, dtype=np.float64)
        points[:, 0] = origin[0] + c * local[:, 0] - s * local[:, 1]
        points[:, 1] = origin[1] + s * local[:, 0] + c * local[:, 1]
        route_error, projected_s, route_segment_ids, route_segments, _ = _project_path(
            points, route_xy, route_distances
        )

        components["route_progress"].append(float(np.clip((projected_s[-1] - current_s) / horizon_m, 0.0, 1.0)))
        components["route_deviation"].append(-float(np.clip(np.mean(route_error) / 1.75, 0.0, 2.0)) / 2.0)

        lane_errors = []
        for point in points:
            location = carla.Location(x=float(point[0]), y=float(point[1]), z=float(ego_tf.location.z))
            waypoint = map_obj.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if waypoint is None:
                lane_errors.append(3.5)
                continue
            lane_tf = waypoint.transform
            dx, dy = point[0] - lane_tf.location.x, point[1] - lane_tf.location.y
            lane_yaw = math.radians(float(lane_tf.rotation.yaw))
            lateral = -math.sin(lane_yaw) * dx + math.cos(lane_yaw) * dy
            lane_errors.append(abs(float(lateral)) / max(float(waypoint.lane_width) * 0.5, 0.5))
        components["lane_deviation"].append(-float(np.clip(np.mean(lane_errors), 0.0, 2.0)) / 2.0)

        path = np.vstack([origin, points])
        path_heading = np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0]))
        route_heading = np.arctan2(
            route_segments[route_segment_ids, 1], route_segments[route_segment_ids, 0]
        )
        heading_error = np.arctan2(np.sin(path_heading - route_heading), np.cos(path_heading - route_heading))
        components["heading_alignment"].append(float(np.mean((1.0 + np.cos(heading_error)) * 0.5)))

        speed_delta = np.diff(np.vstack([np.zeros((1, 2)), speed_wps]), axis=0)
        speeds = np.linalg.norm(speed_delta, axis=1) / dt
        limits = []
        if speed_tree_available:
            for point in points:
                location = carla.Location(x=float(point[0]), y=float(point[1]), z=float(ego_tf.location.z))
                limits.append(float(env._lookup_speed_limit(location) or speed_limit_now))
        else:
            limits = [speed_limit_now] * horizon
        limits = np.maximum(np.asarray(limits), 1.0)
        speed_ratio = speeds / limits
        components["speed_efficiency"].append(float(np.mean(np.clip(speed_ratio, 0.0, 1.0))))
        components["speed_limit"].append(-float(np.mean(np.clip(speed_ratio - 1.0, 0.0, 1.0))))

        for category in ("vehicle", "pedestrian", "static"):
            actors = [item for item in actor_states if item[3] == category]
            components[f"{category}_collision" if category != "static" else "static_obstacle"].append(
                -_collision_risk(points, times, actors, collision_margin)
            )

        if red_light_trigger is None:
            red_risk = 0.0
        else:
            near_signal = np.linalg.norm(points - red_light_trigger[None, :], axis=1) < float(
                reward_cfg.get("traffic_control_radius_m", 5.0)
            )
            red_risk = float(np.any(near_signal & (speeds > float(reward_cfg.get("stop_speed_mps", 0.5)))))
        components["red_light"].append(-red_risk)

        stop_risk = 0.0
        for sign_xy in stop_signs:
            near_sign = np.linalg.norm(points - sign_xy[None, :], axis=1) < float(
                reward_cfg.get("traffic_control_radius_m", 5.0)
            )
            if np.any(near_sign & (speeds > float(reward_cfg.get("stop_speed_mps", 0.5)))):
                stop_risk = 1.0
                break
        components["stop_sign"].append(-stop_risk)

        acceleration = np.diff(speeds) / dt
        jerk = np.diff(acceleration) / dt
        components["acceleration_smoothness"].append(-float(np.mean(np.clip(np.abs(acceleration) / 4.0, 0.0, 1.0))) if len(acceleration) else 0.0)
        components["jerk_smoothness"].append(-float(np.mean(np.clip(np.abs(jerk) / 8.0, 0.0, 1.0))) if len(jerk) else 0.0)
        turn = np.diff(path_heading)
        turn = np.arctan2(np.sin(turn), np.cos(turn))
        components["path_smoothness"].append(-float(np.mean(np.clip(np.abs(turn) / 0.5, 0.0, 1.0))) if len(turn) else 0.0)

    result = {key: np.asarray(values, dtype=np.float32) for key, values in components.items()}
    weights = dict(_DEFAULT_WEIGHTS)
    configured_weights = dict(reward_cfg.get("weights", {}))
    unknown = set(configured_weights) - set(_DEFAULT_WEIGHTS)
    if unknown:
        raise ValueError(f"Unknown simulator reward weight(s): {', '.join(sorted(unknown))}.")
    weights.update(configured_weights)
    if not all(math.isfinite(float(weight)) for weight in weights.values()):
        raise ValueError("Simulator reward weights must all be finite numbers.")
    result["scores"] = sum(float(weights[key]) * values for key, values in result.items())
    if not np.all(np.isfinite(result["scores"])):
        raise ValueError("Simulator GRPO scorer produced non-finite candidate scores.")
    return result
