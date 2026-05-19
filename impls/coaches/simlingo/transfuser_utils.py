from __future__ import annotations

import math

import carla
import numpy as np


def normalize_angle(x):
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi
    return x


def rotate_point(point, angle):
    x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
    y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
    return carla.Vector3D(x_, y_, point.z)


def get_traffic_light_waypoints(traffic_light, carla_map):
    base_transform = traffic_light.get_transform()
    base_loc = traffic_light.get_location()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)

    area_ext = traffic_light.trigger_volume.extent
    x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0)

    area = []
    for x in x_values:
        point = rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
        point_location = area_loc + carla.Location(x=point.x, y=point.y)
        area.append(point_location)

    ini_wps = []
    for pt in area:
        wpx = carla_map.get_waypoint(pt)
        if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
            ini_wps.append(wpx)

    wps = []
    eu_wps = []
    for wpx in ini_wps:
        distance_to_light = base_loc.distance(wpx.transform.location)
        eu_wps.append(wpx)
        next_distance_to_light = distance_to_light + 1.0
        while not wpx.is_intersection:
            next_wp = wpx.next(0.5)[0]
            next_distance_to_light = base_loc.distance(next_wp.transform.location)
            if next_wp and not next_wp.is_intersection and next_distance_to_light <= distance_to_light:
                eu_wps.append(next_wp)
                distance_to_light = next_distance_to_light
                wpx = next_wp
            else:
                break
        if not next_distance_to_light <= distance_to_light and len(eu_wps) >= 4:
            wps.append(eu_wps[-4])
        else:
            wps.append(wpx)

    return area_loc, wps


def inverse_conversion_2d(point, translation, yaw):
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return rotation_matrix.T @ (point - translation)


def preprocess_compass(compass):
    if math.isnan(compass):
        compass = 0.0
    return normalize_angle(compass - np.deg2rad(90.0))
