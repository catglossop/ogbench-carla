from __future__ import annotations

from leaderboard.utils.route_manipulation import downsample_route

from .autonomous_agent import Track


class AutonomousAgent:
    def __init__(self, path_to_conf_file, route_index=None):
        self.track = Track.SENSORS
        self._global_plan = None
        self._global_plan_world_coord = None
        self.org_dense_route_gps = None
        self.org_dense_route_world_coord = None
        self.wallclock_t0 = None

    @staticmethod
    def get_ros_version():
        return -1

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        self.org_dense_route_gps = global_plan_gps
        self.org_dense_route_world_coord = global_plan_world_coord
        ds_ids = downsample_route(global_plan_world_coord, 200)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]

