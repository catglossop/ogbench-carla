"""Minimal leaderboard agent: registers a tiny sensor suite, never produces controls.

Use this as the leaderboard-side agent when you want the gym ``env.step(action)``
to drive the ego vehicle. The agent simply caches the most recent ``input_data``
on ``self.last_input_data`` so the gym wrapper can read it as the observation.

Sensors:
  - one front RGB camera at viz resolution (downscaled in the gym wrapper for RL/VLA)
  - GPS + IMU + Speedometer (zero-cost text sensors used by most leaderboard agents)

To swap in a real visual stack, subclass ``ObservationOnlyAgent.sensors``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track


RGB_FRONT_CAMERA_TAG = "rgb_front"
# CARLA renders at viz resolution; ``carla_utils`` downscales to policy resolution.
VIZ_CAMERA_HEIGHT = 288
VIZ_CAMERA_WIDTH = 512
VIZ_IMAGE_SHAPE_HWC = (VIZ_CAMERA_HEIGHT, VIZ_CAMERA_WIDTH, 3)
CAMERA_HEIGHT = 144
CAMERA_WIDTH = 256
IMAGE_SHAPE_HWC = (CAMERA_HEIGHT, CAMERA_WIDTH, 3)


def get_entry_point() -> str:
    return "ObservationOnlyAgent"


class ObservationOnlyAgent(AutonomousAgent):
    """A leaderboard agent that publishes sensors and stashes inputs but never drives.

    The gym wrapper applies its own VehicleControl after this agent has been called.
    """

    def setup(self, path_to_conf_file: Optional[str]) -> None:
        self.track = Track.SENSORS
        self.last_input_data: Dict[str, Any] = {}
        self.last_timestamp: float = 0.0

    def sensors(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "sensor.camera.rgb",
                "id": RGB_FRONT_CAMERA_TAG,
                "x": 0.7, "y": 0.0, "z": 1.6,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "width": VIZ_CAMERA_WIDTH,
                "height": VIZ_CAMERA_HEIGHT,
                "fov": 90,
            },
            {"type": "sensor.other.gnss", "id": "gps",
             "x": 0.7, "y": 0.0, "z": 1.6},
            {"type": "sensor.other.imu", "id": "imu",
             "x": 0.7, "y": 0.0, "z": 1.6, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            {"type": "sensor.speedometer", "id": "speed", "reading_frequency": 20},
        ]

    def run_step(self, input_data: Dict[str, Any], timestamp: float) -> carla.VehicleControl:
        self.last_input_data = input_data
        self.last_timestamp = timestamp
        return carla.VehicleControl(steer=0.0, throttle=0.0, brake=0.0)

    def destroy(self) -> None:
        self.last_input_data = {}
