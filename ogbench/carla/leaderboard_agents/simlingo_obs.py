"""Leaderboard agent for SimLingo rollouts: registers SimLingo's exact camera spec.

SimLingo was trained with a single front-facing RGB camera at 1024×512, FOV=110°,
mounted at (-1.5, 0, 2.0).  This agent reproduces that exact setup so that the
images fed to the frozen SimLingo VLM match the training distribution.

The gym wrapper reads ``rgb_simlingo`` from ``info["sensors"]`` and passes it
to ``SimLingoBase.get_action_and_features()`` unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track


SIMLINGO_CAMERA_TAG = "rgb_simlingo"
SIMLINGO_CAMERA_WIDTH = 1024
SIMLINGO_CAMERA_HEIGHT = 512
SIMLINGO_IMAGE_SHAPE_HWC = (SIMLINGO_CAMERA_HEIGHT, SIMLINGO_CAMERA_WIDTH, 3)


def get_entry_point() -> str:
    return "SimLingoObsAgent"


class SimLingoObsAgent(AutonomousAgent):
    """Leaderboard agent that registers SimLingo's training camera and stashes inputs."""

    def setup(self, path_to_conf_file: Optional[str]) -> None:
        self.track = Track.SENSORS
        self.last_input_data: Dict[str, Any] = {}
        self.last_timestamp: float = 0.0

    def sensors(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "sensor.camera.rgb",
                "id": SIMLINGO_CAMERA_TAG,
                # SimLingo training camera position (from GlobalConfig.camera_pos_0)
                "x": -1.5, "y": 0.0, "z": 2.0,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "width": SIMLINGO_CAMERA_WIDTH,
                "height": SIMLINGO_CAMERA_HEIGHT,
                "fov": 110,
            },
            {"type": "sensor.other.gnss", "id": "gps",
             "x": 0.0, "y": 0.0, "z": 0.0},
            {"type": "sensor.other.imu", "id": "imu",
             "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            {"type": "sensor.speedometer", "id": "speed", "reading_frequency": 20},
        ]

    def run_step(self, input_data: Dict[str, Any], timestamp: float) -> carla.VehicleControl:
        self.last_input_data = input_data
        self.last_timestamp = timestamp
        return carla.VehicleControl(steer=0.0, throttle=0.0, brake=0.0)

    def destroy(self) -> None:
        self.last_input_data = {}
