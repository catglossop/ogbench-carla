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
# Raised 2026-08-20 to SimLingo's camera RESOLUTION (1024x512) because traffic lights were only a
# few pixels across in the window video the Gemini CAST coach reviews, then completed 2026-08-21 to
# SimLingo's FULL camera -- ``fov``/``x``/``z`` in :meth:`ObservationOnlyAgent.sensors` now match
# ``simlingo_obs.py`` (fov=110, x=-1.5, z=2.0) as well. The two agents therefore render the same
# view; they differ only in that this one downscales a policy copy.
#
# Why the extrinsics had to follow the resolution: the SteerVLA checkpoint is fine-tuned on SimLingo
# frames, so every pixel it has ever seen came from that mount. Matching 1024x512 while keeping
# fov=90 at (0.7, 1.6) produced a correctly-shaped frame of the wrong scene -- narrower, further
# forward, lower. ``fov`` is HORIZONTAL in CARLA, so at 2:1 the vertical field of view goes from
# ~53.1 deg (fov 90) to ~71.1 deg (fov 110); overhead traffic lights sit high in frame and that is
# most of what the extra vertical view buys.
#
# This changes the observation distribution for everything downstream, not just the VLA: replay
# buffers, critic checkpoints and CAST videos collected before 2026-08-21 came from the old mount.
#
# ONE resize for the VLA, NONE for the VLM:
#   * VLM (Gemini CAST coach) reads ``image_viz``, which ``rgb_viz_from_leaderboard_dict`` leaves
#     untouched whenever the sensor already renders at ``VIZ_IMAGE_SHAPE_HWC`` -- so it sees the
#     native 1024x512 frame with no resampling at all.
#   * VLA (Pi0-CoT) reads ``image``, which is ``IMAGE_SHAPE_HWC`` == the model's own 224x224 input.
#     ``downscale_rgb_for_policy`` performs that single squeeze straight from the native frame, and
#     ``vlas.steervla._resize_hl_image`` then becomes a no-op (already 224x224). Previously this was
#     TWO lossy steps (native -> 256x144 -> stretched 224x224); collapsing them keeps small, thin
#     objects far better, which is the whole point of the resolution bump.
VIZ_CAMERA_HEIGHT = 512
VIZ_CAMERA_WIDTH = 1024
VIZ_IMAGE_SHAPE_HWC = (VIZ_CAMERA_HEIGHT, VIZ_CAMERA_WIDTH, 3)
# == vlas.steervla._HL_IMAGE_HW. Keep these two in sync: the point is that no second resize happens.
CAMERA_HEIGHT = 224
CAMERA_WIDTH = 224
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
                # SimLingo's training camera pose (GlobalConfig.camera_pos_0), mirrored from
                # ``simlingo_obs.py``. Keep the two in sync -- see the header comment.
                "x": -1.5, "y": 0.0, "z": 2.0,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "width": VIZ_CAMERA_WIDTH,
                "height": VIZ_CAMERA_HEIGHT,
                "fov": 110,
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
