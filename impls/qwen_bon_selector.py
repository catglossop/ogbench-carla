"""HTTP client for the local Qwen answer-token BoN scoring service."""
from __future__ import annotations

import base64
import io
import json
import urllib.request

import numpy as np
from PIL import Image


class QwenActionSelector:
    def __init__(self, url: str, timeout: float = 300.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def select_candidate(
        self,
        frame: np.ndarray,
        actions: np.ndarray,
        subtasks: list[str],
        routing_command: str,
        speed: float,
        route: str,
    ) -> dict:
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buffer, format="JPEG", quality=92)
        payload = json.dumps(
            {
                "image_jpeg": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "actions": np.asarray(actions, dtype=np.float32).tolist(),
                "subtasks": list(subtasks),
                "routing_command": routing_command,
                "speed": float(speed),
                "route": str(route),
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.url}/score", payload, {"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read())
        if "error" in result:
            raise RuntimeError(result["error"])
        choice = int(result["choice"])
        if not 0 <= choice < len(subtasks):
            raise ValueError(f"Qwen choice {choice} outside candidate range")
        return result

    def train_candidate(
        self,
        frame: np.ndarray,
        action: np.ndarray,
        subtask: str,
        routing_command: str,
        speed: float,
        targets: dict[str, float],
        route: str,
        timestep: int,
    ) -> dict:
        """Submit an executed projected trajectory and its causal rollout targets."""
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buffer, format="JPEG", quality=92)
        payload = json.dumps(
            {
                "image_jpeg": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "subtask": str(subtask),
                "routing_command": str(routing_command),
                "speed": float(speed),
                "targets": {key: float(value) for key, value in targets.items()},
                "route": str(route),
                "timestep": int(timestep),
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.url}/train", payload, {"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read())
        if "error" in result:
            raise RuntimeError(result["error"])
        return result
