"""HTTP client for the local Qwen answer-token BoN scoring service."""
from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class PendingScene:
    image_jpeg: str
    route: str
    future: Future


class QwenActionSelector:
    def __init__(self, url: str, timeout: float = 300.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._scene_executor = None

    @staticmethod
    def _encode_frame(frame):
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buffer, format="JPEG", quality=92)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def prepare_scene(self, frame, route):
        """Begin a fresh description of this exact frame while the actor runs."""
        image_jpeg = self._encode_frame(frame)
        if self._scene_executor is None:
            self._scene_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-scene")
        def request_scene():
            request = urllib.request.Request(
                f"{self.url}/prepare_scene",
                json.dumps({"image_jpeg": image_jpeg, "route": str(route)}).encode(),
                {"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        return PendingScene(image_jpeg, str(route), self._scene_executor.submit(request_scene))

    def close(self):
        if self._scene_executor is not None:
            self._scene_executor.shutdown(wait=True, cancel_futures=True)
            self._scene_executor = None

    def select_candidate(
        self,
        frame: np.ndarray,
        actions: np.ndarray,
        subtasks: list[str],
        routing_command: str,
        speed: float,
        route: str,
        fallback_index: int | None = None,
        prepared_scene: PendingScene | None = None,
    ) -> dict:
        image_jpeg = self._encode_frame(frame)
        preparation = {}
        wait_started = time.perf_counter()
        if prepared_scene is not None:
            if prepared_scene.image_jpeg != image_jpeg or prepared_scene.route != str(route):
                raise ValueError("Prepared scene input changed during candidate sampling")
            preparation = prepared_scene.future.result(timeout=self.timeout)
        prepare_wait_s = time.perf_counter() - wait_started
        payload = json.dumps(
            {
                "image_jpeg": image_jpeg,
                "scene_token": preparation.get("scene_token"),
                "actions": np.asarray(actions, dtype=np.float32).tolist(),
                "subtasks": list(subtasks),
                "routing_command": routing_command,
                "speed": float(speed),
                "route": str(route),
                "fallback_index": fallback_index,
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.url}/score", payload, {"Content-Type": "application/json"}, method="POST"
        )
        try:
            request_started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Qwen scoring request failed with HTTP {exc.code}: {body}"
            ) from exc
        if "error" in result:
            raise RuntimeError(result["error"])
        result.setdefault("timings", {})["client_roundtrip_s"] = time.perf_counter() - request_started
        if prepared_scene is not None:
            result["timings"].update(preparation.get("timings", {}))
            result["timings"]["prepare_client_wait_s"] = prepare_wait_s
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
