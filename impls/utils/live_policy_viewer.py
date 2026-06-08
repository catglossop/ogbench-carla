"""Serve a single overwriting MP4 of the current CARLA rollout for live monitoring.

Starts a background HTTP server (FastAPI + uvicorn) that always serves the latest
``live_policy.mp4`` written by :class:`LivePolicyViewer`.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


class LivePolicyViewer:
    """Accumulate annotated rollout frames and atomically overwrite one MP4 on disk."""

    def __init__(
        self,
        video_path: str | Path,
        *,
        port: int = 8765,
        fps: float = 10.0,
        publish_every_n_steps: int = 5,
    ) -> None:
        self.video_path = Path(video_path)
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self.port = int(port)
        self.fps = float(fps)
        self.publish_every_n_steps = max(1, int(publish_every_n_steps))
        self._lock = threading.Lock()
        self._global_step = 0
        self._updated_at = 0.0
        self._frame_count = 0
        self._server_thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        """Launch the background HTTP server (idempotent)."""
        if self._started:
            return
        self._server_thread = threading.Thread(
            target=self._run_server,
            name="live-policy-viewer",
            daemon=True,
        )
        self._server_thread.start()
        self._started = True
        print(
            f"[live_policy_view] serving {self.video_path} at http://0.0.0.0:{self.port}/",
            flush=True,
        )

    def publish_frames(
        self,
        frames: list[np.ndarray],
        global_step: int,
        *,
        force: bool = False,
    ) -> None:
        """Rewrite ``live_policy.mp4`` from the current episode frame list."""
        self._global_step = int(global_step)
        if not force and (self._global_step % self.publish_every_n_steps) != 0:
            return
        if not frames:
            return
        self._write_video(frames)

    def _write_video(self, frames: list[np.ndarray]) -> None:
        frames = [np.asarray(f, dtype=np.uint8) for f in frames]

        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise ImportError("live_policy_view requires opencv-python.") from exc

        h, w = frames[0].shape[:2]
        tmp_path = self.video_path.with_suffix(".tmp.mp4")
        if self._write_browser_mp4(frames, tmp_path, width=w, height=h):
            os.replace(tmp_path, self.video_path)
        else:
            # Fallback for environments without imageio-ffmpeg / libx264.
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(tmp_path), fourcc, self.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {tmp_path}")
            try:
                for frame in frames:
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    if frame.shape[-1] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame)
            finally:
                writer.release()
            os.replace(tmp_path, self.video_path)
            print(
                "[live_policy_view] wrote mp4v fallback; browsers may not play it. "
                "Install imageio-ffmpeg for H.264 output.",
                flush=True,
            )

        with self._lock:
            self._updated_at = time.time()
            self._frame_count = len(frames)

    def _write_browser_mp4(
        self,
        frames: list[np.ndarray],
        output_path: Path,
        *,
        width: int,
        height: int,
    ) -> bool:
        """Encode H.264/avc1 MP4 for HTML5 video (Chrome/Firefox/Safari)."""
        try:
            import cv2  # type: ignore
            import imageio_ffmpeg
        except ImportError:
            return False

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except OSError:
            return False
        assert proc.stdin is not None
        try:
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                if frame.shape[-1] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except BrokenPipeError:
            proc.stdin.close()
            proc.wait(timeout=30)
            return False
        finally:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        if proc.wait(timeout=120) != 0:
            return False
        return output_path.is_file() and output_path.stat().st_size > 0

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "global_step": self._global_step,
                "frame_count": self._frame_count,
                "updated_at": self._updated_at,
                "video_path": str(self.video_path),
                "port": self.port,
            }

    def _run_server(self) -> None:
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.responses import FileResponse, HTMLResponse
            import uvicorn
        except ImportError as exc:
            raise ImportError(
                "live_policy_view requires fastapi and uvicorn. "
                "Install with: pip install fastapi uvicorn"
            ) from exc

        viewer = self
        app = FastAPI(title="Live CARLA Policy Viewer")

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Live CARLA Policy</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 1.5rem; background: #111; color: #eee; }
    video { max-width: min(100%, 1280px); border: 1px solid #444; background: #000; }
    #meta { color: #aaa; }
  </style>
</head>
<body>
  <h1>Live policy rollout</h1>
  <p id="meta">Waiting for first video&hellip;</p>
  <video id="player" controls autoplay muted playsinline loop></video>
  <p id="err" style="color:#f88"></p>
  <script>
    const player = document.getElementById('player');
    player.addEventListener('error', () => {
      document.getElementById('err').textContent =
        'Video failed to load or decode. If this persists, restart training so live_policy.mp4 is re-encoded as H.264.';
    });
    async function refresh() {
      try {
        const status = await fetch('/status').then(r => r.json());
        const ts = status.updated_at || Date.now() / 1000;
        document.getElementById('meta').textContent =
          `Env step ${status.global_step} | ${status.frame_count} frames | updated ${new Date(ts * 1000).toLocaleTimeString()}`;
        const nextSrc = `/live.mp4?ts=${ts}`;
        if (player.getAttribute('src') !== nextSrc) {
          document.getElementById('err').textContent = '';
          player.src = nextSrc;
          player.load();
          player.play().catch(() => {});
        }
      } catch (err) {
        document.getElementById('meta').textContent = 'Waiting for rollout video...';
      }
    }
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>"""

        @app.get("/status")
        def get_status() -> dict[str, Any]:
            return viewer.status()

        @app.get("/live.mp4")
        def get_video() -> FileResponse:
            if not viewer.video_path.is_file():
                raise HTTPException(status_code=404, detail="Video not ready yet.")
            return FileResponse(
                viewer.video_path,
                media_type="video/mp4",
                filename="live_policy.mp4",
                headers={"Cache-Control": "no-store"},
            )

        uvicorn.run(app, host="0.0.0.0", port=self.port, log_level="warning")
