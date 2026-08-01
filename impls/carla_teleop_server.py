"""Local web UI for the CARLA teleop candidate picker.

Runs a small stdlib HTTP server so a human can view the current camera frame
(with candidate trajectory overlays) and pick one from a browser. Meant to be
reached over an SSH port-forward from this (headless, SSH-only) machine:

    ssh -L 8000:localhost:8000 <this-host>

then open http://localhost:8000 in a local browser. No extra dependencies
(stdlib http.server only); frames are pushed in by the caller as RGB numpy
arrays and encoded to JPEG here.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

import cv2
import numpy as np

# Sentinel returned by wait_for_choice() when the human rejects the whole batch and
# wants a freshly resampled one instead of picking from the current candidates.
REJECT = -1

_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>CARLA teleop</title>
<style>
  body { background:#111; color:#eee; font-family:sans-serif; margin:0; padding:16px; }
  #frame { max-width:100%; border:1px solid #444; }
  #legend { margin-top:12px; display:flex; flex-wrap:wrap; gap:8px; }
  .candidate { border:2px solid #555; border-radius:6px; padding:8px 12px; cursor:pointer;
               background:#222; min-width:160px; }
  .candidate:hover { background:#333; }
  .swatch { display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:6px; }
  .key { opacity:0.6; font-size:0.85em; }
  #timeout { margin-top:8px; opacity:0.8; }
  #reject { margin-top:12px; padding:10px 16px; background:#622; color:#eee; border:1px solid #944;
            border-radius:6px; cursor:pointer; font-size:1em; }
  #reject:hover { background:#833; }
</style>
</head>
<body>
  <img id="frame" src="/frame.jpg">
  <div id="legend"></div>
  <button id="reject" onclick="reject_batch()">🔄 None of these — show new options [space]</button>
  <div id="timeout"></div>
<script>
let deadline = null;

function choose(index) {
  fetch("/choose", {method: "POST", body: JSON.stringify({index: index})});
}

function reject_batch() {
  fetch("/choose", {method: "POST", body: JSON.stringify({reject: true})});
}

document.addEventListener("keydown", (e) => {
  if (e.key === " ") { e.preventDefault(); reject_batch(); return; }
  const n = parseInt(e.key, 10);
  if (!isNaN(n) && n >= 1) choose(n - 1);
});

async function refreshState() {
  const res = await fetch("/state.json");
  const state = await res.json();
  const legend = document.getElementById("legend");
  legend.innerHTML = "";
  state.legend.forEach((c, i) => {
    const div = document.createElement("div");
    div.className = "candidate";
    div.onclick = () => choose(c.index);
    div.innerHTML = `<span class="swatch" style="background:${c.color}"></span>` +
      `<b>${c.index + 1}</b> <span class="key">[press ${c.index + 1}]</span><br>${c.subtask || "(no subtask)"}`;
    legend.appendChild(div);
  });
  deadline = Date.now() + state.timeout_sec * 1000;
}

function tickTimeout() {
  const el = document.getElementById("timeout");
  if (deadline !== null) {
    const remaining = Math.max(0, (deadline - Date.now()) / 1000);
    el.textContent = `auto-pick in ${remaining.toFixed(1)}s if no choice`;
  }
  requestAnimationFrame(tickTimeout);
}

setInterval(() => {
  document.getElementById("frame").src = "/frame.jpg?t=" + Date.now();
}, 200);
setInterval(refreshState, 500);
refreshState();
tickTimeout();
</script>
</body>
</html>
"""


class TeleopServer:
    """Background HTTP server bridging the CARLA stepping loop and a browser client."""

    def __init__(self, port: int = 8000):
        self._port = port
        self._lock = threading.Lock()
        self._frame_jpeg: bytes = b""
        self._legend: list[dict[str, Any]] = []
        self._timeout_sec: float = 8.0
        self._choice_queue: "queue.Queue[int]" = queue.Queue(maxsize=1)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        handler = _make_handler(self)
        self._httpd = ThreadingHTTPServer(("0.0.0.0", self._port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        print(
            f"[carla_teleop_server] listening on 0.0.0.0:{self._port} -- forward with:\n"
            f"    ssh -L {self._port}:localhost:{self._port} <this-host>\n"
            f"then open http://localhost:{self._port}",
            flush=True,
        )

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None

    def publish(self, frame_rgb: np.ndarray, legend: list[dict[str, Any]], timeout_sec: float) -> None:
        """Push a new frame + candidate legend; clears any stale pending choice."""
        frame_bgr = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        with self._lock:
            if ok:
                self._frame_jpeg = buf.tobytes()
            self._legend = legend
            self._timeout_sec = timeout_sec
        while not self._choice_queue.empty():
            try:
                self._choice_queue.get_nowait()
            except queue.Empty:
                break

    def wait_for_choice(self, timeout_sec: float, default_index: int = 0) -> int:
        """Block for a human choice: a candidate index, or REJECT to resample the batch.

        Falls back to ``default_index`` if nothing arrives in time -- this still
        applies after any number of rejects, so an unattended session can't stall
        forever on one decision point.
        """
        try:
            return self._choice_queue.get(timeout=timeout_sec)
        except queue.Empty:
            return default_index

    def _snapshot(self) -> tuple[bytes, list[dict[str, Any]], float]:
        with self._lock:
            return self._frame_jpeg, list(self._legend), self._timeout_sec

    def _submit_choice(self, index: int) -> None:
        try:
            self._choice_queue.put_nowait(index)
        except queue.Full:
            pass


def _make_handler(server: TeleopServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A002 -- silence per-request logging
            pass

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                self._send_html(_PAGE)
            elif self.path.startswith("/frame.jpg"):
                jpeg, _legend, _timeout = server._snapshot()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            elif self.path.startswith("/state.json"):
                _jpeg, legend, timeout_sec = server._snapshot()
                body = json.dumps({"legend": legend, "timeout_sec": timeout_sec}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/choose":
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(body)
                except Exception:
                    payload = {}
                if payload.get("reject"):
                    server._submit_choice(REJECT)
                else:
                    try:
                        index = int(payload.get("index"))
                    except Exception:
                        index = None
                    if index is not None and index >= 0:
                        server._submit_choice(index)
                self.send_response(204)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler
