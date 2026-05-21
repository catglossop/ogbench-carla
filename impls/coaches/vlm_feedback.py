"""VLM coaches that review driving rollout videos and annotate good/bad moments.

Supports:
  - Google Gemini (``gemini-3.5-flash`` by default)
  - Perceptron video QA API

Run from ``impls/``::

    python -m coaches.vlm_feedback \\
        --video /path/to/rollout.mp4 \\
        --metadata /path/to/metadata.json \\
        --provider gemini \\
        --output /path/to/annotated.mp4

Output videos are twice as wide as the input: original footage on the left,
black panel on the right with timed coach annotations.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tyro

ProviderName = Literal["gemini", "perceptron"]

# Placeholders — override via environment variables in real runs.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
PERCEPTRON_API_KEY = os.environ.get("PERCEPTRON_API_KEY", "YOUR_PERCEPTRON_API_KEY_HERE")
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"

# How long each GOOD/BAD label stays on screen after its timestamp (seconds).
ANNOTATION_DISPLAY_SEC = 1.0


@dataclass(frozen=True)
class CoachEvent:
    """A single good or bad driving moment identified by the VLM."""

    timestamp_sec: float
    label: Literal["GOOD", "BAD"]
    description: str
    correction: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CoachEvent:
        label = str(raw.get("label", "")).strip().upper()
        if label not in ("GOOD", "BAD"):
            raise ValueError(f"Event label must be GOOD or BAD, got {raw.get('label')!r}.")
        return cls(
            timestamp_sec=float(raw["timestamp_sec"]),
            label=label,  # type: ignore[arg-type]
            description=str(raw.get("description", "")).strip(),
            correction=str(raw.get("correction", "")).strip(),
        )


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Load optional JSON metadata; empty dict if the file is missing or blank."""
    path = Path(metadata_path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Metadata file must contain a JSON object, got {type(data).__name__}.")
    return data


def build_coaching_prompt(metadata: dict[str, Any]) -> str:
    """Prompt shared by Gemini and Perceptron coaches."""
    metadata_block = json.dumps(metadata, indent=2) if metadata else "{}"
    return textwrap.dedent(
        f"""
        You are reviewing a driving rollout video for an autonomous vehicle policy.

        Optional metadata (may be empty for now):
        ```json
        {metadata_block}
        ```

        Watch the full video and identify moments where driving behavior was clearly
        good or clearly bad (lane keeping, speed, turns, collisions/near-misses,
        stopping, yielding, etc.).

        Return ONLY valid JSON with this schema (no markdown fences):
        {{
          "events": [
            {{
              "timestamp_sec": 12.5,
              "label": "GOOD",
              "description": "Brief explanation of what went well."
            }},
            {{
              "timestamp_sec": 18.0,
              "label": "BAD",
              "description": "Brief explanation of what went wrong.",
              "correction": "Concrete instruction for how to fix the behavior."
            }}
          ]
        }}

        Rules:
        - Use seconds from the start of the video for ``timestamp_sec``.
        - ``label`` must be exactly ``GOOD`` or ``BAD``.
        - Include ``correction`` only for ``BAD`` events (empty string otherwise).
        - Prefer 3–12 high-signal events rather than narrating every second.
        """
    ).strip()


def _extract_json_payload(text: str) -> dict[str, Any]:
    """Parse model output that may include markdown fences or extra prose."""
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if brace_match:
            stripped = brace_match.group(0)
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("Coach response must be a JSON object.")
    return payload


def parse_coach_response(text: str) -> list[CoachEvent]:
    """Convert raw model text into structured coach events."""
    payload = _extract_json_payload(text)
    raw_events = payload.get("events", [])
    if not isinstance(raw_events, list):
        raise ValueError("Coach response JSON must contain an 'events' list.")
    events = [CoachEvent.from_dict(item) for item in raw_events]
    return sorted(events, key=lambda event: event.timestamp_sec)


class VLMCOach(ABC):
    """Base interface for video-based VLM coaches."""

    @abstractmethod
    def analyze(self, video_path: str | Path, metadata: dict[str, Any]) -> list[CoachEvent]:
        """Return timestamped GOOD/BAD events for a rollout video."""


class GeminiVLMCOach(VLMCOach):
    """Coach backed by the Google Gemini API."""

    def __init__(
        self,
        *,
        api_key: str = GEMINI_API_KEY,
        model: str = DEFAULT_GEMINI_MODEL,
    ) -> None:
        self.api_key = api_key
        self.model = model

    def analyze(self, video_path: str | Path, metadata: dict[str, Any]) -> list[CoachEvent]:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Gemini coach requires the google-genai package. Install with: pip install google-genai"
            ) from exc

        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")

        client = genai.Client(api_key=self.api_key)
        uploaded = client.files.upload(file=str(path))
        while uploaded.state.name == "PROCESSING":
            time.sleep(1.0)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state.name != "ACTIVE":
            raise RuntimeError(f"Gemini file upload failed with state={uploaded.state.name!r}.")

        prompt = build_coaching_prompt(metadata)
        response = client.models.generate_content(
            model=self.model,
            contents=[uploaded, prompt],
        )
        text = getattr(response, "text", None) or str(response)
        return parse_coach_response(text)


class PerceptronVLMCOach(VLMCOach):
    """Coach backed by the Perceptron video QA API."""

    def __init__(self, *, api_key: str = PERCEPTRON_API_KEY) -> None:
        self.api_key = api_key

    def analyze(self, video_path: str | Path, metadata: dict[str, Any]) -> list[CoachEvent]:
        try:
            from perceptron import question, video
        except ImportError as exc:
            raise ImportError(
                "Perceptron coach requires the perceptron package. Install with: pip install perceptron"
            ) from exc

        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")

        # Perceptron clients typically read credentials from the environment.
        if self.api_key and not self.api_key.startswith("YOUR_"):
            os.environ.setdefault("PERCEPTRON_API_KEY", self.api_key)

        prompt = build_coaching_prompt(metadata)
        result = question(
            video(str(path)),
            prompt,
            reasoning=True,
        )
        text = getattr(result, "text", None) or str(result)
        return parse_coach_response(text)


def create_coach(provider: ProviderName, **kwargs: Any) -> VLMCOach:
    """Factory for ``gemini`` or ``perceptron`` coaches."""
    if provider == "gemini":
        return GeminiVLMCOach(**kwargs)
    if provider == "perceptron":
        return PerceptronVLMCOach(**kwargs)
    raise ValueError(f"Unknown provider {provider!r}; expected 'gemini' or 'perceptron'.")


def _wrap_text(text: str, *, width: int = 48) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return textwrap.wrap(cleaned, width=width)


def _active_events(events: list[CoachEvent], timestamp_sec: float, display_sec: float) -> list[CoachEvent]:
    return [
        event
        for event in events
        if timestamp_sec >= event.timestamp_sec and timestamp_sec < event.timestamp_sec + display_sec
    ]


def _extend_frame(frame) -> Any:
    """Place the video on the left; pad the right half with black pixels."""
    import numpy as np

    h, w = frame.shape[:2]
    extended = np.zeros((h, w * 2, 3), dtype=frame.dtype)
    extended[:, :w, :] = frame
    return extended


def _annotation_lines(event: CoachEvent, *, wrap_width: int) -> list[str]:
    """Structured annotation text for the right-hand panel."""
    lines = [f"Status: {event.label}"]
    if event.description:
        lines.append("Description of behavior:")
        lines.extend(_wrap_text(event.description, width=wrap_width))
    if event.label == "BAD" and event.correction:
        lines.append("Correction:")
        lines.extend(_wrap_text(event.correction, width=wrap_width))
    return lines


def _draw_side_panel_annotations(frame, event: CoachEvent | None):
    """Draw coach text on the black right-hand panel of a doubled-width frame."""
    import cv2  # type: ignore

    annotated = frame.copy()
    h, w = annotated.shape[:2]
    panel_x0 = w // 2
    panel_w = w - panel_x0
    if event is None or panel_w <= 0:
        return annotated

    wrap_width = max(24, panel_w // 10)
    lines = _annotation_lines(event, wrap_width=wrap_width)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    line_height = 18
    x = panel_x0 + 12
    y = 28
    status_color = (0, 200, 0) if event.label == "GOOD" else (0, 0, 255)
    body_color = (255, 255, 255)

    for line in lines:
        if y + line_height > h - 8:
            break
        color = status_color if line.startswith("Status:") else body_color
        weight = 2 if line.startswith("Status:") else 1
        cv2.putText(
            annotated,
            line,
            (x, y),
            font,
            font_scale,
            color,
            weight,
            cv2.LINE_AA,
        )
        y += line_height

    return annotated


def annotate_video(
    video_path: str | Path,
    events: list[CoachEvent],
    output_path: str | Path,
    *,
    display_sec: float = ANNOTATION_DISPLAY_SEC,
) -> Path:
    """Write a video with the frame width doubled; annotations appear on the right panel."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("Video annotation requires opencv-python.") from exc

    src = Path(video_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {src}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 10.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = width * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (out_width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open video writer: {dst}")

    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp_sec = frame_idx / fps
            active = _active_events(events, timestamp_sec, display_sec)
            event = active[-1] if active else None
            extended = _extend_frame(frame)
            annotated = _draw_side_panel_annotations(extended, event)
            writer.write(annotated)
            frame_idx += 1
    finally:
        capture.release()
        writer.release()

    return dst


def main(
    video: Path,
    metadata: Path,
    provider: ProviderName,
    output: Path | None = None,
    display_sec: float = ANNOTATION_DISPLAY_SEC,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> Path:
    """Run VLM coaching on a video and write an annotated output clip."""
    metadata_dict = load_metadata(metadata)
    coach = create_coach(provider, model=gemini_model) if provider == "gemini" else create_coach(provider)

    print(f"[vlm_feedback] provider={provider} video={video}", flush=True)
    events = coach.analyze(video, metadata_dict)
    print(f"[vlm_feedback] parsed {len(events)} events", flush=True)
    for event in events:
        suffix = f" | correction: {event.correction}" if event.correction else ""
        print(
            f"  {event.timestamp_sec:6.2f}s {event.label:4s} {event.description}{suffix}",
            flush=True,
        )

    if output is None:
        output = video.with_name(f"{video.stem}_annotated{video.suffix}")
    annotated_path = annotate_video(video, events, output, display_sec=display_sec)
    print(f"[vlm_feedback] wrote annotated video to {annotated_path}", flush=True)
    return annotated_path


if __name__ == "__main__":
    tyro.cli(main)
