"""VLM coaches that review driving rollout videos and annotate good/bad moments.

Supports:
  - Google Gemini (``gemini-3.5-flash`` by default)
  - Perceptron video QA API

Run from ``impls/``::

    python -m coaches.vlm_feedback \\
        --video /path/to/rollout.mp4 \\
        --metadata /path/to/episode_0001.json \\
        --provider gemini \\
        --output /path/to/annotated.mp4

    # Attach trajectory plots (speed, controls, collisions, route progress) to the prompt:
    python -m coaches.vlm_feedback ... --include-plots-in-prompt

Plots are always written under ``<output_dir>/<metadata_stem>_plots/`` (override with ``--plots-dir``).
When ``--include-plots-in-prompt`` is off, the prompt still receives the JSON (video-aligned steps)
plus a field guide describing each key.

By default, BAD events are mapped to 10-step action chunks (0.5s each) within ±2 chunks and written
to ``<metadata_stem>_chunk_feedback.json``. Use ``--no-chunk-feedback`` to skip.

Output videos are twice as wide as the input: original footage on the left,
black panel on the right with timed coach annotations and per-chunk feedback
(lateral/longitudinal coaching for each 0.5s action chunk near BAD events).
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

from impls.coaches.action_chunk_feedback import (
    DEFAULT_ACTION_CHUNK_STEPS,
    DEFAULT_BAD_EVENT_RADIUS_CHUNKS,
    DEFAULT_CHUNK_DURATION_SEC,
    active_chunk_at_time,
    affected_chunks_from_bad_events,
    assemble_chunk_feedback_json,
    build_action_chunk_specs,
    build_chunk_feedback_prompt,
    format_chunk_feedback_lines,
    parse_chunk_feedback_response,
)
from impls.coaches.trajectory_plots import (
    TRAJECTORY_JSON_DESCRIPTION,
    compact_metadata_for_prompt,
    generate_trajectory_plots,
    is_trajectory_metadata,
    load_trajectory_metadata,
)

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
    """Load trajectory JSON metadata from a rollout episode file."""
    return load_trajectory_metadata(metadata_path)


def build_coaching_prompt(
    metadata: dict[str, Any],
    *,
    include_plots: bool = False,
) -> str:
    """Prompt shared by Gemini and Perceptron coaches."""
    if is_trajectory_metadata(metadata):
        prompt_metadata = compact_metadata_for_prompt(metadata)
        metadata_block = json.dumps(prompt_metadata, indent=2)
        metadata_intro = TRAJECTORY_JSON_DESCRIPTION
        if include_plots:
            metadata_intro += (
                "\n\nA combined trajectory plot image is attached (speed, controls, collisions, "
                "route progress vs video time in seconds). Use it together with the video and JSON."
            )
        else:
            metadata_intro += (
                "\n\nUse the JSON below as structured context; timestamps in ``video_timestamp_sec`` "
                "align with the rollout video."
            )
    else:
        metadata_block = json.dumps(metadata, indent=2) if metadata else "{}"
        metadata_intro = "Optional metadata (may be empty):"

    return textwrap.dedent(
        f"""
        You are reviewing a driving rollout video for an autonomous vehicle policy.

        {metadata_intro}
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
        - For ``BAD`` events, phrase ``correction`` using controlled driving verbs:
          combine lateral guidance (``adjust left`` / ``adjust right``) and/or longitudinal
          guidance (``accelerate`` / ``decelerate``). You may add brief extra detail after that.
        - Prefer 3–12 high-signal events rather than narrating every second.
        - When trajectory metadata is provided, cross-check timestamps with
          ``video_timestamp_sec``, speed, controls, collisions, and route progress.
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
    def analyze(
        self,
        video_path: str | Path,
        metadata: dict[str, Any],
        *,
        plot_paths: list[Path] | None = None,
        include_plots_in_prompt: bool = False,
    ) -> list[CoachEvent]:
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

    def analyze(
        self,
        video_path: str | Path,
        metadata: dict[str, Any],
        *,
        plot_paths: list[Path] | None = None,
        include_plots_in_prompt: bool = False,
    ) -> list[CoachEvent]:
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

        contents: list[Any] = [uploaded]
        if include_plots_in_prompt and plot_paths:
            for plot_path in plot_paths:
                plot_upload = client.files.upload(file=str(plot_path))
                while plot_upload.state.name == "PROCESSING":
                    time.sleep(0.5)
                    plot_upload = client.files.get(name=plot_upload.name)
                if plot_upload.state.name != "ACTIVE":
                    raise RuntimeError(
                        f"Gemini plot upload failed for {plot_path} "
                        f"with state={plot_upload.state.name!r}."
                    )
                contents.append(plot_upload)

        prompt = build_coaching_prompt(metadata, include_plots=include_plots_in_prompt)
        contents.append(prompt)
        response = client.models.generate_content(
            model=self.model,
            contents=contents,
        )
        text = getattr(response, "text", None) or str(response)
        return parse_coach_response(text)

    def complete_text(self, prompt: str) -> str:
        """Text-only completion (used for per-chunk feedback refinement)."""
        from google import genai

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return getattr(response, "text", None) or str(response)


class PerceptronVLMCOach(VLMCOach):
    """Coach backed by the Perceptron video QA API."""

    def __init__(self, *, api_key: str = PERCEPTRON_API_KEY) -> None:
        self.api_key = api_key

    def analyze(
        self,
        video_path: str | Path,
        metadata: dict[str, Any],
        *,
        plot_paths: list[Path] | None = None,
        include_plots_in_prompt: bool = False,
    ) -> list[CoachEvent]:
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

        prompt = build_coaching_prompt(metadata, include_plots=include_plots_in_prompt)
        if include_plots_in_prompt and plot_paths:
            plot_note = ", ".join(str(p) for p in plot_paths)
            prompt += (
                f"\n\nNote: trajectory plots were generated at {plot_note} but cannot be "
                "attached via the Perceptron video API; rely on the JSON metadata above."
            )
        result = question(
            video(str(path)),
            prompt,
            reasoning=True,
        )
        text = getattr(result, "text", None) or str(result)
        return parse_coach_response(text)

    def complete_text(self, prompt: str) -> str:
        raise NotImplementedError(
            "Perceptron chunk-feedback refinement is not supported; use provider=gemini."
        )


def generate_action_chunk_feedback(
    coach: VLMCOach,
    metadata: dict[str, Any],
    events: list[CoachEvent],
    *,
    steps_per_chunk: int = DEFAULT_ACTION_CHUNK_STEPS,
    chunk_duration_sec: float = DEFAULT_CHUNK_DURATION_SEC,
    bad_event_radius_chunks: int = DEFAULT_BAD_EVENT_RADIUS_CHUNKS,
) -> dict[str, Any]:
    """Map episode BAD events to per-chunk feedback (null outside the affected window)."""
    chunk_specs = build_action_chunk_specs(
        metadata,
        steps_per_chunk=steps_per_chunk,
        chunk_duration_sec=chunk_duration_sec,
    )
    num_chunks = len(chunk_specs)
    bad_timestamps = [e.timestamp_sec for e in events if e.label == "BAD"]
    affected = affected_chunks_from_bad_events(
        bad_timestamps,
        num_chunks=num_chunks,
        radius_chunks=bad_event_radius_chunks,
        chunk_duration_sec=chunk_duration_sec,
    )

    chunk_feedback: list[dict[str, Any] | None] = [None] * num_chunks
    if affected and bad_timestamps:
        bad_payload = [
            {
                "timestamp_sec": e.timestamp_sec,
                "description": e.description,
                "correction": e.correction,
            }
            for e in events
            if e.label == "BAD"
        ]
        prompt = build_chunk_feedback_prompt(
            bad_events=bad_payload,
            chunk_specs=chunk_specs,
            affected_chunk_indices=affected,
            metadata=metadata,
            steps_per_chunk=steps_per_chunk,
            chunk_duration_sec=chunk_duration_sec,
            bad_event_radius_chunks=bad_event_radius_chunks,
        )
        if not hasattr(coach, "complete_text"):
            raise RuntimeError(f"Coach {type(coach).__name__} does not support text completion.")
        response_text = coach.complete_text(prompt)  # type: ignore[attr-defined]
        chunk_feedback = parse_chunk_feedback_response(
            response_text,
            num_chunks=num_chunks,
            affected_chunk_indices=affected,
        )

    return assemble_chunk_feedback_json(
        metadata,
        events=events,
        chunk_feedback=chunk_feedback,
        chunk_specs=chunk_specs,
        affected_chunk_indices=affected,
        steps_per_chunk=steps_per_chunk,
        chunk_duration_sec=chunk_duration_sec,
        bad_event_radius_chunks=bad_event_radius_chunks,
    )


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


def _draw_side_panel_annotations(
    frame,
    event: CoachEvent | None,
    *,
    chunk: dict[str, Any] | None = None,
):
    """Draw coach event pulse and/or action-chunk feedback on the right-hand panel."""
    import cv2  # type: ignore

    annotated = frame.copy()
    h, w = annotated.shape[:2]
    panel_x0 = w // 2
    panel_w = w - panel_x0
    if panel_w <= 0:
        return annotated
    if event is None and chunk is None:
        return annotated

    wrap_width = max(24, panel_w // 10)
    lines: list[tuple[str, tuple[int, int, int], int]] = []

    if event is not None:
        status_color = (0, 200, 0) if event.label == "GOOD" else (0, 0, 255)
        lines.append((f"Event: {event.label}", status_color, 2))
        if event.description:
            lines.append(("Description:", (255, 255, 255), 1))
            for part in _wrap_text(event.description, width=wrap_width):
                lines.append((part, (255, 255, 255), 1))
        if event.label == "BAD" and event.correction:
            lines.append(("Correction:", (255, 255, 255), 1))
            for part in _wrap_text(event.correction, width=wrap_width):
                lines.append((part, (255, 255, 255), 1))
        if chunk is not None:
            lines.append(("", (255, 255, 255), 1))

    if chunk is not None:
        chunk_color = (255, 200, 80)
        for idx, line in enumerate(format_chunk_feedback_lines(chunk, wrap_width=wrap_width)):
            color = chunk_color if idx == 0 else (220, 220, 220)
            weight = 2 if idx == 0 else 1
            lines.append((line, color, weight))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    line_height = 17
    x = panel_x0 + 12
    y = 24

    for line, color, weight in lines:
        if not line:
            y += line_height // 2
            continue
        if y + line_height > h - 8:
            break
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


def _draw_chunk_badge_on_video(frame, chunk: dict[str, Any] | None):
    """Small chunk-index badge on the source video (left half)."""
    if chunk is None:
        return frame
    import cv2  # type: ignore

    annotated = frame.copy()
    h, w = annotated.shape[:2]
    video_w = w // 2 if w > h else w
    label = f"Chunk {chunk.get('chunk_index', '?')}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    pad = 4
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    x0, y0 = 8, 8
    x1 = min(video_w - 8, x0 + tw + 2 * pad)
    y1 = y0 + th + baseline + 2 * pad
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (40, 40, 40), thickness=-1)
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 200, 80), thickness=1)
    cv2.putText(
        annotated,
        label,
        (x0 + pad, y1 - baseline - pad),
        font,
        font_scale,
        (255, 200, 80),
        thickness,
        cv2.LINE_AA,
    )
    return annotated


def annotate_video(
    video_path: str | Path,
    events: list[CoachEvent],
    output_path: str | Path,
    *,
    display_sec: float = ANNOTATION_DISPLAY_SEC,
    chunk_feedback_json: dict[str, Any] | None = None,
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
            chunk = (
                active_chunk_at_time(chunk_feedback_json, timestamp_sec)
                if chunk_feedback_json is not None
                else None
            )
            extended = _extend_frame(frame)
            extended = _draw_chunk_badge_on_video(extended, chunk)
            annotated = _draw_side_panel_annotations(extended, event, chunk=chunk)
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
    include_plots_in_prompt: bool = False,
    plots_dir: Path | None = None,
    chunk_feedback: bool = True,
    chunk_feedback_output: Path | None = None,
    action_chunk_steps: int = DEFAULT_ACTION_CHUNK_STEPS,
    action_chunk_duration_sec: float = DEFAULT_CHUNK_DURATION_SEC,
    bad_event_radius_chunks: int = DEFAULT_BAD_EVENT_RADIUS_CHUNKS,
) -> Path:
    """Run VLM coaching on a video and write an annotated output clip."""
    metadata_dict = load_metadata(metadata)
    resolved_plots_dir = plots_dir
    if resolved_plots_dir is None:
        stem = metadata.stem if metadata else video.stem
        resolved_plots_dir = (output.parent if output is not None else video.parent) / f"{stem}_plots"

    plot_paths: list[Path] = []
    if is_trajectory_metadata(metadata_dict):
        plot_map = generate_trajectory_plots(metadata_dict, resolved_plots_dir)
        plot_paths = [plot_map["combined"]]
        print(f"[vlm_feedback] wrote trajectory plots to {resolved_plots_dir}", flush=True)
        for name, path in plot_map.items():
            print(f"  {name}: {path}", flush=True)

    coach = create_coach(provider, model=gemini_model) if provider == "gemini" else create_coach(provider)

    print(f"[vlm_feedback] provider={provider} video={video}", flush=True)
    if include_plots_in_prompt:
        print("[vlm_feedback] including trajectory plots in coaching prompt", flush=True)
    events = coach.analyze(
        video,
        metadata_dict,
        plot_paths=plot_paths if include_plots_in_prompt else None,
        include_plots_in_prompt=include_plots_in_prompt,
    )
    print(f"[vlm_feedback] parsed {len(events)} events", flush=True)
    for event in events:
        suffix = f" | correction: {event.correction}" if event.correction else ""
        print(
            f"  {event.timestamp_sec:6.2f}s {event.label:4s} {event.description}{suffix}",
            flush=True,
        )

    if output is None:
        output = video.with_name(f"{video.stem}_annotated{video.suffix}")

    chunk_json: dict[str, Any] | None = None
    if chunk_feedback and is_trajectory_metadata(metadata_dict):
        chunk_json = generate_action_chunk_feedback(
            coach,
            metadata_dict,
            events,
            steps_per_chunk=action_chunk_steps,
            chunk_duration_sec=action_chunk_duration_sec,
            bad_event_radius_chunks=bad_event_radius_chunks,
        )
        chunk_out = chunk_feedback_output
        if chunk_out is None:
            chunk_out = (output.parent if output is not None else metadata.parent) / f"{metadata.stem}_chunk_feedback.json"
        chunk_out.parent.mkdir(parents=True, exist_ok=True)
        chunk_out.write_text(json.dumps(chunk_json, indent=2), encoding="utf-8")
        n_with_feedback = sum(1 for c in chunk_json["action_chunks"] if c["feedback"] is not None)
        print(
            f"[vlm_feedback] wrote chunk feedback to {chunk_out} "
            f"({n_with_feedback}/{len(chunk_json['action_chunks'])} chunks)",
            flush=True,
        )

    annotated_path = annotate_video(
        video,
        events,
        output,
        display_sec=display_sec,
        chunk_feedback_json=chunk_json,
    )
    print(f"[vlm_feedback] wrote annotated video to {annotated_path}", flush=True)

    return annotated_path


if __name__ == "__main__":
    tyro.cli(main)
