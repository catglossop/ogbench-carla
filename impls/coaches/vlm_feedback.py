"""VLM coaches that review driving rollout videos and annotate good/bad moments.

Supports:
  - Google Gemini (``gemini-2.0-flash`` by default)
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

ProviderName = Literal["gemini", "perceptron"]

# Placeholders — override via environment variables in real runs.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
PERCEPTRON_API_KEY = os.environ.get("PERCEPTRON_API_KEY", "YOUR_PERCEPTRON_API_KEY_HERE")
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

DEFAULT_ACTION_CHUNK_STEPS = 10
DEFAULT_CHUNK_DURATION_SEC = 0.5
DEFAULT_BAD_EVENT_RADIUS_CHUNKS = 2

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


def _build_task_overview_block(metadata: dict[str, Any]) -> str:
    """Render the episode's full routing-command plan as an overall 'task' for the VLM.

    ``metadata["route_command_plan"]`` is the ordered list of maneuvers the ego will be told
    to perform over the whole route (precomputed at episode start, e.g. ``follow the road``
    → ``go right at the next intersection`` → ``follow the road``). Giving the VLM the whole
    plan up front lets it judge whether the policy is progressing through the route correctly
    rather than only reacting to the single command visible at each instant.
    """
    plan = metadata.get("route_command_plan") or []
    if not isinstance(plan, list) or not plan:
        return ""
    lines: list[str] = []
    for i, item in enumerate(plan):
        if not isinstance(item, dict):
            continue
        command = str(item.get("command", "")).strip()
        if not command:
            continue
        dist = item.get("start_distance_m")
        if dist is not None:
            lines.append(f"  {i + 1}. {command} (after ~{float(dist):.0f} m along the route)")
        else:
            lines.append(f"  {i + 1}. {command}")
    if not lines:
        return ""
    body = "\n".join(lines)
    return (
        "\nOverall task — the full sequence of routing commands the ego is given over this "
        "route, in order (the video may cover only part of it):\n"
        f"{body}\n"
    )


def _build_route_progress_block(metadata: dict[str, Any]) -> str:
    """Render route-completion context so the VLM rewards/penalizes making progress.

    Completing the assigned route is a primary objective: an agent that stalls short of the
    goal — most commonly by stopping prematurely when the path ahead is clear — should be
    flagged BAD even if its moment-to-moment driving looks smooth.
    """
    start = metadata.get("route_progress_start_pct")
    end = metadata.get("route_progress_end_pct")
    delta = metadata.get("route_progress_delta_pct")
    completed = metadata.get("route_completed")
    mean_end_speed = metadata.get("mean_end_speed_mps")
    if end is None and start is None:
        return ""
    lines = ["\nRoute progress over this window (route completion is a primary objective):"]
    if start is not None:
        lines.append(f"  - start: {float(start):.1f}% complete")
    if end is not None:
        lines.append(f"  - end:   {float(end):.1f}% complete")
    if delta is not None:
        lines.append(f"  - advanced {float(delta):.1f}% of the route during this window")
    if mean_end_speed is not None:
        lines.append(f"  - mean ego speed over the last few steps: {float(mean_end_speed):.2f} m/s")
    if completed is False and end is not None and float(end) < 99.5:
        lines.append(
            "  - NOTE: the route is NOT complete. If the ego is stopped or barely moving with a "
            "clear gap ahead (no leading vehicle within stopping distance, green/no light, no "
            "pedestrian or yield obligation), treat that as BAD — it should accelerate and make "
            "forward progress. Only reward stopping when there is a real reason (red light, stop "
            "sign, leading vehicle close ahead, pedestrian/cyclist crossing, or a yield)."
        )
    return "\n".join(lines) + "\n"


def _build_per_timestamp_block(metadata: dict[str, Any]) -> str:
    """Render a timestamp-keyed JSON block of per-step trajectory data for the prompt.

    Top-level keys are video seconds (as strings); each value bundles the vehicle state and
    controls already recorded per step with the executed subtask, chain-of-thought
    reasoning, and the prompt the policy received at that moment. Steps that were not
    captured in the video (``video_timestamp_sec is None``) are skipped, since the VLM can
    only cross-reference moments it can actually see.
    """
    steps = metadata.get("steps", []) or []
    per_timestamp: dict[str, Any] = {}
    for s in steps:
        if not isinstance(s, dict):
            continue
        ts = s.get("video_timestamp_sec")
        if ts is None:
            continue
        key = f"{float(ts):.2f}"
        per_timestamp[key] = {
            "episode_step": s.get("episode_step"),
            "ego_speed_mps": s.get("ego_speed_mps"),
            "control_throttle": s.get("control_throttle"),
            "control_steer": s.get("control_steer"),
            "control_brake": s.get("control_brake"),
            "collision": s.get("collision"),
            "route_progress_pct": s.get("route_progress_pct"),
            "subtask": s.get("subtask", ""),
            "reasoning": s.get("reasoning", ""),
            "prompt": s.get("prompt", ""),
        }
    if not per_timestamp:
        return ""
    return (
        "\nPer-timestamp trajectory data (keys are video seconds; each value is the vehicle "
        "state/controls plus the executed subtask, chain-of-thought reasoning, and the prompt "
        "the policy received at that moment):\n"
        f"```json\n{json.dumps(per_timestamp, indent=2)}\n```\n"
    )


def build_coaching_prompt(
    metadata: dict[str, Any],
    *,
    include_plots: bool = False,
) -> str:
    """Prompt shared by Gemini and Perceptron coaches."""
    # Summarise metadata compactly: top-level fields + collision events only.
    summary_keys = ("episode", "route", "episode_steps", "success",
                    "termination_reason", "route_progress_start_pct",
                    "route_progress_end_pct", "route_progress_delta_pct",
                    "route_completed", "collision_events")
    summary = {k: metadata[k] for k in summary_keys if k in metadata}
    metadata_block = json.dumps(summary, indent=2) if summary else "{}"

    # Overall task: the full ordered routing-command plan for the episode (highest-level
    # context — what the whole route asks of the ego, not just the current command).
    task_overview_block = _build_task_overview_block(metadata)

    # Route-completion context: emphasize that failing to advance the route (e.g. stopping
    # prematurely with a clear gap ahead while progress is well under 100%) is a bad behavior.
    progress_block = _build_route_progress_block(metadata)

    # Per-timestamp trajectory data: key each in-video step by its video timestamp so the
    # VLM can line up the executed subtask, chain-of-thought reasoning, and the prompt the
    # policy received with the exact moment in the video (alongside the speed/controls/
    # collision/route data already recorded per step).
    per_timestamp_block = _build_per_timestamp_block(metadata)

    collision_events = metadata.get("collision_events", [])
    if collision_events:
        collision_lines = "\n".join(
            f"  t={e.get('video_timestamp_sec', '?'):.2f}s  "
            f"new_event={e.get('new_event')}  contact_active={e.get('contact_active')}"
            for e in collision_events
        )
        collision_section = (
            f"\nCollision log (from on-board sensors — cross-reference with the video):\n"
            f"{collision_lines}\n"
        )
    else:
        collision_section = "\nNo collisions were recorded by the on-board sensors.\n"

    return textwrap.dedent(
        f"""
        You are reviewing a driving rollout video for an autonomous vehicle policy.
        {task_overview_block}
        Episode summary:
        ```json
        {metadata_block}
        ```
        {collision_section}{progress_block}{per_timestamp_block}
        Carefully watch the full video and identify moments where driving behavior was clearly
        good or clearly bad (lane keeping, speed, turns, collisions/near-misses,
        stopping, yielding, route progress, etc.). The collision log above shows ground-truth
        sensor data — use it to anchor your feedback to the correct timestamps.
        
        ** Step 1 **
        Determine the state of the vehicle and of the other agents (vehicles, pedestrians, cyclists, etc.) in the video. 
        Ask these questions to guide your analysis:
        - What lane in the vehicle in? 
        - What other vehicles are there? What lanes are they in? 
        - Is there a leading vehicle? If so, how far ahead is it? Is it stopped or moving? 
        - Are there any pedestrians or cyclists in the video? If so, what are they doing? 
        - Are there any stop signs or traffic lights in the video? What are their states?
        - According to the overall task (the ordered routing-command plan above) and the current
          routing command in the per-timestamp data, what maneuver should the vehicle be
          performing right now (following the road, turning left/right at the intersection,
          changing lanes, etc.), and is it in the correct lane and position to do so?

        ** Step 2 **
        Analyze the vehicle's behavior in the video. For example, you can ask these questions to guide your analysis: 
        - Does the vehicle's behavior conflict with the route command plan? (if yes, BAD; if no, GOOD)
        - Is the vehicle maintaining a safe distance from the front car? (if yes, GOOD; if no, BAD)
        - Is the vehicle maintaining a safe speed? (if yes, GOOD; if no, BAD)
        - Does the vehicle crash with any other vehicles or structures? This is directly from the collision log above. (if yes, BAD; if no, GOOD)
        - Did the vehicle get dangerously close to any other vehicles and/or collide with any other vehicles? (if yes, BAD; if no, GOOD)
        - Does the vehicle leave unnecessary gaps between itself and the front car? For example, at a stop light, the car should move forward to be closer to the front car to not leave an unnecessary gap (if yes, BAD; if no, GOOD)
        - Does the vehicle properly wait for a gap in the traffic before changing lanes? (if yes, GOOD; if no, BAD)
        - Does the vehicle properly yield to other vehicles when necessary? (if yes, GOOD; if no, BAD)
        - Does the vehicle properly stop at red lights and stop signs? (if yes, GOOD; if no, BAD)
        - Does the vehicle properly follow the route and traffic laws? (if yes, GOOD; if no, BAD)
        - Does the vehicle yield to pedestrians and cyclists when necessary? (if yes, GOOD; if no, BAD)
        - Does the vehicle follow the rules of the road (turning from left or right most lane when making a turn, stopping at stop signs and red lights, etc.)? (if yes, GOOD; if no, BAD)
        - Is the vehicle making progress along the route toward completion? Completing the route
          is a primary objective (see the route-progress section above). If route progress is
          below 100% and the vehicle is stopped or crawling with a clear gap ahead and NO valid
          reason to stop (no red light/stop sign, no close leading vehicle, no pedestrian/cyclist,
          no yield), that is BAD — it should accelerate and move forward. Conversely, resuming
          motion and advancing the route when the way is clear is GOOD. (Do NOT penalize stopping
          that is justified by a red light, stop sign, close leading vehicle, or a pedestrian/yield.)
        - If the vehicle encounters an obstacle blocking the entire route, does it stop entirely before the obstruction? (if yes, GOOD; if no, BAD)
        - If the vehicle encounters an obstacle blocking part of the route (one lane), does it stop and wait for a gap to go around the obstruction? (if yes, GOOD; if no, BAD)

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

    def complete_text(self, prompt: str) -> str:
        """Text-only completion (used for per-chunk feedback refinement)."""
        raise NotImplementedError(f"{type(self).__name__} does not support text-only completion.")


# ── Gemini REST API helpers (Python-3.8-compatible; no google-genai package needed) ──

_GEMINI_UPLOAD_BASE = "https://generativelanguage.googleapis.com/upload/v1beta"
_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _gemini_upload_file(path: Path, api_key: str) -> dict[str, Any]:
    """Upload *path* to the Gemini Files API via the resumable-upload protocol.

    Returns the file metadata dict (keys: ``name``, ``uri``, ``state``, …).
    """
    import mimetypes

    import requests

    mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    file_size = path.stat().st_size

    # Step 1 — initiate resumable upload session.
    start_resp = requests.post(
        f"{_GEMINI_UPLOAD_BASE}/files",
        params={"uploadType": "resumable", "key": api_key},
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": path.name}},
        timeout=30,
    )
    start_resp.raise_for_status()
    upload_url = start_resp.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        raise RuntimeError("Gemini upload: missing X-Goog-Upload-URL in response headers.")

    # Step 2 — stream the file content (SDK uses POST, not PUT).
    with path.open("rb") as fh:
        data = fh.read()
    upload_resp = requests.post(
        upload_url,
        headers={
            "Content-Length": str(file_size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
            "Content-Type": mime_type,
        },
        data=data,
        timeout=120,
    )
    upload_resp.raise_for_status()
    body = upload_resp.json()
    # The Files API wraps the metadata under a "file" key on upload.
    return body.get("file", body)


def _gemini_get_file(name: str, api_key: str) -> dict[str, Any]:
    """Fetch current metadata for a Gemini-uploaded file by its resource name."""
    import requests

    resp = requests.get(
        f"{_GEMINI_API_BASE}/{name}",
        params={"key": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


_GEMINI_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _gemini_generate_content(model: str, contents: list[Any], api_key: str, max_retries: int = 5) -> str:
    """Call ``models.generateContent`` and return the response text.

    Retries up to *max_retries* times on transient errors (429/5xx) with
    exponential backoff.
    """
    import requests

    delay = 5.0
    for attempt in range(max_retries + 1):
        resp = requests.post(
            f"{_GEMINI_API_BASE}/models/{model}:generateContent",
            params={"key": api_key},
            json={"contents": contents},
            timeout=120,
        )
        if resp.status_code not in _GEMINI_RETRYABLE_STATUS:
            break
        if attempt == max_retries:
            break
        print(
            f"[vlm_feedback] Gemini {resp.status_code} on attempt {attempt + 1}/{max_retries};"
            f" retrying in {delay:.0f}s…",
            flush=True,
        )
        time.sleep(delay)
        delay = min(delay * 2, 60.0)

    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response structure: {data}") from exc


class GeminiVLMCOach(VLMCOach):
    """Coach backed by the Google Gemini REST API (no google-genai package required)."""

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
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")

        # Upload video and wait for it to become ACTIVE.
        # The upload response may omit "state" when the file is already ready;
        # in that case do a GET to get the authoritative status.
        uploaded = _gemini_upload_file(path, self.api_key)
        if uploaded.get("state") is None:
            uploaded = _gemini_get_file(uploaded["name"], self.api_key)
        while uploaded.get("state") == "PROCESSING":
            time.sleep(1.0)
            uploaded = _gemini_get_file(uploaded["name"], self.api_key)
        if uploaded.get("state") != "ACTIVE":
            raise RuntimeError(f"Gemini file upload failed with state={uploaded.get('state')!r}.")

        parts: list[Any] = [
            {
                "file_data": {
                    "mime_type": uploaded.get("mimeType", "video/mp4"),
                    "file_uri": uploaded["uri"],
                }
            }
        ]

        if include_plots_in_prompt and plot_paths:
            for plot_path in plot_paths:
                plot_uploaded = _gemini_upload_file(Path(plot_path), self.api_key)
                if plot_uploaded.get("state") is None:
                    plot_uploaded = _gemini_get_file(plot_uploaded["name"], self.api_key)
                while plot_uploaded.get("state") == "PROCESSING":
                    time.sleep(0.5)
                    plot_uploaded = _gemini_get_file(plot_uploaded["name"], self.api_key)
                if plot_uploaded.get("state") != "ACTIVE":
                    raise RuntimeError(
                        f"Gemini plot upload failed for {plot_path} "
                        f"with state={plot_uploaded.get('state')!r}."
                    )
                parts.append(
                    {
                        "file_data": {
                            "mime_type": plot_uploaded.get("mimeType", "image/png"),
                            "file_uri": plot_uploaded["uri"],
                        }
                    }
                )

        prompt = build_coaching_prompt(metadata)
        parts.append({"text": prompt})

        text = _gemini_generate_content(self.model, [{"parts": parts}], self.api_key)
        return parse_coach_response(text)

    def complete_text(self, prompt: str) -> str:
        """Text-only completion (used for per-chunk feedback refinement)."""
        return _gemini_generate_content(
            self.model,
            [{"parts": [{"text": prompt}]}],
            self.api_key,
        )


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


def generate_action_chunk_feedback(
    coach: VLMCOach,
    metadata: dict[str, Any],
    events: list[CoachEvent],
    *,
    steps_per_chunk: int = 10,
    chunk_duration_sec: float = 0.5,
    bad_event_radius_chunks: int = 2,
) -> dict[str, Any]:
    """Map episode BAD events → per-action-chunk lateral/longitudinal feedback BoW labels.

    Returns the assembled chunk-feedback JSON dict (see coaches/action_chunk_feedback.py).
    Bad events near a GOOD/BAD boundary are expanded by ``bad_event_radius_chunks`` chunks
    on each side; all other chunks receive ``null`` feedback.
    """
    from coaches.action_chunk_feedback import (
        affected_chunks_from_bad_events,
        assemble_chunk_feedback_json,
        build_action_chunk_specs,
        build_chunk_feedback_prompt,
        parse_chunk_feedback_response,
    )

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
        response_text = coach.complete_text(prompt)
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
    import tyro

    tyro.cli(main)
