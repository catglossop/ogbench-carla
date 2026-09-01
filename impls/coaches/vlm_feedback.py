"""VLM coaches that review driving rollout videos and annotate good/bad moments.

Supports:
  - Google Gemini (``gemini-2.0-flash`` by default)
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


# Decimal places for every numeric in the per-timestamp block. The values are float32 and were
# previously dumped at full repr precision (``-0.6499999761581421`` for a steer of -0.65), which
# reads as significance that isn't there. 2dp is safe for the env reward too: across ~342k recorded
# steps only 0.1% of nonzero rewards fall below 0.005.
TELEMETRY_DECIMALS = 2

# Speed/control fields are emitted on every Nth timestamp; every other field appears on all of
# them. The block only ever holds video-aligned steps (one per ``video_frame_stride`` env steps),
# so N=2 means speed and controls land every 2 frames — every 4 env steps at the default stride.
# Reward and route progress stay on every row: they are the signals the credit pass leans on, and
# they are also attached to the prompt as a plot.
CONTROL_SAMPLE_EVERY = 2

_CONTROL_FIELDS = ("ego_speed_mps", "control_throttle", "control_steer", "control_brake")


def _round(value: Any, places: int = TELEMETRY_DECIMALS) -> Any:
    """Round a numeric to ``places`` dp, passing anything that isn't a number straight through."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), places)
    return value


def _build_per_timestamp_block(metadata: dict[str, Any]) -> str:
    """Render a timestamp-keyed JSON block of per-step trajectory data for the prompt.

    Top-level keys are video seconds (as strings); each value bundles the vehicle state and
    controls already recorded per step with the env reward earned at that step, the executed
    subtask, chain-of-thought reasoning, and the prompt the policy received at that moment. Steps
    that were not captured in the video (``video_timestamp_sec is None``) are skipped, since the
    VLM can only cross-reference moments it can actually see.

    Two densities, so the block stays readable without dropping what matters: ``reward_total`` and
    ``route_progress_pct`` (plus the step index, collision flag and CoT text) appear on every row,
    while speed and the control channels are thinned to every ``CONTROL_SAMPLE_EVERY``-th row —
    they vary smoothly and are plotted in full alongside the video.
    """
    steps = metadata.get("steps", []) or []
    per_timestamp: dict[str, Any] = {}
    kept = 0
    for s in steps:
        if not isinstance(s, dict):
            continue
        ts = s.get("video_timestamp_sec")
        if ts is None:
            continue
        entry: dict[str, Any] = {"episode_step": s.get("episode_step")}
        if kept % max(1, CONTROL_SAMPLE_EVERY) == 0:
            for field in _CONTROL_FIELDS:
                entry[field] = _round(s.get(field))
        entry["collision"] = s.get("collision")
        entry["route_progress_pct"] = _round(s.get("route_progress_pct"))
        # Env reward actually earned at this step — the objective the RL side optimizes.
        entry["reward_total"] = _round(s.get("reward_total"))
        entry["subtask"] = s.get("subtask", "")
        entry["reasoning"] = s.get("reasoning", "")
        entry["prompt"] = s.get("prompt", "")
        per_timestamp[f"{float(ts):.2f}"] = entry
        kept += 1
    if not per_timestamp:
        return ""
    return (
        "\nPer-timestamp trajectory data (keys are video seconds; each value is the vehicle "
        "state/controls plus the executed subtask, chain-of-thought reasoning, and the prompt "
        "the policy received at that moment). ``reward_total`` and ``route_progress_pct`` are "
        "given on EVERY timestamp; ``ego_speed_mps`` and the ``control_*`` channels are sampled "
        f"every {CONTROL_SAMPLE_EVERY} timestamps to keep this readable — where they are absent "
        "they simply were not sampled, so read them off the neighbouring timestamps or the "
        "attached plot rather than assuming a change. All numbers are rounded to "
        f"{TELEMETRY_DECIMALS} decimal places. ``reward_total`` is the environment reward earned "
        "at that step — the objective the policy is being trained on. It is dominated by route "
        "progress and is reduced by collisions, route/traffic infractions, and harsh steering or "
        "braking; a sustained near-zero or negative stretch marks behavior the environment itself "
        "scores as bad. Use it as corroborating evidence, not as the verdict: it cannot see "
        "everything you can in the video, so flag behavior that looks wrong even where the reward "
        "does not, and say so when the two disagree:\n"
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
                    "route_completed", "collision_events",
                    # Env reward over the window (absent for producers that don't record it).
                    "window_reward_total", "window_reward_mean")
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

        def _collision_time(event: dict[str, Any]) -> str:
            # A collision step whose frame failed to extract carries no timestamp; formatting it
            # as a float would raise and take the whole window's review down with it.
            ts = event.get("video_timestamp_sec")
            return f"{float(ts):.2f}s" if ts is not None else "unknown (not captured in video)"

        collision_lines = "\n".join(
            f"  t={_collision_time(e)}  "
            f"new_event={e.get('new_event')}  contact_active={e.get('contact_active')}"
            for e in collision_events
        )
        collision_section = (
            f"\nCollision log (from on-board sensors — cross-reference with the video):\n"
            f"{collision_lines}\n"
        )
    else:
        collision_section = "\nNo collisions were recorded by the on-board sensors.\n"

    # Attached alongside the video when the caller uploads plots. Without this the images arrive
    # unannounced and the model has no reason to read them as the same window it is watching.
    plots_section = (
        "\nA plot is attached with this video: env reward per step (top) and route progress "
        "(bottom), both against the SAME video-time axis as the video and your event timestamps. "
        "Light vertical rules are action-chunk boundaries; red rules are collisions. Use it to "
        "locate flat-zero reward stretches and places where progress stops climbing, then look at "
        "those moments in the video to say what caused them.\n"
        if include_plots else ""
    )

    # Bounded record of what earlier windows of this run already corrected, so successive reviews
    # don't flip the same behaviour back and forth. Written by coaches.correction_memory; absent
    # until something has actually been corrected.
    memory_block = str(metadata.get("correction_memory") or "")

    return textwrap.dedent(
        f"""
        You are reviewing a driving rollout video for an autonomous vehicle policy. You are seeing the view of the ego vehicle from the perspective of the ego vehicle's camera.
        {task_overview_block}
        Episode summary:
        ```json
        {metadata_block}
        ```
        {collision_section}{progress_block}{memory_block}{plots_section}{per_timestamp_block}
        Carefully watch the full video and identify moments where driving behavior was clearly
        good or clearly bad (lane keeping, speed, turns, collisions/near-misses,
        stopping, yielding, route progress, etc.). The collision log above shows ground-truth
        sensor data — use it to anchor your feedback to the correct timestamps. IMPORTANT: the 
        subtask should always aline with the route command plan. For example, if the route command plan
        if to turn left at the intersection, "turn right" would be a blatant violation of the route command plan.
        If this is violated, make sure it is in the description of the event.
        
        Everything that is violated in the below steps should be in the description of the event.
        
        ** Step 1 **
        Determine the state of the vehicle and of the other agents (vehicles, pedestrians, cyclists, etc.) in the video. 
        Ask these questions to guide your analysis:
        - What lane in the vehicle in? 
        - What other vehicles are there? What lanes are they in? 
        - Is there a leading vehicle? If so, how far ahead is it? Is it stopped or moving? 
        - Are there any pedestrians or cyclists in the video? If so, what are they doing? 
        - If you can see traffic lights or stop signs, where are they and what are their colors? Timing is important here.
        - IMPORTANT: Always also infer the traffic light states from the traffic flow. Vision can be unreliable, so you must use the traffic flow to infer the traffic light states.
          For example: 
            * cross-traffic moving through the junction => our direction is almost certainly red
            * the queue ahead of us discharging, or the lead vehicle pulling away => green
            * the lead vehicle stopped at the line with no obstruction ahead of it => red
            * pedestrians crossing our path => our direction is red
          Traffic flow is observable from the same camera the car has, so an inference of this
          kind is legitimate; a colour you cannot actually see is not. If neither the lamp nor
          the traffic flow settles it, say the signal state is UNKNOWN and do not raise a
          GOOD/BAD event that depends on it.
        - Where is the egovehicle in relation to other vehicles, pedestrians, lanes, crosswalks etc.?
        - According to the overall task (the ordered routing-command plan above) and the current
          routing command in the per-timestamp data, what maneuver should the vehicle be
          performing right now (following the road, turning left/right at the intersection,
          changing lanes, etc.), and is it in the correct lane and position to do so?

        ** Step 2 **
        Analyze the vehicle's behavior in the video. For example, you can ask these questions to guide your analysis: 
        - Does the vehicle's behavior conflict with the route command plan? (if yes, BAD; if no, GOOD). 
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
        - Is the vehicle an appropriate distance from the crosswalk at a red light? If not, it should move forward to be closer to the crosswalk to not leave an unnecessary gap.
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

    def complete_image_text(self, image: Any, prompt: str) -> str:
        """Single-image + text completion (used by the GRPO VLM critic to score candidates)."""
        raise NotImplementedError(f"{type(self).__name__} does not support image+text completion.")


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

        prompt = build_coaching_prompt(
            metadata, include_plots=bool(include_plots_in_prompt and plot_paths)
        )
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

    def complete_image_text(self, image: Any, prompt: str) -> str:
        """Single-frame + text completion via an inline JPEG part."""
        import base64
        import io

        import numpy as np
        from PIL import Image

        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        parts = [{"inline_data": {"mime_type": "image/jpeg", "data": b64}}, {"text": prompt}]
        return _gemini_generate_content(self.model, [{"parts": parts}], self.api_key)


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
    import tyro

    tyro.cli(main)
