"""Map episode-level VLM coach events to per-action-chunk feedback."""

from __future__ import annotations

import json
import math
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Literal

LateralFeedback = Literal["adjust left", "adjust right"]
LongitudinalFeedback = Literal["accelerate", "decelerate"]

DEFAULT_ACTION_CHUNK_STEPS = 10
DEFAULT_CHUNK_DURATION_SEC = 0.5
DEFAULT_BAD_EVENT_RADIUS_CHUNKS = 2

LATERAL_OPTIONS = ("adjust left", "adjust right")
LONGITUDINAL_OPTIONS = ("accelerate", "decelerate")


@dataclass(frozen=True)
class ActionChunkSpec:
    chunk_index: int
    episode_step_start: int
    episode_step_end: int
    video_time_start_sec: float
    video_time_end_sec: float


@dataclass(frozen=True)
class ChunkFeedback:
    lateral: LateralFeedback | None
    longitudinal: LongitudinalFeedback | None
    detail: str = ""

    def to_json(self) -> dict[str, Any] | None:
        if self.lateral is None and self.longitudinal is None and not self.detail.strip():
            return None
        out: dict[str, Any] = {
            "lateral": self.lateral,
            "longitudinal": self.longitudinal,
        }
        if self.detail.strip():
            out["detail"] = self.detail.strip()
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ChunkFeedback | None:
        if raw is None:
            return None
        lateral_raw = raw.get("lateral")
        longitudinal_raw = raw.get("longitudinal")
        lateral: LateralFeedback | None = None
        longitudinal: LongitudinalFeedback | None = None
        if lateral_raw is not None and str(lateral_raw).strip():
            lateral_val = str(lateral_raw).strip().lower()
            if lateral_val not in LATERAL_OPTIONS:
                raise ValueError(f"lateral must be one of {LATERAL_OPTIONS}, got {lateral_raw!r}.")
            lateral = lateral_val  # type: ignore[assignment]
        if longitudinal_raw is not None and str(longitudinal_raw).strip():
            long_val = str(longitudinal_raw).strip().lower()
            if long_val not in LONGITUDINAL_OPTIONS:
                raise ValueError(
                    f"longitudinal must be one of {LONGITUDINAL_OPTIONS}, got {longitudinal_raw!r}."
                )
            longitudinal = long_val  # type: ignore[assignment]
        detail = str(raw.get("detail", "")).strip()
        if lateral is None and longitudinal is None and not detail:
            return None
        return cls(lateral=lateral, longitudinal=longitudinal, detail=detail)


def num_action_chunks(episode_steps: int, *, steps_per_chunk: int = DEFAULT_ACTION_CHUNK_STEPS) -> int:
    return max(1, math.ceil(int(episode_steps) / int(steps_per_chunk)))


def timestamp_to_chunk_index(
    timestamp_sec: float,
    *,
    chunk_duration_sec: float = DEFAULT_CHUNK_DURATION_SEC,
    num_chunks: int,
) -> int:
    idx = int(float(timestamp_sec) // float(chunk_duration_sec))
    return max(0, min(int(num_chunks) - 1, idx))


def episode_step_to_chunk_index(
    episode_step: int,
    *,
    steps_per_chunk: int = DEFAULT_ACTION_CHUNK_STEPS,
) -> int:
    return max(0, (int(episode_step) - 1) // int(steps_per_chunk))


def build_action_chunk_specs(
    metadata: dict[str, Any],
    *,
    steps_per_chunk: int = DEFAULT_ACTION_CHUNK_STEPS,
    chunk_duration_sec: float = DEFAULT_CHUNK_DURATION_SEC,
) -> list[ActionChunkSpec]:
    episode_steps = int(metadata.get("episode_steps", len(metadata.get("steps", []))))
    n_chunks = num_action_chunks(episode_steps, steps_per_chunk=steps_per_chunk)
    specs: list[ActionChunkSpec] = []
    for chunk_index in range(n_chunks):
        ep_start = chunk_index * steps_per_chunk + 1
        ep_end = min((chunk_index + 1) * steps_per_chunk, episode_steps)
        specs.append(
            ActionChunkSpec(
                chunk_index=chunk_index,
                episode_step_start=ep_start,
                episode_step_end=ep_end,
                video_time_start_sec=chunk_index * chunk_duration_sec,
                video_time_end_sec=(chunk_index + 1) * chunk_duration_sec,
            )
        )
    return specs


def affected_chunks_from_bad_events(
    bad_timestamps_sec: list[float],
    *,
    num_chunks: int,
    radius_chunks: int = DEFAULT_BAD_EVENT_RADIUS_CHUNKS,
    chunk_duration_sec: float = DEFAULT_CHUNK_DURATION_SEC,
) -> set[int]:
    affected: set[int] = set()
    for ts in bad_timestamps_sec:
        center = timestamp_to_chunk_index(
            ts,
            chunk_duration_sec=chunk_duration_sec,
            num_chunks=num_chunks,
        )
        for offset in range(-int(radius_chunks), int(radius_chunks) + 1):
            idx = center + offset
            if 0 <= idx < num_chunks:
                affected.add(idx)
    return affected


def _extract_json_payload(text: str) -> dict[str, Any]:
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
        raise ValueError("Chunk feedback response must be a JSON object.")
    return payload


def build_chunk_feedback_prompt(
    *,
    bad_events: list[dict[str, Any]],
    chunk_specs: list[ActionChunkSpec],
    affected_chunk_indices: set[int],
    metadata: dict[str, Any],
    steps_per_chunk: int = DEFAULT_ACTION_CHUNK_STEPS,
    chunk_duration_sec: float = DEFAULT_CHUNK_DURATION_SEC,
    bad_event_radius_chunks: int = DEFAULT_BAD_EVENT_RADIUS_CHUNKS,
) -> str:
    affected_specs = [s for s in chunk_specs if s.chunk_index in affected_chunk_indices]
    affected_payload = [
        {
            "chunk_index": s.chunk_index,
            "episode_step_start": s.episode_step_start,
            "episode_step_end": s.episode_step_end,
            "video_time_start_sec": s.video_time_start_sec,
            "video_time_end_sec": s.video_time_end_sec,
        }
        for s in affected_specs
    ]
    return textwrap.dedent(
        f"""
        You previously reviewed a driving rollout and flagged BAD moments. Convert that
        feedback into per-action-chunk coaching for the policy.

        Each action chunk spans {steps_per_chunk} env steps ({chunk_duration_sec}s).
        Only the chunks listed below need feedback; all other chunks will be set to null.

        Episode summary:
        ```json
        {json.dumps({k: metadata.get(k) for k in ("episode", "route", "episode_steps", "success")}, indent=2)}
        ```

        BAD events from the episode review:
        ```json
        {json.dumps(bad_events, indent=2)}
        ```

        Action chunks that need feedback (within {bad_event_radius_chunks} chunks of a BAD event):
        ```json
        {json.dumps(affected_payload, indent=2)}
        ```

        For each listed chunk, return structured feedback using ONLY these controlled terms:
        - lateral: "adjust left", "adjust right", or null
        - longitudinal: "accelerate", "decelerate", or null
        - detail: optional free-text elaboration (may be empty)

        Every chunk must include at least one of lateral or longitudinal when feedback is required.
        Chunks before/after the BAD moment may need anticipatory or recovery guidance.

        Return ONLY valid JSON (no markdown fences):
        {{
          "chunk_feedback": [
            {{
              "chunk_index": 0,
              "lateral": "adjust right",
              "longitudinal": "decelerate",
              "detail": "Optional short context tied to the nearest BAD event."
            }}
          ]
        }}

        Rules:
        - Include one entry per affected chunk_index listed above.
        - Use null (not "none") when a axis does not apply.
        - Keep detail under 25 words when present.
        """
    ).strip()


def parse_chunk_feedback_response(
    text: str,
    *,
    num_chunks: int,
    affected_chunk_indices: set[int],
) -> list[dict[str, Any] | None]:
    payload = _extract_json_payload(text)
    raw_items = payload.get("chunk_feedback", [])
    if not isinstance(raw_items, list):
        raise ValueError("Chunk feedback JSON must contain a 'chunk_feedback' list.")

    by_index: dict[int, ChunkFeedback | None] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        idx = int(item["chunk_index"])
        by_index[idx] = ChunkFeedback.from_dict(item)

    missing = affected_chunk_indices - set(by_index)
    if missing:
        raise ValueError(f"Chunk feedback response missing chunk indices: {sorted(missing)}")

    out: list[dict[str, Any] | None] = []
    for chunk_index in range(num_chunks):
        if chunk_index not in affected_chunk_indices:
            out.append(None)
            continue
        fb = by_index.get(chunk_index)
        out.append(None if fb is None else fb.to_json())
    return out


def assemble_chunk_feedback_json(
    metadata: dict[str, Any],
    *,
    events: list[Any],
    chunk_feedback: list[dict[str, Any] | None],
    chunk_specs: list[ActionChunkSpec],
    affected_chunk_indices: set[int],
    steps_per_chunk: int = DEFAULT_ACTION_CHUNK_STEPS,
    chunk_duration_sec: float = DEFAULT_CHUNK_DURATION_SEC,
    bad_event_radius_chunks: int = DEFAULT_BAD_EVENT_RADIUS_CHUNKS,
) -> dict[str, Any]:
    bad_events = [
        {
            "timestamp_sec": float(e.timestamp_sec),
            "description": e.description,
            "correction": e.correction,
            "center_chunk_index": timestamp_to_chunk_index(
                e.timestamp_sec,
                chunk_duration_sec=chunk_duration_sec,
                num_chunks=len(chunk_specs),
            ),
        }
        for e in events
        if getattr(e, "label", None) == "BAD"
    ]
    chunks_out: list[dict[str, Any]] = []
    for spec, feedback in zip(chunk_specs, chunk_feedback):  # strict= requires Python 3.10+
        chunks_out.append(
            {
                "chunk_index": spec.chunk_index,
                "episode_step_start": spec.episode_step_start,
                "episode_step_end": spec.episode_step_end,
                "video_time_start_sec": spec.video_time_start_sec,
                "video_time_end_sec": spec.video_time_end_sec,
                "near_bad_event": spec.chunk_index in affected_chunk_indices,
                "feedback": feedback,
            }
        )
    return {
        "episode": metadata.get("episode"),
        "route": metadata.get("route"),
        "episode_steps": metadata.get("episode_steps"),
        "action_chunk_steps": steps_per_chunk,
        "action_chunk_duration_sec": chunk_duration_sec,
        "bad_event_radius_chunks": bad_event_radius_chunks,
        "bad_events": bad_events,
        "action_chunks": chunks_out,
    }


def active_chunk_at_time(
    chunk_feedback_json: dict[str, Any],
    timestamp_sec: float,
) -> dict[str, Any] | None:
    """Return the action-chunk record active at ``timestamp_sec``, if it has feedback."""
    for chunk in chunk_feedback_json.get("action_chunks", []):
        if chunk.get("feedback") is None:
            continue
        t0 = float(chunk["video_time_start_sec"])
        t1 = float(chunk["video_time_end_sec"])
        if t0 <= float(timestamp_sec) < t1:
            return chunk
    return None


def format_chunk_feedback_lines(
    chunk: dict[str, Any],
    *,
    wrap_width: int = 48,
) -> list[str]:
    """Render chunk feedback for the video annotation panel."""
    import textwrap

    fb = chunk.get("feedback") or {}
    idx = chunk.get("chunk_index", "?")
    t0 = float(chunk.get("video_time_start_sec", 0.0))
    t1 = float(chunk.get("video_time_end_sec", 0.0))
    lines = [f"Chunk {idx} ({t0:.1f}-{t1:.1f}s)"]
    lateral = fb.get("lateral")
    longitudinal = fb.get("longitudinal")
    if lateral:
        lines.append(f"Lateral: {lateral}")
    if longitudinal:
        lines.append(f"Longitudinal: {longitudinal}")
    detail = str(fb.get("detail", "")).strip()
    if detail:
        lines.append("Detail:")
        lines.extend(textwrap.wrap(detail, width=wrap_width))
    return lines


def chunk_feedback_to_text(feedback: dict[str, Any] | None) -> str:
    """Convert structured chunk feedback to a short corrective phrase."""
    if not feedback:
        return ""
    parts: list[str] = []
    lateral = feedback.get("lateral")
    longitudinal = feedback.get("longitudinal")
    if lateral:
        parts.append(f"{lateral}.")
    if longitudinal:
        parts.append(f"{longitudinal}.")
    detail = str(feedback.get("detail", "")).strip()
    if detail:
        if not detail.endswith("."):
            detail += "."
        parts.append(detail)
    return " ".join(parts).strip()


def chunk_feedback_to_language_label(
    feedback: dict[str, Any] | None,
) -> tuple[str, "np.ndarray"]:
    """Map chunk feedback to delta-commentary bag-of-words (DSRL critic input)."""
    from coaches.expert_label import NUM_DELTA_COMMENTARY_WORDS, delta_commentary_to_bow
    import numpy as np

    text = chunk_feedback_to_text(feedback)
    if not text:
        return "", np.zeros(NUM_DELTA_COMMENTARY_WORDS, dtype=np.float32)
    return text, delta_commentary_to_bow(text)


def language_label_for_episode_step(
    chunk_feedback_json: dict[str, Any] | None,
    episode_step: int,
    *,
    action_chunk_steps: int = DEFAULT_ACTION_CHUNK_STEPS,
) -> tuple[str, "np.ndarray"]:
    """Look up chunk feedback for an env step and encode as language BOW."""
    if chunk_feedback_json is None:
        from coaches.expert_label import NUM_DELTA_COMMENTARY_WORDS
        import numpy as np

        return "", np.zeros(NUM_DELTA_COMMENTARY_WORDS, dtype=np.float32)
    chunk_index = max(0, (int(episode_step) - 1) // int(action_chunk_steps))
    for chunk in chunk_feedback_json.get("action_chunks", []):
        if int(chunk.get("chunk_index", -1)) == chunk_index:
            return chunk_feedback_to_language_label(chunk.get("feedback"))
    return chunk_feedback_to_language_label(None)
