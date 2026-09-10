"""Episode-level strategy review: what the vehicle actually tried, and what it scored.

``correction_memory`` remembers individual corrections *within* a run, as longitudinal mode
transitions plus a few notes. That keeps consecutive windows from contradicting each other, but it
is deliberately myopic: every entry is a local fix, and nothing ever asks whether the accumulated
fixes added up to a route that scored well.

This module closes that loop. After an episode ends it makes ONE extra VLM call over:

  * the full-episode rollout video (not a 150-step window -- the whole route attempt), and
  * every correction the CAST reviewer made during that episode, re-timed into the episode's
    own clock so a correction can be pointed at in the video it belongs to, and
  * the leaderboard driving score the episode actually earned.

The VLM answers in one or two sentences: what strategy the vehicle followed, which corrections
mattered, and how that relates to the score. That sentence goes into the same
:class:`CorrectionMemory` bank the window prompts already read, so the NEXT episode's labeling is
written with "last time we kept braking early at junctions and scored 43" in view rather than
starting from scratch.

Re-timing note. Window artifacts carry event timestamps in *window* seconds, but their action
chunks carry BOTH ``episode_step_start/end`` and ``video_time_start/end_sec``. That pair defines a
linear map from window time to episode step, per window, derived from the artifact itself rather
than assumed -- so it stays correct regardless of frame-sampling or window length changes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# One or two sentences. This is a nudge that has to share a small word budget with the correction
# log, not an essay.
DEFAULT_MAX_SENTENCE_WORDS = 60


def _fps_from_chunks(chunks: list[dict[str, Any]]) -> float | None:
    """Env steps per second of window video, derived from a window's own chunk table.

    Each chunk states the episode-step span it covers and the window-video seconds it occupies, so
    the ratio is the sampling rate that window was written at. Taken over the widest chunk for
    stability; ``None`` when the artifact cannot support the calculation.
    """
    best = None
    for c in chunks:
        try:
            ds = float(c["episode_step_end"]) - float(c["episode_step_start"])
            dt = float(c["video_time_end_sec"]) - float(c["video_time_start_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if ds > 0 and dt > 0 and (best is None or ds > best[0]):
            best = (ds, ds / dt)
    return None if best is None else best[1]


def episode_steps_per_video_second(
    video_path: str | Path, max_episode_step: int
) -> float | None:
    """Env steps per second of the FULL-episode video, probed from the file itself.

    The episode video is written at a fixed frame rate from a SUBSAMPLED frame list -- roughly 197
    frames for a 400-step episode -- so its playback fps is not the env-step rate, and a
    correction at episode step 300 is not at t=30 s. Reading the real frame count and fps out of
    the container gives the true mapping without threading the frame list through main_carla, and
    it cannot drift if the sampling changes.

    Returns ``None`` when the video is unreadable, in which case the caller falls back to showing
    raw episode steps -- a missing timestamp is better than a confidently wrong one pointing the
    reviewer at the wrong moment.
    """
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()
    except Exception:  # noqa: BLE001 - probing is best-effort
        return None
    if n_frames <= 0 or fps <= 0 or max_episode_step <= 0:
        return None
    duration_sec = n_frames / fps
    if duration_sec <= 0:
        return None
    return float(max_episode_step) / duration_sec


def collect_episode_corrections(artifact_dir: str | Path, episode: int) -> list[dict[str, Any]]:
    """Every CAST event for ``episode``, re-timed into episode steps and sorted.

    Reads the per-window ``cast_relabel.json`` artifacts rather than requiring the session to have
    kept them in memory, so this works equally well after the fact on a finished run.
    """
    root = Path(artifact_dir)
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for win_dir in sorted(root.glob(f"ep{int(episode):04d}_win*")):
        path = win_dir / "cast_relabel.json"
        if not path.is_file():
            continue  # a window whose VLM review failed leaves the dir but no artifact
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        chunks = data.get("action_chunks") or []
        events = data.get("events") or []
        if not events:
            continue
        # Anchor: the earliest episode step this window covers, and the window-video second that
        # corresponds to it.
        try:
            base_step = min(float(c["episode_step_start"]) for c in chunks)
            base_time = min(float(c["video_time_start_sec"]) for c in chunks)
        except (KeyError, TypeError, ValueError):
            base_step, base_time = None, 0.0
        rate = _fps_from_chunks(chunks)
        for ev in events:
            try:
                t = float(ev.get("timestamp_sec"))
            except (TypeError, ValueError):
                continue
            step = None
            if base_step is not None and rate:
                step = base_step + (t - base_time) * rate
            out.append(
                {
                    "window_index": data.get("window_index"),
                    "episode_step": None if step is None else round(step),
                    "window_time_sec": t,
                    "label": str(ev.get("label", "")),
                    "description": str(ev.get("description", "")),
                    "correction": str(ev.get("correction", "")),
                }
            )
    # Unknown-step events sort last rather than being dropped -- they still say what was corrected.
    out.sort(key=lambda e: (e["episode_step"] is None, e["episode_step"] or 0, e["window_time_sec"]))
    return out


def format_corrections_block(
    corrections: list[dict[str, Any]], *, episode_fps: float | None = None, max_events: int = 40
) -> str:
    """Render corrections as a timestamped table keyed to the FULL-episode video."""
    if not corrections:
        return "(no corrections were recorded for this episode)"
    lines = []
    for e in corrections[:max_events]:
        step = e.get("episode_step")
        if step is not None and episode_fps:
            when = f"t={step / episode_fps:6.1f}s (episode step {step})"
        elif step is not None:
            when = f"episode step {step}"
        else:
            when = f"window {e.get('window_index')} t={e['window_time_sec']:.1f}s"
        text = e["description"].strip()
        if e["correction"].strip():
            text += f"  ->  CORRECTION: {e['correction'].strip()}"
        lines.append(f"- [{e['label'] or '?'}] {when}: {text}")
    if len(corrections) > max_events:
        lines.append(f"- (+{len(corrections) - max_events} further events omitted)")
    return "\n".join(lines)


def build_strategy_prompt(
    *,
    route: str,
    route_goal: str,
    driving_score: float,
    route_completion: float | None,
    corrections_block: str,
    max_words: int = DEFAULT_MAX_SENTENCE_WORDS,
) -> str:
    """Prompt for the episode-level strategy summary."""
    score_line = f"Driving score for this episode: {driving_score:.2f} / 100."
    if route_completion is not None:
        score_line += f" Route completion: {route_completion:.1f}%."
    goal_line = f"The route's objective was: {route_goal}." if route_goal else ""
    return (
        "You are reviewing a COMPLETE driving episode, not a short window. The attached video is "
        "the whole route attempt.\n\n"
        f"Route: `{route}`. {goal_line}\n"
        f"{score_line}\n\n"
        "During the episode a reviewer flagged the following moments and, where the behavior was "
        "bad, wrote a correction. Timestamps are in the attached video's own clock:\n\n"
        f"{corrections_block}\n\n"
        "Summarise, in AT MOST "
        f"{max_words} words and as one or two plain sentences:\n"
        "  1. the strategy the vehicle actually followed over this episode (not a list of "
        "events -- the pattern, e.g. 'crept to every junction and waited for a full gap');\n"
        "  2. whether that strategy is what produced the score above, naming the ONE behavior "
        "that cost the most (or, on a high score, the one that earned it);\n"
        "  3. what the next episode should do differently, stated as a strategy rather than a "
        "single fix.\n\n"
        "Be concrete and causal. Do not restate the score. Do not hedge. Return ONLY the "
        "sentences, with no preamble, bullet points, or markdown."
    )


def _tidy_sentence(text: str, max_words: int) -> str:
    """Collapse a model reply to a single clean line within budget."""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    s = re.sub(r"^(here('s| is)[^:]*:|summary:|strategy:)\s*", "", s, flags=re.IGNORECASE).strip()
    s = s.strip("`*_ ")
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words]).rstrip(",;:") + "…"
    return s


def summarize_episode_strategy(
    coach: Any,
    *,
    video_path: str | Path | None,
    corrections: list[dict[str, Any]],
    route: str,
    route_goal: str = "",
    driving_score: float = 0.0,
    route_completion: float | None = None,
    episode_fps: float | None = None,
    max_words: int = DEFAULT_MAX_SENTENCE_WORDS,
) -> str:
    """One sentence describing the episode's strategy in the context of its score.

    Returns ``""`` on any failure -- this is an enrichment, and must never take down an episode
    that has already been driven and scored.
    """
    if episode_fps is None and video_path and corrections:
        steps = [c["episode_step"] for c in corrections if c.get("episode_step") is not None]
        if steps:
            episode_fps = episode_steps_per_video_second(video_path, max(steps))
    prompt = build_strategy_prompt(
        route=route,
        route_goal=route_goal,
        driving_score=driving_score,
        route_completion=route_completion,
        corrections_block=format_corrections_block(corrections, episode_fps=episode_fps),
        max_words=max_words,
    )
    try:
        path = Path(video_path) if video_path else None
        if path is not None and path.is_file():
            # Goes through the coach's own retrying upload, so a transient INTERNAL on the
            # episode video costs a retry rather than the whole summary.
            reply = coach.analyze_video_text(str(path), prompt)
        else:
            # No video (or no video-capable coach): the correction table alone still supports a
            # useful summary, and is far better than skipping the episode entirely.
            reply = coach.complete_text(prompt)
    except Exception as exc:  # noqa: BLE001 - enrichment must never break the run
        print(f"[strategy_memory] episode summary failed (non-fatal): {exc}", flush=True)
        return ""
    return _tidy_sentence(reply, max_words)
