"""Episode-level strategy review: what the vehicle actually tried, and what it scored.

After an episode ends this makes ONE extra VLM call over:

  * the full-episode rollout video (not a 150-step window -- the whole route attempt),
  * every correction the CAST reviewer made during that episode, re-timed into the episode's
    own clock so a correction can be pointed at in the video it belongs to,
  * the subtasks the policy actually executed, and the routing commands it was carrying out,
  * and the leaderboard driving score the episode earned.

The VLM answers in one or two sentences: what strategy the vehicle followed, which behaviour cost
or earned the most, and what to do differently. That sentence goes into :class:`StrategyMemory`,
whose rendered block is injected into BOTH the window-review and the credit prompt, so the NEXT
episode's labeling is written with "last time we crept to every junction and scored 27" in view
rather than starting from scratch.

This replaced ``coaches/correction_memory.py``, which remembered individual within-window
corrections as longitudinal mode transitions plus notes, carried vehicle state and crash ageing
across windows, and pruned and coach-summarised itself to fit a word budget. Five interacting
mechanisms whose combined output was a table of ``stop -> accelerate: 7x`` -- an episode summary
says the same thing causally, in one sentence, and can be read.

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

# Same derivation the window review uses, so both judge the episode against one objective.
from coaches.vlm_feedback import describe_route_goal

# One or two sentences. This is a nudge that has to share a small word budget with the correction
# log, not an essay.
DEFAULT_MAX_SENTENCE_WORDS = 60

# How many episode summaries to carry. The most recent matter most, but a few older ones
# keep a trend visible ("scored 43, then 60, then 43 again") that one sentence cannot show.
DEFAULT_MAX_ENTRIES = 8


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


def collect_episode_chunks(artifact_dir: str | Path, episode: int) -> list[dict[str, Any]]:
    """Every action chunk of ``episode``: the subtask the policy EXECUTED and its verdict.

    The events collected above are the reviewer's commentary. These are the policy's own
    decisions -- ``original_subtask`` is what it actually chose to do over those steps, and
    ``suggested_subtasks`` is what the reviewer relabelled it to. Summarising "strategy" without
    them describes the critic rather than the driver.
    """
    root = Path(artifact_dir)
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for win_dir in sorted(root.glob(f"ep{int(episode):04d}_win*")):
        path = win_dir / "cast_relabel.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for c in data.get("action_chunks") or []:
            suggested = c.get("suggested_subtasks") or []
            if isinstance(suggested, str):
                suggested = [suggested]
            out.append(
                {
                    "episode_step_start": c.get("episode_step_start"),
                    "episode_step_end": c.get("episode_step_end"),
                    "label": str(c.get("label", "")),
                    "credit_source": str(c.get("credit_source", "")),
                    "executed_subtask": strip_cot(str(c.get("original_subtask", ""))),
                    "corrected_subtask": strip_cot(str(suggested[0])) if suggested else "",
                }
            )
    out.sort(key=lambda c: (c["episode_step_start"] is None, c["episode_step_start"] or 0))
    return out


def strip_cot(text: str) -> str:
    """Drop CoT sentinels and collapse whitespace, so the table reads as plain subtasks."""
    t = re.sub(r"<[^>]{0,40}>", " ", str(text or ""))
    return re.sub(r"\s+", " ", t).strip()


def format_executed_block(
    chunks: list[dict[str, Any]], *, episode_fps: float | None = None, max_runs: int = 25
) -> str:
    """What the policy executed, with consecutive identical subtasks collapsed into runs.

    Collapsing matters: a 400-step episode is ~40 chunks and the same subtask usually persists
    across many of them, so an uncollapsed list reads as noise and buries the handful of places
    the behaviour actually changed -- which is exactly what a strategy summary needs to see.
    """
    if not chunks:
        return "(no executed subtasks were recorded for this episode)"
    runs: list[dict[str, Any]] = []
    for c in chunks:
        key = (c["executed_subtask"], c["label"], c["corrected_subtask"])
        if runs and runs[-1]["key"] == key:
            runs[-1]["end"] = c["episode_step_end"]
            runs[-1]["n"] += 1
            continue
        runs.append(
            {
                "key": key,
                "start": c["episode_step_start"],
                "end": c["episode_step_end"],
                "n": 1,
                **c,
            }
        )
    lines = []
    for r in runs[:max_runs]:
        start, end = r["start"], r["end"]
        if episode_fps and start is not None and end is not None:
            when = f"t={start / episode_fps:5.1f}-{end / episode_fps:5.1f}s"
        else:
            when = f"steps {start}-{end}"
        label = r["label"] or "?"
        if r["credit_source"]:
            label += f"/{r['credit_source']}"
        line = f"- {when} [{label}] executed: \"{r['executed_subtask'] or '(none)'}\""
        if r["corrected_subtask"] and r["corrected_subtask"] != r["executed_subtask"]:
            line += f"  ->  relabelled to: \"{r['corrected_subtask']}\""
        if r["n"] > 1:
            line += f"  ({r['n']} consecutive chunks)"
        lines.append(line)
    if len(runs) > max_runs:
        lines.append(f"- (+{len(runs) - max_runs} further runs omitted)")
    return "\n".join(lines)


def format_route_plan_block(route_command_plan: list[dict[str, Any]] | None) -> str:
    """The routing commands the episode was given, in order."""
    if not route_command_plan:
        return ""
    lines = []
    for i, item in enumerate(route_command_plan):
        if not isinstance(item, dict):
            continue
        cmd = str(item.get("command", "")).strip()
        if not cmd:
            continue
        dist = item.get("start_distance_m")
        lines.append(
            f"  {i + 1}. {cmd}" + (f" (from ~{float(dist):.0f} m along the route)" if dist is not None else "")
        )
    if not lines:
        return ""
    return (
        "\nRouting commands this episode was given, in order — the task the vehicle was actually "
        "asked to carry out:\n" + "\n".join(lines) + "\n"
    )


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
    executed_block: str = "",
    route_plan_block: str = "",
    max_words: int = DEFAULT_MAX_SENTENCE_WORDS,
) -> str:
    """Prompt for the episode-level strategy summary.

    Mirrors the review prompt's framing deliberately: the episode is judged against the same
    PRIMARY GOAL, derived the same way (scenario family + the routing-command plan). A strategy
    summary written against a different notion of the objective than the one the window reviews
    used would push the labelling in a direction the reviews never intended.
    """
    if not route_goal:
        route_goal = describe_route_goal(route)
    score_line = f"Driving score for this episode: {driving_score:.2f} / 100."
    if route_completion is not None:
        score_line += f" Route completion: {route_completion:.1f}%."
    goal_block = (
        f"PRIMARY GOAL of this episode (route `{route}`): {route_goal}.\n"
        "This is what the vehicle was TRYING TO ACHIEVE. The routing commands below are the means "
        "to it, not the end — an episode that obeyed every command but never accomplished the "
        "goal has FAILED, whatever else it did well.\n"
        if route_goal
        else f"Route: `{route}`.\n"
    )
    return (
        "You are reviewing a COMPLETE driving episode, not a short window. The attached video is "
        "the whole route attempt.\n\n"
        f"{goal_block}"
        f"{route_plan_block}\n"
        f"{score_line}\n\n"
        "What the policy ACTUALLY DID — the subtask it chose over each stretch, the reviewer's "
        "verdict on it, and where the reviewer relabelled it. Consecutive identical subtasks are "
        "collapsed into one line:\n\n"
        f"{executed_block}\n\n"
        "Moments the reviewer flagged, with the correction where the behaviour was bad. "
        "Timestamps are in the attached video's own clock:\n\n"
        f"{corrections_block}\n\n"
        "Summarise, in AT MOST "
        f"{max_words} words and as one or two plain sentences:\n"
        "  1. the strategy the vehicle actually followed over this episode — read it off the "
        "executed subtasks above, as a pattern rather than a list (e.g. 'crept to every junction "
        "and waited for a full gap');\n"
        "  2. whether that strategy achieved the PRIMARY GOAL and produced the score above, "
        "naming the ONE behaviour that cost the most (or, on a high score, the one that earned "
        "it);\n"
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
    chunks: list[dict[str, Any]] | None = None,
    route_command_plan: list[dict[str, Any]] | None = None,
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
        executed_block=format_executed_block(chunks or [], episode_fps=episode_fps),
        route_plan_block=format_route_plan_block(route_command_plan),
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


class StrategyMemory:
    """Bounded, persistent record of how previous episodes of this run played out.

    Deliberately simple: a list of ``{episode, driving_score, sentence}``, newest last, capped by
    count. It replaced ``CorrectionMemory``, which tracked longitudinal mode transitions, free-text
    notes, vehicle-state carry-over and crash ageing, and then pruned and coach-summarised itself
    to fit a word budget -- five interacting mechanisms whose combined output was a table of
    ``stop -> accelerate: 7x``. An episode summary says the same thing causally and in one
    sentence, so none of that machinery is needed to earn its keep.
    """

    def __init__(self, path: str | Path | None = None, *, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self.path = Path(path) if path else None
        self.max_entries = max(1, int(max_entries))
        self.strategies: list[dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        if not self.path or not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a corrupt cache must not stop a run
            print(f"[strategy_memory] could not read {self.path} ({exc}); starting empty.", flush=True)
            return
        self.strategies = list(raw.get("strategies") or [])

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {"version": 1, "max_entries": self.max_entries,
                     "strategies": self.strategies, "rendered": self.render()},
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - best effort
            print(f"[strategy_memory] could not write {self.path} ({exc}).", flush=True)

    def add_strategy(self, sentence: str, *, episode: int, driving_score: float) -> None:
        """Record an episode's summary and the score it produced."""
        text = " ".join(str(sentence or "").split()).strip()
        if not text:
            return
        self.strategies.append(
            {"episode": int(episode), "driving_score": float(driving_score), "sentence": text}
        )
        del self.strategies[: -self.max_entries]
        self.save()

    def render(self) -> str:
        """The block injected into both prompts. Empty until an episode has finished."""
        if not self.strategies:
            return ""
        lines = [
            f"- episode {e['episode']} (score {e['driving_score']:.1f}): {e['sentence']}"
            for e in self.strategies
        ]
        return (
            "\nStrategy memory — how PREVIOUS episodes of this run played out, and what they "
            "scored. Use it to steer the high-level strategy of your labelling: if an approach "
            "already scored badly, do not keep correcting toward it; if one scored well, keep "
            "your corrections consistent with it. These are outcomes, not instructions about "
            "this window — the video always wins.\n" + "\n".join(lines) + "\n"
        )
