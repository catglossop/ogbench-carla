"""A small, bounded memory of what earlier windows already corrected.

Each CAST window is reviewed in isolation: the review call sees one video, the credit call sees
that call's events and nothing else (:meth:`vlm_feedback.GeminiVLMCOach.complete_text` carries no
history). With no memory across windows the coach can correct a chunk toward "come to a stop" in
window 12 and correct the same situation toward "accelerate and make progress" in window 13, and
the HL dataset ends up training both directions of the same decision.

This keeps a compact record of the corrections already made — as *mode transitions*
(``stop -> accelerate``) with counts, plus a few short notes — and injects it into both prompts so
later windows stay consistent with earlier ones. It is deliberately tiny: the whole rendered block
is capped at :data:`DEFAULT_MAX_WORDS` words, pruned oldest-note-first and, if still over budget,
summarized by the coach itself.

The rendered text is stashed on the window metadata as ``correction_memory``, so both prompt
builders pick it up without a signature change and every window artifact records exactly the
memory that was in play when it was reviewed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from subtask_diversity import subtask_categories

# Total word budget for the rendered block. Small on purpose: this is a consistency nudge, not a
# second source of instructions competing with the video.
DEFAULT_MAX_WORDS = 300

# Transitions are tracked on the longitudinal axis (what the vehicle does about speed), which is
# where the flip-flopping actually happens. Lateral tags are recorded in notes, not counted.
_LONGITUDINAL = ("stop", "decelerate", "maintain", "accelerate", "reverse")

# Keep the transition table bounded no matter how long a run goes.
_MAX_TRANSITIONS = 8
_MAX_NOTES = 4
# How many windows a crash keeps being mentioned before it is dropped even if the ego never
# visibly recovers. Without a cap a single unrecovered crash would caption every remaining window
# of the episode; with it, the note fades on its own.
_MAX_CRASH_CARRY_WINDOWS = 4
# What counts as "moving again" when deciding a crash is behind us: the ego must be above this
# speed at the end of a window AND have made at least ``_RECOVERED_PROGRESS_PCT`` of route
# progress during it. Speed alone is not enough -- spinning against a wall clears a speed gate.
_RECOVERED_SPEED_MPS = 1.0
_RECOVERED_PROGRESS_PCT = 0.5


def _longitudinal_mode(text: str) -> str:
    """Reduce a subtask phrase to one longitudinal mode, or ``""`` when it says nothing about speed.

    ``subtask_categories`` can return several tags ("remains stopped ... then accelerates"); the
    first match in :data:`_LONGITUDINAL` order wins so the phrase is summarized by its most
    restrictive intent, which is the one a later window must not silently reverse.
    """
    cats = subtask_categories(text or "")
    for mode in _LONGITUDINAL:
        if mode in cats:
            return mode
    return ""


def _word_count(text: str) -> int:
    return len(text.split())


class CorrectionMemory:
    """Bounded, persistent record of the corrections made so far in a run."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_words: int = DEFAULT_MAX_WORDS,
        coach: Any = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.max_words = max(40, int(max_words))
        self.coach = coach
        # "from->to" -> {"count": int, "last_window": int}
        self.transitions: dict[str, dict[str, int]] = {}
        self.notes: list[str] = []
        self.summary: str = ""
        self.windows_seen: int = 0
        # How the ego was doing at the end of the most recent window, and whether a crash from an
        # EARLIER window is still hanging over it. The reviewer sees only the current window's
        # video, so without this it re-reads a stopped car as a fresh decision to stop rather than
        # as the aftermath of a collision two windows ago.
        self.vehicle_state: str = ""
        self.crash_note: str = ""
        self.crash_age: int = 0
        self.load()

    # ── persistence ──────────────────────────────────────────────────────────────
    def load(self) -> None:
        if not self.path or not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a corrupt cache must not stop a run
            print(f"[correction_memory] could not read {self.path} ({exc}); starting empty.", flush=True)
            return
        self.transitions = dict(raw.get("transitions") or {})
        self.notes = list(raw.get("notes") or [])
        self.summary = str(raw.get("summary") or "")
        self.windows_seen = int(raw.get("windows_seen") or 0)
        self.vehicle_state = str(raw.get("vehicle_state") or "")
        self.crash_note = str(raw.get("crash_note") or "")
        self.crash_age = int(raw.get("crash_age") or 0)

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "max_words": self.max_words,
                        "windows_seen": self.windows_seen,
                        "transitions": self.transitions,
                        "notes": self.notes,
                        "summary": self.summary,
                        "vehicle_state": self.vehicle_state,
                        "crash_note": self.crash_note,
                        "crash_age": self.crash_age,
                        "rendered": self.render(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - best effort
            print(f"[correction_memory] could not write {self.path} ({exc}).", flush=True)

    # ── accumulation ─────────────────────────────────────────────────────────────
    def observe_window(
        self,
        cast_json: dict[str, Any],
        *,
        window_index: int,
        route: str = "",
        vehicle_state: dict[str, Any] | None = None,
    ) -> None:
        """Fold one window's corrections into the memory, then re-fit it to the word budget.

        ``vehicle_state`` is the window's closing ego state (see :meth:`observe_vehicle_state`).
        It is folded in unconditionally -- a window can end with the car wrecked and still contain
        no corrections at all, and that is exactly the window whose aftermath the next one needs
        to know about.
        """
        if vehicle_state is not None:
            self.observe_vehicle_state(vehicle_state, window_index=window_index)
        chunks = cast_json.get("action_chunks") or []
        seen_this_window: dict[str, int] = {}
        for chunk in chunks:
            if str(chunk.get("label") or "").strip().upper() != "BAD":
                continue
            suggested = chunk.get("suggested_subtasks") or []
            if not suggested:
                continue
            src = _longitudinal_mode(str(chunk.get("original_subtask") or ""))
            dst = _longitudinal_mode(str(suggested[0]))
            if not src or not dst or src == dst:
                continue  # nothing to remember: no speed intent, or the intent did not change
            key = f"{src} -> {dst}"
            seen_this_window[key] = seen_this_window.get(key, 0) + 1

        if not seen_this_window:
            # No corrections, but the vehicle state may have changed; keep it on disk.
            if vehicle_state is not None:
                self._fit_budget()
                self.save()
            return
        self.windows_seen += 1
        for key, n in seen_this_window.items():
            entry = self.transitions.setdefault(key, {"count": 0, "last_window": 0})
            entry["count"] = int(entry["count"]) + n
            entry["last_window"] = int(window_index)

        # One short note per window, naming its dominant correction — enough context to tell a
        # deliberate repeat from an oscillation, without storing prose per chunk.
        dominant = max(seen_this_window.items(), key=lambda kv: kv[1])
        where = f" on {route}" if route else ""
        self.notes.append(
            f"w{int(window_index)}{where}: {dominant[1]}x {dominant[0]}."
        )
        # Only the last _MAX_NOTES ever render, so don't let the stored list grow for a whole run.
        self.notes = self.notes[-_MAX_NOTES:]
        self._fit_budget()
        self.save()

    def observe_vehicle_state(self, state: dict[str, Any], *, window_index: int = 0) -> None:
        """Record how the ego ended this window, and age any crash carried from an earlier one.

        The crash note is what the user asked for: a collision in one window stays visible in the
        next, and the next, until it stops being relevant. "Relevant" ends one of two ways --

          * **Recovery.** The ego is moving again (``end_speed_mps`` over ``_RECOVERED_SPEED_MPS``)
            AND made real route progress during the window. Both are required: a car spinning its
            wheels against a barrier clears a speed test but is still stuck.
          * **Age.** ``_MAX_CRASH_CARRY_WINDOWS`` windows pass regardless, so an ego that never
            recovers does not caption the entire rest of the episode with the same sentence.

        A fresh crash always resets the counter, so repeated collisions keep the note alive.
        """
        collided = bool(state.get("collided"))
        stuck = bool(state.get("stuck"))
        speed = float(state.get("end_speed_mps") or 0.0)
        progress = float(state.get("route_progress_delta_pct") or 0.0)
        recovered = speed >= _RECOVERED_SPEED_MPS and progress >= _RECOVERED_PROGRESS_PCT

        if collided or stuck:
            what = "collided and is stuck in it" if stuck else "collided"
            self.crash_note = f"the ego {what} in window {int(window_index)}"
            self.crash_age = 1
        elif self.crash_age > 0:
            if recovered or self.crash_age >= _MAX_CRASH_CARRY_WINDOWS:
                self.crash_note = ""
                self.crash_age = 0
            else:
                self.crash_age += 1

        parts = [
            f"ended at {speed:.1f} m/s",
            (
                f"route progress {float(state.get('route_progress_end_pct') or 0.0):.1f}%"
                f" ({progress:+.1f} this window)"
            ),
        ]
        if stuck:
            parts.append("STUCK in a collision")
        elif collided:
            parts.append("collided")
        self.vehicle_state = "; ".join(parts)

    def render_vehicle_block(self) -> str:
        """The carry-over paragraph, or ``""`` when there is nothing to carry."""
        if not self.vehicle_state and not self.crash_note:
            return ""
        lines = []
        if self.vehicle_state:
            lines.append(f"- At the end of the PREVIOUS window: {self.vehicle_state}.")
        if self.crash_note:
            windows = "window" if self.crash_age == 1 else "windows"
            lines.append(
                f"- Still relevant: {self.crash_note} ({self.crash_age} {windows} ago). Read this "
                "window as the aftermath — if the ego is stopped or crawling, that may be damage "
                "or an obstruction rather than a fresh decision to stop, and the correction is "
                "whatever gets it back onto the route."
            )
        return (
            "\nVehicle state carried from the previous window (the video below starts mid-story, "
            "so this is what happened just before it):\n" + "\n".join(lines) + "\n"
        )

    # ── rendering + budget ───────────────────────────────────────────────────────
    def render(self) -> str:
        """The block injected into both prompts.

        Two independent halves: the vehicle-state carry-over (present as soon as ONE window has
        been observed, corrections or not) and the correction log (present once something has
        actually been corrected). Either can be empty.
        """
        vehicle_block = self.render_vehicle_block()
        if self.summary:
            body = self.summary
        elif self.transitions:
            ranked = sorted(
                self.transitions.items(),
                key=lambda kv: (-int(kv[1]["count"]), -int(kv[1]["last_window"])),
            )[:_MAX_TRANSITIONS]
            lines = [
                f"- {key}: {v['count']}x (latest window {v['last_window']})" for key, v in ranked
            ]
            if self.notes:
                lines.append("Recent: " + " ".join(self.notes[-_MAX_NOTES:]))
            body = "\n".join(lines)
        else:
            return vehicle_block
        return vehicle_block + (
            "\nCorrection memory — longitudinal changes earlier windows of THIS run already made, "
            f"as `was -> corrected to` with how often ({self.windows_seen} windows so far). Stay "
            "consistent with it: do not reverse a correction that was already made in a comparable "
            "situation, and if this scene really does call for the opposite, say so in the "
            "rationale. It is a summary of past decisions, not an instruction about this window — "
            "the video always wins.\n"
            f"{body}\n"
        )

    def _fit_budget(self) -> None:
        """Prune, then summarize, until :meth:`render` fits in ``max_words``."""
        while _word_count(self.render()) > self.max_words and self.notes:
            self.notes.pop(0)  # oldest note first; the transition counts are the durable part
        if _word_count(self.render()) <= self.max_words:
            return
        # Still over: the transition table itself is long. Ask the coach to compress it once and
        # keep the prose from then on (further windows fold into it via the notes path above).
        compacted = self._summarize_with_coach()
        if compacted:
            self.summary = compacted
            self.notes = []
        else:
            # No coach (or it failed): keep the most frequent transitions and drop the tail.
            ranked = sorted(
                self.transitions.items(),
                key=lambda kv: (-int(kv[1]["count"]), -int(kv[1]["last_window"])),
            )
            self.transitions = dict(ranked[: max(1, _MAX_TRANSITIONS // 2)])

    def _summarize_with_coach(self) -> str:
        if self.coach is None or not hasattr(self.coach, "complete_text"):
            return ""
        raw = "\n".join(
            f"- {k}: {v['count']}x (latest window {v['last_window']})"
            for k, v in self.transitions.items()
        )
        prompt = (
            "Summarize this log of driving-policy corrections into at most "
            f"{max(20, self.max_words - 60)} words of plain prose. Keep which behaviours were "
            "changed into which, and roughly how often; drop everything else. No preamble, no "
            "markdown, just the summary.\n\n" + raw + "\n" + " ".join(self.notes)
        )
        try:
            text = " ".join(str(self.coach.complete_text(prompt)).split())
        except Exception as exc:  # noqa: BLE001 - falls back to deterministic pruning
            print(f"[correction_memory] summarization failed ({exc}); pruning instead.", flush=True)
            return ""
        words = text.split()
        return " ".join(words[: max(20, self.max_words - 60)])
