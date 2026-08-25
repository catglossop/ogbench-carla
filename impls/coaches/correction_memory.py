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
    ) -> None:
        """Fold one window's corrections into the memory, then re-fit it to the word budget."""
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

    # ── rendering + budget ───────────────────────────────────────────────────────
    def render(self) -> str:
        """The block injected into both prompts. Empty until something has been corrected."""
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
            return ""
        return (
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
