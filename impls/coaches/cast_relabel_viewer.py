"""Offline viewer for the inputs the CAST relabel VLM calls actually receive.

The CAST relabel pipeline makes two Gemini calls per window (see :mod:`coaches.cast_relabel`):

1. **Window review** (``coach.analyze``) — the window video *plus* a text prompt carrying the
   route plan, window summary, collision log, and a per-timestamp block of telemetry / executed
   subtask / CoT reasoning / policy prompt.
2. **Credit + relabel** (``coach.complete_text``) — a fresh, text-only call carrying only the
   events from step 1, a per-chunk timing + reward + original-subtask table, and the seed phrase
   banks.

Neither call is inspectable from the artifacts alone: the prompts are built on the fly and never
written to disk. This renders one self-contained HTML page that plays the exact video the VLM saw,
scrubs the per-step telemetry against it, and shows both prompts verbatim — alongside automated
checks for the failure modes that silently corrupt the inputs (a video clock that does not start
at zero, an uneven frame cadence, chunks with no frames, events outside the video).

Usage::

    .venv/bin/python impls/coaches/cast_relabel_viewer.py <path> [-o out.html]

``<path>`` may be a single window dir (one holding ``trajectory.json``), a run's ``cast_relabel/``
dir, or a run dir containing one. Multiple paths are merged into one page.

Legacy artifacts (written before the window-relative clock fix) are rebased for display, and the
Checks panel reports the original offset so you can see what the VLM was actually sent.
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

_IMPLS_ROOT = Path(__file__).resolve().parent.parent
if str(_IMPLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPLS_ROOT))

from coaches.action_chunk_feedback import build_action_chunk_specs
from coaches.cast_relabel import (
    SEED_SUBTASKS,
    build_credit_relabel_prompt,
    retime_chunk_specs,
    strip_cot_sentinels,
)
from coaches.vlm_feedback import CoachEvent, build_coaching_prompt

DEFAULT_MAX_WINDOWS = 8


# ── artifact discovery ───────────────────────────────────────────────────────────────


def find_window_dirs(path: Path) -> list[Path]:
    """Resolve a window dir, a ``cast_relabel/`` dir, or a run dir into window dirs."""
    path = path.expanduser().resolve()
    if (path / "trajectory.json").is_file():
        return [path]
    roots = [path]
    if (path / "cast_relabel").is_dir():
        roots.append(path / "cast_relabel")
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        found.extend(sorted(p.parent for p in root.glob("*/trajectory.json")))
    # De-duplicate while preserving order (a run dir and its cast_relabel/ can both match).
    seen: set[Path] = set()
    out: list[Path] = []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ── clock rebasing ───────────────────────────────────────────────────────────────────


def rebase_steps(metadata: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    """Return ``(steps, raw_offset_sec)`` with every timestamp relative to the window's video.

    ``main_carla`` stamps ``video_frame_index`` from a counter that is never reset, so windows
    written before the fix carry a run-global clock while the video they were sent with always
    starts at t=0. Subtracting the window's first frame index restores the clock the VLM saw.
    Already-correct artifacts have offset 0 and pass through unchanged.
    """
    steps = [dict(s) for s in (metadata.get("steps") or []) if isinstance(s, dict)]
    fps = float(metadata.get("video_fps") or 10.0) or 10.0
    indices = [
        int(s["video_frame_index"]) for s in steps if s.get("video_frame_index") is not None
    ]
    base = min(indices) if indices else 0
    for s in steps:
        idx = s.get("video_frame_index")
        if idx is None:
            s["video_frame_index"] = None
            s["video_timestamp_sec"] = None
            s["in_video"] = False
        else:
            rel = int(idx) - base
            s["video_frame_index"] = rel
            s["video_timestamp_sec"] = round(rel / fps, 3)
            s["in_video"] = True
    return steps, round(base / fps, 3)


# ── checks ───────────────────────────────────────────────────────────────────────────


def _check(cid: str, title: str, status: str, detail: str, extra: Any = None) -> dict[str, Any]:
    return {"id": cid, "title": title, "status": status, "detail": detail, "extra": extra}


def run_checks(
    metadata: dict[str, Any],
    steps: list[dict[str, Any]],
    chunks_payload: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    raw_offset_sec: float,
    duration_sec: float,
    n_frames: int,
    n_sentinel: int,
    credit_prompt: str = "",
) -> list[dict[str, Any]]:
    """Validate the inputs against the ways the two prompts can silently disagree with the video."""
    checks: list[dict[str, Any]] = []
    fps = float(metadata.get("video_fps") or 10.0)
    stride = int(metadata.get("video_frame_stride") or 2)
    chunk_steps = int(metadata.get("action_chunk_steps") or 10)

    # 1. Does the telemetry clock match the video the VLM watches?
    if raw_offset_sec > 0.0:
        checks.append(
            _check(
                "clock",
                "Video clock alignment",
                "fail",
                f"As written, this window's per-step timestamps start at {raw_offset_sec:.1f}s "
                f"and run past the end of a {duration_sec:.1f}s video. The review prompt labels "
                f"them 'video seconds', so every telemetry row was misaligned by "
                f"{raw_offset_sec:.1f}s when this window was sent. Shown below rebased to 0.",
                {"offset_sec": raw_offset_sec},
            )
        )
    else:
        checks.append(
            _check(
                "clock",
                "Video clock alignment",
                "pass",
                f"Timestamps start at 0.0s and span {duration_sec:.1f}s of video — the same clock "
                "the VLM reads its event timestamps off.",
            )
        )

    # 2. Frame cadence: one frame every `stride` env steps, or extra collision/violation frames?
    in_video = [s for s in steps if s.get("video_frame_index") is not None]
    gaps = [
        int(b["episode_step"]) - int(a["episode_step"])
        for a, b in itertools.pairwise(in_video)
        if a.get("episode_step") is not None and b.get("episode_step") is not None
    ]
    off_cadence = [g for g in gaps if g != stride]
    if off_cadence:
        checks.append(
            _check(
                "cadence",
                "Frame cadence",
                "warn",
                f"{len(off_cadence)} of {len(gaps)} frame gaps are not the expected {stride} env "
                f"steps (saw {sorted(set(off_cadence))}). main_carla captures an extra frame on "
                "every collision / traffic-violation step, so a uniform "
                "chunk_index × chunk_duration grid mis-times every chunk after the first "
                "off-cadence frame. Chunk times here are re-derived from the frames themselves.",
                {"gaps": gaps},
            )
        )
    else:
        checks.append(
            _check(
                "cadence",
                "Frame cadence",
                "pass",
                f"All {len(gaps)} frame gaps are exactly {stride} env steps — the uniform grid and "
                "the real video agree.",
                {"gaps": gaps},
            )
        )

    # 3. Every chunk needs at least one frame, or the VLM cannot see what it is judging.
    empty = [c["chunk_index"] for c in chunks_payload if c.get("_n_frames", 0) == 0]
    checks.append(
        _check(
            "coverage",
            "Chunk frame coverage",
            "fail" if empty else "pass",
            f"Chunks with no recorded frame: {empty}." if empty
            else f"All {len(chunks_payload)} chunks contain at least one video frame "
                 f"({chunk_steps} env steps each).",
        )
    )

    # 4. Events the review call returned have to land inside the video.
    out_of_range = [
        e for e in events
        if e.get("timestamp_sec") is not None
        and not (-0.01 <= float(e["timestamp_sec"]) <= duration_sec + 0.01)
    ]
    if not events:
        checks.append(
            _check("events", "Event timestamps", "warn", "No events — the review call returned "
                   "nothing, so every chunk falls through to 'no signal'.")
        )
    else:
        checks.append(
            _check(
                "events",
                "Event timestamps",
                "fail" if out_of_range else "pass",
                f"{len(out_of_range)} of {len(events)} events fall outside the "
                f"{duration_sec:.1f}s video." if out_of_range
                else f"All {len(events)} events land inside the {duration_sec:.1f}s video.",
            )
        )

    # 5. Original subtasks: needed by the credit call to keep GOOD chunks and smooth transitions.
    missing_subtask = [c["chunk_index"] for c in chunks_payload if not c.get("original_subtask")]
    checks.append(
        _check(
            "originals",
            "Original subtasks in credit prompt",
            "fail" if missing_subtask else "pass",
            f"{len(missing_subtask)} chunks carry no original_subtask ({missing_subtask}) — the "
            "credit call cannot keep or smooth what it cannot see." if missing_subtask
            else f"All {len(chunks_payload)} chunks pass their executed subtask to the credit call.",
        )
    )

    # 6. Per-step CoT text: the richest signal in the review prompt.
    n = len(steps) or 1
    have_subtask = sum(1 for s in steps if str(s.get("subtask") or "").strip())
    have_reasoning = sum(1 for s in steps if str(s.get("reasoning") or "").strip())
    have_prompt = sum(1 for s in steps if str(s.get("prompt") or "").strip())
    worst = min(have_subtask, have_reasoning, have_prompt) / n
    checks.append(
        _check(
            "cot",
            "Per-step CoT capture",
            "pass" if worst > 0.95 else ("warn" if worst > 0.0 else "fail"),
            f"subtask {have_subtask}/{n} · reasoning {have_reasoning}/{n} · policy prompt "
            f"{have_prompt}/{n} steps. Only steps captured in the video reach the prompt "
            f"({len(in_video)} of {n}).",
        )
    )

    # 7. Reward is the credit call's only hard per-chunk signal — and it has to survive the trip
    #    into the chunk table, not just exist on the steps.
    have_reward = sum(1 for s in steps if s.get("reward_total") is not None)
    rewards = [float(s["reward_total"]) for s in steps if s.get("reward_total") is not None]
    all_zero = bool(rewards) and all(abs(r) < 1e-9 for r in rewards)
    in_prompt = "reward_total_sum" in credit_prompt
    if not in_prompt and have_reward:
        status, detail = "fail", (
            f"reward_total is on {have_reward}/{n} steps but no chunk in the credit prompt carries "
            "reward_total_sum — the table lost the one hard per-chunk signal the prompt then "
            "explains how to weigh."
        )
    else:
        status = "warn" if (have_reward < n or all_zero) else "pass"
        detail = (
            f"reward_total present on {have_reward}/{n} steps and folded into the chunk table"
            + (", but every value is exactly 0.0 — the credit call gets no reward signal to "
               "corroborate its labels (check debug_task, which replaces env reward)."
               if all_zero else ".")
        )
    checks.append(_check("reward", "Env reward signal", status, detail))

    # 8. PaliGemma <loc> sentinels must not reach a prompt as if they were prose.
    checks.append(
        _check(
            "sentinels",
            "CoT sentinel hygiene",
            "warn" if n_sentinel else "pass",
            f"{n_sentinel} captured subtask/reasoning fields still carry raw <locNNNN> sentinels "
            "in this artifact — they were sent to the review call as-is. Stripped here (and in the "
            "live pipeline now), so the prompts below show the clean text."
            if n_sentinel
            else "No <locNNNN> sentinels in the captured subtask / reasoning text.",
        )
    )

    # 9. Frame count vs. what the mp4 should hold.
    checks.append(
        _check(
            "frames",
            "Video length",
            "pass" if n_frames else "warn",
            f"{n_frames} frames at {fps:g} fps = {duration_sec:.1f}s covering "
            f"{len(steps)} env steps.",
        )
    )
    return checks


def project_hl_samples(cast_json: dict[str, Any] | None) -> dict[str, Any]:
    """Mirror ``_resolve_hl_targets`` so the page can show what this window would store."""
    out = {"correct": 0, "reinforce": 0, "skipped_no_suggestion": 0, "total": 0}
    if not cast_json:
        return out
    for chunk in cast_json.get("action_chunks", []) or []:
        label = str(chunk.get("label") or "").strip().upper()
        if label == "BAD":
            if chunk.get("suggested_subtasks"):
                out["correct"] += 1
            else:
                out["skipped_no_suggestion"] += 1
        else:
            out["reinforce"] += 1
    out["total"] = out["correct"] + out["reinforce"]
    return out


# ── window assembly ──────────────────────────────────────────────────────────────────


def build_window_payload(
    win_dir: Path, *, embed_video: bool = True, embed_plot: bool = True
) -> dict[str, Any]:
    """Load one window dir and rebuild everything both VLM calls were given."""
    metadata = json.loads((win_dir / "trajectory.json").read_text(encoding="utf-8"))
    cast_path = win_dir / "cast_relabel.json"
    cast_json = json.loads(cast_path.read_text(encoding="utf-8")) if cast_path.is_file() else None

    steps, raw_offset_sec = rebase_steps(metadata)
    metadata = dict(metadata)
    # Apply the same CoT-sentinel strip the live pipeline now does, so the prompts rendered here
    # match what a fresh run would send. ``n_sentinel`` records how much the artifact had raw.
    n_sentinel = sum(
        1 for s in steps for k in ("subtask", "reasoning")
        if strip_cot_sentinels(s.get(k)) != str(s.get(k) or "").strip()
    )
    for s in steps:
        for k in ("subtask", "reasoning"):
            s[k] = strip_cot_sentinels(s.get(k))
    metadata["chunk_original_subtask"] = {
        k: strip_cot_sentinels(v)
        for k, v in (metadata.get("chunk_original_subtask") or {}).items()
    }
    metadata["steps"] = steps

    fps = float(metadata.get("video_fps") or 10.0) or 10.0
    n_frames = sum(1 for s in steps if s.get("video_frame_index") is not None)
    duration_sec = round(n_frames / fps, 3) if n_frames else 0.0

    chunk_specs = build_action_chunk_specs(
        metadata,
        steps_per_chunk=int(metadata.get("action_chunk_steps") or 10),
        chunk_duration_sec=float(metadata.get("action_chunk_duration_sec") or 0.5),
    )
    chunk_specs = retime_chunk_specs(chunk_specs, metadata)

    events_raw = list((cast_json or {}).get("events") or [])
    events = [CoachEvent.from_dict(e) for e in events_raw]
    num_suggestions = int((cast_json or {}).get("num_subtask_suggestions") or 3)

    # The reward/progress figure the review call now uploads with the video. Rendered here from
    # the same metadata, so what the page shows is what the coach is handed.
    plot_b64 = ""
    if embed_plot:
        try:
            from coaches.trajectory_plots import generate_reward_progress_plot

            with tempfile.TemporaryDirectory() as td:
                png = generate_reward_progress_plot(
                    metadata,
                    Path(td) / "reward_progress.png",
                    chunk_steps=int(metadata.get("action_chunk_steps") or 10),
                )
                plot_b64 = base64.b64encode(png.read_bytes()).decode("ascii")
        except Exception as exc:  # noqa: BLE001 - the plot is an extra here too
            print(f"[viewer] reward/progress plot failed for {win_dir.name}: {exc}", file=sys.stderr)

    # The two prompts, rebuilt exactly as the pipeline builds them.
    review_prompt = build_coaching_prompt(metadata, include_plots=bool(plot_b64))
    credit_prompt = build_credit_relabel_prompt(
        events=events,
        chunk_specs=chunk_specs,
        metadata=metadata,
        seed_subtasks=SEED_SUBTASKS,
        num_suggestions=num_suggestions,
    )

    originals = metadata.get("chunk_original_subtask") or {}
    credits = {
        int(c.get("chunk_index", -1)): c for c in (cast_json or {}).get("action_chunks", []) or []
    }
    chunks_payload: list[dict[str, Any]] = []
    for spec in chunk_specs:
        span = steps[max(0, spec.episode_step_start - 1): spec.episode_step_end]
        frames = [s for s in span if s.get("video_frame_index") is not None]
        rewards = [float(s["reward_total"]) for s in span if s.get("reward_total") is not None]
        credit = credits.get(spec.chunk_index, {})
        chunks_payload.append(
            {
                "chunk_index": spec.chunk_index,
                "episode_step_start": spec.episode_step_start,
                "episode_step_end": spec.episode_step_end,
                "abs_step_start": span[0].get("episode_step") if span else None,
                "abs_step_end": span[-1].get("episode_step") if span else None,
                "t0": spec.video_time_start_sec,
                "t1": spec.video_time_end_sec,
                "original_subtask": str(originals.get(str(spec.chunk_index), "") or ""),
                "label": credit.get("label"),
                "credit_source": str(credit.get("credit_source") or ""),
                "rationale": str(credit.get("rationale") or ""),
                "suggested_subtasks": list(credit.get("suggested_subtasks") or []),
                "suggested_reasoning": str(credit.get("suggested_reasoning") or ""),
                "reward_sum": round(sum(rewards), 4) if rewards else None,
                "_n_frames": len(frames),
            }
        )

    checks = run_checks(
        metadata,
        steps,
        chunks_payload,
        events_raw,
        raw_offset_sec=raw_offset_sec,
        duration_sec=duration_sec,
        n_frames=n_frames,
        n_sentinel=n_sentinel,
        credit_prompt=credit_prompt,
    )

    video_b64 = ""
    video_path = win_dir / "rollout.mp4"
    video_bytes = 0
    if embed_video and video_path.is_file():
        raw = video_path.read_bytes()
        video_bytes = len(raw)
        video_b64 = base64.b64encode(raw).decode("ascii")

    summary_keys = (
        "episode", "route", "window_index", "episode_steps", "step_offset", "success",
        "termination_reason", "route_progress_start_pct", "route_progress_end_pct",
        "route_progress_delta_pct", "route_completed", "mean_end_speed_mps",
        "window_reward_total", "window_reward_mean", "action_chunk_steps", "video_fps",
        "video_frame_stride",
    )
    return {
        "name": win_dir.name,
        "run": win_dir.parent.parent.name,
        "path": str(win_dir),
        "summary": {k: metadata.get(k) for k in summary_keys},
        "route_command_plan": metadata.get("route_command_plan") or [],
        # Cross-window correction memory as it stood when this window was reviewed (empty for
        # windows recorded before the memory existed).
        "memory": str(metadata.get("correction_memory") or "").strip(),
        "collision_events": metadata.get("collision_events") or [],
        "fps": fps,
        "duration": duration_sec,
        "n_frames": n_frames,
        "raw_offset_sec": raw_offset_sec,
        "steps": [
            {
                "episode_step": s.get("episode_step"),
                "t": s.get("video_timestamp_sec"),
                "frame": s.get("video_frame_index"),
                "speed": s.get("ego_speed_mps"),
                "throttle": s.get("control_throttle"),
                "steer": s.get("control_steer"),
                "brake": s.get("control_brake"),
                "reward": s.get("reward_total"),
                "progress": s.get("route_progress_pct"),
                "collision": bool(s.get("collision")),
                "subtask": str(s.get("subtask") or ""),
                "reasoning": str(s.get("reasoning") or ""),
                "prompt": str(s.get("prompt") or ""),
            }
            for s in steps
        ],
        "chunks": chunks_payload,
        "events": events_raw,
        "checks": checks,
        "hl": project_hl_samples(cast_json),
        "prompts": {"review": review_prompt, "credit": credit_prompt},
        "video": video_b64,
        "plot": plot_b64,
        "video_bytes": video_bytes,
        "has_cast": cast_json is not None,
    }


# ── page ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE = r"""<title>CAST Relabel Inspector</title>
<style>
:root {
  --paper: #f2f5f8;
  --surface: #ffffff;
  --surface-2: #e8edf2;
  --ink: #0e141b;
  --ink-2: #46535f;
  --ink-3: #7b8794;
  --line: #d3dbe3;
  --line-strong: #b6c2cd;
  --accent: #12768a;
  --accent-soft: #d6eef2;
  --good: #24855b;
  --bad: #c2392f;
  --precursor: #c26a14;
  --null: #7b8794;
  --warn-bg: #fdf3e2;
  --fail-bg: #fdecea;
  --pass-bg: #e9f4ee;
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
  --sans: ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #0e141b;
    --surface: #161e27;
    --surface-2: #1e2833;
    --ink: #e6edf3;
    --ink-2: #a8b6c2;
    --ink-3: #74838f;
    --line: #26313d;
    --line-strong: #3a4854;
    --accent: #3fb6cd;
    --accent-soft: #10333c;
    --good: #4cbf85;
    --bad: #e8695e;
    --precursor: #e9963c;
    --null: #74838f;
    --warn-bg: #2a2415;
    --fail-bg: #2c1a19;
    --pass-bg: #14261d;
  }
}
:root[data-theme="dark"] {
  --paper: #0e141b;
  --surface: #161e27;
  --surface-2: #1e2833;
  --ink: #e6edf3;
  --ink-2: #a8b6c2;
  --ink-3: #74838f;
  --line: #26313d;
  --line-strong: #3a4854;
  --accent: #3fb6cd;
  --accent-soft: #10333c;
  --good: #4cbf85;
  --bad: #e8695e;
  --precursor: #e9963c;
  --null: #74838f;
  --warn-bg: #2a2415;
  --fail-bg: #2c1a19;
  --pass-bg: #14261d;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
h1, h2, h3 { margin: 0; letter-spacing: -0.015em; text-wrap: balance; }
.label {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.num { font-variant-numeric: tabular-nums; font-family: var(--mono); }

/* ── top bar ─────────────────────────────────────────── */
header {
  position: sticky; top: 0; z-index: 20;
  display: flex; flex-wrap: wrap; align-items: center; gap: 16px;
  padding: 12px 20px;
  background: var(--surface);
  border-bottom: 1px solid var(--line);
}
header h1 { font-size: 15px; font-weight: 650; }
header h1 span { color: var(--accent); }
select, button {
  font: inherit; color: var(--ink);
  background: var(--surface-2);
  border: 1px solid var(--line-strong);
  border-radius: 5px; padding: 5px 9px; cursor: pointer;
}
select { font-family: var(--mono); font-size: 12px; }
button:hover { border-color: var(--accent); }
button:focus-visible, select:focus-visible, .chunk:focus-visible, tr:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px;
}
.stats { display: flex; gap: 18px; margin-left: auto; flex-wrap: wrap; }
.stat { display: flex; flex-direction: column; }
.stat b { font-family: var(--mono); font-size: 14px; font-weight: 600; font-variant-numeric: tabular-nums; }

/* ── shell ───────────────────────────────────────────── */
main {
  display: grid;
  grid-template-columns: minmax(380px, 460px) minmax(0, 1fr);
  gap: 20px; padding: 20px; align-items: start;
}
@media (max-width: 980px) { main { grid-template-columns: minmax(0, 1fr); } }
.rail { position: sticky; top: 62px; display: flex; flex-direction: column; gap: 14px; }
.card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}
.card > h3 {
  font-size: 11px; font-family: var(--mono); letter-spacing: 0.11em; text-transform: uppercase;
  color: var(--ink-3); padding: 10px 14px; border-bottom: 1px solid var(--line);
  background: var(--surface-2);
}
.card .body { padding: 14px; }
video { width: 100%; display: block; background: #000; }

/* ── timeline ────────────────────────────────────────── */
.timeline { padding: 12px 14px 14px; }
.track { position: relative; height: 30px; border-radius: 4px; overflow: hidden; background: var(--surface-2); }
.chunk {
  position: absolute; top: 0; bottom: 0;
  border-right: 1px solid var(--paper);
  cursor: pointer; border-radius: 0;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 9px; color: #fff;
  background: var(--null); opacity: 0.82;
}
.chunk:hover { opacity: 1; }
.chunk.good { background: var(--good); }
.chunk.bad { background: var(--bad); }
.chunk.precursor { background: var(--precursor); }
.chunk.active { opacity: 1; box-shadow: inset 0 0 0 2px var(--ink); }
.pins { position: relative; height: 16px; margin-top: 3px; }
.pin {
  position: absolute; top: 0; width: 2px; height: 10px; border-radius: 1px; cursor: pointer;
  transform: translateX(-1px);
}
.pin.GOOD { background: var(--good); }
.pin.BAD { background: var(--bad); }
.playhead {
  position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--accent);
  transform: translateX(-1px); pointer-events: none; z-index: 3;
}
.axis { display: flex; justify-content: space-between; margin-top: 2px; }

/* ── now panel ───────────────────────────────────────── */
.now dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; align-items: baseline; }
.now dt { font-family: var(--mono); font-size: 10px; letter-spacing: 0.09em; text-transform: uppercase; color: var(--ink-3); }
.now dd { margin: 0; font-family: var(--mono); font-size: 12px; font-variant-numeric: tabular-nums; }
.now .text { grid-column: 1 / -1; font-family: var(--sans); font-size: 12.5px; color: var(--ink-2);
  border-left: 2px solid var(--line-strong); padding-left: 9px; margin: 2px 0 4px; }
.bar { height: 5px; border-radius: 3px; background: var(--surface-2); overflow: hidden; min-width: 60px; }
.bar i { display: block; height: 100%; background: var(--accent); }

/* ── tabs ────────────────────────────────────────────── */
.tabs { display: flex; gap: 2px; border-bottom: 1px solid var(--line); flex-wrap: wrap; }
.tab {
  background: none; border: none; border-bottom: 2px solid transparent; border-radius: 0;
  padding: 8px 12px; font-family: var(--mono); font-size: 11px; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--ink-3);
}
.tab:hover { color: var(--ink); }
.tab[aria-selected="true"] { color: var(--accent); border-bottom-color: var(--accent); }
.panel { display: none; padding: 16px; }
.panel.on { display: block; }

/* ── tables ──────────────────────────────────────────── */
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 6px 9px; border-bottom: 1px solid var(--line); vertical-align: top; }
th {
  font-family: var(--mono); font-size: 10px; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--ink-3); font-weight: 500; position: sticky; top: 0; background: var(--surface); z-index: 1;
}
td.n { font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
tbody tr { cursor: pointer; }
tbody tr:hover { background: var(--surface-2); }
tbody tr.active { background: var(--accent-soft); }
tbody tr.novideo td.n { color: var(--ink-3); }
.tall { max-height: 620px; overflow-y: auto; }

.tag {
  display: inline-block; font-family: var(--mono); font-size: 10px; letter-spacing: 0.06em;
  padding: 1px 6px; border-radius: 3px; color: #fff; white-space: nowrap;
}
.tag.good { background: var(--good); }
.tag.bad { background: var(--bad); }
.tag.precursor { background: var(--precursor); }
.tag.null { background: var(--null); }

/* ── charts ──────────────────────────────────────────── */
.chartwrap { margin-bottom: 18px; }
.charthead { display: flex; align-items: baseline; gap: 10px; margin-bottom: 4px; }
.charthead .chip:last-child { margin-left: auto; }
.key { font-family: var(--mono); font-size: 10px; color: var(--ink-2); display: inline-flex; align-items: center; gap: 4px; }
.key i { width: 9px; height: 2px; border-radius: 1px; display: inline-block; }
.chart {
  width: 100%; height: 108px; display: block; cursor: crosshair;
  background: var(--surface-2); border: 1px solid var(--line); border-radius: 5px;
}
.chart .band { opacity: 0.13; }
.chart .band.good { fill: var(--good); }
.chart .band.bad { fill: var(--bad); }
.chart .band.precursor { fill: var(--precursor); }
.chart .band.null { fill: transparent; }
.chart .grid, .chart .zero {
  stroke: var(--line-strong); stroke-width: 1; vector-effect: non-scaling-stroke;
}
.chart .grid { stroke-dasharray: 2 4; opacity: 0.7; }
.chart .evtline { stroke-width: 1; vector-effect: non-scaling-stroke; opacity: 0.75; stroke-dasharray: 3 3; }
.chart .evtline.GOOD { stroke: var(--good); }
.chart .evtline.BAD { stroke: var(--bad); }
.chart .line { fill: none; stroke-width: 1.6; vector-effect: non-scaling-stroke;
  stroke-linejoin: round; stroke-linecap: round; }
.chart .area { opacity: 0.14; stroke: none; }
.chart .chartph { stroke: var(--ink); stroke-width: 1.5; vector-effect: non-scaling-stroke; }

/* ── checks ──────────────────────────────────────────── */
.checks { display: flex; flex-direction: column; gap: 10px; }
.check { border: 1px solid var(--line); border-left-width: 4px; border-radius: 6px; padding: 10px 13px; }
.check.pass { border-left-color: var(--good); background: var(--pass-bg); }
.check.warn { border-left-color: var(--precursor); background: var(--warn-bg); }
.check.fail { border-left-color: var(--bad); background: var(--fail-bg); }
.check h4 { margin: 0 0 3px; font-size: 13px; display: flex; gap: 8px; align-items: center; }
.check p { margin: 0; font-size: 12.5px; color: var(--ink-2); }
.cadence { display: flex; align-items: flex-end; gap: 1px; height: 26px; margin-top: 8px; }
.cadence i { flex: 1 1 2px; background: var(--good); border-radius: 1px 1px 0 0; min-width: 2px; }
.cadence i.off { background: var(--bad); }

/* ── prompts ─────────────────────────────────────────── */
pre {
  margin: 0; font-family: var(--mono); font-size: 11.5px; line-height: 1.55;
  white-space: pre-wrap; word-break: break-word;
  background: var(--surface-2); border: 1px solid var(--line); border-radius: 6px;
  padding: 13px; max-height: 640px; overflow: auto;
}
.promptbar { display: flex; align-items: center; gap: 12px; margin: 0 0 8px; }
.plot {
  display: block; width: 100%; max-width: 100%; height: auto; margin: 0 0 14px;
  border: 1px solid var(--line); border-radius: 6px; background: #fff;
}
.chip { font-family: var(--mono); font-size: 11px; color: var(--ink-3); }
.muted { color: var(--ink-3); }
.evt { border-left: 3px solid var(--line-strong); padding: 2px 0 2px 11px; margin-bottom: 12px; cursor: pointer; }
.evt.GOOD { border-left-color: var(--good); }
.evt.BAD { border-left-color: var(--bad); }
.evt b { font-family: var(--mono); font-size: 12px; }
.evt p { margin: 2px 0 0; font-size: 12.5px; color: var(--ink-2); }
.plan { margin: 0; padding-left: 18px; font-size: 12.5px; color: var(--ink-2); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>

<header>
  <h1>CAST Relabel <span>Inspector</span></h1>
  <select id="winsel" aria-label="Window"></select>
  <button id="theme" title="Toggle theme">◑</button>
  <div class="stats" id="stats"></div>
</header>

<main>
  <div class="rail">
    <div class="card">
      <h3>Window video — exactly what the review call watched</h3>
      <video id="vid" controls preload="metadata"></video>
      <div class="timeline">
        <div class="label" style="margin-bottom:6px">Action chunks · credit assignment</div>
        <div class="track" id="track"></div>
        <div class="pins" id="pins"></div>
        <div class="axis label"><span>0.0s</span><span id="tmid"></span><span id="tend"></span></div>
      </div>
    </div>
    <div class="card now">
      <h3>At the playhead</h3>
      <div class="body" id="now"></div>
    </div>
  </div>

  <div class="card">
    <div class="tabs" role="tablist" id="tabs"></div>
    <div id="panels"></div>
  </div>
</main>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = (v, d = 2) => (v === null || v === undefined) ? '—' : Number(v).toFixed(d);

let W = null;          // active window
let stepByFrame = new Map();

/* ── theme toggle ─────────────────────────────────────── */
$('theme').onclick = () => {
  const cur = document.documentElement.getAttribute('data-theme');
  const dark = cur ? cur === 'dark'
    : window.matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark');
};

/* ── window selector ──────────────────────────────────── */
const sel = $('winsel');
DATA.windows.forEach((w, i) => {
  const o = document.createElement('option');
  o.value = i;
  const bad = w.chunks.filter(c => c.label === 'BAD').length;
  o.textContent = `${w.name}  ·  ${w.chunks.length} chunks, ${bad} BAD`;
  sel.appendChild(o);
});
sel.onchange = () => load(Number(sel.value));

/* ── chunk helpers ────────────────────────────────────── */
const chunkClass = (c) => c.label === 'GOOD' ? 'good'
  : c.label === 'BAD' ? (c.credit_source === 'precursor' ? 'precursor' : 'bad') : 'null';
const chunkName = (c) => c.label === 'BAD'
  ? (c.credit_source === 'precursor' ? 'BAD · precursor' : 'BAD · direct')
  : (c.label || 'no signal');

/* ── render ───────────────────────────────────────────── */
function load(i) {
  W = DATA.windows[i];
  stepByFrame = new Map();
  W.steps.forEach(s => { if (s.frame !== null) stepByFrame.set(s.frame, s); });

  const v = $('vid');
  v.src = W.video ? 'data:video/mp4;base64,' + W.video : '';
  v.load();

  renderStats(); renderTimeline(); renderTabs(); renderNow(0);
}

function renderStats() {
  const bad = W.chunks.filter(c => c.label === 'BAD');
  const fails = W.checks.filter(c => c.status === 'fail').length;
  const warns = W.checks.filter(c => c.status === 'warn').length;
  const cells = [
    ['route', W.summary.route ?? '—'],
    ['episode / window', `${W.summary.episode} / ${W.summary.window_index}`],
    ['env steps', W.summary.episode_steps],
    ['video', `${fmt(W.duration, 1)}s · ${W.n_frames}f`],
    ['events', W.events.length],
    ['BAD chunks', `${bad.length}/${W.chunks.length}`],
    ['HL samples', `${W.hl.total} (${W.hl.correct} corrective)`],
    ['checks', fails ? `${fails} fail` : (warns ? `${warns} warn` : 'all pass')],
  ];
  $('stats').innerHTML = cells.map(([k, val]) =>
    `<div class="stat"><span class="label">${esc(k)}</span><b>${esc(val)}</b></div>`).join('');
}

function renderTimeline() {
  const dur = W.duration || 1;
  $('track').innerHTML = W.chunks.map(c => {
    const left = (c.t0 / dur) * 100, wid = Math.max(((c.t1 - c.t0) / dur) * 100, 0.4);
    const title = `Chunk ${c.chunk_index} · ${fmt(c.t0, 2)}–${fmt(c.t1, 2)}s · ${chunkName(c)}`
      + (c.original_subtask ? `\nwas: ${c.original_subtask}` : '')
      + (c.suggested_subtasks.length ? `\nnow: ${c.suggested_subtasks[0]}` : '');
    return `<div class="chunk ${chunkClass(c)}" data-t="${c.t0}" data-c="${c.chunk_index}"
      tabindex="0" role="button" style="left:${left}%;width:${wid}%" title="${esc(title)}">${c.chunk_index}</div>`;
  }).join('') + '<div class="playhead" id="ph" style="left:0"></div>';

  $('pins').innerHTML = W.events.map((e, i) => {
    const left = Math.max(0, Math.min(100, (Number(e.timestamp_sec) / dur) * 100));
    return `<div class="pin ${esc(e.label)}" data-t="${e.timestamp_sec}" title="${esc(e.label + ' @ ' + fmt(e.timestamp_sec, 1) + 's — ' + (e.description || ''))}"
      style="left:${left}%"></div>`;
  }).join('');

  $('tmid').textContent = fmt(dur / 2, 1) + 's';
  $('tend').textContent = fmt(dur, 1) + 's';

  document.querySelectorAll('.chunk, .pin').forEach(el => {
    el.onclick = () => seek(Number(el.dataset.t));
    el.onkeydown = (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); seek(Number(el.dataset.t)); } };
  });
}

function seek(t) { const v = $('vid'); v.currentTime = Math.max(0, t + 0.001); v.pause(); renderNow(v.currentTime); }

$('vid').addEventListener('timeupdate', () => renderNow($('vid').currentTime));
$('vid').addEventListener('seeked', () => renderNow($('vid').currentTime));

function renderNow(t) {
  const dur = W.duration || 1;
  const ph = $('ph'); if (ph) ph.style.left = ((t / dur) * 100) + '%';

  const frame = Math.max(0, Math.min(W.n_frames - 1, Math.round(t * W.fps)));
  const s = stepByFrame.get(frame);
  const c = W.chunks.find(k => t >= k.t0 && t < k.t1) || W.chunks[W.chunks.length - 1];

  document.querySelectorAll('.chunk').forEach(el =>
    el.classList.toggle('active', c && Number(el.dataset.c) === c.chunk_index));
  document.querySelectorAll('#steptable tbody tr').forEach(tr =>
    tr.classList.toggle('active', s && Number(tr.dataset.step) === s.episode_step));
  updateCharts(t, s);

  if (!s) { $('now').innerHTML = '<p class="muted">No recorded frame at this time.</p>'; return; }
  const pct = (x) => `<span class="bar" style="display:inline-block;width:70px;vertical-align:middle">
      <i style="width:${Math.round(Math.max(0, Math.min(1, x)) * 100)}%"></i></span>`;
  $('now').innerHTML = `
    <dl>
      <dt>t / step</dt><dd>${fmt(s.t, 2)}s · ep step ${s.episode_step} · frame ${s.frame}</dd>
      <dt>speed</dt><dd>${fmt(s.speed, 2)} m/s</dd>
      <dt>throttle</dt><dd>${pct(s.throttle)} ${fmt(s.throttle, 2)}</dd>
      <dt>brake</dt><dd>${pct(s.brake)} ${fmt(s.brake, 2)}</dd>
      <dt>steer</dt><dd>${fmt(s.steer, 3)}</dd>
      <dt>reward</dt><dd>${fmt(s.reward, 4)}</dd>
      <dt>progress</dt><dd>${fmt(s.progress, 2)}%${s.collision ? ' · <b style="color:var(--bad)">COLLISION</b>' : ''}</dd>
      <dt>chunk</dt><dd>${c ? c.chunk_index : '—'} <span class="tag ${c ? chunkClass(c) : 'null'}">${esc(c ? chunkName(c) : '—')}</span></dd>
    </dl>
    <div class="label" style="margin-top:12px">Executed subtask</div>
    <div class="text">${esc(s.subtask) || '<span class="muted">—</span>'}</div>
    <div class="label">CoT reasoning</div>
    <div class="text">${esc(s.reasoning) || '<span class="muted">—</span>'}</div>
    <div class="label">Policy prompt</div>
    <div class="text">${esc(s.prompt) || '<span class="muted">—</span>'}</div>
    ${c && c.suggested_subtasks.length ? `<div class="label">Relabeled to</div>
      <div class="text" style="border-left-color:var(--accent)">${esc(c.suggested_subtasks[0])}</div>` : ''}`;
}

/* ── tabs ─────────────────────────────────────────────── */
const TABS = [
  ['checks', 'Checks', renderChecks],
  ['chunks', 'Chunks', renderChunks],
  ['steps', 'Telemetry', renderSteps],
  ['events', 'Events & context', renderEvents],
  ['review', 'Review prompt', () => renderPrompt('review', 'Call 1 — window review (video + this text)')],
  ['credit', 'Credit prompt', () => renderPrompt('credit', 'Call 2 — credit + relabel (text only, no video, no history)')],
];
let activeTab = 'checks';

function renderTabs() {
  $('tabs').innerHTML = TABS.map(([id, name]) =>
    `<button class="tab" role="tab" data-tab="${id}" aria-selected="${id === activeTab}">${name}</button>`).join('');
  $('panels').innerHTML = TABS.map(([id]) =>
    `<div class="panel ${id === activeTab ? 'on' : ''}" id="p-${id}"></div>`).join('');
  document.querySelectorAll('.tab').forEach(b => b.onclick = () => {
    activeTab = b.dataset.tab;
    document.querySelectorAll('.tab').forEach(x => x.setAttribute('aria-selected', x.dataset.tab === activeTab));
    document.querySelectorAll('.panel').forEach(x => x.classList.toggle('on', x.id === 'p-' + activeTab));
    draw(activeTab);
  });
  TABS.forEach(([id]) => draw(id));
}
function draw(id) { const t = TABS.find(x => x[0] === id); if (t) t[2](); }

function renderChecks() {
  const cadence = (W.checks.find(c => c.id === 'cadence') || {}).extra;
  const stride = W.summary.video_frame_stride;
  const bars = cadence && cadence.gaps
    ? `<div class="cadence" title="Env steps between consecutive recorded frames">` +
      cadence.gaps.map(g => `<i class="${g === stride ? '' : 'off'}" style="height:${Math.min(100, g / stride * 50)}%" title="${g} steps"></i>`).join('') +
      `</div>` : '';
  $('p-checks').innerHTML = `<div class="checks">` + W.checks.map(c => `
    <div class="check ${c.status}">
      <h4>${esc(c.title)} <span class="tag ${c.status === 'pass' ? 'good' : c.status === 'warn' ? 'precursor' : 'bad'}">${c.status}</span></h4>
      <p>${esc(c.detail)}</p>
      ${c.id === 'cadence' ? bars : ''}
    </div>`).join('') + `</div>
    <div class="check" style="margin-top:10px;border-left-color:var(--accent)">
      <h4>What this window would store</h4>
      <p>${W.hl.total} HL samples — ${W.hl.correct} corrective (BAD, subtask replaced),
      ${W.hl.reinforce} reinforcing (GOOD/unlabeled, original subtask kept),
      ${W.hl.skipped_no_suggestion} BAD chunks skipped for having no suggestion.</p>
    </div>`;
}

function renderChunks() {
  $('p-chunks').innerHTML = `<div class="scroll"><table>
    <thead><tr><th>#</th><th>video t</th><th>env steps</th><th>verdict</th><th>reward</th>
      <th>executed subtask</th><th>relabeled to</th><th>rationale</th></tr></thead>
    <tbody>` + W.chunks.map(c => `
      <tr data-t="${c.t0}">
        <td class="n">${c.chunk_index}</td>
        <td class="n">${fmt(c.t0, 2)}–${fmt(c.t1, 2)}</td>
        <td class="n">${c.abs_step_start}–${c.abs_step_end}<br><span class="muted">${c._n_frames} frames</span></td>
        <td><span class="tag ${chunkClass(c)}">${esc(chunkName(c))}</span></td>
        <td class="n">${fmt(c.reward_sum, 3)}</td>
        <td>${esc(c.original_subtask) || '<span class="muted">—</span>'}</td>
        <td>${c.suggested_subtasks.length ? esc(c.suggested_subtasks[0]) : '<span class="muted">kept</span>'}
          ${c.suggested_reasoning ? `<br><span class="muted">${esc(c.suggested_reasoning)}</span>` : ''}</td>
        <td class="muted">${esc(c.rationale)}</td>
      </tr>`).join('') + `</tbody></table></div>`;
  $('p-chunks').querySelectorAll('tbody tr').forEach(tr => tr.onclick = () => seek(Number(tr.dataset.t)));
}

/* ── telemetry charts ─────────────────────────────────── */
const VW = 1000, VH = 120;   // chart viewBox; x is stretched, strokes stay 1px

/* `in_prompt` mirrors how each series reaches the review call: 'dense' = on every timestamp,
   'sampled' = every CONTROL_SAMPLE_EVERY-th, 'plot' = also attached as an image. */
const SERIES = [
  { key: 'speed', title: 'Ego speed', unit: 'm/s', fill: true, in_prompt: 'sampled',
    lines: [{ f: s => s.speed, color: 'var(--accent)' }] },
  { key: 'control', title: 'Throttle / brake', unit: '', lo: 0, hi: 1, in_prompt: 'sampled',
    lines: [{ f: s => s.throttle, color: 'var(--good)', name: 'throttle' },
            { f: s => s.brake, color: 'var(--bad)', name: 'brake' }] },
  { key: 'steer', title: 'Steer', unit: '', lo: -1, hi: 1, zero: true, in_prompt: 'sampled',
    lines: [{ f: s => s.steer, color: 'var(--accent)' }] },
  { key: 'reward', title: 'Env reward', unit: '/step', zero: true, in_prompt: 'dense+plot',
    lines: [{ f: s => s.reward, color: 'var(--precursor)' }] },
  { key: 'progress', title: 'Route progress', unit: '%', in_prompt: 'dense+plot',
    lines: [{ f: s => s.progress, color: 'var(--accent)' }] },
];

const PROMPT_BADGE = {
  'sampled': ['every 2nd timestamp', 'sampled'],
  'dense+plot': ['every timestamp + attached plot', 'dense'],
};

function scaleOf(spec, pts) {
  let lo = spec.lo, hi = spec.hi;
  if (lo === undefined || hi === undefined) {
    const vals = pts.flatMap(p => p.v.filter(v => v !== null && v !== undefined));
    let mn = vals.length ? Math.min(...vals) : 0, mx = vals.length ? Math.max(...vals) : 1;
    if (spec.zero) { const m = Math.max(Math.abs(mn), Math.abs(mx), 1e-6); mn = -m; mx = m; }
    if (mx - mn < 1e-9) { mx = mn + 1; }
    const pad = (mx - mn) * 0.08;
    lo = spec.lo !== undefined ? spec.lo : mn - pad;
    hi = spec.hi !== undefined ? spec.hi : mx + pad;
  }
  return { lo, hi };
}

function renderSteps() {
  const dur = W.duration || 1;
  const pts = W.steps.filter(s => s.t !== null)
    .map(s => ({ t: s.t, v: SERIES.map(sp => sp.lines.map(l => l.f(s))) }));

  const bands = W.chunks.map(c =>
    `<rect x="${(c.t0 / dur) * VW}" y="0" width="${Math.max(((c.t1 - c.t0) / dur) * VW, 1)}"
       height="${VH}" class="band ${chunkClass(c)}"></rect>`).join('');
  const evtMarks = W.events.map(e =>
    `<line class="evtline ${esc(e.label)}" x1="${(Number(e.timestamp_sec) / dur) * VW}"
       x2="${(Number(e.timestamp_sec) / dur) * VW}" y1="0" y2="${VH}"></line>`).join('');

  const charts = SERIES.map((spec, si) => {
    const series = spec.lines.map((l, li) => pts.map(p => ({ t: p.t, y: p.v[si][li] })));
    const { lo, hi } = scaleOf(spec, pts.map(p => ({ v: p.v[si] })));
    const X = t => (t / dur) * VW;
    const Y = y => VH - ((Number(y) - lo) / (hi - lo)) * VH;
    const paths = series.map((pp, li) => {
      const usable = pp.filter(p => p.y !== null && p.y !== undefined && !Number.isNaN(Number(p.y)));
      if (!usable.length) return '';
      const d = usable.map((p, i) => `${i ? 'L' : 'M'}${X(p.t).toFixed(2)},${Y(p.y).toFixed(2)}`).join('');
      const area = spec.fill
        ? `<path class="area" d="${d}L${X(usable[usable.length - 1].t).toFixed(2)},${VH}L${X(usable[0].t).toFixed(2)},${VH}Z"
             style="fill:${spec.lines[li].color}"></path>` : '';
      return `${area}<path class="line" d="${d}" style="stroke:${spec.lines[li].color}"></path>`;
    }).join('');
    const zero = (spec.zero && lo < 0 && hi > 0)
      ? `<line class="zero" x1="0" x2="${VW}" y1="${Y(0)}" y2="${Y(0)}"></line>` : '';
    const legend = spec.lines.filter(l => l.name).map(l =>
      `<span class="key"><i style="background:${l.color}"></i>${esc(l.name)}</span>`).join('');
    const badge = PROMPT_BADGE[spec.in_prompt] || ['', ''];
    return `<div class="chartwrap" data-series="${spec.key}">
      <div class="charthead">
        <span class="label">${esc(spec.title)}${spec.unit ? ' · ' + esc(spec.unit) : ''}</span>
        <span class="tag ${badge[1] === 'dense' ? 'good' : 'null'}" title="${esc(badge[0])}">${esc(badge[1])}</span>
        ${legend}
        <span class="chip num" id="v-${spec.key}"></span>
        <span class="chip muted num">${fmt(lo, 2)} … ${fmt(hi, 2)}</span>
      </div>
      <svg class="chart" viewBox="0 0 ${VW} ${VH}" preserveAspectRatio="none" data-dur="${dur}">
        <g class="bands">${bands}</g>
        <line class="grid" x1="0" x2="${VW}" y1="${VH / 2}" y2="${VH / 2}"></line>
        ${zero}${evtMarks}${paths}
        <line class="chartph" x1="0" x2="0" y1="0" y2="${VH}"></line>
      </svg>
    </div>`;
  }).join('');

  $('p-steps').innerHTML = `
    <p class="muted" style="margin:0 0 14px">Chunk verdicts shade the background; vertical rules are
    the review call's events. Click anywhere to seek. Only the
    ${pts.length} of ${W.steps.length} steps captured as frames reach the review prompt — and of
    those, <b>dense</b> series appear on every timestamp (plus the attached plot) while
    <b>sampled</b> ones are thinned to every 2nd.</p>
    ${charts}
    <details style="margin-top:16px"><summary class="label" style="cursor:pointer">Per-step table</summary>
      <div class="scroll tall" style="margin-top:10px"><table id="steptable">
      <thead><tr><th>step</th><th>video t</th><th>speed</th><th>thr</th><th>brk</th><th>steer</th>
        <th>reward</th><th>prog</th><th>subtask</th></tr></thead>
      <tbody>` + W.steps.map(s => `
        <tr data-step="${s.episode_step}" data-t="${s.t ?? ''}" class="${s.t === null ? 'novideo' : ''}">
          <td class="n">${s.episode_step}</td>
          <td class="n">${s.t === null ? '<span class="muted">not in video</span>' : fmt(s.t, 2) + 's'}</td>
          <td class="n">${fmt(s.speed, 2)}</td><td class="n">${fmt(s.throttle, 2)}</td>
          <td class="n">${fmt(s.brake, 2)}</td><td class="n">${fmt(s.steer, 3)}</td>
          <td class="n">${fmt(s.reward, 4)}</td><td class="n">${fmt(s.progress, 2)}</td>
          <td>${esc(s.subtask.slice(0, 90))}</td>
        </tr>`).join('') + `</tbody></table></div></details>`;

  $('p-steps').querySelectorAll('.chart').forEach(svg => {
    svg.onclick = (ev) => {
      const r = svg.getBoundingClientRect();
      seek(((ev.clientX - r.left) / r.width) * Number(svg.dataset.dur));
    };
  });
  $('p-steps').querySelectorAll('tbody tr').forEach(tr => tr.onclick = () => {
    if (tr.dataset.t) seek(Number(tr.dataset.t));
  });
  updateCharts($('vid').currentTime || 0, null);
}

function updateCharts(t, s) {
  const x = ((t / (W.duration || 1)) * VW).toFixed(2);
  document.querySelectorAll('.chartph').forEach(l => { l.setAttribute('x1', x); l.setAttribute('x2', x); });
  if (!s) return;
  const vals = {
    speed: fmt(s.speed, 2), control: `${fmt(s.throttle, 2)} / ${fmt(s.brake, 2)}`,
    steer: fmt(s.steer, 3), reward: fmt(s.reward, 4), progress: fmt(s.progress, 2),
  };
  for (const k in vals) { const el = $('v-' + k); if (el) el.textContent = vals[k]; }
}

function renderEvents() {
  const plan = W.route_command_plan.map((p, i) =>
    `<li>${esc(p.command || '')}${p.start_distance_m != null ? ` <span class="muted">(after ~${Math.round(p.start_distance_m)} m)</span>` : ''}</li>`).join('');
  $('p-events').innerHTML = `
    <div class="label">Events returned by the review call</div>
    <div style="margin:10px 0 20px">` + (W.events.length ? W.events.map(e => `
      <div class="evt ${esc(e.label)}" data-t="${e.timestamp_sec}">
        <b>${fmt(e.timestamp_sec, 1)}s · ${esc(e.label)}</b>
        <p>${esc(e.description)}</p>
        ${e.correction ? `<p><span class="label">fix</span> ${esc(e.correction)}</p>` : ''}
      </div>`).join('') : '<p class="muted">No events.</p>') + `</div>
    <div class="label">Correction memory carried into both calls</div>
    <div style="margin:8px 0 20px">${W.memory
      ? `<pre style="max-height:260px">${esc(W.memory)}</pre>
         <p class="muted" style="margin:6px 0 0">${W.memory.split(/\s+/).length} words of a 300-word budget.</p>`
      : '<p class="muted">Empty — nothing had been corrected yet when this window was reviewed (or the window predates the memory cache).</p>'}</div>
    <div class="label">Overall task — routing command plan</div>
    <ol class="plan" style="margin:8px 0 20px">${plan || '<li class="muted">none</li>'}</ol>
    <div class="label">Window summary passed to both calls</div>
    <div class="scroll"><table><tbody>` +
    Object.entries(W.summary).map(([k, v]) =>
      `<tr><td class="n muted">${esc(k)}</td><td class="n">${esc(v === null ? 'null' : v)}</td></tr>`).join('') +
    `</tbody></table></div>
    <div class="label" style="margin-top:20px">Collision log</div>
    <p class="muted">${W.collision_events.length ? esc(JSON.stringify(W.collision_events)) : 'No collisions recorded by the on-board sensors.'}</p>`;
  $('p-events').querySelectorAll('.evt').forEach(el => el.onclick = () => seek(Number(el.dataset.t)));
}

function renderPrompt(which, caption) {
  const text = W.prompts[which];
  const el = $('p-' + which);
  const attached = (which === 'review' && W.plot)
    ? `<div class="label" style="margin-bottom:6px">Attached to this call as an image, with the video</div>
       <img class="plot" alt="Env reward and route progress vs video time"
            src="data:image/png;base64,${W.plot}">` : '';
  el.innerHTML = `<div class="promptbar">
      <span class="label">${esc(caption)}</span>
      <span class="chip">${text.length.toLocaleString()} chars</span>
      <button data-copy>Copy</button>
    </div>${attached}<pre>${esc(text)}</pre>`;
  el.querySelector('[data-copy]').onclick = (ev) => {
    navigator.clipboard.writeText(text).then(() => { ev.target.textContent = 'Copied'; });
  };
}

load(0);
</script>
"""


def render_page(windows: list[dict[str, Any]]) -> str:
    payload = json.dumps({"windows": windows}, ensure_ascii=False)
    # The payload lives in a <script type="application/json"> block; only `</script>` can break out.
    payload = payload.replace("</", "<\\/")
    return _TEMPLATE.replace("__PAYLOAD__", payload)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=Path,
                    help="window dir, cast_relabel/ dir, or run dir")
    ap.add_argument("-o", "--out", type=Path, default=Path("cast_relabel_viewer.html"))
    ap.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS,
                    help=f"cap embedded windows (default {DEFAULT_MAX_WINDOWS}); videos dominate size")
    ap.add_argument("--no-video", action="store_true", help="skip embedding the mp4s")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip rendering the reward/progress figure the review call attaches")
    args = ap.parse_args()

    win_dirs: list[Path] = []
    for p in args.paths:
        found = find_window_dirs(p)
        if not found:
            print(f"[viewer] no window dirs (trajectory.json) under {p}", file=sys.stderr)
        win_dirs.extend(found)
    if not win_dirs:
        print("[viewer] nothing to render.", file=sys.stderr)
        return 1

    dropped = max(0, len(win_dirs) - args.max_windows)
    if dropped:
        print(f"[viewer] {len(win_dirs)} windows found; embedding the first {args.max_windows} "
              f"(raise --max-windows to include the other {dropped}).")
        win_dirs = win_dirs[: args.max_windows]

    windows: list[dict[str, Any]] = []
    for d in win_dirs:
        try:
            windows.append(build_window_payload(
                d, embed_video=not args.no_video, embed_plot=not args.no_plot
            ))
            print(f"[viewer] loaded {d}")
        except Exception as exc:  # noqa: BLE001 - one bad window must not lose the rest
            print(f"[viewer] skipped {d}: {exc}", file=sys.stderr)
    if not windows:
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_page(windows), encoding="utf-8")
    mb = args.out.stat().st_size / 1e6
    fails = sum(1 for w in windows for c in w["checks"] if c["status"] == "fail")
    print(f"[viewer] wrote {args.out} ({mb:.1f} MB, {len(windows)} windows, {fails} failing checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
