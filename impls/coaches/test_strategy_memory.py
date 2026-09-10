"""Standalone checks for episode-level strategy memory (no VLM, no CARLA, no video).

    JAX_PLATFORMS=cpu WANDB_MODE=disabled PYTHONPATH=impls uv run python \\
        impls/coaches/test_strategy_memory.py

Covers the 2026-09-10 feature: after each episode, one VLM call reviews the whole rollout plus
every correction made during it, in the context of the driving score, and the resulting sentence
goes into the CorrectionMemory bank the window prompts already read.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


from coaches.correction_memory import DEFAULT_MAX_WORDS, CorrectionMemory
from coaches.strategy_memory import (
    _tidy_sentence,
    collect_episode_chunks,
    collect_episode_corrections,
    format_corrections_block,
    format_executed_block,
    format_route_plan_block,
    summarize_episode_strategy,
)


def write_window(root: Path, ep: int, win: int, *, step0: int, events: list[dict], ok=True):
    d = root / f"ep{ep:04d}_win{win:04d}"
    d.mkdir(parents=True, exist_ok=True)
    if not ok:  # a window whose VLM review failed leaves the dir but no artifact
        return d
    # 10 env steps per window-video second, offset by step0.
    chunks = [
        {
            "chunk_index": i,
            "episode_step_start": step0 + i * 10,
            "episode_step_end": step0 + i * 10 + 9,
            "video_time_start_sec": i * 1.0,
            "video_time_end_sec": i * 1.0 + 0.9,
            "label": "GOOD" if i < 2 else "BAD",
            "credit_source": "" if i < 2 else "precursor",
            "original_subtask": "hold lane and follow traffic" if i < 2 else "come to a stop",
            "suggested_subtasks": [] if i < 2 else ["take the gap and complete the turn"],
        }
        for i in range(3)
    ]
    (d / "cast_relabel.json").write_text(
        json.dumps({"episode": ep, "window_index": win, "action_chunks": chunks, "events": events})
    )
    return d


print("\n[1] corrections are collected and re-timed into the EPISODE frame")
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    write_window(root, 1, 0, step0=1, events=[
        {"timestamp_sec": 0.0, "label": "GOOD", "description": "smooth start", "correction": ""},
        {"timestamp_sec": 2.0, "label": "BAD", "description": "braked early", "correction": "take the gap"},
    ])
    write_window(root, 1, 1, step0=101, events=[
        {"timestamp_sec": 1.0, "label": "BAD", "description": "stalled", "correction": "accelerate"},
    ])
    write_window(root, 1, 2, step0=201, events=[], ok=False)   # failed review
    write_window(root, 2, 0, step0=1, events=[
        {"timestamp_sec": 0.0, "label": "GOOD", "description": "other episode", "correction": ""},
    ])

    got = collect_episode_corrections(root, 1)
    check("only this episode's windows are read", len(got) == 3, f"n={len(got)}")
    check("a window with no artifact is skipped, not fatal", all(g["description"] != "" for g in got))
    steps = [g["episode_step"] for g in got]
    check("window-relative times become episode steps", steps == [1, 21, 111], str(steps))
    check("events are ordered along the episode", steps == sorted(steps))
    check("corrections are carried", got[1]["correction"] == "take the gap", got[1]["correction"])
    check("the other episode is not mixed in", all("other episode" not in g["description"] for g in got))

    empty = collect_episode_corrections(root, 99)
    check("an episode with no windows yields nothing", empty == [])
    check("a missing directory is handled", collect_episode_corrections(root / "nope", 1) == [])

print("\n[2] the corrections block is timestamped for the reviewer")
blk = format_corrections_block(got, episode_fps=10.0)
check("uses video seconds when the mapping is known", "t=   0.1s" in blk or "t=  0.1s" in blk, blk.splitlines()[0])
check("shows the episode step too", "episode step 21" in blk)
check("labels each event", "[BAD]" in blk and "[GOOD]" in blk)
check("renders the correction", "CORRECTION: take the gap" in blk)
blk_nofps = format_corrections_block(got, episode_fps=None)
check("falls back to raw steps without a mapping", "episode step 21" in blk_nofps and "t=" not in blk_nofps)
check("an empty list is stated, not blank", "no corrections" in format_corrections_block([]))

print("\n[3] the reply is reduced to one clean sentence")
check("whitespace collapses", _tidy_sentence("a\n\n  b   c", 50) == "a b c")
check("a preamble is stripped", _tidy_sentence("Here's the summary: crept everywhere.", 50) == "crept everywhere.")
check("markdown is stripped", _tidy_sentence("**bold**", 50) == "bold")
long = _tidy_sentence(" ".join(["word"] * 100), 10)
check("over-long replies are truncated to budget", len(long.split()) <= 11, f"{len(long.split())} words")


class FakeCoach:
    def __init__(self, reply="Crept to every junction and waited for a full gap; cost the exit."):
        self.reply, self.calls = reply, []

    def analyze_video_text(self, path, prompt):
        self.calls.append(("video", path, prompt))
        return self.reply

    def complete_text(self, prompt):
        self.calls.append(("text", None, prompt))
        return self.reply


print("\n[4] the summary call")
with tempfile.TemporaryDirectory() as td:
    vid = Path(td) / "ep0001.mp4"
    vid.write_bytes(b"\x00" * 64)
    c = FakeCoach()
    out = summarize_episode_strategy(
        c, video_path=vid, corrections=got, route="highway-exit-002",
        route_goal="LEAVE THE HIGHWAY", driving_score=43.5, route_completion=88.0,
    )
    check("returns the sentence", out.startswith("Crept to every junction"), out[:40])
    check("the video path is used", c.calls[0][0] == "video")
    prompt = c.calls[0][2]
    check("the score is in the prompt", "43.50" in prompt)
    check("route completion is in the prompt", "88.0%" in prompt)
    check("the route goal is in the prompt", "LEAVE THE HIGHWAY" in prompt)
    check("the corrections are in the prompt", "take the gap" in prompt)
    check("it asks for a strategy, not an event list", "as a pattern rather than a list" in prompt)

    c2 = FakeCoach()
    summarize_episode_strategy(c2, video_path=None, corrections=got, route="r", driving_score=1.0)
    check("falls back to text-only without a video", c2.calls[0][0] == "text")

    class Boom(FakeCoach):
        def analyze_video_text(self, path, prompt):
            raise RuntimeError("gemini down")

    quiet = summarize_episode_strategy(
        Boom(), video_path=vid, corrections=got, route="r", driving_score=1.0
    )
    check("a VLM failure returns empty, never raises", quiet == "")

print("\n[5] memory storage, rendering and pruning")
check("the word budget was raised for this", DEFAULT_MAX_WORDS >= 900, str(DEFAULT_MAX_WORDS))
with tempfile.TemporaryDirectory() as td:
    m = CorrectionMemory(Path(td) / "mem.json", max_words=DEFAULT_MAX_WORDS)
    m.add_strategy("Crept to every junction.", episode=1, driving_score=27.5)
    m.add_strategy("Took gaps promptly.", episode=2, driving_score=71.0)
    r = m.render()
    check("strategies appear in the rendered block", "Crept to every junction." in r)
    check("each is paired with its score", "score 27.5" in r and "score 71.0" in r)
    check("the episode number is shown", "episode 2" in r)
    check("the block explains how to use it", "steer the high-level strategy" in r)
    check("blank sentences are ignored", (m.add_strategy("   ", episode=3, driving_score=0.0), len(m.strategies))[1] == 2)

    for i in range(3, 12):
        m.add_strategy(f"strategy number {i}.", episode=i, driving_score=float(i))
    check("the list is capped", len(m.strategies) <= 5, f"n={len(m.strategies)}")
    check("the newest survives", any("strategy number 11" in s["sentence"] for s in m.strategies))
    check("the oldest was dropped", not any("Crept to every junction" in s["sentence"] for s in m.strategies))

    m2 = CorrectionMemory(Path(td) / "mem.json", max_words=DEFAULT_MAX_WORDS)
    check("strategies persist across reload", len(m2.strategies) == len(m.strategies))

print("\n[6] the policy's OWN behaviour is included, not just the reviewer's commentary")
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    write_window(root, 1, 0, step0=1, events=[
        {"timestamp_sec": 0.0, "label": "GOOD", "description": "ok", "correction": ""},
    ])
    ch = collect_episode_chunks(root, 1)
    check("chunks are collected", len(ch) == 3, f"n={len(ch)}")
    check("the EXECUTED subtask is captured", ch[0]["executed_subtask"] == "hold lane and follow traffic", ch[0]["executed_subtask"])
    check("the relabelled subtask is captured", ch[2]["corrected_subtask"] == "take the gap and complete the turn")
    check("the verdict and credit source come too", (ch[2]["label"], ch[2]["credit_source"]) == ("BAD", "precursor"))
    check("chunks are ordered along the episode", [c["episode_step_start"] for c in ch] == [1, 11, 21])

    blk = format_executed_block(ch, episode_fps=10.0)
    check("identical consecutive subtasks collapse", "(2 consecutive chunks)" in blk, blk.splitlines()[0])
    check("the collapsed run spans both chunks", "0.1-  2.0s" in blk or "0.1-2.0s" in blk.replace(" ", ""))
    check("a relabel is shown", "relabelled to:" in blk)
    check("an empty list is stated", "no executed subtasks" in format_executed_block([]))

print("\n[7] routing commands and the scenario goal")
plan = [{"command": "follow the road", "start_distance_m": 0},
        {"command": "go left at the next intersection", "start_distance_m": 78}]
rb = format_route_plan_block(plan)
check("commands render in order with distances", "1. follow the road" in rb and "~78 m" in rb)
check("an absent plan renders nothing", format_route_plan_block(None) == "")

from coaches.strategy_memory import build_strategy_prompt

prompt = build_strategy_prompt(
    route="signalized-junction-left-turn-001", route_goal="", driving_score=43.5,
    route_completion=88.0, corrections_block="<C>",
    executed_block=format_executed_block(ch, episode_fps=10.0),
    route_plan_block=rb,
)
check("the goal is DERIVED from the scenario name", "turn left at a signalised junction" in prompt)
check("it is framed as the PRIMARY GOAL, as in the review prompt", "PRIMARY GOAL of this episode" in prompt)
check("obeying commands but failing the goal is called out", "never accomplished the" in prompt)
check("routing commands are in the prompt", "go left at the next intersection" in prompt)
check("executed subtasks are in the prompt", "hold lane and follow traffic" in prompt)
check("it asks the summary to read strategy off the executed subtasks", "read it off the executed subtasks" in prompt)

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
    raise SystemExit(1)
print("all checks passed")
