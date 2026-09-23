"""Standalone checks on the CAST review prompt contract (no VLM, no CARLA, no model).

    JAX_PLATFORMS=cpu PYTHONPATH=impls uv run python impls/coaches/test_vlm_prompt.py

Each rule here is only useful if the DATA it refers to is actually in the prompt, so every check
pairs the instruction with the field it depends on. Covers the 2026-09-09 additions: the episode's
PRIMARY GOAL, the no-yielding gap rule, and the subtask / reasoning / routing-command disagreement
check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


from coaches.vlm_feedback import build_coaching_prompt, describe_route_goal

MD = {
    "route": "signalized-junction-left-turn-001",
    "route_command_plan": [
        {"command": "follow the road", "start_distance_m": 0},
        {"command": "go left at the next intersection", "start_distance_m": 78},
    ],
    "route_distance_end_m": 84.0,
    "route_total_distance_m": 210.0,
    "steps": [
        {
            "video_timestamp_sec": 1.0,
            "episode_step": 1,
            # the disagreement the rule exists to catch
            "subtask": "turn right at the intersection",
            "reasoning": "I should turn left here to follow the route",
            "prompt": "go left at the next intersection",
            "route_progress_pct": 12.0,
            "reward_total": 0.1,
        }
    ],
}

# ── 1. the data the rules depend on is actually rendered ──────────────────────────────
print("\n[1] per-timestamp data")
p = build_coaching_prompt(MD)
check("executed subtask is in the prompt", "turn right at the intersection" in p)
check("chain-of-thought reasoning is in the prompt", "I should turn left here" in p)
check("routing command is in the prompt", "go left at the next intersection" in p)
check("the routing-command plan is rendered", "78 m along the route" in p)

# ── 2. the disagreement rule ──────────────────────────────────────────────────────────
print("\n[2] subtask / reasoning / routing-command disagreement")
check("rule is present", "MANDATORY: check the executed SUBTASK" in p)
check("authority order is stated", "ROUTING COMMAND first" in p)
check(
    "the subtask-wrong case is spelled out",
    'the subtask\n              says "turn right"' in p or '"turn right"' in p,
)
check("the both-wrong case is covered", "BOTH are wrong" in p)
check("the bad-reasoning case is covered", "the reasoning contradicts it" in p)
check("judged against the command in force at that moment", "AT THAT TIMESTAMP" in p)
check("wording differences are excluded", "pure wording differences" in p)

# ── 3. it reaches every stage that emits events ───────────────────────────────────────
print("\n[3] stage coverage")
for stage in ("both", "events"):
    check(
        f"stage={stage} carries the rule",
        "MANDATORY: check the executed SUBTASK" in build_coaching_prompt(MD, stage=stage),
    )
# The scene stage answers in prose and emits no events, so the rule is not required there.

# ── 4. the other two prompt additions still hold ──────────────────────────────────────
print("\n[4] the other prompt rules")
check("PRIMARY GOAL is stated", "PRIMARY GOAL of this episode" in p)
check(
    "the goal is the route's, not a generic one",
    describe_route_goal("signalized-junction-left-turn-001") in p,
)
check("traffic does not yield / take the gap", "does NOT yield" in p)
check("failing the goal is the top checklist item", "FIRST AND MOST IMPORTANT" in p)

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
    raise SystemExit(1)
print("all checks passed")
