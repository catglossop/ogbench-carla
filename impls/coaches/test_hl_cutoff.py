"""Standalone checks for the cast_relabel HL supervision cutoff.

Run directly (no model, no CARLA, no VLM)::

    JAX_PLATFORMS=cpu PYTHONPATH=impls uv run python impls/coaches/test_hl_cutoff.py

Covers the 2026-09-07 rule change: leaving the route still cuts immediately, but a collision only
cuts once the ego is STUCK in it -- incidental contact it drives away from is kept.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coaches.cast_relabel import (
    DEFAULT_HL_COLLISION_STUCK_TICKS,
    OnlineCastRelabelSession,
    resolve_window_outcome,
)

FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


class _Session:
    """The real cutoff methods, without building a whole session (which needs a VLM provider)."""

    def __init__(
        self,
        stuck_ticks=DEFAULT_HL_COLLISION_STUCK_TICKS,
        on_collision=True,
        on_off_route=True,
        on_route_divergence=True,
        div_speed=2.0,
        div_ticks=60,
        div_progress=2.0,
    ):
        self.hl_stop_after_failure = True
        self.store_hl_dataset = True
        self.hl_stop_on_collision = on_collision
        self.hl_stop_on_off_route = on_off_route
        self.hl_collision_stuck_ticks = stuck_ticks
        self.hl_stop_on_route_divergence = on_route_divergence
        self.hl_divergence_speed_mps = div_speed
        self.hl_divergence_ticks = div_ticks
        self.hl_divergence_progress_m = div_progress
        self._hl_cutoff_step = None
        self._hl_cutoff_reason = ""
        self._last_collision_step = None
        self._moving_run_start_step = None
        self._moving_run_start_route_m = 0.0

    _is_stuck_collision = OnlineCastRelabelSession._is_stuck_collision
    _maybe_trip_hl_cutoff = OnlineCastRelabelSession._maybe_trip_hl_cutoff
    _route_divergence_stall_step = OnlineCastRelabelSession._route_divergence_stall_step


def step(
    n,
    *,
    collision_delta=0.0,
    outside_route_delta=0.0,
    stuck=0,
    termination_reason=None,
    route_deviation_delta=0.0,
    speed=0.0,
    route_m=0.0,
):
    return {
        "episode_step": n,
        "collision_delta": collision_delta,
        "outside_route_delta": outside_route_delta,
        "crash_stuck_ticks": stuck,
        "termination_reason": termination_reason,
        "route_deviation_delta": route_deviation_delta,
        "ego_speed_mps": speed,
        "route_distance_m": route_m,
    }


# ── 1. a graze the ego drives away from is KEPT ───────────────────────────────────────
print("\n[1] incidental collision -> no cutoff")
s = _Session()
s._maybe_trip_hl_cutoff(step(100, collision_delta=1.0))          # impact
for i in range(1, 6):
    s._maybe_trip_hl_cutoff(step(100 + i, stuck=i))              # briefly stopped
for i in range(6, 40):
    s._maybe_trip_hl_cutoff(step(100 + i, stuck=0))              # moving again -> ticks reset
check("counted collision alone does not cut", s._hl_cutoff_step is None, f"cutoff={s._hl_cutoff_step}")
check("but the impact step is remembered", s._last_collision_step == 100)

# ── 2. a collision it gets stuck in DOES cut, backdated to the impact ──────────────────
print("\n[2] stuck collision -> cutoff, backdated")
s = _Session()
s._maybe_trip_hl_cutoff(step(200, collision_delta=1.0))
for i in range(1, 26):
    s._maybe_trip_hl_cutoff(step(200 + i, stuck=i))
check("cuts once stuck for the threshold", s._hl_cutoff_reason == "collision", s._hl_cutoff_reason)
check(
    "backdated to the impact, not to the detection step",
    s._hl_cutoff_step == 200,
    f"cutoff={s._hl_cutoff_step} (detector fired at {200 + DEFAULT_HL_COLLISION_STUCK_TICKS})",
)

print("\n[2b] stuck with no counted impact seen -> backdate to start of the stuck run")
s = _Session()
for i in range(1, 26):
    s._maybe_trip_hl_cutoff(step(300 + i, stuck=i))
check(
    "cut at the start of the stuck run",
    s._hl_cutoff_step == 300 + DEFAULT_HL_COLLISION_STUCK_TICKS - DEFAULT_HL_COLLISION_STUCK_TICKS,
    f"cutoff={s._hl_cutoff_step}",
)

# ── 3. leaving the route still cuts immediately ───────────────────────────────────────
print("\n[3] off-route -> immediate cutoff (unchanged)")
s = _Session()
s._maybe_trip_hl_cutoff(step(50, outside_route_delta=1.0))
check("cuts at once", s._hl_cutoff_step == 50 and s._hl_cutoff_reason == "off_route",
      f"{s._hl_cutoff_step}/{s._hl_cutoff_reason}")

print("\n[3b] off-route wins over a simultaneous stuck collision")
s = _Session()
s._maybe_trip_hl_cutoff(step(60, outside_route_delta=1.0, collision_delta=1.0, stuck=99))
check("reason is off_route", s._hl_cutoff_reason == "off_route", s._hl_cutoff_reason)

# ── 4. first cutoff of the episode wins, and it does NOT move ─────────────────────────
print("\n[4] latching")
s = _Session()
s._maybe_trip_hl_cutoff(step(10, outside_route_delta=1.0))
s._maybe_trip_hl_cutoff(step(20, outside_route_delta=1.0))
check("later failures do not move the cutoff", s._hl_cutoff_step == 10)

# ── 5. legacy behavior is still reachable ─────────────────────────────────────────────
print("\n[5] hl_collision_stuck_ticks=0 restores the old any-contact rule")
s = _Session(stuck_ticks=0)
s._maybe_trip_hl_cutoff(step(70, collision_delta=1.0))
check("any counted collision cuts", s._hl_cutoff_step == 70 and s._hl_cutoff_reason == "collision")

print("\n[5b] hl_stop_on_collision=False ignores collisions entirely")
s = _Session(on_collision=False)
for i in range(1, 40):
    s._maybe_trip_hl_cutoff(step(80 + i, collision_delta=1.0 if i == 1 else 0.0, stuck=i))
check("no cutoff", s._hl_cutoff_step is None, f"cutoff={s._hl_cutoff_step}")

# ── 6. crash_stuck termination remains a fallback ─────────────────────────────────────
print("\n[6] termination_reason=crash_stuck fallback (step records without the tick field)")
s = _Session()
s._maybe_trip_hl_cutoff(step(400, collision_delta=1.0))
s._maybe_trip_hl_cutoff(step(430, termination_reason="crash_stuck"))
check("cuts", s._hl_cutoff_reason == "crash_stuck", s._hl_cutoff_reason)
check("backdated to the impact", s._hl_cutoff_step == 400, f"cutoff={s._hl_cutoff_step}")

# ── 7. outcome tagging agrees with the cutoff ─────────────────────────────────────────
print("\n[7] resolve_window_outcome uses the same standard")
check(
    "survivable collision is NOT catastrophic",
    resolve_window_outcome({"collision_events": [{"new_event": True}], "max_crash_stuck_ticks": 4}) == "",
)
check(
    "survivable collision in a completed route still reads as success",
    resolve_window_outcome(
        {"collision_events": [{"new_event": True}], "max_crash_stuck_ticks": 4, "route_completed": True}
    ) == "route_completed",
)
check(
    "stuck collision IS catastrophic",
    resolve_window_outcome({"max_crash_stuck_ticks": 25}) == "collision",
)
check(
    "stuck collision beats route_completed",
    resolve_window_outcome({"max_crash_stuck_ticks": 25, "route_completed": True}) == "collision",
)
check(
    "threshold is honoured",
    resolve_window_outcome({"max_crash_stuck_ticks": 5}, collision_stuck_ticks=5) == "collision"
    and resolve_window_outcome({"max_crash_stuck_ticks": 4}, collision_stuck_ticks=5) == "",
)
check("off_route unchanged", resolve_window_outcome({}, "off_route") == "off_route")
check("route divergence unchanged", resolve_window_outcome({}, "off route at t=3.0s") == "route_divergence")

# ── 8. the per-episode reset ──────────────────────────────────────────────────────────
print("\n[8] begin_episode clears the cutoff (was latched for the whole RUN)")
src = Path("impls/coaches/cast_relabel.py").read_text()
begin = src[src.index("    def begin_episode("):src.index("    def record_frame(")]
check("begin_episode resets _hl_cutoff_step", "self._hl_cutoff_step = None" in begin)
check("begin_episode resets _hl_cutoff_reason", 'self._hl_cutoff_reason = ""' in begin)
check("begin_episode resets _last_collision_step", "self._last_collision_step = None" in begin)

# ── 9. programmatic route-divergence detector ─────────────────────────────────────────
print("\n[9] route divergence: moving but not progressing along the plan")

# Driving hard for 3 s while the route barely advances -> diverged, backdated to the start.
s = _Session()
for n in range(81):
    s._maybe_trip_hl_cutoff(step(n, speed=6.0, route_m=10.0 + 0.001 * n))
check(
    "sustained motion with no route progress trips",
    s._hl_cutoff_step is not None and "route_divergence" in s._hl_cutoff_reason,
    f"step={s._hl_cutoff_step} reason={s._hl_cutoff_reason[:40]}",
)
check("cutoff is backdated to where the run began", s._hl_cutoff_step == 0, f"got {s._hl_cutoff_step}")

# Normal driving: route advances in step with the odometer -> never trips.
s = _Session()
for n in range(400):
    s._maybe_trip_hl_cutoff(step(n, speed=6.0, route_m=0.30 * n))
check("normal driving does not trip", s._hl_cutoff_step is None)

# Stopped at a light for a long time: zero progress, but not moving -> must NOT trip.
s = _Session()
for n in range(400):
    s._maybe_trip_hl_cutoff(step(n, speed=0.0, route_m=25.0))
check("a long legitimate stop does not trip", s._hl_cutoff_step is None)

# Crawling below the speed threshold in queued traffic -> must NOT trip.
s = _Session()
for n in range(400):
    s._maybe_trip_hl_cutoff(step(n, speed=1.0, route_m=25.0))
check("crawling below the speed threshold does not trip", s._hl_cutoff_step is None)

# Just under the tick threshold -> not yet.
s = _Session()
for n in range(59):
    s._maybe_trip_hl_cutoff(step(n, speed=6.0, route_m=10.0))
check("does not trip before the tick threshold", s._hl_cutoff_step is None)

# The switch turns it off.
s = _Session(on_route_divergence=False)
for n in range(200):
    s._maybe_trip_hl_cutoff(step(n, speed=6.0, route_m=10.0))
check("hl_stop_on_route_divergence=False disables it", s._hl_cutoff_step is None)

# ── 10. the leaderboard's own InRouteTest counter ─────────────────────────────────────
print("\n[10] route_deviation_delta (InRouteTest) trips immediately")
s = _Session()
s._maybe_trip_hl_cutoff(step(10, speed=6.0, route_m=5.0))
s._maybe_trip_hl_cutoff(step(11, route_deviation_delta=1.0, speed=6.0, route_m=5.0))
check(
    "InRouteTest delta trips at that step",
    s._hl_cutoff_step == 11 and "InRouteTest" in s._hl_cutoff_reason,
    f"step={s._hl_cutoff_step} reason={s._hl_cutoff_reason[:40]}",
)
s = _Session(on_route_divergence=False)
s._maybe_trip_hl_cutoff(step(11, route_deviation_delta=1.0))
check("disabled switch also gates InRouteTest", s._hl_cutoff_step is None)

# ── 11. the new state is reset per episode ────────────────────────────────────────────
print("\n[11] begin_episode clears the divergence run state")
check("begin_episode resets _moving_run_start_step", "self._moving_run_start_step = None" in begin)


print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
    raise SystemExit(1)
print("all checks passed")
