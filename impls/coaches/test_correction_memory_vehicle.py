"""Standalone checks for the vehicle-state carry-over in CorrectionMemory.

    JAX_PLATFORMS=cpu PYTHONPATH=impls uv run python impls/coaches/test_correction_memory_vehicle.py

A collision in one window must stay visible in the next windows until the ego demonstrably
recovers (moving AND making route progress) or the carry ages out.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coaches.correction_memory import (
    _MAX_CRASH_CARRY_WINDOWS,
    CorrectionMemory,
)

FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def state(*, collided=False, stuck=False, speed=6.0, progress=3.0, end_pct=50.0):
    return {
        "collided": collided,
        "stuck": stuck,
        "end_speed_mps": speed,
        "route_progress_delta_pct": progress,
        "route_progress_end_pct": end_pct,
    }


# ── 1. a crash is carried forward ─────────────────────────────────────────────────────
print("\n[1] crash carries into later windows")
m = CorrectionMemory(path=None)
m.observe_vehicle_state(state(), window_index=1)
check("clean window -> no crash note", m.crash_note == "" and "Still relevant" not in m.render())

m.observe_vehicle_state(state(collided=True, stuck=True, speed=0.0, progress=0.0), window_index=2)
check("crash recorded", m.crash_age == 1 and "stuck" in m.crash_note, m.crash_note)
check("rendered in the block", "Still relevant" in m.render_vehicle_block())

m.observe_vehicle_state(state(speed=0.2, progress=0.0), window_index=3)
check("still carried while the ego has not recovered", m.crash_age == 2, f"age={m.crash_age}")
check("names how long ago", "2 windows ago" in m.render_vehicle_block(), m.render_vehicle_block()[:200])

# ── 2. recovery clears it ─────────────────────────────────────────────────────────────
print("\n[2] recovery clears the carry")
m.observe_vehicle_state(state(speed=7.0, progress=4.0), window_index=4)
check("cleared once moving AND progressing", m.crash_age == 0 and m.crash_note == "")
check("block no longer mentions it", "Still relevant" not in m.render_vehicle_block())

print("\n[2b] speed alone does NOT count as recovery (wheels spinning on a wall)")
m = CorrectionMemory(path=None)
m.observe_vehicle_state(state(collided=True, stuck=True, speed=0.0, progress=0.0), window_index=1)
m.observe_vehicle_state(state(speed=8.0, progress=0.0), window_index=2)
check("still carried", m.crash_age == 2, f"age={m.crash_age}")

print("\n[2c] progress alone does NOT count either")
m = CorrectionMemory(path=None)
m.observe_vehicle_state(state(collided=True, speed=0.0, progress=0.0), window_index=1)
m.observe_vehicle_state(state(speed=0.1, progress=5.0), window_index=2)
check("still carried", m.crash_age == 2, f"age={m.crash_age}")

# ── 3. it ages out even without recovery ──────────────────────────────────────────────
print(f"\n[3] ages out after {_MAX_CRASH_CARRY_WINDOWS} windows")
m = CorrectionMemory(path=None)
m.observe_vehicle_state(state(collided=True, stuck=True, speed=0.0, progress=0.0), window_index=1)
ages = []
for w in range(2, 2 + _MAX_CRASH_CARRY_WINDOWS + 2):
    m.observe_vehicle_state(state(speed=0.0, progress=0.0), window_index=w)
    ages.append(m.crash_age)
check("eventually cleared without recovery", ages[-1] == 0, f"ages={ages}")
check("did not clear immediately", ages[0] != 0, f"ages={ages}")

# ── 4. a fresh crash resets the clock ─────────────────────────────────────────────────
print("\n[4] repeated crashes keep the note alive")
m = CorrectionMemory(path=None)
m.observe_vehicle_state(state(collided=True, speed=0.0, progress=0.0), window_index=1)
m.observe_vehicle_state(state(speed=0.0, progress=0.0), window_index=2)
m.observe_vehicle_state(state(collided=True, speed=0.0, progress=0.0), window_index=3)
check("reset to 1 on the new crash", m.crash_age == 1, f"age={m.crash_age}")
check("names the newest window", "window 3" in m.crash_note, m.crash_note)

# ── 5. the block reaches render() with no corrections at all ──────────────────────────
print("\n[5] vehicle block renders independently of the correction log")
m = CorrectionMemory(path=None)
check("empty before anything is observed", m.render() == "")
m.observe_vehicle_state(state(collided=True, speed=0.0, progress=0.0), window_index=1)
r = m.render()
check("present with zero corrections", "PREVIOUS window" in r and "Still relevant" in r)
check("no correction-log header yet", "Correction memory" not in r)

# ── 6. observe_window folds state in, corrections or not ──────────────────────────────
print("\n[6] observe_window(vehicle_state=...)")
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "correction_memory.json"
    m = CorrectionMemory(path=path)
    m.observe_window({"action_chunks": []}, window_index=1,
                     vehicle_state=state(collided=True, stuck=True, speed=0.0, progress=0.0))
    check("state folded in despite no corrections", m.crash_age == 1)
    check("persisted to disk", path.is_file())
    reloaded = CorrectionMemory(path=path)
    check("survives a reload", reloaded.crash_age == 1 and "stuck" in reloaded.crash_note,
          f"age={reloaded.crash_age} note={reloaded.crash_note!r}")
    check("reloaded block renders", "Still relevant" in reloaded.render())

# ── 7. budget fitting still terminates with the block present ─────────────────────────
print("\n[7] word budget")
m = CorrectionMemory(path=None, max_words=40)
m.observe_vehicle_state(state(collided=True, stuck=True, speed=0.0, progress=0.0), window_index=1)
for w in range(2, 12):
    m.observe_window(
        {"action_chunks": [{"label": "BAD", "original_subtask": "The vehicle accelerates.",
                            "suggested_subtasks": ["The vehicle comes to a stop."]}]},
        window_index=w,
    )
check("_fit_budget terminates", True)
check("block survives pruning", "PREVIOUS window" in m.render())

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
    raise SystemExit(1)
print("all checks passed")
