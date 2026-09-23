"""Standalone checks for CAST window-drop accounting (no VLM, no CARLA, no wandb).

    JAX_PLATFORMS=cpu WANDB_MODE=disabled PYTHONPATH=impls uv run python \\
        impls/coaches/test_window_drop_accounting.py

A window whose VLM call fails is swallowed as non-fatal: it produces no HL samples, no
cast_relabel.json and no debug video, and before this the run's metrics said nothing about it.
On 2026-09-10 that hid 69-75% of windows being lost to Gemini file-upload failures -- the policy
saw a quarter of its intended supervision while every chart looked normal.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


from coaches.cast_relabel import OnlineCastRelabelSession
from configs.steervla_cast_relabel_hl750x25_adaptive_config import get_config

print("\n[1] counters start clean and track outcomes")
with tempfile.TemporaryDirectory() as td:
    s = OnlineCastRelabelSession(get_config().cast_relabel, save_dir=td, action_chunk_steps=10)
    check("starts at zero", (s._windows_attempted, s._windows_succeeded, s._windows_dropped) == (0, 0, 0))

    s._record_window_outcome(ok=True, global_step=10)
    s._record_window_outcome(ok=True, global_step=20)
    check("successes counted", (s._windows_attempted, s._windows_succeeded, s._windows_dropped) == (2, 2, 0))

    s._record_window_outcome(
        ok=False, exc=RuntimeError("Gemini file upload failed with state='FAILED'."), global_step=30
    )
    check("drops counted", (s._windows_attempted, s._windows_succeeded, s._windows_dropped) == (3, 2, 1))
    check("drop rate is dropped/attempted", abs(s._windows_dropped / s._windows_attempted - 1 / 3) < 1e-9)

print("\n[2] drop reasons are bucketed usefully")
cases = {
    "upload_failed": RuntimeError("Gemini file upload failed with state='FAILED'."),
    "rate_limited": RuntimeError("HTTP 429 quota exceeded"),
    "timeout": TimeoutError("request timed out"),
    "model_or_file_not_found": RuntimeError("404 model not found"),
}
for expected, exc in cases.items():
    got = OnlineCastRelabelSession._drop_reason(exc)
    check(f"{expected}", got == expected, f"got {got}")
check(
    "an unrecognised error keeps its type name",
    OnlineCastRelabelSession._drop_reason(ValueError("something else")) == "ValueError",
)

print("\n[3] the accounting is wired into BOTH catch sites")
src = Path("impls/coaches/cast_relabel.py").read_text()
check("sync path records success", src.count("self._record_window_outcome(ok=True") == 2)
check("both paths record failure", src.count("self._record_window_outcome(ok=False") == 2)
check(
    "the drop count is surfaced in the console line too",
    src.count("dropped {self._windows_dropped}/{self._windows_attempted} windows so far") == 2,
)
check("wandb series is under cast/", '"cast/window_drop_rate"' in src)
check("per-reason series too", 'f"cast/window_drop_reason/{k}"' in src)

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
    raise SystemExit(1)
print("all checks passed")
