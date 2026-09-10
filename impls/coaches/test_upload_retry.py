"""Standalone checks for Gemini file-upload retry (no network, no VLM, no CARLA).

    JAX_PLATFORMS=cpu WANDB_MODE=disabled PYTHONPATH=impls uv run python \\
        impls/coaches/test_upload_retry.py

The failure this exists for: the upload SUCCEEDS (bytes intact, sha256 present, correct mime) and
Gemini's own processing then fails with google.rpc code 13 INTERNAL. That is worth resending;
code 3 INVALID_ARGUMENT is not, because the file itself is what is being rejected.
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


import coaches.vlm_feedback as vf

vf.time.sleep = lambda *_a, **_k: None  # no real backoff in tests

VIDEO = str(Path(tempfile.mkdtemp()) / "rollout.mp4")
Path(VIDEO).write_bytes(b"\x00" * 4096)

ACTIVE = {"name": "files/ok", "state": "ACTIVE", "mimeType": "video/mp4", "uri": "u"}


def failed(code, msg="boom"):
    return {"name": f"files/bad{code}", "state": "FAILED", "error": {"code": code, "message": msg}}


def make_coach():
    c = vf.GeminiVLMCOach.__new__(vf.GeminiVLMCOach)
    c.api_key = "k"
    return c


def run(sequence):
    """Drive _upload_media against a scripted sequence of upload results."""
    calls = {"n": 0, "deleted": []}

    def fake_upload(_path, _key):
        r = sequence[min(calls["n"], len(sequence) - 1)]
        calls["n"] += 1
        return r

    vf._gemini_upload_file = fake_upload
    vf._gemini_get_file = lambda name, key: next(
        (r for r in sequence if r.get("name") == name), ACTIVE
    )
    vf._gemini_delete_file_quiet = lambda name, key: calls["deleted"].append(name)
    try:
        make_coach()._upload_media(VIDEO, None, False)
        return "ok", calls
    except Exception as exc:  # noqa: BLE001
        return exc, calls


print("\n[1] success paths")
outcome, calls = run([ACTIVE])
check("succeeds first try without retrying", outcome == "ok" and calls["n"] == 1, f"attempts={calls['n']}")

outcome, calls = run([failed(13), failed(13), ACTIVE])
check("INTERNAL (13) is retried and then succeeds", outcome == "ok", str(outcome)[:60])
check("it took three attempts", calls["n"] == 3, f"attempts={calls['n']}")
check("each failed File is cleaned up", len(calls["deleted"]) == 2, str(calls["deleted"]))

for code, label in ((14, "UNAVAILABLE"), (4, "DEADLINE_EXCEEDED"), (8, "RESOURCE_EXHAUSTED")):
    outcome, calls = run([failed(code), ACTIVE])
    check(f"{label} ({code}) is retried", outcome == "ok" and calls["n"] == 2, str(outcome)[:50])

print("\n[2] a bad file is NOT retried")
outcome, calls = run([failed(3, "Unsupported video format.")])
check("INVALID_ARGUMENT (3) raises immediately", isinstance(outcome, RuntimeError))
check("no retry was attempted", calls["n"] == 1, f"attempts={calls['n']}")
check("the reason is preserved", "Unsupported video format" in str(outcome))

outcome, calls = run([failed(9, "precondition")])
check("FAILED_PRECONDITION (9) is not retried", isinstance(outcome, RuntimeError) and calls["n"] == 1)

print("\n[3] the retry budget is bounded")
outcome, calls = run([failed(13)])
check("persistent INTERNAL eventually raises", isinstance(outcome, RuntimeError), str(outcome)[:60])
check(
    "it stops at max_retries + 1 attempts",
    calls["n"] == vf._GEMINI_UPLOAD_MAX_RETRIES + 1,
    f"attempts={calls['n']} budget={vf._GEMINI_UPLOAD_MAX_RETRIES + 1}",
)
check("the final error still carries the full detail", "full File resource" in str(outcome))

print("\n[4] transport errors retry too")


def flaky_transport():
    state = {"n": 0}

    def up(_p, _k):
        state["n"] += 1
        if state["n"] == 1:
            raise ConnectionError("connection reset")
        return ACTIVE

    return up, state


up, state = flaky_transport()
vf._gemini_upload_file = up
vf._gemini_get_file = lambda n, k: ACTIVE
vf._gemini_delete_file_quiet = lambda n, k: None
try:
    make_coach()._upload_media(VIDEO, None, False)
    check("a dropped connection is retried", state["n"] == 2, f"attempts={state['n']}")
except Exception as exc:  # noqa: BLE001
    check("a dropped connection is retried", False, str(exc)[:60])

Path(VIDEO).unlink(missing_ok=True)
print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
    raise SystemExit(1)
print("all checks passed")
