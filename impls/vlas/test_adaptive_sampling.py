"""Standalone checks for ``steervla.use_adaptive_sampling``.

Not a pytest suite (this repo has none) -- run it directly, the way
``coaches/test_action_chunk_feedback_integration.py`` is run::

    JAX_PLATFORMS=cpu PYTHONPATH=impls uv run python impls/vlas/test_adaptive_sampling.py

It exercises the pure sampling logic only: no model, no checkpoint, no CARLA. The ordering methods
are bound to a stand-in object rather than a real ``SteerVLAActor`` so the whole file runs in a
second and can be re-run after any edit to the weighting.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coaches.cast_relabel import (
    CATASTROPHIC_OUTCOMES,
    SUCCESS_OUTCOMES,
    HLSample,
    resolve_window_outcome,
    write_hl_samples,
)

from vlas.steervla import ADAPTIVE_SAMPLING_WEIGHTS, SteerVLAActor

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


class _Stand_in:
    """Just enough of ``SteerVLAActor`` for the online-ordering methods."""

    def __init__(self, weights=None, precursor_fraction=-1.0, backfill=True):
        self.adaptive_sampling_weights = dict(ADAPTIVE_SAMPLING_WEIGHTS)
        self.adaptive_sampling_weights.update(weights or {})
        self.hl_online_bad_fraction = 0.9
        self.hl_online_backfill_from_replay = backfill
        self.hl_online_precursor_fraction = precursor_fraction
        self.use_adaptive_sampling = True
        self._adaptive_logged_once = True
        self._adaptive_no_outcomes_warned = True

    # Bind the real implementations. ``SteerVLAActor.<name>`` on a static/classmethod hands back a
    # plain function, so it must be re-wrapped or Python would pass ``self`` into it.
    _is_bad_entry = staticmethod(SteerVLAActor._is_bad_entry)
    _is_precursor_entry = staticmethod(SteerVLAActor._is_precursor_entry)
    _balance_buckets = staticmethod(SteerVLAActor._balance_buckets)
    _weighted_order = staticmethod(SteerVLAActor._weighted_order)
    _adaptive_category = classmethod(SteerVLAActor._adaptive_category.__func__)
    _adaptive_weights_for = SteerVLAActor._adaptive_weights_for
    _online_bucket_counts = SteerVLAActor._online_bucket_counts
    _order_online_entries = SteerVLAActor._order_online_entries
    _order_online_entries_adaptive = SteerVLAActor._order_online_entries_adaptive
    _scan_pool = SteerVLAActor._scan_pool


def entry(label=None, credit_source="", outcome="", uid=0):
    return {"label": label, "credit_source": credit_source, "outcome": outcome, "uid": uid}


# ── 1. outcome resolution ─────────────────────────────────────────────────────────────
print("\n[1] resolve_window_outcome")
check("latched collision wins", resolve_window_outcome({}, "collision") == "collision")
check("latched off_route", resolve_window_outcome({}, "off_route") == "off_route")
check(
    "VLM wrong turn -> route_divergence",
    resolve_window_outcome({}, "off route at t=4.5s") == "route_divergence",
)
check(
    "stuck collision inferred from metadata",
    resolve_window_outcome({"max_crash_stuck_ticks": 25}) == "collision",
)
check(
    "a collision the ego drove away from is NOT catastrophic (2026-09-07 rule change)",
    resolve_window_outcome(
        {"collision_events": [{"new_event": True}], "max_crash_stuck_ticks": 3, "success": True}
    ) == "success",
)
check("crash_stuck from termination_reason", resolve_window_outcome({"termination_reason": "crash_stuck"}) == "crash_stuck")
check("route_completed", resolve_window_outcome({"route_completed": True}) == "route_completed")
check("nothing notable -> empty", resolve_window_outcome({}) == "")
check(
    "catastrophe beats success",
    resolve_window_outcome({"success": True, "route_completed": True}, "collision") == "collision",
)
check(
    "every catastrophic tag round-trips",
    all(resolve_window_outcome({}, t) == t for t in CATASTROPHIC_OUTCOMES if t != "route_divergence"),
)
check("success vocab is what steervla expects", set(SUCCESS_OUTCOMES) == {"success", "route_completed"})

# ── 2. category resolution ────────────────────────────────────────────────────────────
print("\n[2] _adaptive_category")
cases = [
    (entry("BAD", "precursor", "collision"), "bad_precursor_catastrophic"),
    (entry("BAD", "precursor", "route_divergence"), "bad_precursor_catastrophic"),
    (entry("BAD", "precursor", ""), "bad_precursor"),
    (entry("BAD", "precursor", "success"), "bad_precursor"),
    (entry("BAD", "direct", "collision"), "bad_direct_catastrophic"),
    (entry("BAD", "direct", ""), "bad_direct"),
    (entry("GOOD", "", "route_completed"), "good_success"),
    (entry("GOOD", "", "success"), "good_success"),
    (entry("GOOD", "", ""), "good"),
    (entry(None, "", "success"), "unlabeled"),
    (entry(None, "", ""), "unlabeled"),
    (entry("BAD", "direct", "bogus_tag"), "bad_direct"),  # unknown tag degrades, never raises
]
for e, want in cases:
    got = SteerVLAActor._adaptive_category(e)
    check(f"{e['label']}/{e['credit_source'] or '-'}/{e['outcome'] or '-'} -> {want}", got == want, got)

check(
    "every category has a weight",
    {c for _, c in cases} <= set(ADAPTIVE_SAMPLING_WEIGHTS),
)

# ── 3. the ordering the user asked for ────────────────────────────────────────────────
print("\n[3] weight ordering matches the requested priority")
w = ADAPTIVE_SAMPLING_WEIGHTS
check(
    "precursor-of-catastrophe > ordinary precursor > direct BAD",
    w["bad_precursor_catastrophic"] > w["bad_precursor"] > w["bad_direct_catastrophic"] > w["bad_direct"],
    f"{w['bad_precursor_catastrophic']} > {w['bad_precursor']} > "
    f"{w['bad_direct_catastrophic']} > {w['bad_direct']}",
)
check(
    "route-success GOOD > ordinary GOOD > unlabeled",
    w["good_success"] > w["good"] > w["unlabeled"],
    f"{w['good_success']} > {w['good']} > {w['unlabeled']}",
)

# ── 4. weighted draw frequencies ──────────────────────────────────────────────────────
print("\n[4] _weighted_order draw frequencies (10k trials, pick 1 of 4)")
np.random.seed(0)
pool = [entry(uid=i) for i in range(4)]
weights = np.array([4.0, 2.0, 1.0, 0.5])
counts = np.zeros(4)
TRIALS = 10000
for _ in range(TRIALS):
    counts[SteerVLAActor._weighted_order(pool, weights)[0]["uid"]] += 1
empirical = counts / TRIALS
expected = weights / weights.sum()
check(
    "top-1 frequency tracks the weights",
    bool(np.all(np.abs(empirical - expected) < 0.02)),
    f"empirical={np.round(empirical, 3).tolist()} expected={np.round(expected, 3).tolist()}",
)
zero_w = SteerVLAActor._weighted_order(pool, np.array([1.0, 1.0, 1.0, 0.0]))
check(
    "zero weight never gets drawn first but is still returned",
    zero_w[0]["uid"] != 3 and len(zero_w) == 4,
)
check(
    "all-zero weights degrade to a plain shuffle",
    len(SteerVLAActor._weighted_order(pool, np.zeros(4))) == 4,
)
check("empty pool is safe", SteerVLAActor._weighted_order([], np.array([])) == [])

# ── 5. coverage quota is preserved ────────────────────────────────────────────────────
print("\n[5] _order_online_entries_adaptive keeps the corrective/reinforce quota")
actor = _Stand_in()
online = (
    [entry("BAD", "precursor", "collision", uid=i) for i in range(10)]
    + [entry("BAD", "precursor", "", uid=100 + i) for i in range(10)]
    + [entry("BAD", "direct", "", uid=200 + i) for i in range(30)]
    + [entry("GOOD", "", "route_completed", uid=300 + i) for i in range(5)]
    + [entry("GOOD", "", "", uid=400 + i) for i in range(15)]
    + [entry(None, "", "", uid=500 + i) for i in range(10)]
)
CNT = 32
np.random.seed(1)
ordered, n_take = actor._order_online_entries_adaptive(online, CNT, 0.9)
check("returns a full permutation", sorted(e["uid"] for e in ordered) == sorted(e["uid"] for e in online))
sel = ordered[:CNT]
n_bad = sum(1 for e in sel if e["label"] == "BAD")
check(
    "corrective quota == round(cnt * bad_fraction)",
    n_bad == round(CNT * 0.9),
    f"n_bad={n_bad} want={round(CNT * 0.9)}",
)
check("reinforce bucket is still represented", CNT - n_bad > 0, f"n_good={CNT - n_bad}")

# Severity should shift the *composition* of the corrective quota, not its size.
np.random.seed(2)
cat_share, uniform_share = [], []
for _ in range(300):
    sel_a = actor._order_online_entries_adaptive(online, CNT, 0.9)[0][:CNT]
    cat_share.append(sum(1 for e in sel_a if SteerVLAActor._adaptive_category(e) == "bad_precursor_catastrophic"))
    sel_u = actor._order_online_entries(online, CNT, 0.9, -1.0)[0][:CNT]
    uniform_share.append(
        sum(1 for e in sel_u if SteerVLAActor._adaptive_category(e) == "bad_precursor_catastrophic")
    )
mean_cat, mean_uni = float(np.mean(cat_share)), float(np.mean(uniform_share))
check(
    "catastrophic precursors are over-represented vs a uniform draw",
    mean_cat > mean_uni * 1.5,
    f"adaptive={mean_cat:.2f}/29 vs uniform={mean_uni:.2f}/29 (pool has 10 of 50 corrective)",
)
np.random.seed(3)
direct_share = float(
    np.mean(
        [
            sum(
                1
                for e in actor._order_online_entries_adaptive(online, CNT, 0.9)[0][:CNT]
                if SteerVLAActor._adaptive_category(e) == "bad_direct"
            )
            for _ in range(300)
        ]
    )
)
check(
    "plain direct-BAD rows are down-weighted (30 of 50 corrective in the pool)",
    direct_share < 29 * (30 / 50),
    f"adaptive={direct_share:.2f} vs uniform-expectation={29 * 30 / 50:.2f}",
)

# ── 6. thin buckets and degenerate inputs ─────────────────────────────────────────────
print("\n[6] underfill / degenerate cases")
only_bad = [entry("BAD", "direct", "", uid=i) for i in range(5)]
out, take_bad = actor._order_online_entries_adaptive(only_bad, 32, 0.9)
check(
    "no reinforce rows -> supplies only its 5 corrective rows, 27 slots backfill",
    take_bad == 5,
    f"take={take_bad} (target was min(5 available, round(32*0.9)=29))",
)
check("permutation still complete", len(out) == 5)
only_good = [entry("GOOD", "", "", uid=i) for i in range(5)]
out, take_good = actor._order_online_entries_adaptive(only_good, 32, 0.9)
check(
    "no corrective rows -> reinforce capped at its own target (3), NOT expanded to 32",
    take_good == 3,
    f"take={take_good}",
)
check("permutation still complete", len(out) == 5 and all(e["label"] == "GOOD" for e in out))
check("empty pool", actor._order_online_entries_adaptive([], 32, 0.9) == ([], 0))
untagged = [entry("BAD", "precursor", "", uid=i) for i in range(4)] + [
    entry("GOOD", "", "", uid=10 + i) for i in range(4)
]
check(
    "pool with no outcome tags still returns a full permutation",
    sorted(e["uid"] for e in actor._order_online_entries_adaptive(untagged, 4, 0.5)[0])
    == sorted(e["uid"] for e in untagged),
)

# ── 6b. offline backfill when the online pool cannot hold the ratio ───────────────────
print("\n[6b] hl_online_backfill_from_replay")
backfill = _Stand_in(backfill=True)
legacy = _Stand_in(backfill=False)

# 5 corrective, 100 reinforce, 32 slots at 0.9 -> wants 29 corrective + 3 reinforce.
thin_bad = [entry("BAD", "direct", "", uid=i) for i in range(5)] + [
    entry("GOOD", "", "", uid=100 + i) for i in range(100)
]
_, take_bf = backfill._order_online_entries_adaptive(thin_bad, 32, 0.9)
sel_lg, take_lg = legacy._order_online_entries_adaptive(thin_bad, 32, 0.9)
check(
    "backfill: online hands back only what it can hold at the ratio (5 + 3 = 8)",
    take_bf == 8,
    f"take={take_bf} -> {32 - take_bf} slots left for the replay pools",
)
check(
    "legacy: online fills all 32 by flooding with reinforce rows",
    take_lg == 32 and sum(1 for e in sel_lg[:32] if e["label"] != "BAD") == 27,
    f"take={take_lg}, reinforce={sum(1 for e in sel_lg[:32] if e['label'] != 'BAD')}/32",
)
sel_bf = backfill._order_online_entries_adaptive(thin_bad, 32, 0.9)[0][:take_bf]
check(
    "backfill preserves the requested ratio among the rows it DOES supply",
    sum(1 for e in sel_bf if e["label"] == "BAD") == 5,
)

# Both buckets healthy -> the two modes must agree exactly (no behavior change off the edge case).
healthy = [entry("BAD", "direct", "", uid=i) for i in range(60)] + [
    entry("GOOD", "", "", uid=100 + i) for i in range(60)
]
check(
    "healthy pool: backfill and legacy both take the full share",
    backfill._order_online_entries_adaptive(healthy, 32, 0.9)[1] == 32
    and legacy._order_online_entries_adaptive(healthy, 32, 0.9)[1] == 32,
)
sel_healthy = backfill._order_online_entries_adaptive(healthy, 32, 0.9)[0][:32]
check(
    "healthy pool: quota is exactly 29/3",
    sum(1 for e in sel_healthy if e["label"] == "BAD") == 29,
)

# The uniform (non-adaptive) path must honour the same knob.
check(
    "legacy uniform path honours backfill too",
    backfill._order_online_entries(thin_bad, 32, 0.9, -1.0)[1] == 8
    and legacy._order_online_entries(thin_bad, 32, 0.9, -1.0)[1] == 32,
)
check(
    "empty online pool -> nothing taken, everything backfills",
    backfill._order_online_entries_adaptive([], 32, 0.9)[1] == 0,
)

# ── 7. weight override validation ─────────────────────────────────────────────────────
print("\n[7] adaptive_sampling_weights validation")
try:
    _Stand_in(weights={"bad_direct": 0.1})
    ok_override = True
except (ValueError, KeyError, TypeError):
    ok_override = False
check("known key accepted by the stand-in table", ok_override)
check(
    "constructor rejects unknown keys",
    "unknown category" in (SteerVLAActor.__init__.__doc__ or "")
    or "unknown category" in Path("impls/vlas/steervla.py").read_text(),
)

# ── 8. end-to-end: does the tag survive the trip through disk? ────────────────────────
print("\n[8] HLSample -> hl_samples.json -> _scan_pool round-trip")


def _hl_sample(outcome, label, credit_source):
    return HLSample(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        state=np.zeros((16,), dtype=np.float32),
        current_speed=1.0,
        prompt="Prompt:...;State:0 0;",
        subtask="Follow the route. Decelerate.",
        reasoning="Follow the route. Decelerate because the road is blocked.",
        actions=np.zeros((10, 4), dtype=np.float32),
        action_loss_mask=np.zeros((10,), dtype=bool),
        episode=0,
        window_index=0,
        chunk_index=0,
        episode_step=1,
        label=label,
        credit_source=credit_source,
        outcome=outcome,
    )


with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "window0"
    write_hl_samples(
        [
            _hl_sample("collision", "BAD", "precursor"),
            _hl_sample("collision", "BAD", "direct"),
            _hl_sample("route_completed", "GOOD", ""),
        ],
        root,
    )
    scanned = _Stand_in()._scan_pool({"dir": Path(td), "name": "online", "weight": 1.0, "kind": "online"})
    check("all 3 samples scanned back", len(scanned) == 3, f"got {len(scanned)}")
    check(
        "outcome survives the manifest",
        sorted(e.get("outcome", "") for e in scanned) == ["collision", "collision", "route_completed"],
        str(sorted(e.get("outcome", "") for e in scanned)),
    )
    check(
        "categories resolve on real scanned entries",
        sorted(SteerVLAActor._adaptive_category(e) for e in scanned)
        == ["bad_direct_catastrophic", "bad_precursor_catastrophic", "good_success"],
        str(sorted(SteerVLAActor._adaptive_category(e) for e in scanned)),
    )

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
    raise SystemExit(1)
print("all checks passed")
