# prior-14k on the full Bench2Drive benchmark (220 routes)

_Generated 2026-08-30 09:50 local._

Checkpoint `pi05_cot_simplified_0823_154520/14000` (the commentary run), frozen, greedy CoT, seed 0, scored over **all 220 Bench2Drive routes** -- unlike `checkpoint_comparison_norm_ll_heavy.md`, these are unbiased benchmark scores.

| Run | DS | SD | SEM | RC | IP | SR (DS==100) | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| base policy (HANDOVER reference) | 64.24 | - | - | - | - | 37.73% | 220 |
| prior-14k, apmq3/apc5 (default cadence) | 70.46 | 31.11 | 2.11 | 85.90 | 0.818 | 44.04% | 218 |
| prior-14k, apmq1/apc1 (re-reason every step) | 70.47 | 31.59 | 2.55 | 85.96 | 0.823 | 44.16% | 154 |
| ll_heavy_unnorm 10000, apmq3/apc5 | 67.57 | 31.50 | 2.13 | 87.92 | 0.767 | 40.64% | 219 |

## apc1 vs apc5, paired on the routes both scored

> Per-route differences, so route difficulty cancels. A `Δ mean` within about 2x`SEM` of zero means the two cadences are indistinguishable.

| n | DS apc1 | DS apc5 | Δ mean | paired SD | paired SEM | mean abs Δ | routes ≥40 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 153 | 70.27 | 72.55 | -2.28 | 29.33 | 2.37 | 17.1 | 30/153 |

Route completion: apc1 85.87 vs apc5 86.21 · infraction penalty: apc1 0.822 vs apc5 0.843.

## Status breakdown — prior-14k, apmq3/apc5 (default cadence)

| Status | Routes |
|---|---:|
| Completed | 162 |
| Failed - TickRuntime | 35 |
| Failed - Agent deviated from the route | 14 |
| Failed | 6 |
| Failed - Agent got blocked | 3 |

## Status breakdown — prior-14k, apmq1/apc1 (re-reason every step)

| Status | Routes |
|---|---:|
| Completed | 109 |
| Failed - TickRuntime | 22 |
| Failed - Agent deviated from the route | 13 |
| Failed | 10 |
| Failed - Agent got blocked | 1 |

## Status breakdown — ll_heavy_unnorm 10000, apmq3/apc5

| Status | Routes |
|---|---:|
| Completed | 170 |
| Failed - TickRuntime | 28 |
| Failed - Agent deviated from the route | 16 |
| Failed - Agent got blocked | 5 |

---

> Aggregates exclude `NoRecord*`/`Timeout*` harness failures. `Failed - TickRuntime` is the 4000-tick (200 s) cap, not a crash, and those routes ARE scored and included.
>
> Regenerate: `.venv/bin/python .run_carla/gen_full_report.py`
