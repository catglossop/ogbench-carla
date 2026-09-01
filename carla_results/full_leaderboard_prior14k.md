# prior-14k on the full Bench2Drive benchmark (220 routes)

_Generated 2026-09-01 10:02 local._

Checkpoint `pi05_cot_simplified_0823_154520/14000` (the commentary run), frozen, greedy CoT, seed 0, scored over **all 220 Bench2Drive routes** -- unlike `checkpoint_comparison_norm_ll_heavy.md`, these are unbiased benchmark scores.

| Run | DS | SD | SEM | RC | IP | SR (DS==100) | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| base policy (HANDOVER reference) | 64.24 | - | - | - | - | 37.73% | 220 |
| prior-14k, seed 0 (apmq3/apc5) | 70.46 | 31.11 | 2.11 | 85.90 | 0.818 | 44.04% | 218 |
| prior-14k, seed 1 | 74.20 | 30.66 | 2.08 | 89.40 | 0.828 | 50.00% | 218 |
| prior-14k, seed 2 | 70.86 | 31.54 | 2.14 | 86.00 | 0.823 | 44.95% | 218 |
| prior-14k, apmq1/apc1 (re-reason every step) | 70.47 | 31.59 | 2.55 | 85.96 | 0.823 | 44.16% | 154 |
| ll_heavy_unnorm 10000, apmq3/apc5 | 67.57 | 31.50 | 2.13 | 87.92 | 0.767 | 40.64% | 219 |

## apc1 vs apc5, paired on the routes both scored

> Per-route differences, so route difficulty cancels. A `Δ mean` within about 2x`SEM` of zero means the two cadences are indistinguishable.

| n | DS apc1 | DS apc5 | Δ mean | paired SD | paired SEM | mean abs Δ | routes ≥40 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 153 | 70.27 | 72.55 | -2.28 | 29.33 | 2.37 | 17.1 | 30/153 |

Route completion: apc1 85.87 vs apc5 86.21 · infraction penalty: apc1 0.822 vs apc5 0.843.

## Seed sweep — the benchmark's own noise floor

> Same checkpoint, same config, same routes; only `env.reset(seed)` differs. The spread here bounds what a single-run comparison between two checkpoints can resolve.

Means: seed 0 **70.46** · seed 1 **74.20** · seed 2 **70.86** — **spread 3.74 DS**

| Pair | Δ mean | paired SEM | sigma | identical routes | mean abs Δ | n |
|---|---:|---:|---:|---:|---:|---:|
| seed 1 − seed 0 | +3.62 | 1.96 | 1.8 | 114/217 | 16.1 | 217 |
| seed 2 − seed 0 | +0.44 | 1.81 | 0.2 | 109/217 | 15.3 | 217 |
| seed 2 − seed 1 | -3.34 | 1.92 | 1.7 | 107/218 | 16.0 | 218 |

Roughly half of Bench2Drive is fully deterministic across seeds; all of the variance concentrates in the remainder, and it moves route **completion** far more than infraction penalty.

## Status breakdown — prior-14k, seed 0 (apmq3/apc5)

| Status | Routes |
|---|---:|
| Completed | 162 |
| Failed - TickRuntime | 35 |
| Failed - Agent deviated from the route | 14 |
| Failed | 6 |
| Failed - Agent got blocked | 3 |

## Status breakdown — prior-14k, seed 1

| Status | Routes |
|---|---:|
| Completed | 173 |
| Failed - TickRuntime | 28 |
| Failed - Agent deviated from the route | 13 |
| Failed - Agent got blocked | 4 |

## Status breakdown — prior-14k, seed 2

| Status | Routes |
|---|---:|
| Completed | 161 |
| Failed - TickRuntime | 39 |
| Failed - Agent deviated from the route | 14 |
| Failed | 2 |
| Failed - Agent got blocked | 2 |

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
