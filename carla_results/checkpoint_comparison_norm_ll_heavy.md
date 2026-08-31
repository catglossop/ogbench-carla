# SteerVLA `norm_ll_heavy` checkpoint comparison — Bench2Drive 20-route subset

_Generated 2026-08-30 09:50 local._

Checkpoints under test — `ll_heavy_bs1152`, the first family trained with `skip_norm_stats=False` (predicts in quantile-normalized action space; OpenPI `Unnormalize` runs at inference, auto-detected from the checkpoint's `norm_stats.json`):

```
gs://cat-logs/pi05_steervla_cot_simplified_reasoning_norm_ll_heavy
  /ll_heavy_bs1152/ll_heavy_bs1152_20260826_185850/{6000,8000,10000,16000}
```

Protocol is identical to the earlier sweep (HANDOVER_ckpt_comparison.md): frozen policy, `actions_per_model_query=3`, `actions_per_cot=5`, greedy CoT, same 20 routes, same seed. The only config differences are the ones this checkpoint family requires — `actor_config=pi05_steervla_cot_simplified_reasoning_norm_ll_heavy` and `proprio_norm=False` (raw m/s and degrees, which is what its state norm stats were computed on).

`base-fam *` columns are the base policy's own training run (`pi05_steervla_cot_simplified_reasoning_no_ego_history` v1 @ `20260718_201640`) scored under this harness. **`base-fam 6000` is the base policy itself** -- a like-for-like re-run of the `base @6000` row, so the gap between those two lines measures run-to-run CARLA variance, not a policy difference. It uses the pre-norm-stats config (`actor_config=...no_ego_history`, `proprio_norm=True`, no `Unnormalize`).

`prior *` columns are the earlier `pi05_cot_simplified_0823_154520` family, included for context.

## Progress

| Checkpoint | Scored | Harness failures |
|---|---:|---:|
| ll_heavy 6000 | 20/20 | 0 |
| ll_heavy 8000 | 20/20 | 0 |
| ll_heavy 10000 | 20/20 | 0 |
| ll_heavy 16000 | 20/20 | 0 |
| ll_heavy 10000 unnorm | 20/20 | 0 |
| base-fam 6000 (re-run) | 20/20 | 0 |
| base-fam 8000 | 20/20 | 0 |
| base-fam 10000 | 20/20 | 0 |
| prior 4000 | 20/20 | 0 |
| prior 6000 | 20/20 | 0 |
| prior 8000 | 20/20 | 0 |
| prior 10000 | 20/20 | 0 |
| prior 12000 | 20/20 | 0 |
| prior 14000 | 20/20 | 0 |
| base-fam 10000 (rep2) | 20/20 | 0 |
| prior 10000 (rep2) | 20/20 | 0 |
| prior 14000 (rep2) | 20/20 | 0 |

## Summary — all 20 routes

> Selection-biased (10 base-fail / 10 base-pass); not an overall Bench2Drive score.

> `SD` is the sample standard deviation of Driving Score **across the routes** in this subset; `SEM = SD/sqrt(n)` is the standard error on the mean DS in the same row. SD is dominated by how much the routes differ in difficulty, so it is large by construction and is *not* a run-to-run noise estimate — compare SD between columns, not against zero. SEM is the one to use when asking whether two columns differ: a gap smaller than roughly the two SEMs combined is not resolvable from these runs.

| Checkpoint | DS | SD | SEM | RC | IP | SR (DS==100) | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| base @6000 | 86.03 | 25.37 | 5.67 | - | - | 50.00% | 20 |
| ll_heavy 6000 | 69.88 | 30.44 | 6.81 | 95.09 | 0.727 | 40.00% | 20 |
| ll_heavy 8000 | 71.66 | 30.91 | 6.91 | 92.98 | 0.753 | 45.00% | 20 |
| ll_heavy 10000 | 65.52 | 35.41 | 7.92 | 97.87 | 0.668 | 45.00% | 20 |
| ll_heavy 16000 | 65.77 | 34.51 | 7.72 | 96.96 | 0.672 | 45.00% | 20 |
| ll_heavy 10000 unnorm | 81.74 | 29.69 | 6.64 | 94.09 | 0.868 | 65.00% | 20 |
| base-fam 6000 (re-run) | 68.54 | 30.34 | 6.78 | 92.59 | 0.737 | 25.00% | 20 |
| base-fam 8000 | 68.79 | 36.81 | 8.23 | 97.03 | 0.718 | 45.00% | 20 |
| base-fam 10000 | 78.54 | 29.88 | 6.68 | 97.16 | 0.805 | 50.00% | 20 |
| prior 4000 | 72.69 | 28.71 | 6.42 | 89.88 | 0.802 | 35.00% | 20 |
| prior 6000 | 71.10 | 28.58 | 6.39 | 92.38 | 0.779 | 35.00% | 20 |
| prior 8000 | 75.49 | 26.40 | 5.90 | 93.09 | 0.824 | 40.00% | 20 |
| prior 10000 | 79.91 | 28.60 | 6.39 | 96.10 | 0.819 | 50.00% | 20 |
| prior 12000 | 78.38 | 29.73 | 6.65 | 97.11 | 0.802 | 50.00% | 20 |
| prior 14000 | 78.61 | 31.19 | 6.97 | 94.83 | 0.810 | 55.00% | 20 |
| base-fam 10000 (rep2) | 74.92 | 28.50 | 6.37 | 95.88 | 0.782 | 30.00% | 20 |
| prior 10000 (rep2) | 73.68 | 31.04 | 6.94 | 95.35 | 0.759 | 45.00% | 20 |
| prior 14000 (rep2) | 81.47 | 21.89 | 4.89 | 95.67 | 0.858 | 45.00% | 20 |

## Base policy FAILED these 10 — is the checkpoint better?

> `SD` is the sample standard deviation of Driving Score **across the routes** in this subset; `SEM = SD/sqrt(n)` is the standard error on the mean DS in the same row. SD is dominated by how much the routes differ in difficulty, so it is large by construction and is *not* a run-to-run noise estimate — compare SD between columns, not against zero. SEM is the one to use when asking whether two columns differ: a gap smaller than roughly the two SEMs combined is not resolvable from these runs.

| Checkpoint | DS | SD | SEM | RC | IP | SR (DS==100) | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| base @6000 | 72.07 | 30.42 | 9.62 | - | - | 0.00% | 10 |
| ll_heavy 6000 | 63.76 | 33.96 | 10.74 | 90.18 | 0.693 | 30.00% | 10 |
| ll_heavy 8000 | 68.38 | 34.76 | 10.99 | 85.97 | 0.757 | 40.00% | 10 |
| ll_heavy 10000 | 57.43 | 39.35 | 12.44 | 95.73 | 0.600 | 40.00% | 10 |
| ll_heavy 16000 | 61.96 | 36.20 | 11.45 | 96.31 | 0.642 | 40.00% | 10 |
| ll_heavy 10000 unnorm | 71.48 | 36.65 | 11.59 | 88.19 | 0.816 | 50.00% | 10 |
| base-fam 6000 (re-run) | 68.45 | 33.44 | 10.57 | 89.65 | 0.759 | 10.00% | 10 |
| base-fam 8000 | 64.52 | 37.83 | 11.96 | 94.06 | 0.705 | 30.00% | 10 |
| base-fam 10000 | 79.33 | 30.66 | 9.70 | 94.32 | 0.833 | 40.00% | 10 |
| prior 4000 | 76.37 | 28.96 | 9.16 | 89.51 | 0.832 | 30.00% | 10 |
| prior 6000 | 73.45 | 29.21 | 9.24 | 89.15 | 0.826 | 30.00% | 10 |
| prior 8000 | 78.12 | 23.71 | 7.50 | 92.52 | 0.856 | 30.00% | 10 |
| prior 10000 | 77.91 | 33.69 | 10.65 | 92.19 | 0.819 | 40.00% | 10 |
| prior 12000 | 76.36 | 31.72 | 10.03 | 94.22 | 0.800 | 40.00% | 10 |
| prior 14000 | 72.65 | 35.62 | 11.27 | 89.65 | 0.775 | 40.00% | 10 |
| base-fam 10000 (rep2) | 73.57 | 31.22 | 9.87 | 94.67 | 0.772 | 10.00% | 10 |
| prior 10000 (rep2) | 79.66 | 32.17 | 10.17 | 91.78 | 0.837 | 50.00% | 10 |
| prior 14000 (rep2) | 84.34 | 20.34 | 6.43 | 91.34 | 0.930 | 40.00% | 10 |

## Base policy PASSED these 10 — regression check

> `SD` is the sample standard deviation of Driving Score **across the routes** in this subset; `SEM = SD/sqrt(n)` is the standard error on the mean DS in the same row. SD is dominated by how much the routes differ in difficulty, so it is large by construction and is *not* a run-to-run noise estimate — compare SD between columns, not against zero. SEM is the one to use when asking whether two columns differ: a gap smaller than roughly the two SEMs combined is not resolvable from these runs.

| Checkpoint | DS | SD | SEM | RC | IP | SR (DS==100) | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| base @6000 | 100.00 | 0.00 | 0.00 | - | - | 100.00% | 10 |
| ll_heavy 6000 | 76.00 | 26.84 | 8.49 | 100.00 | 0.760 | 50.00% | 10 |
| ll_heavy 8000 | 74.94 | 28.02 | 8.86 | 100.00 | 0.749 | 50.00% | 10 |
| ll_heavy 10000 | 73.60 | 30.89 | 9.77 | 100.00 | 0.736 | 50.00% | 10 |
| ll_heavy 16000 | 69.57 | 34.22 | 10.82 | 97.60 | 0.702 | 50.00% | 10 |
| ll_heavy 10000 unnorm | 92.00 | 16.87 | 5.33 | 100.00 | 0.920 | 80.00% | 10 |
| base-fam 6000 (re-run) | 68.63 | 28.72 | 9.08 | 95.52 | 0.715 | 40.00% | 10 |
| base-fam 8000 | 73.06 | 37.27 | 11.79 | 100.00 | 0.731 | 60.00% | 10 |
| base-fam 10000 | 77.76 | 30.73 | 9.72 | 100.00 | 0.778 | 60.00% | 10 |
| prior 4000 | 69.02 | 29.51 | 9.33 | 90.25 | 0.772 | 40.00% | 10 |
| prior 6000 | 68.75 | 29.32 | 9.27 | 95.62 | 0.731 | 40.00% | 10 |
| prior 8000 | 72.86 | 29.90 | 9.45 | 93.66 | 0.792 | 50.00% | 10 |
| prior 10000 | 81.90 | 24.14 | 7.63 | 100.00 | 0.819 | 60.00% | 10 |
| prior 12000 | 80.40 | 29.15 | 9.22 | 100.00 | 0.804 | 60.00% | 10 |
| prior 14000 | 84.56 | 26.56 | 8.40 | 100.00 | 0.846 | 70.00% | 10 |
| base-fam 10000 (rep2) | 76.28 | 27.12 | 8.58 | 97.08 | 0.792 | 50.00% | 10 |
| prior 10000 (rep2) | 67.69 | 30.32 | 9.59 | 98.92 | 0.680 | 40.00% | 10 |
| prior 14000 (rep2) | 78.60 | 24.07 | 7.61 | 100.00 | 0.786 | 50.00% | 10 |

## Run-to-run repeatability (independent repeats)

> Each repeat re-scores the **same checkpoint** with the same config, the same 20 routes and the same `--seed 0`. The only thing that differs is CARLA's own nondeterminism, so these deltas are a direct estimate of the noise floor on this subset. The paired statistics below are computed per route and then averaged, which cancels route difficulty entirely -- a far tighter test than comparing the two means.

| Checkpoint | DS orig | DS rep2 | Δ mean | paired SD | paired SEM | mean abs Δ | routes ≥40 | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base-fam 10000 | 78.54 | 74.92 | -3.62 | 15.22 | 3.40 | 8.3 | 0/20 | 20 |
| prior 10000 | 79.91 | 73.68 | -6.23 | 20.46 | 4.57 | 9.0 | 2/20 | 20 |
| prior 14000 | 78.61 | 81.47 | +2.86 | 30.08 | 6.73 | 20.3 | 6/20 | 20 |

`Δ mean` is the mean per-route change (repeat minus original); `paired SEM` is the standard error on it. A `Δ mean` within about 2×`paired SEM` of zero means the two runs are consistent. `mean abs Δ` and `routes ≥40` show how much individual routes move even when the means agree -- the quantity that limits what a single 20-route run can resolve.

## Per-route Driving Score

> Split by family so the tables stay readable; `base` repeats in each as the common reference. Column headers are training steps, `·r2` marks an independent repeat run.

### ll_heavy (`ll_heavy_bs1152`; `unnorm` = the skip_norm_stats=True twin)

| Route | base | 6000 | 8000 | 10000 | 16000 | 10000·unnorm |
|---|---:|---:|---:|---:|---:|---:|
| parking-crossing-pedestrian-005 | 58.46 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| invading-turn-002 | 96.43 | 60.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| vehicle-opens-door-two-ways-001 | 94.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| accident-two-ways-004 | 59.96 | 34.98 | 35.56 | 34.40 | 37.89 | 34.98 |
| vehicle-turning-route-pedestrian-005 | 10.65 | 12.96 | 14.03 | 3.20 | 8.88 | 25.00 |
| t_-junction-002 | 87.58 | 48.11 | 47.29 | 19.62 | 70.00 | 24.93 |
| hazard-at-side-lane-004 | 85.20 | 60.00 | 60.00 | 36.00 | 21.60 | 100.00 |
| non-signalized-junction-right-turn-003 | 33.83 | 23.76 | 29.70 | 60.00 | 39.00 | 31.20 |
| merger-into-slow-traffic-004 | 98.22 | 97.77 | 100.00 | 21.12 | 42.24 | 98.66 |
| interurban-actor-flow-004 | 96.33 | 100.00 | 97.24 | 100.00 | 100.00 | 100.00 |
| blocked-intersection-005 | 100.00 | 65.00 | 33.08 | 60.00 | 55.25 | 100.00 |
| construction-obstacle-003 | 100.00 | 39.00 | 36.00 | 14.04 | 23.40 | 60.00 |
| control-loss-001 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| crossing-bicycle-flow-001 | 100.00 | 100.00 | 60.32 | 42.00 | 21.05 | 100.00 |
| dynamic-object-crossing-001 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| hard-break-route-003 | 100.00 | 60.00 | 100.00 | 60.00 | 60.00 | 60.00 |
| hazard-at-side-lane-two-ways-001 | 100.00 | 60.00 | 60.00 | 100.00 | 100.00 | 100.00 |
| highway-cut-in-001 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| highway-exit-004 | 100.00 | 36.00 | 60.00 | 60.00 | 36.00 | 100.00 |
| interurban-advanced-actor-flow-001 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

### base-fam (`..._no_ego_history` v1 -- the base policy's own run)

| Route | base | 6000 (re-run) | 8000 | 10000 | 10000·r2 |
|---|---:|---:|---:|---:|---:|
| parking-crossing-pedestrian-005 | 58.46 | 96.58 | 100.00 | 100.00 | 100.00 |
| invading-turn-002 | 96.43 | 94.64 | 100.00 | 100.00 | 96.43 |
| vehicle-opens-door-two-ways-001 | 94.00 | 100.00 | 100.00 | 100.00 | 97.00 |
| accident-two-ways-004 | 59.96 | 45.70 | 44.73 | 34.40 | 34.40 |
| vehicle-turning-route-pedestrian-005 | 10.65 | 36.00 | 17.50 | 25.00 | 10.56 |
| t_-junction-002 | 87.58 | 70.00 | 39.80 | 100.00 | 62.29 |
| hazard-at-side-lane-004 | 85.20 | 36.00 | 36.00 | 90.41 | 60.00 |
| non-signalized-junction-right-turn-003 | 33.83 | 11.00 | 11.23 | 48.02 | 80.00 |
| merger-into-slow-traffic-004 | 98.22 | 98.22 | 98.66 | 98.22 | 98.66 |
| interurban-actor-flow-004 | 96.33 | 96.33 | 97.24 | 97.24 | 96.33 |
| blocked-intersection-005 | 100.00 | 59.19 | 42.00 | 100.00 | 100.00 |
| construction-obstacle-003 | 100.00 | 36.00 | 21.60 | 60.00 | 60.00 |
| control-loss-001 | 100.00 | 35.13 | 100.00 | 100.00 | 70.75 |
| crossing-bicycle-flow-001 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| dynamic-object-crossing-001 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| hard-break-route-003 | 100.00 | 100.00 | 100.00 | 60.00 | 60.00 |
| hazard-at-side-lane-two-ways-001 | 100.00 | 60.00 | 60.00 | 36.00 | 36.00 |
| highway-cut-in-001 | 100.00 | 60.00 | 100.00 | 100.00 | 100.00 |
| highway-exit-004 | 100.00 | 36.00 | 7.01 | 21.60 | 36.00 |
| interurban-advanced-actor-flow-001 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

### prior (`pi05_cot_simplified_0823_154520` = the commentary run)

| Route | base | 4000 | 6000 | 8000 | 10000 | 12000 | 14000 | 10000·r2 | 14000·r2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parking-crossing-pedestrian-005 | 58.46 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 50.00 | 100.00 | 100.00 |
| invading-turn-002 | 96.43 | 100.00 | 100.00 | 100.00 | 100.00 | 60.00 | 100.00 | 100.00 | 100.00 |
| vehicle-opens-door-two-ways-001 | 94.00 | 91.01 | 100.00 | 100.00 | 91.01 | 100.00 | 100.00 | 100.00 | 100.00 |
| accident-two-ways-004 | 59.96 | 36.73 | 34.40 | 57.33 | 19.24 | 28.58 | 39.05 | 20.64 | 68.00 |
| vehicle-turning-route-pedestrian-005 | 10.65 | 100.00 | 38.23 | 36.00 | 100.00 | 100.00 | 12.78 | 100.00 | 48.54 |
| t_-junction-002 | 87.58 | 55.99 | 70.00 | 73.28 | 50.82 | 60.00 | 100.00 | 41.49 | 70.00 |
| hazard-at-side-lane-004 | 85.20 | 60.00 | 60.00 | 60.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| non-signalized-junction-right-turn-003 | 33.83 | 25.88 | 36.00 | 60.00 | 22.14 | 20.44 | 28.80 | 39.04 | 60.00 |
| merger-into-slow-traffic-004 | 98.22 | 97.77 | 98.66 | 98.22 | 98.66 | 98.22 | 98.66 | 98.22 | 98.66 |
| interurban-actor-flow-004 | 96.33 | 96.33 | 97.24 | 96.33 | 97.24 | 96.33 | 97.24 | 97.24 | 98.16 |
| blocked-intersection-005 | 100.00 | 32.42 | 53.79 | 36.61 | 100.00 | 100.00 | 100.00 | 21.91 | 100.00 |
| construction-obstacle-003 | 100.00 | 21.60 | 21.60 | 36.00 | 39.00 | 14.04 | 100.00 | 39.00 | 36.00 |
| control-loss-001 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| crossing-bicycle-flow-001 | 100.00 | 56.16 | 56.16 | 100.00 | 100.00 | 70.00 | 25.64 | 100.00 | 70.00 |
| dynamic-object-crossing-001 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| hard-break-route-003 | 100.00 | 60.00 | 100.00 | 100.00 | 100.00 | 60.00 | 100.00 | 60.00 | 60.00 |
| hazard-at-side-lane-two-ways-001 | 100.00 | 60.00 | 60.00 | 60.00 | 60.00 | 100.00 | 100.00 | 60.00 | 60.00 |
| highway-cut-in-001 | 100.00 | 60.00 | 60.00 | 60.00 | 60.00 | 100.00 | 60.00 | 60.00 | 60.00 |
| highway-exit-004 | 100.00 | 100.00 | 36.00 | 36.00 | 60.00 | 60.00 | 60.00 | 36.00 | 100.00 |
| interurban-advanced-actor-flow-001 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

## Per-route status (runs scored here)

> `OK` = Completed · `tick-cap` = hit the harness 4000-tick (200 s) cap (still scored) · `deviate` = left the route · `blocked` = AgentBlockedTest · `timeout` = route timeout · `harness` = NoRecord/Timeout, excluded from aggregates · `-` = not scored.

### ll_heavy (`ll_heavy_bs1152`; `unnorm` = the skip_norm_stats=True twin)

| Route | 6000 | 8000 | 10000 | 16000 | 10000·unnorm |
|---|---|---|---|---|---|
| parking-crossing-pedestrian-005 | OK | OK | OK | OK | OK |
| invading-turn-002 | OK | OK | OK | OK | OK |
| vehicle-opens-door-two-ways-001 | OK | OK | OK | OK | OK |
| accident-two-ways-004 | tick-cap | tick-cap | tick-cap | tick-cap | tick-cap |
| vehicle-turning-route-pedestrian-005 | OK | blocked | OK | OK | OK |
| t_-junction-002 | OK | OK | OK | OK | tick-cap |
| hazard-at-side-lane-004 | OK | OK | OK | OK | OK |
| non-signalized-junction-right-turn-003 | tick-cap | tick-cap | OK | OK | OK |
| merger-into-slow-traffic-004 | deviate | OK | OK | OK | deviate |
| interurban-actor-flow-004 | OK | deviate | OK | OK | OK |
| blocked-intersection-005 | OK | OK | OK | OK | OK |
| construction-obstacle-003 | OK | OK | OK | OK | OK |
| control-loss-001 | OK | OK | OK | OK | OK |
| crossing-bicycle-flow-001 | OK | OK | OK | blocked | OK |
| dynamic-object-crossing-001 | OK | OK | OK | OK | OK |
| hard-break-route-003 | OK | OK | OK | OK | OK |
| hazard-at-side-lane-two-ways-001 | OK | OK | OK | OK | OK |
| highway-cut-in-001 | OK | OK | OK | OK | OK |
| highway-exit-004 | OK | OK | OK | OK | OK |
| interurban-advanced-actor-flow-001 | OK | OK | OK | OK | OK |

### base-fam (`..._no_ego_history` v1 -- the base policy's own run)

| Route | 6000 (re-run) | 8000 | 10000 | 10000·r2 |
|---|---|---|---|---|
| parking-crossing-pedestrian-005 | tick-cap | OK | OK | OK |
| invading-turn-002 | tick-cap | OK | OK | tick-cap |
| vehicle-opens-door-two-ways-001 | OK | OK | OK | tick-cap |
| accident-two-ways-004 | tick-cap | tick-cap | tick-cap | tick-cap |
| vehicle-turning-route-pedestrian-005 | OK | OK | OK | OK |
| t_-junction-002 | OK | OK | OK | OK |
| hazard-at-side-lane-004 | OK | OK | tick-cap | OK |
| non-signalized-junction-right-turn-003 | tick-cap | OK | OK | OK |
| merger-into-slow-traffic-004 | deviate | deviate | deviate | deviate |
| interurban-actor-flow-004 | deviate | deviate | deviate | deviate |
| blocked-intersection-005 | OK | OK | OK | OK |
| construction-obstacle-003 | OK | OK | OK | OK |
| control-loss-001 | blocked | OK | OK | tick-cap |
| crossing-bicycle-flow-001 | OK | OK | OK | OK |
| dynamic-object-crossing-001 | OK | OK | OK | OK |
| hard-break-route-003 | OK | OK | OK | OK |
| hazard-at-side-lane-two-ways-001 | OK | OK | OK | OK |
| highway-cut-in-001 | OK | OK | OK | OK |
| highway-exit-004 | OK | OK | OK | OK |
| interurban-advanced-actor-flow-001 | OK | OK | OK | OK |

### prior (`pi05_cot_simplified_0823_154520` = the commentary run)

| Route | 4000 | 6000 | 8000 | 10000 | 12000 | 14000 | 10000·r2 | 14000·r2 |
|---|---|---|---|---|---|---|---|---|
| parking-crossing-pedestrian-005 | OK | OK | OK | OK | OK | OK | OK | OK |
| invading-turn-002 | OK | OK | OK | OK | OK | OK | OK | OK |
| vehicle-opens-door-two-ways-001 | OK | OK | OK | OK | OK | OK | OK | OK |
| accident-two-ways-004 | tick-cap | tick-cap | tick-cap | tick-cap | tick-cap | tick-cap | tick-cap | tick-cap |
| vehicle-turning-route-pedestrian-005 | OK | tick-cap | OK | OK | OK | tick-cap | OK | tick-cap |
| t_-junction-002 | OK | OK | deviate | deviate | OK | OK | OK | OK |
| hazard-at-side-lane-004 | OK | OK | OK | OK | OK | OK | OK | OK |
| non-signalized-junction-right-turn-003 | tick-cap | OK | OK | OK | OK | OK | tick-cap | OK |
| merger-into-slow-traffic-004 | deviate | deviate | deviate | deviate | deviate | deviate | deviate | deviate |
| interurban-actor-flow-004 | deviate | deviate | deviate | deviate | deviate | deviate | deviate | deviate |
| blocked-intersection-005 | tick-cap | OK | tick-cap | OK | OK | OK | tick-cap | OK |
| construction-obstacle-003 | OK | OK | OK | OK | OK | OK | OK | OK |
| control-loss-001 | OK | OK | OK | OK | OK | OK | OK | OK |
| crossing-bicycle-flow-001 | deviate | deviate | OK | OK | OK | OK | OK | OK |
| dynamic-object-crossing-001 | OK | OK | OK | OK | OK | OK | OK | OK |
| hard-break-route-003 | OK | OK | OK | OK | OK | OK | OK | OK |
| hazard-at-side-lane-two-ways-001 | OK | OK | OK | OK | OK | OK | OK | OK |
| highway-cut-in-001 | OK | OK | OK | OK | OK | OK | OK | OK |
| highway-exit-004 | OK | OK | OK | OK | OK | OK | OK | OK |
| interurban-advanced-actor-flow-001 | OK | OK | OK | OK | OK | OK | OK | OK |

---

> **Selection bias.** These 20 routes were deliberately picked so the base policy fails 10 and passes 10. Across all 220 Bench2Drive routes the base policy scores DS 64.24 / SR 37.73%. These columns are valid only as per-route comparators, not as absolute Bench2Drive scores.
>
> Aggregates exclude `NoRecord*`/`Timeout*` harness failures (HANDOVER §10.1). Success = Driving Score == 100 (HANDOVER §9). `Failed - TickRuntime` is the harness's 4000-tick (200 s) cap, not a crash, and its routes ARE scored and included.
>
> Regenerate: `.venv/bin/python .run_carla/gen_llheavy_report.py`
