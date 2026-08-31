# SteerVLA checkpoint comparison — Bench2Drive 20-route subset

## Progress

| Checkpoint | Scored | Harness failures |
|---|---:|---:|
| 4000 | 20/20 | 0 |
| 6000 | 20/20 | 0 |
| 8000 | 20/20 | 0 |
| 10000 | 20/20 | 0 |
| 12000 | 20/20 | 0 |
| 14000 | 13/20 | 0 |

## Summary — all 20 routes

> Selection-biased (10 base-fail / 10 base-pass); not an overall Bench2Drive score.

| Checkpoint | DS | RC | IP | SR (DS==100) | n |
|---|---:|---:|---:|---:|---:|
| base @6000 | 86.03 | - | - | 50.00% | 20 |
| 4000 | 72.69 | 89.88 | 0.802 | 35.00% | 20 |
| 6000 | 71.10 | 92.38 | 0.779 | 35.00% | 20 |
| 8000 | 75.49 | 93.09 | 0.824 | 40.00% | 20 |
| 10000 | 79.91 | 96.10 | 0.819 | 50.00% | 20 |
| 12000 | 78.38 | 97.11 | 0.802 | 50.00% | 20 |
| 14000 | 78.96 | 92.04 | 0.827 | 53.85% | 13 |

## Base policy FAILED these 10 — is the checkpoint better?

| Checkpoint | DS | RC | IP | SR (DS==100) | n |
|---|---:|---:|---:|---:|---:|
| base @6000 | 72.07 | - | - | 0.00% | 10 |
| 4000 | 76.37 | 89.51 | 0.832 | 30.00% | 10 |
| 6000 | 73.45 | 89.15 | 0.826 | 30.00% | 10 |
| 8000 | 78.12 | 92.52 | 0.856 | 30.00% | 10 |
| 10000 | 77.91 | 92.19 | 0.819 | 40.00% | 10 |
| 12000 | 76.36 | 94.22 | 0.800 | 40.00% | 10 |
| 14000 | 72.65 | 89.65 | 0.775 | 40.00% | 10 |

## Base policy PASSED these 10 — regression check

| Checkpoint | DS | RC | IP | SR (DS==100) | n |
|---|---:|---:|---:|---:|---:|
| base @6000 | 100.00 | - | - | 100.00% | 10 |
| 4000 | 69.02 | 90.25 | 0.772 | 40.00% | 10 |
| 6000 | 68.75 | 95.62 | 0.731 | 40.00% | 10 |
| 8000 | 72.86 | 93.66 | 0.792 | 50.00% | 10 |
| 10000 | 81.90 | 100.00 | 0.819 | 60.00% | 10 |
| 12000 | 80.40 | 100.00 | 0.804 | 60.00% | 10 |
| 14000 | 100.00 | 100.00 | 1.000 | 100.00% | 3 |

## Per-route Driving Score

| Route | base | 4000 | 6000 | 8000 | 10000 | 12000 | 14000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| parking-crossing-pedestrian-005 | 58.46 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 50.00 |
| invading-turn-002 | 96.43 | 100.00 | 100.00 | 100.00 | 100.00 | 60.00 | 100.00 |
| vehicle-opens-door-two-ways-001 | 94.00 | 91.01 | 100.00 | 100.00 | 91.01 | 100.00 | 100.00 |
| accident-two-ways-004 | 59.96 | 36.73 | 34.40 | 57.33 | 19.24 | 28.58 | 39.05 |
| vehicle-turning-route-pedestrian-005 | 10.65 | 100.00 | 38.23 | 36.00 | 100.00 | 100.00 | 12.78 |
| t_-junction-002 | 87.58 | 55.99 | 70.00 | 73.28 | 50.82 | 60.00 | 100.00 |
| hazard-at-side-lane-004 | 85.20 | 60.00 | 60.00 | 60.00 | 100.00 | 100.00 | 100.00 |
| non-signalized-junction-right-turn-003 | 33.83 | 25.88 | 36.00 | 60.00 | 22.14 | 20.44 | 28.80 |
| merger-into-slow-traffic-004 | 98.22 | 97.77 | 98.66 | 98.22 | 98.66 | 98.22 | 98.66 |
| interurban-actor-flow-004 | 96.33 | 96.33 | 97.24 | 96.33 | 97.24 | 96.33 | 97.24 |
| blocked-intersection-005 | 100.00 | 32.42 | 53.79 | 36.61 | 100.00 | 100.00 | 100.00 |
| construction-obstacle-003 | 100.00 | 21.60 | 21.60 | 36.00 | 39.00 | 14.04 | 100.00 |
| control-loss-001 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| crossing-bicycle-flow-001 | 100.00 | 56.16 | 56.16 | 100.00 | 100.00 | 70.00 | - |
| dynamic-object-crossing-001 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | - |
| hard-break-route-003 | 100.00 | 60.00 | 100.00 | 100.00 | 100.00 | 60.00 | - |
| hazard-at-side-lane-two-ways-001 | 100.00 | 60.00 | 60.00 | 60.00 | 60.00 | 100.00 | - |
| highway-cut-in-001 | 100.00 | 60.00 | 60.00 | 60.00 | 60.00 | 100.00 | - |
| highway-exit-004 | 100.00 | 100.00 | 36.00 | 36.00 | 60.00 | 60.00 | - |
| interurban-advanced-actor-flow-001 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | - |

## Per-route status

| Route | 4000 | 6000 | 8000 | 10000 | 12000 | 14000 |
|---|---|---|---|---|---|---|
| parking-crossing-pedestrian-005 | Completed | Completed | Completed | Completed | Completed | Completed |
| invading-turn-002 | Completed | Completed | Completed | Completed | Completed | Completed |
| vehicle-opens-door-two-ways-001 | Completed | Completed | Completed | Completed | Completed | Completed |
| accident-two-ways-004 | Failed - TickRuntime | Failed - TickRuntime | Failed - TickRuntime | Failed - TickRuntime | Failed - TickRuntime | Failed - TickRuntime |
| vehicle-turning-route-pedestrian-005 | Completed | Failed - TickRuntime | Completed | Completed | Completed | Failed - TickRuntime |
| t_-junction-002 | Completed | Completed | Failed - Agent deviated from the route | Failed - Agent deviated from the route | Completed | Completed |
| hazard-at-side-lane-004 | Completed | Completed | Completed | Completed | Completed | Completed |
| non-signalized-junction-right-turn-003 | Failed - TickRuntime | Completed | Completed | Completed | Completed | Completed |
| merger-into-slow-traffic-004 | Failed - Agent deviated from the route | Failed - Agent deviated from the route | Failed - Agent deviated from the route | Failed - Agent deviated from the route | Failed - Agent deviated from the route | Failed - Agent deviated from the route |
| interurban-actor-flow-004 | Failed - Agent deviated from the route | Failed - Agent deviated from the route | Failed - Agent deviated from the route | Failed - Agent deviated from the route | Failed - Agent deviated from the route | Failed - Agent deviated from the route |
| blocked-intersection-005 | Failed - TickRuntime | Completed | Failed - TickRuntime | Completed | Completed | Completed |
| construction-obstacle-003 | Completed | Completed | Completed | Completed | Completed | Completed |
| control-loss-001 | Completed | Completed | Completed | Completed | Completed | Completed |
| crossing-bicycle-flow-001 | Failed - Agent deviated from the route | Failed - Agent deviated from the route | Completed | Completed | Completed | - |
| dynamic-object-crossing-001 | Completed | Completed | Completed | Completed | Completed | - |
| hard-break-route-003 | Completed | Completed | Completed | Completed | Completed | - |
| hazard-at-side-lane-two-ways-001 | Completed | Completed | Completed | Completed | Completed | - |
| highway-cut-in-001 | Completed | Completed | Completed | Completed | Completed | - |
| highway-exit-004 | Completed | Completed | Completed | Completed | Completed | - |
| interurban-advanced-actor-flow-001 | Completed | Completed | Completed | Completed | Completed | - |

---

> **Selection bias.** These 20 routes were deliberately picked so the base policy fails 10 and passes 10. Across all 220 Bench2Drive routes the base policy scores DS 64.24 / SR 37.73%. These columns are valid only as per-route comparators, not as absolute Bench2Drive scores.
>
> Aggregates exclude `NoRecord*`/`Timeout*` harness failures (HANDOVER §10.1). Success = Driving Score == 100 (HANDOVER §9).
