# prior-14k — per-route results across three seeds

_Generated 2026-09-01 10:11 local._

Checkpoint (the commentary run, step 14000):

```
gs://cat-logs/pi05_steervla_cot_simplified_reasoning_commentary/pi05_steervla_cot_simplfied_reasoning_commentary_0823/pi05_steervla_cot_simplfied_reasoning_commentary_0823_20260823_154520/14000
```

Frozen policy, `actions_per_model_query=3`, `actions_per_cot=5`, greedy CoT, all 220 Bench2Drive routes, run three times with **seeds 0, 1 and 2**. Nothing differs between the columns except `env.reset(seed)` — so where two seeds disagree on a route, that difference is the simulator, not the policy.

## Seed agreement

| | Routes | Share |
|---|---:|---:|
| Scored in all three seeds | 217 | — |
| **Identical DS in all three** | **86** | **39.6%** |
| Differ between seeds | 131 | 60.4% |
| Perfect (DS 100) in all three | 67 | 30.9% |
| Below 100 in all three | 79 | 36.4% |
| **Flip between perfect and not** | **71** | **32.7%** |

Mean spread across all routes **23.66 DS**; among the 131 that vary, mean spread **39.2 DS** (max 97).

**Roughly half the benchmark (40%) is fully deterministic.** The seed-to-seed variance reported elsewhere comes almost entirely from the other 131 routes, where a single route can swing by up to 97 DS. That is why a two-run comparison of checkpoints can manufacture a several-point difference: it depends which of these routes happened to land well.

### How far apart the seeds get

| Spread (max−min DS) | Routes |
|---|---:|
| identical | 86 |
| ≤10 | 19 |
| 10–25 | 17 |
| 25–50 | 61 |
| 50–75 | 22 |
| >75 | 14 |

## ⚠ Where the seeds disagree most (30 of 131)

> Largest spread first. These routes dominate the benchmark's noise floor.

| Route | seed 0 | seed 1 | seed 2 | spread | status (seed 0) |
|---|---:|---:|---:|---:|---|
| **hard-break-route-002** | 3.07 | 100.00 | 3.98 | **96.93** | Failed - Agent got blocked |
| **vanilla-signalized-turn-encounter-red-light-003** | 6.27 | 100.00 | 6.27 | **93.73** | Failed - TickRuntime |
| **hard-break-route-003** | 7.59 | 100.00 | 21.11 | **92.41** | Failed - TickRuntime |
| **vehicle-turning-route-pedestrian-004** | 100.00 | 12.96 | 100.00 | **87.04** | Completed |
| **parked-obstacle-two-ways-005** | 100.00 | 36.00 | 14.19 | **85.81** | Completed |
| **vanilla-signalized-turn-encounter-green-light-004** | 9.22 | 94.59 | 9.22 | **85.37** | Failed - TickRuntime |
| **vehicle-turning-route-pedestrian-003** | 15.01 | 21.25 | 100.00 | **84.99** | Failed - Agent deviated from the route |
| **parked-obstacle-001** | 100.00 | 18.20 | 100.00 | **81.80** | Completed |
| **crossing-bicycle-flow-004** | 100.00 | 18.51 | 50.18 | **81.49** | Completed |
| **accident-003** | 100.00 | 19.60 | 60.00 | **80.40** | Completed |
| **parked-obstacle-003** | 20.49 | 100.00 | 100.00 | **79.51** | Failed - TickRuntime |
| **parked-obstacle-004** | 20.84 | 100.00 | 60.00 | **79.16** | Failed - TickRuntime |
| **parked-obstacle-005** | 100.00 | 100.00 | 21.60 | **78.40** | Completed |
| **signalized-junction-left-turn-004** | 100.00 | 36.34 | 22.88 | **77.12** | Completed |
| **vanilla-non-signalized-turn-005** | 60.00 | 26.63 | 100.00 | **73.37** | Completed |
| **t_-junction-005** | 25.51 | 96.25 | 25.09 | **71.16** | Failed - TickRuntime |
| **blocked-intersection-004** | 100.00 | 42.00 | 29.57 | **70.42** | Completed |
| **blocked-intersection-001** | 30.26 | 100.00 | 60.00 | **69.74** | Failed - TickRuntime |
| **vehicle-turning-route-pedestrian-002** | 100.00 | 100.00 | 30.28 | **69.72** | Completed |
| **parked-obstacle-002** | 33.18 | 100.00 | 100.00 | **66.82** | Failed - TickRuntime |
| **hazard-at-side-lane-003** | 100.00 | 33.44 | 92.88 | **66.56** | Completed |
| **signalized-junction-left-turn-enter-flow-003** | 36.00 | 33.49 | 100.00 | **66.51** | Completed |
| **blocked-intersection-002** | 34.50 | 89.56 | 100.00 | **65.50** | Failed - TickRuntime |
| **vehicle-turning-route-002** | 60.00 | 100.00 | 35.00 | **65.00** | Completed |
| **non-signalized-junction-right-turn-005** | 36.00 | 35.50 | 100.00 | **64.50** | Completed |
| **highway-exit-004** | 60.00 | 100.00 | 36.00 | **64.00** | Completed |
| **hazard-at-side-lane-004** | 100.00 | 100.00 | 38.18 | **61.82** | Completed |
| **construction-obstacle-003** | 100.00 | 100.00 | 42.33 | **57.67** | Completed |
| **opposite-vehicle-running-red-light-002** | 24.45 | 28.80 | 80.00 | **55.55** | Failed - Agent deviated from the route |
| **pedestrian-crossing-005** | 45.50 | 70.00 | 100.00 | **54.50** | Completed |

## Consistently below 100 DS (79)

> Sub-perfect in every seed — these are real policy weaknesses, not luck. Sorted by mean DS.

| Route | seed 0 | seed 1 | seed 2 | mean | infractions (seed 0) |
|---|---:|---:|---:|---:|---|
| **parking-exit-003** | 1.50 | 1.50 | 1.50 | **1.50** | — |
| **parking-exit-002** | 1.52 | 1.52 | 1.52 | **1.52** | — |
| **vanilla-signalized-turn-encounter-green-light-003** | 6.85 | 3.47 | 3.47 | **4.60** | min-speed |
| **vanilla-non-signalized-turn-encounter-stopsign-002** | 8.55 | 18.46 | 17.41 | **14.80** | min-speed×4, veh-collision×2 |
| **parking-exit-004** | 14.04 | 45.47 | 1.66 | **20.39** | min-speed×20, veh-collision×3, static-collision |
| **vehicle-turning-route-005** | 15.19 | 19.63 | 28.80 | **21.20** | min-speed×8, static-collision, veh-collision, stop-sign |
| **non-signalized-junction-right-turn-003** | 36.00 | 7.36 | 24.48 | **22.61** | min-speed×20, veh-collision×2 |
| **accident-two-ways-003** | 24.80 | 26.59 | 16.49 | **22.62** | min-speed×7, veh-collision, off-lane |
| **construction-obstacle-two-ways-003** | 28.82 | 28.82 | 11.60 | **23.08** | min-speed×5 |
| **vanilla-signalized-turn-encounter-red-light-005** | 31.84 | 7.42 | 31.51 | **23.59** | min-speed×8, static-collision |
| **signalized-junction-left-turn-003** | 21.62 | 35.57 | 23.92 | **27.04** | min-speed×11, veh-collision×2 |
| **vanilla-signalized-turn-encounter-green-light-002** | 60.00 | 8.29 | 13.14 | **27.14** | min-speed×18, veh-collision |
| **enter-actor-flow-005** | 28.63 | 27.39 | 28.63 | **28.22** | min-speed×4 |
| **construction-obstacle-two-ways-001** | 28.82 | 28.82 | 27.31 | **28.32** | min-speed×4 |
| **accident-005** | 24.85 | 30.11 | 30.11 | **28.36** | min-speed×4 |
| **accident-two-ways-002** | 15.38 | 56.54 | 13.58 | **28.50** | min-speed×12, veh-collision×2, static-collision, off-lane |
| **construction-obstacle-004** | 29.39 | 28.63 | 29.39 | **29.14** | min-speed×5 |
| **vehicle-turning-route-pedestrian-005** | 12.78 | 38.23 | 36.87 | **29.29** | min-speed×6, veh-collision×2 |
| **enter-actor-flow-003** | 31.52 | 36.00 | 21.12 | **29.55** | min-speed×4 |
| **signalized-junction-left-turn-001** | 30.37 | 30.37 | 29.47 | **30.07** | min-speed×8, veh-collision |
| **construction-obstacle-two-ways-002** | 33.14 | 25.35 | 33.14 | **30.54** | min-speed×6 |
| **opposite-vehicle-running-red-light-003** | 32.74 | 32.74 | 31.55 | **32.34** | min-speed×10, veh-collision, route-deviation |
| **vanilla-non-signalized-turn-encounter-stopsign-001** | 47.35 | 13.71 | 36.00 | **32.35** | min-speed×9, off-lane, route-deviation |
| **construction-obstacle-two-ways-005** | 34.88 | 33.36 | 31.85 | **33.36** | min-speed×5 |
| **non-signalized-junction-left-turn-enter-flow-002** | 31.94 | 26.26 | 42.58 | **33.59** | min-speed×8, veh-collision |
| **pedestrian-crossing-001** | 48.00 | 28.80 | 28.80 | **35.20** | min-speed×20, veh-collision, stop-sign |
| **non-signalized-junction-left-turn-enter-flow-001** | 48.00 | 10.95 | 48.00 | **35.65** | min-speed×21, veh-collision, stop-sign |
| **accident-two-ways-005** | 27.40 | 38.40 | 42.25 | **36.02** | min-speed×8, static-collision |
| **t_-junction-003** | 37.48 | 37.48 | 37.48 | **37.48** | min-speed×8, red-light, route-deviation |
| **vanilla-signalized-turn-encounter-green-light-004** | 9.22 | 94.59 | 9.22 | **37.68** | min-speed |
| **construction-obstacle-002** | 12.96 | 65.00 | 36.00 | **37.99** | min-speed×20, veh-collision×4 |
| **non-signalized-junction-left-turn-enter-flow-005** | 70.00 | 22.21 | 22.21 | **38.14** | min-speed×19, red-light |
| **non-signalized-junction-right-turn-004** | 60.00 | 34.52 | 20.41 | **38.31** | min-speed×20, veh-collision |
| **signalized-junction-left-turn-005** | 27.55 | 28.77 | 60.00 | **38.77** | min-speed×8, veh-collision |
| **non-signalized-junction-left-turn-004** | 32.80 | 31.97 | 53.29 | **39.35** | min-speed×8, veh-collision, route-deviation |
| **opposite-vehicle-taking-priority-004** | 11.32 | 47.31 | 60.00 | **39.54** | min-speed×12, veh-collision×2, static-collision, stop-sign, off-lane, blocked |
| **construction-obstacle-001** | 28.82 | 65.00 | 28.82 | **40.88** | min-speed×4 |
| **signalized-junction-left-turn-002** | 29.60 | 60.00 | 33.80 | **41.13** | min-speed×5 |
| **non-signalized-junction-left-turn-enter-flow-004** | 27.37 | 48.94 | 47.54 | **41.28** | min-speed×4 |
| **signalized-junction-right-turn-004** | 42.00 | 42.00 | 42.00 | **42.00** | min-speed×21, veh-collision, red-light |
| **vanilla-signalized-turn-encounter-green-light-001** | 60.00 | 60.00 | 8.86 | **42.95** | min-speed×20, veh-collision |
| **opposite-vehicle-taking-priority-003** | 36.28 | 36.00 | 60.00 | **44.09** | min-speed×10, static-collision, blocked |
| **opposite-vehicle-running-red-light-002** | 24.45 | 28.80 | 80.00 | **44.42** | min-speed×8, veh-collision, stop-sign, route-deviation |
| **non-signalized-junction-left-turn-001** | 60.00 | 60.00 | 18.27 | **46.09** | min-speed×21, veh-collision |
| **accident-two-ways-001** | 34.64 | 65.00 | 39.00 | **46.21** | min-speed×6 |
| **parked-obstacle-two-ways-003** | 65.00 | 39.00 | 38.14 | **47.38** | min-speed×18, static-collision |
| **t_-junction-005** | 25.51 | 96.25 | 25.09 | **48.95** | min-speed×7, static-collision |
| **accident-001** | 49.48 | 50.62 | 49.48 | **49.86** | min-speed×8 |
| **signalized-junction-left-turn-enter-flow-004** | 42.00 | 70.00 | 42.00 | **51.33** | min-speed×17, veh-collision, red-light |
| **signalized-junction-right-turn-005** | 42.00 | 70.00 | 42.00 | **51.33** | min-speed×21, veh-collision, red-light |
| **opposite-vehicle-taking-priority-002** | 60.00 | 35.10 | 60.00 | **51.70** | min-speed×18, veh-collision |
| **non-signalized-junction-right-turn-001** | 36.00 | 60.00 | 60.00 | **52.00** | min-speed×19, veh-collision×2 |
| **static-cut-in-001** | 60.00 | 60.00 | 36.00 | **52.00** | min-speed×19, veh-collision |
| **non-signalized-junction-left-turn-002** | 60.00 | 50.73 | 52.35 | **54.36** | min-speed×21, veh-collision |
| **highway-cut-in-001** | 60.00 | 60.00 | 60.00 | **60.00** | min-speed×19, veh-collision |
| **interurban-actor-flow-003** | 60.00 | 60.00 | 60.00 | **60.00** | min-speed×20, veh-collision, route-deviation |
| **non-signalized-junction-left-turn-005** | 60.00 | 60.00 | 60.00 | **60.00** | min-speed×21, veh-collision |
| **non-signalized-junction-left-turn-enter-flow-003** | 60.00 | 60.00 | 60.00 | **60.00** | min-speed×18, veh-collision |
| **parked-obstacle-two-ways-002** | 60.00 | 60.00 | 60.00 | **60.00** | min-speed×20, veh-collision |
| **signalized-junction-left-turn-enter-flow-001** | 60.00 | 60.00 | 60.00 | **60.00** | min-speed×19, veh-collision |
| **static-cut-in-004** | 60.00 | 60.00 | 60.00 | **60.00** | min-speed×20, veh-collision |
| **vehicle-turning-route-003** | 60.00 | 60.00 | 60.00 | **60.00** | min-speed×18, veh-collision |
| **vanilla-signalized-turn-encounter-red-light-004** | 70.00 | 70.00 | 42.00 | **60.67** | min-speed×18, red-light |
| **accident-two-ways-004** | 65.00 | 77.70 | 39.64 | **60.78** | min-speed×19, static-collision |
| **vanilla-non-signalized-turn-encounter-stopsign-005** | 48.00 | 77.36 | 60.00 | **61.79** | min-speed×18, veh-collision, stop-sign |
| **highway-exit-002** | 77.88 | 33.85 | 77.88 | **63.20** | min-speed×12, route-deviation |
| **signalized-junction-left-turn-enter-flow-005** | 62.02 | 70.00 | 66.01 | **66.01** | min-speed×16, red-light |
| **crossing-bicycle-flow-002** | 70.00 | 70.00 | 70.00 | **70.00** | min-speed×17, red-light |
| **signalized-junction-left-turn-enter-flow-002** | 70.00 | 70.00 | 70.00 | **70.00** | min-speed×19, red-light |
| **yield-to-emergency-vehicle-001** | 70.00 | 70.00 | 70.00 | **70.00** | min-speed×19, no-yield |
| **yield-to-emergency-vehicle-002** | 70.00 | 70.00 | 70.00 | **70.00** | min-speed×19, no-yield |
| **yield-to-emergency-vehicle-003** | 70.00 | 70.00 | 70.00 | **70.00** | min-speed×19, no-yield |
| **yield-to-emergency-vehicle-004** | 70.00 | 70.00 | 70.00 | **70.00** | min-speed×19, no-yield |
| **yield-to-emergency-vehicle-005** | 70.00 | 70.00 | 70.00 | **70.00** | min-speed×19, no-yield |
| **highway-exit-001** | 79.77 | 76.14 | 71.59 | **75.83** | min-speed×14, route-deviation |
| **interurban-actor-flow-002** | 92.17 | 92.17 | 93.15 | **92.50** | min-speed×16, route-deviation |
| **interurban-actor-flow-005** | 92.17 | 93.15 | 92.17 | **92.50** | min-speed×16, route-deviation |
| **interurban-actor-flow-004** | 97.24 | 97.24 | 96.33 | **96.94** | min-speed×19, route-deviation |
| **merger-into-slow-traffic-004** | 97.77 | 98.66 | 98.22 | **98.22** | min-speed×20, route-deviation |

### What costs the points on those routes

| Infraction | Occurrences | Routes affected |
|---|---:|---:|
| min-speed | 1033 | 77 |
| veh-collision | 50 | 38 |
| route-deviation | 12 | 12 |
| static-collision | 10 | 10 |
| red-light | 9 | 9 |
| stop-sign | 6 | 6 |
| no-yield | 5 | 5 |
| off-lane | 4 | 4 |
| blocked | 2 | 2 |

## Perfect in all three seeds (67)

> DS 100 every time — solved, and stable.

- `accident-002` · `blocked-intersection-003` · `blocked-intersection-005`
- `construction-obstacle-005` · `control-loss-001` · `control-loss-002`
- `control-loss-004` · `control-loss-005` · `crossing-bicycle-flow-005`
- `dynamic-object-crossing-001` · `dynamic-object-crossing-002` · `dynamic-object-crossing-003`
- `dynamic-object-crossing-004` · `hard-break-route-004` · `hazard-at-side-lane-001`
- `hazard-at-side-lane-two-ways-003` · `hazard-at-side-lane-two-ways-005` · `highway-cut-in-003`
- `highway-exit-005` · `interurban-actor-flow-001` · `interurban-advanced-actor-flow-001`
- `interurban-advanced-actor-flow-002` · `interurban-advanced-actor-flow-003` · `interurban-advanced-actor-flow-004`
- `interurban-advanced-actor-flow-005` · `invading-turn-002` · `invading-turn-003`
- `invading-turn-004` · `invading-turn-005` · `merger-into-slow-traffic-001`
- `merger-into-slow-traffic-003` · `merger-into-slow-traffic-005` · `merger-into-slow-traffic-v2-001`
- `merger-into-slow-traffic-v2-002` · `merger-into-slow-traffic-v2-005` · `non-signalized-junction-right-turn-002`
- `opposite-vehicle-running-red-light-001` · `opposite-vehicle-running-red-light-005` · `opposite-vehicle-taking-priority-001`
- `parked-obstacle-two-ways-001` · `parking-crossing-pedestrian-001` · `parking-crossing-pedestrian-002`
- `parking-crossing-pedestrian-005` · `parking-cut-in-001` · `parking-cut-in-002`
- `parking-cut-in-003` · `parking-cut-in-004` · `parking-cut-in-005`
- `pedestrian-crossing-003` · `sequential-lane-change-001` · `sequential-lane-change-002`
- `sequential-lane-change-003` · `sequential-lane-change-004` · `sequential-lane-change-005`
- `static-cut-in-003` · `static-cut-in-005` · `t_-junction-001`
- `t_-junction-004` · `vanilla-non-signalized-turn-001` · `vanilla-non-signalized-turn-002`
- `vanilla-non-signalized-turn-003` · `vanilla-non-signalized-turn-004` · `vanilla-signalized-turn-encounter-green-light-005`
- `vehicle-opens-door-two-ways-001` · `vehicle-opens-door-two-ways-002` · `vehicle-opens-door-two-ways-003`
- `vehicle-opens-door-two-ways-005`

---

> Regenerate: `.venv/bin/python .run_carla/gen_prior14k_routes.py`
