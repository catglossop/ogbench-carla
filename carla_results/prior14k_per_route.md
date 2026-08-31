# prior-14k — per-route Bench2Drive performance

*Generated 2026-08-30 13:51 local.*

Checkpoint (the commentary run, step 14000):

```
gs://cat-logs/pi05_steervla_cot_simplified_reasoning_commentary/pi05_steervla_cot_simplfied_reasoning_commentary_0823/pi05_steervla_cot_simplfied_reasoning_commentary_0823_20260823_154520/14000
```

Frozen policy, `actions_per_model_query=3`, `actions_per_cot=5`, greedy CoT, seed 0, all 220 Bench2Drive routes. **Success = Driving Score exactly 100** (full route, zero infractions); `DS = RC × IP`, so any infraction at all drops a route below 100.

## Summary


|                    | Routes  | Share     |
| ------------------ | ------- | --------- |
| **Scored**         | 218     | —         |
| Perfect (DS = 100) | 96      | 44.0%     |
| **Below 100 DS**   | **122** | **56.0%** |


Mean DS 70.46 · SD 31.11 · SEM 2.11 · mean RC 85.90 · mean IP 0.818

### What costs the points


| Infraction       | Total occurrences | Routes affected |
| ---------------- | ----------------- | --------------- |
| min-speed        | 1696              | 119             |
| veh-collision    | 74                | 59              |
| route-deviation  | 18                | 18              |
| static-collision | 16                | 16              |
| red-light        | 16                | 16              |
| stop-sign        | 7                 | 7               |
| off-lane         | 6                 | 6               |
| no-yield         | 5                 | 5               |
| blocked          | 3                 | 3               |
| ped-collision    | 2                 | 2               |




## ⚠ Routes below 100 DS (122)

> Worst first. These are where the policy loses score.


| Route                                                  | DS        | RC     | IP    | Status                                 | Infractions                                                                   |
| ------------------------------------------------------ | --------- | ------ | ----- | -------------------------------------- | ----------------------------------------------------------------------------- |
| **parking-exit-003**                                   | **1.50**  | 1.50   | 1.000 | Failed - TickRuntime                   | —                                                                             |
| **parking-exit-002**                                   | **1.52**  | 1.52   | 1.000 | Failed - TickRuntime                   | —                                                                             |
| **hard-break-route-002**                               | **3.07**  | 4.73   | 0.650 | Failed - Agent got blocked             | static-collision, blocked                                                     |
| **vanilla-signalized-turn-encounter-red-light-003**    | **6.27**  | 6.27   | 1.000 | Failed - TickRuntime                   | min-speed                                                                     |
| **vanilla-signalized-turn-encounter-green-light-003**  | **6.85**  | 6.85   | 1.000 | Failed - TickRuntime                   | min-speed                                                                     |
| **hard-break-route-003**                               | **7.59**  | 7.59   | 1.000 | Failed - TickRuntime                   | min-speed                                                                     |
| **vanilla-non-signalized-turn-encounter-stopsign-002** | **8.55**  | 23.74  | 0.360 | Failed - TickRuntime                   | min-speed×4, veh-collision×2                                                  |
| **vanilla-signalized-turn-encounter-green-light-004**  | **9.22**  | 9.22   | 1.000 | Failed - TickRuntime                   | min-speed                                                                     |
| **opposite-vehicle-taking-priority-004**               | **11.32** | 66.86  | 0.169 | Failed - Agent got blocked             | min-speed×12, veh-collision×2, static-collision, stop-sign, off-lane, blocked |
| **vehicle-turning-route-pedestrian-005**               | **12.78** | 35.51  | 0.360 | Failed - TickRuntime                   | min-speed×6, veh-collision×2                                                  |
| **construction-obstacle-002**                          | **12.96** | 100.00 | 0.130 | Completed                              | min-speed×20, veh-collision×4                                                 |
| **parking-exit-004**                                   | **14.04** | 100.00 | 0.140 | Completed                              | min-speed×20, veh-collision×3, static-collision                               |
| **vehicle-turning-route-pedestrian-003**               | **15.01** | 41.70  | 0.360 | Failed - Agent deviated from the route | min-speed×7, veh-collision×2, route-deviation                                 |
| **vehicle-turning-route-005**                          | **15.19** | 48.67  | 0.312 | Failed - TickRuntime                   | min-speed×8, static-collision, veh-collision, stop-sign                       |
| **accident-two-ways-002**                              | **15.38** | 70.47  | 0.218 | Failed - TickRuntime                   | min-speed×12, veh-collision×2, static-collision, off-lane                     |
| **parked-obstacle-003**                                | **20.49** | 34.15  | 0.600 | Failed - TickRuntime                   | min-speed×6, veh-collision                                                    |
| **parked-obstacle-004**                                | **20.84** | 34.74  | 0.600 | Failed - TickRuntime                   | min-speed×5, veh-collision                                                    |
| **signalized-junction-left-turn-003**                  | **21.62** | 60.05  | 0.360 | Failed - TickRuntime                   | min-speed×11, veh-collision×2                                                 |
| **opposite-vehicle-running-red-light-002**             | **24.45** | 50.93  | 0.480 | Failed - Agent deviated from the route | min-speed×8, veh-collision, stop-sign, route-deviation                        |
| **accident-two-ways-003**                              | **24.80** | 45.80  | 0.541 | Failed                                 | min-speed×7, veh-collision, off-lane                                          |
| **accident-005**                                       | **24.85** | 24.85  | 1.000 | Failed                                 | min-speed×4                                                                   |
| **t_-junction-005**                                    | **25.51** | 39.24  | 0.650 | Failed - TickRuntime                   | min-speed×7, static-collision                                                 |
| **non-signalized-junction-left-turn-enter-flow-004**   | **27.37** | 27.37  | 1.000 | Failed - TickRuntime                   | min-speed×4                                                                   |
| **accident-two-ways-005**                              | **27.40** | 42.15  | 0.650 | Failed                                 | min-speed×8, static-collision                                                 |
| **signalized-junction-left-turn-005**                  | **27.55** | 45.91  | 0.600 | Failed - TickRuntime                   | min-speed×8, veh-collision                                                    |
| **enter-actor-flow-005**                               | **28.63** | 28.63  | 1.000 | Failed - TickRuntime                   | min-speed×4                                                                   |
| **construction-obstacle-001**                          | **28.82** | 28.82  | 1.000 | Failed - TickRuntime                   | min-speed×4                                                                   |
| **construction-obstacle-two-ways-001**                 | **28.82** | 28.82  | 1.000 | Failed - TickRuntime                   | min-speed×4                                                                   |
| **construction-obstacle-two-ways-003**                 | **28.82** | 28.82  | 1.000 | Failed - TickRuntime                   | min-speed×5                                                                   |
| **construction-obstacle-004**                          | **29.39** | 29.39  | 1.000 | Failed - TickRuntime                   | min-speed×5                                                                   |
| **signalized-junction-left-turn-002**                  | **29.60** | 29.60  | 1.000 | Failed - TickRuntime                   | min-speed×5                                                                   |
| **blocked-intersection-001**                           | **30.26** | 30.26  | 1.000 | Failed - TickRuntime                   | min-speed×5                                                                   |
| **signalized-junction-left-turn-001**                  | **30.37** | 50.61  | 0.600 | Failed - TickRuntime                   | min-speed×8, veh-collision                                                    |
| **enter-actor-flow-003**                               | **31.52** | 31.52  | 1.000 | Failed - TickRuntime                   | min-speed×4                                                                   |
| **vanilla-signalized-turn-encounter-red-light-005**    | **31.84** | 48.98  | 0.650 | Failed - TickRuntime                   | min-speed×8, static-collision                                                 |
| **non-signalized-junction-left-turn-enter-flow-002**   | **31.94** | 53.23  | 0.600 | Failed - TickRuntime                   | min-speed×8, veh-collision                                                    |
| **opposite-vehicle-running-red-light-003**             | **32.74** | 54.56  | 0.600 | Failed - Agent deviated from the route | min-speed×10, veh-collision, route-deviation                                  |
| **non-signalized-junction-left-turn-004**              | **32.80** | 54.66  | 0.600 | Failed - Agent deviated from the route | min-speed×8, veh-collision, route-deviation                                   |
| **construction-obstacle-two-ways-002**                 | **33.14** | 33.14  | 1.000 | Failed - TickRuntime                   | min-speed×6                                                                   |
| **parked-obstacle-002**                                | **33.18** | 33.18  | 1.000 | Failed - TickRuntime                   | min-speed×6                                                                   |
| **blocked-intersection-002**                           | **34.50** | 54.45  | 0.634 | Failed - TickRuntime                   | min-speed×8, static-collision, off-lane                                       |
| **accident-two-ways-001**                              | **34.64** | 34.64  | 1.000 | Failed                                 | min-speed×6                                                                   |
| **construction-obstacle-two-ways-005**                 | **34.88** | 34.88  | 1.000 | Failed - TickRuntime                   | min-speed×5                                                                   |
| **non-signalized-junction-right-turn-001**             | **36.00** | 100.00 | 0.360 | Completed                              | min-speed×19, veh-collision×2                                                 |
| **non-signalized-junction-right-turn-003**             | **36.00** | 100.00 | 0.360 | Completed                              | min-speed×20, veh-collision×2                                                 |
| **non-signalized-junction-right-turn-005**             | **36.00** | 100.00 | 0.360 | Completed                              | min-speed×19, veh-collision×2                                                 |
| **signalized-junction-left-turn-enter-flow-003**       | **36.00** | 100.00 | 0.360 | Completed                              | min-speed×20, veh-collision×2                                                 |
| **opposite-vehicle-taking-priority-003**               | **36.28** | 55.81  | 0.650 | Failed - Agent got blocked             | min-speed×10, static-collision, blocked                                       |
| **t_-junction-003**                                    | **37.48** | 53.55  | 0.700 | Failed - Agent deviated from the route | min-speed×8, red-light, route-deviation                                       |
| **signalized-junction-left-turn-enter-flow-004**       | **42.00** | 100.00 | 0.420 | Completed                              | min-speed×17, veh-collision, red-light                                        |
| **signalized-junction-right-turn-004**                 | **42.00** | 100.00 | 0.420 | Completed                              | min-speed×21, veh-collision, red-light                                        |
| **signalized-junction-right-turn-005**                 | **42.00** | 100.00 | 0.420 | Completed                              | min-speed×21, veh-collision, red-light                                        |
| **pedestrian-crossing-005**                            | **45.50** | 100.00 | 0.455 | Completed                              | min-speed×19, static-collision, red-light                                     |
| **merger-into-slow-traffic-v2-003**                    | **46.81** | 46.81  | 1.000 | Failed - Agent deviated from the route | min-speed×7, route-deviation                                                  |
| **vanilla-non-signalized-turn-encounter-stopsign-001** | **47.35** | 54.14  | 0.875 | Failed - Agent deviated from the route | min-speed×9, off-lane, route-deviation                                        |
| **non-signalized-junction-left-turn-enter-flow-001**   | **48.00** | 100.00 | 0.480 | Completed                              | min-speed×21, veh-collision, stop-sign                                        |
| **pedestrian-crossing-001**                            | **48.00** | 100.00 | 0.480 | Completed                              | min-speed×20, veh-collision, stop-sign                                        |
| **vanilla-non-signalized-turn-encounter-stopsign-005** | **48.00** | 100.00 | 0.480 | Completed                              | min-speed×18, veh-collision, stop-sign                                        |
| **accident-001**                                       | **49.48** | 49.48  | 1.000 | Failed - TickRuntime                   | min-speed×8                                                                   |
| **parking-crossing-pedestrian-003**                    | **50.00** | 100.00 | 0.500 | Completed                              | min-speed×19, ped-collision                                                   |
| **parking-crossing-pedestrian-004**                    | **50.00** | 100.00 | 0.500 | Completed                              | min-speed×20, ped-collision                                                   |
| **t_-junction-002**                                    | **50.82** | 72.60  | 0.700 | Failed - Agent deviated from the route | min-speed×12, red-light, route-deviation                                      |
| **parking-exit-005**                                   | **51.22** | 100.00 | 0.512 | Completed                              | min-speed×19, static-collision, off-lane                                      |
| **crossing-bicycle-flow-003**                          | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×18, veh-collision                                                   |
| **enter-actor-flow-004**                               | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×21, veh-collision                                                   |
| **hard-break-route-001**                               | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×19, veh-collision                                                   |
| **hazard-at-side-lane-002**                            | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×20, veh-collision                                                   |
| **hazard-at-side-lane-005**                            | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×18, veh-collision                                                   |
| **hazard-at-side-lane-two-ways-001**                   | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×19, veh-collision                                                   |
| **highway-cut-in-001**                                 | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×19, veh-collision                                                   |
| **highway-cut-in-002**                                 | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×20, veh-collision                                                   |
| **highway-cut-in-005**                                 | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×19, veh-collision                                                   |
| **highway-exit-003**                                   | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×20, veh-collision, route-deviation                                  |
| **highway-exit-004**                                   | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×20, veh-collision, route-deviation                                  |
| **interurban-actor-flow-003**                          | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×20, veh-collision, route-deviation                                  |
| **invading-turn-001**                                  | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×19, veh-collision                                                   |
| **merger-into-slow-traffic-002**                       | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×19, veh-collision, route-deviation                                  |
| **non-signalized-junction-left-turn-001**              | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×21, veh-collision                                                   |
| **non-signalized-junction-left-turn-002**              | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×21, veh-collision                                                   |
| **non-signalized-junction-left-turn-005**              | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×21, veh-collision                                                   |
| **non-signalized-junction-left-turn-enter-flow-003**   | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×18, veh-collision                                                   |
| **non-signalized-junction-right-turn-004**             | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×20, veh-collision                                                   |
| **opposite-vehicle-taking-priority-002**               | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×18, veh-collision                                                   |
| **parked-obstacle-two-ways-002**                       | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×20, veh-collision                                                   |
| **signalized-junction-left-turn-enter-flow-001**       | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×19, veh-collision                                                   |
| **static-cut-in-001**                                  | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×19, veh-collision                                                   |
| **static-cut-in-004**                                  | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×20, veh-collision                                                   |
| **vanilla-non-signalized-turn-005**                    | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×18, veh-collision                                                   |
| **vanilla-non-signalized-turn-encounter-stopsign-003** | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×20, veh-collision                                                   |
| **vanilla-non-signalized-turn-encounter-stopsign-004** | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×18, veh-collision                                                   |
| **vanilla-signalized-turn-encounter-green-light-001**  | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×20, veh-collision                                                   |
| **vanilla-signalized-turn-encounter-green-light-002**  | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×18, veh-collision                                                   |
| **vehicle-turning-route-002**                          | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×20, veh-collision                                                   |
| **vehicle-turning-route-003**                          | **60.00** | 100.00 | 0.600 | Completed                              | min-speed×18, veh-collision                                                   |
| **control-loss-003**                                   | **61.10** | 61.10  | 1.000 | Failed - TickRuntime                   | min-speed×11                                                                  |
| **signalized-junction-left-turn-enter-flow-005**       | **62.02** | 88.60  | 0.700 | Failed - TickRuntime                   | min-speed×16, red-light                                                       |
| **accident-two-ways-004**                              | **65.00** | 100.00 | 0.650 | Completed                              | min-speed×19, static-collision                                                |
| **opposite-vehicle-running-red-light-004**             | **65.00** | 100.00 | 0.650 | Completed                              | min-speed×21, static-collision                                                |
| **parked-obstacle-two-ways-003**                       | **65.00** | 100.00 | 0.650 | Completed                              | min-speed×18, static-collision                                                |
| **parking-exit-001**                                   | **65.00** | 100.00 | 0.650 | Completed                              | min-speed×20, static-collision                                                |
| **crossing-bicycle-flow-002**                          | **70.00** | 100.00 | 0.700 | Completed                              | min-speed×17, red-light                                                       |
| **enter-actor-flow-001**                               | **70.00** | 100.00 | 0.700 | Completed                              | min-speed×19, red-light                                                       |
| **non-signalized-junction-left-turn-enter-flow-005**   | **70.00** | 100.00 | 0.700 | Completed                              | min-speed×19, red-light                                                       |
| **pedestrian-crossing-002**                            | **70.00** | 100.00 | 0.700 | Completed                              | min-speed×17, red-light                                                       |
| **signalized-junction-left-turn-enter-flow-002**       | **70.00** | 100.00 | 0.700 | Completed                              | min-speed×19, red-light                                                       |
| **vanilla-signalized-turn-encounter-red-light-001**    | **70.00** | 100.00 | 0.700 | Completed                              | min-speed×19, red-light                                                       |
| **vanilla-signalized-turn-encounter-red-light-002**    | **70.00** | 100.00 | 0.700 | Completed                              | min-speed×19, red-light                                                       |
| **vanilla-signalized-turn-encounter-red-light-004**    | **70.00** | 100.00 | 0.700 | Completed                              | min-speed×18, red-light                                                       |
| **vehicle-turning-route-pedestrian-001**               | **70.00** | 100.00 | 0.700 | Completed                              | min-speed×20, red-light                                                       |
| **yield-to-emergency-vehicle-001**                     | **70.00** | 100.00 | 0.700 | Completed                              | min-speed×19, no-yield                                                        |
| **yield-to-emergency-vehicle-002**                     | **70.00** | 100.00 | 0.700 | Completed                              | min-speed×19, no-yield                                                        |
| **yield-to-emergency-vehicle-003**                     | **70.00** | 100.00 | 0.700 | Completed                              | min-speed×19, no-yield                                                        |
| **yield-to-emergency-vehicle-004**                     | **70.00** | 100.00 | 0.700 | Completed                              | min-speed×19, no-yield                                                        |
| **yield-to-emergency-vehicle-005**                     | **70.00** | 100.00 | 0.700 | Completed                              | min-speed×19, no-yield                                                        |
| **highway-exit-002**                                   | **77.88** | 77.88  | 1.000 | Failed - Agent deviated from the route | min-speed×12, route-deviation                                                 |
| **highway-exit-001**                                   | **79.77** | 79.77  | 1.000 | Failed - Agent deviated from the route | min-speed×14, route-deviation                                                 |
| **pedestrian-crossing-004**                            | **80.00** | 100.00 | 0.800 | Completed                              | min-speed×19, stop-sign                                                       |
| **hazard-at-side-lane-two-ways-002**                   | **89.34** | 89.34  | 1.000 | Failed - TickRuntime                   | min-speed×16                                                                  |
| **interurban-actor-flow-002**                          | **92.17** | 92.17  | 1.000 | Failed - Agent deviated from the route | min-speed×16, route-deviation                                                 |
| **interurban-actor-flow-005**                          | **92.17** | 92.17  | 1.000 | Failed - Agent deviated from the route | min-speed×16, route-deviation                                                 |
| **interurban-actor-flow-004**                          | **97.24** | 97.24  | 1.000 | Failed - Agent deviated from the route | min-speed×19, route-deviation                                                 |
| **merger-into-slow-traffic-004**                       | **97.77** | 97.77  | 1.000 | Failed - Agent deviated from the route | min-speed×20, route-deviation                                                 |




## Perfect routes (96)

> DS 100 — full completion, no infractions.

- `accident-002` · `accident-003` · `blocked-intersection-003`
- `blocked-intersection-004` · `blocked-intersection-005` · `construction-obstacle-003`
- `construction-obstacle-005` · `construction-obstacle-two-ways-004` · `control-loss-001`
- `control-loss-002` · `control-loss-004` · `control-loss-005`
- `crossing-bicycle-flow-001` · `crossing-bicycle-flow-004` · `crossing-bicycle-flow-005`
- `dynamic-object-crossing-001` · `dynamic-object-crossing-002` · `dynamic-object-crossing-003`
- `dynamic-object-crossing-004` · `dynamic-object-crossing-005` · `hard-break-route-004`
- `hard-break-route-005` · `hazard-at-side-lane-001` · `hazard-at-side-lane-003`
- `hazard-at-side-lane-004` · `hazard-at-side-lane-two-ways-003` · `hazard-at-side-lane-two-ways-004`
- `hazard-at-side-lane-two-ways-005` · `highway-cut-in-003` · `highway-cut-in-004`
- `highway-exit-005` · `interurban-actor-flow-001` · `interurban-advanced-actor-flow-001`
- `interurban-advanced-actor-flow-002` · `interurban-advanced-actor-flow-003` · `interurban-advanced-actor-flow-004`
- `interurban-advanced-actor-flow-005` · `invading-turn-002` · `invading-turn-003`
- `invading-turn-004` · `invading-turn-005` · `merger-into-slow-traffic-001`
- `merger-into-slow-traffic-003` · `merger-into-slow-traffic-005` · `merger-into-slow-traffic-v2-001`
- `merger-into-slow-traffic-v2-002` · `merger-into-slow-traffic-v2-004` · `merger-into-slow-traffic-v2-005`
- `non-signalized-junction-left-turn-003` · `non-signalized-junction-right-turn-002` · `opposite-vehicle-running-red-light-001`
- `opposite-vehicle-running-red-light-005` · `opposite-vehicle-taking-priority-001` · `opposite-vehicle-taking-priority-005`
- `parked-obstacle-001` · `parked-obstacle-005` · `parked-obstacle-two-ways-001`
- `parked-obstacle-two-ways-004` · `parked-obstacle-two-ways-005` · `parking-crossing-pedestrian-001`
- `parking-crossing-pedestrian-002` · `parking-crossing-pedestrian-005` · `parking-cut-in-001`
- `parking-cut-in-002` · `parking-cut-in-003` · `parking-cut-in-004`
- `parking-cut-in-005` · `pedestrian-crossing-003` · `sequential-lane-change-001`
- `sequential-lane-change-002` · `sequential-lane-change-003` · `sequential-lane-change-004`
- `sequential-lane-change-005` · `signalized-junction-left-turn-004` · `signalized-junction-right-turn-001`
- `signalized-junction-right-turn-002` · `signalized-junction-right-turn-003` · `static-cut-in-002`
- `static-cut-in-003` · `static-cut-in-005` · `t_-junction-001`
- `t_-junction-004` · `vanilla-non-signalized-turn-001` · `vanilla-non-signalized-turn-002`
- `vanilla-non-signalized-turn-003` · `vanilla-non-signalized-turn-004` · `vanilla-signalized-turn-encounter-green-light-005`
- `vehicle-opens-door-two-ways-001` · `vehicle-opens-door-two-ways-002` · `vehicle-opens-door-two-ways-003`
- `vehicle-opens-door-two-ways-004` · `vehicle-opens-door-two-ways-005` · `vehicle-turning-route-001`
- `vehicle-turning-route-004` · `vehicle-turning-route-pedestrian-002` · `vehicle-turning-route-pedestrian-004`

---

> Regenerate: `.venv/bin/python .run_carla/gen_prior14k_routes.py`

