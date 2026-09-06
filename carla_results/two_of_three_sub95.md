# Routes where 2 or more of {base policy, SimLingo, SteerVLA} score below 95 DS

_Base = prior-14k checkpoint, mean of seeds 0/1/2 (`carla_results/prior14k_per_route.md`)._
_SimLingo = `~/carla_results/merged.json`; SteerVLA = `~/carla_results/steervla (3).json`; all 220 Bench2Drive routes, joined on leaderboard `route_id` via `ogbench.carla.route_registry`._
_Regenerate: `python3 .run_carla/gen_two_of_three_sub95.py`_

**69 confirmed** routes, plus **12 undetermined** (below) where prior-14k per-route numbers are not on this machine.

## Confirmed

| Route | Base (prior-14k) | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `enter-actor-flow-002` | n/a | 32.59 | 42.00 | SimLingo + SteerVLA |
| `signalized-junction-left-turn-001` | 30.07 | 23.15 | 60.00 | Base + SimLingo + SteerVLA |
| `non-signalized-junction-right-turn-003` | 22.61 | 36.00 | 60.00 | Base + SimLingo + SteerVLA |
| `signalized-junction-left-turn-003` | 27.04 | 42.00 | 60.00 | Base + SimLingo + SteerVLA |
| `opposite-vehicle-taking-priority-003` | 44.09 | 60.00 | 31.92 | Base + SimLingo + SteerVLA |
| `signalized-junction-left-turn-002` | 41.13 | 36.00 | 60.00 | Base + SimLingo + SteerVLA |
| `non-signalized-junction-left-turn-enter-flow-002` | 33.59 | 48.00 | 60.00 | Base + SimLingo + SteerVLA |
| `accident-two-ways-002` | 28.50 | 100.00 | 13.79 | Base + SteerVLA |
| `parking-exit-002` | 1.52 | 42.00 | 100.00 | Base + SimLingo |
| `accident-005` | 28.36 | 60.00 | 60.00 | Base + SimLingo + SteerVLA |
| `merger-into-slow-traffic-v2-005` | 100.00 | 6.40 | 45.50 | SimLingo + SteerVLA |
| `non-signalized-junction-left-turn-003` | n/a | 42.00 | 60.00 | SimLingo + SteerVLA |
| `non-signalized-junction-right-turn-005` | 57.17 | 60.00 | 36.00 | Base + SimLingo + SteerVLA |
| `vehicle-turning-route-pedestrian-005` | 29.29 | 100.00 | 25.00 | Base + SteerVLA |
| `opposite-vehicle-running-red-light-005` | 100.00 | 32.30 | 25.04 | SimLingo + SteerVLA |
| `signalized-junction-left-turn-005` | 38.77 | 60.00 | 60.00 | Base + SimLingo + SteerVLA |
| `opposite-vehicle-taking-priority-004` | 39.54 | 60.00 | 60.00 | Base + SimLingo + SteerVLA |
| `opposite-vehicle-running-red-light-003` | 32.34 | 100.00 | 28.46 | Base + SteerVLA |
| `parking-exit-003` | 1.50 | 60.00 | 100.00 | Base + SimLingo |
| `signalized-junction-right-turn-004` | 42.00 | 60.00 | 60.00 | Base + SimLingo + SteerVLA |
| `non-signalized-junction-left-turn-004` | 39.35 | 100.00 | 31.97 | Base + SteerVLA |
| `signalized-junction-right-turn-005` | 51.33 | 60.00 | 60.00 | Base + SimLingo + SteerVLA |
| `enter-actor-flow-005` | 28.22 | 100.00 | 46.57 | Base + SteerVLA |
| `enter-actor-flow-004` | n/a | 60.00 | 60.00 | SimLingo + SteerVLA |
| `signalized-junction-left-turn-enter-flow-001` | 60.00 | 60.00 | 60.00 | Base + SimLingo + SteerVLA |
| `static-cut-in-001` | 52.00 | 33.69 | 100.00 | Base + SimLingo |
| `construction-obstacle-two-ways-001` | 28.32 | 60.00 | 100.00 | Base + SimLingo |
| `construction-obstacle-004` | 29.14 | 60.00 | 100.00 | Base + SimLingo |
| `enter-actor-flow-003` | 29.55 | 60.00 | 100.00 | Base + SimLingo |
| `accident-two-ways-003` | 22.63 | 70.00 | 100.00 | Base + SimLingo |
| `crossing-bicycle-flow-004` | 56.23 | 100.00 | 40.27 | Base + SteerVLA |
| `non-signalized-junction-left-turn-enter-flow-005` | 38.14 | 60.00 | 100.00 | Base + SimLingo |
| `non-signalized-junction-right-turn-004` | 38.31 | 60.00 | 100.00 | Base + SimLingo |
| `highway-exit-002` | 63.20 | 60.00 | 75.87 | Base + SimLingo + SteerVLA |
| `yield-to-emergency-vehicle-005` | 70.00 | 70.00 | 60.00 | Base + SimLingo + SteerVLA |
| `construction-obstacle-001` | 40.88 | 60.00 | 100.00 | Base + SimLingo |
| `non-signalized-junction-left-turn-enter-flow-004` | 41.28 | 60.00 | 100.00 | Base + SimLingo |
| `signalized-junction-left-turn-enter-flow-005` | 66.01 | 100.00 | 35.55 | Base + SteerVLA |
| `accident-two-ways-005` | 36.02 | 100.00 | 70.00 | Base + SteerVLA |
| `vanilla-signalized-turn-encounter-red-light-003` | 37.51 | 70.00 | 100.00 | Base + SimLingo |
| `parked-obstacle-001` | 72.73 | 36.00 | 100.00 | Base + SimLingo |
| `yield-to-emergency-vehicle-001` | 70.00 | 70.00 | 70.00 | Base + SimLingo + SteerVLA |
| `yield-to-emergency-vehicle-003` | 70.00 | 70.00 | 70.00 | Base + SimLingo + SteerVLA |
| `yield-to-emergency-vehicle-004` | 70.00 | 70.00 | 70.00 | Base + SimLingo + SteerVLA |
| `signalized-junction-left-turn-enter-flow-004` | 51.33 | 100.00 | 60.00 | Base + SteerVLA |
| `non-signalized-junction-right-turn-001` | 52.00 | 60.00 | 100.00 | Base + SimLingo |
| `signalized-junction-left-turn-004` | 53.07 | 100.00 | 60.00 | Base + SteerVLA |
| `non-signalized-junction-left-turn-002` | 54.36 | 100.00 | 60.00 | Base + SteerVLA |
| `pedestrian-crossing-001` | 35.20 | 100.00 | 80.00 | Base + SteerVLA |
| `non-signalized-junction-left-turn-enter-flow-001` | 35.65 | 80.00 | 100.00 | Base + SimLingo |
| `signalized-junction-left-turn-enter-flow-003` | 56.50 | 60.00 | 100.00 | Base + SimLingo |
| `construction-obstacle-003` | 80.78 | 39.00 | 100.00 | Base + SimLingo |
| `static-cut-in-004` | 60.00 | 60.00 | 100.00 | Base + SimLingo |
| `parked-obstacle-004` | 60.28 | 60.00 | 100.00 | Base + SimLingo |
| `vanilla-non-signalized-turn-encounter-stopsign-005` | 61.79 | 60.00 | 100.00 | Base + SimLingo |
| `opposite-vehicle-running-red-light-002` | 44.42 | 80.00 | 100.00 | Base + SimLingo |
| `highway-exit-004` | 65.33 | 60.00 | 100.00 | Base + SimLingo |
| `vanilla-signalized-turn-encounter-green-light-004` | 37.68 | 87.81 | 100.00 | Base + SimLingo |
| `blocked-intersection-004` | 57.19 | 70.00 | 100.00 | Base + SimLingo |
| `highway-exit-001` | 75.83 | 76.14 | 76.14 | Base + SimLingo + SteerVLA |
| `vanilla-signalized-turn-encounter-red-light-004` | 60.67 | 70.00 | 100.00 | Base + SimLingo |
| `parked-obstacle-005` | 73.87 | 60.00 | 100.00 | Base + SimLingo |
| `parked-obstacle-002` | 77.73 | 60.00 | 100.00 | Base + SimLingo |
| `pedestrian-crossing-004` | n/a | 80.00 | 80.00 | SimLingo + SteerVLA |
| `yield-to-emergency-vehicle-002` | 70.00 | 70.00 | 100.00 | Base + SimLingo |
| `pedestrian-crossing-005` | 71.83 | 70.00 | 100.00 | Base + SimLingo |
| `interurban-actor-flow-003` | 60.00 | 92.77 | 100.00 | Base + SimLingo |
| `interurban-actor-flow-005` | 92.50 | 93.15 | 92.17 | Base + SimLingo + SteerVLA |
| `interurban-actor-flow-002` | 92.50 | 100.00 | 92.17 | Base + SteerVLA |

## Undetermined — exactly one baseline below 95, base policy unknown

> These join the list iff the prior-14k 3-seed mean is below 95. They fall in the "flips between perfect and not" bucket, which `prior14k_per_route.md` does not enumerate numerically.

| Route | SimLingo | SteerVLA |
|---|---:|---:|
| `merger-into-slow-traffic-v2-003` | 100.00 | 46.13 |
| `highway-exit-003` | 54.72 | 100.00 |
| `hazard-at-side-lane-002` | 60.00 | 100.00 |
| `hazard-at-side-lane-005` | 100.00 | 60.00 |
| `signalized-junction-right-turn-003` | 60.00 | 100.00 |
| `vehicle-turning-route-001` | 100.00 | 63.75 |
| `crossing-bicycle-flow-003` | 70.00 | 100.00 |
| `enter-actor-flow-001` | 70.00 | 100.00 |
| `t_-junction-002` | 70.00 | 100.00 |
| `vanilla-signalized-turn-encounter-red-light-002` | 70.00 | 100.00 |
| `vanilla-non-signalized-turn-encounter-stopsign-003` | 100.00 | 80.00 |
| `hard-break-route-005` | 100.00 | 92.69 |

## Grouped by route prefix

> Family means are over the flagged routes only, not all 5 in the family. `n/a` routes are excluded from the base mean.


### `signalized-junction-left-turn` — 5/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `signalized-junction-left-turn-001` | 30.07 | 23.15 | 60.00 | Base + SimLingo + SteerVLA |
| `signalized-junction-left-turn-002` | 41.13 | 36.00 | 60.00 | Base + SimLingo + SteerVLA |
| `signalized-junction-left-turn-003` | 27.04 | 42.00 | 60.00 | Base + SimLingo + SteerVLA |
| `signalized-junction-left-turn-004` | 53.07 | 100.00 | 60.00 | Base + SteerVLA |
| `signalized-junction-left-turn-005` | 38.77 | 60.00 | 60.00 | Base + SimLingo + SteerVLA |
| **family mean** | **38.02** | **52.23** | **60.00** | |

### `yield-to-emergency-vehicle` — 5/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `yield-to-emergency-vehicle-001` | 70.00 | 70.00 | 70.00 | Base + SimLingo + SteerVLA |
| `yield-to-emergency-vehicle-002` | 70.00 | 70.00 | 100.00 | Base + SimLingo |
| `yield-to-emergency-vehicle-003` | 70.00 | 70.00 | 70.00 | Base + SimLingo + SteerVLA |
| `yield-to-emergency-vehicle-004` | 70.00 | 70.00 | 70.00 | Base + SimLingo + SteerVLA |
| `yield-to-emergency-vehicle-005` | 70.00 | 70.00 | 60.00 | Base + SimLingo + SteerVLA |
| **family mean** | **70.00** | **70.00** | **74.00** | |

### `enter-actor-flow` — 4/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `enter-actor-flow-002` | n/a | 32.59 | 42.00 | SimLingo + SteerVLA |
| `enter-actor-flow-003` | 29.55 | 60.00 | 100.00 | Base + SimLingo |
| `enter-actor-flow-004` | n/a | 60.00 | 60.00 | SimLingo + SteerVLA |
| `enter-actor-flow-005` | 28.22 | 100.00 | 46.57 | Base + SteerVLA |
| **family mean** | **28.88** | **63.15** | **62.14** | |

### `non-signalized-junction-right-turn` — 4/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `non-signalized-junction-right-turn-001` | 52.00 | 60.00 | 100.00 | Base + SimLingo |
| `non-signalized-junction-right-turn-003` | 22.61 | 36.00 | 60.00 | Base + SimLingo + SteerVLA |
| `non-signalized-junction-right-turn-004` | 38.31 | 60.00 | 100.00 | Base + SimLingo |
| `non-signalized-junction-right-turn-005` | 57.17 | 60.00 | 36.00 | Base + SimLingo + SteerVLA |
| **family mean** | **42.52** | **54.00** | **74.00** | |

### `non-signalized-junction-left-turn-enter-flow` — 4/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `non-signalized-junction-left-turn-enter-flow-001` | 35.65 | 80.00 | 100.00 | Base + SimLingo |
| `non-signalized-junction-left-turn-enter-flow-002` | 33.59 | 48.00 | 60.00 | Base + SimLingo + SteerVLA |
| `non-signalized-junction-left-turn-enter-flow-004` | 41.28 | 60.00 | 100.00 | Base + SimLingo |
| `non-signalized-junction-left-turn-enter-flow-005` | 38.14 | 60.00 | 100.00 | Base + SimLingo |
| **family mean** | **37.17** | **62.00** | **90.00** | |

### `signalized-junction-left-turn-enter-flow` — 4/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `signalized-junction-left-turn-enter-flow-001` | 60.00 | 60.00 | 60.00 | Base + SimLingo + SteerVLA |
| `signalized-junction-left-turn-enter-flow-003` | 56.50 | 60.00 | 100.00 | Base + SimLingo |
| `signalized-junction-left-turn-enter-flow-004` | 51.33 | 100.00 | 60.00 | Base + SteerVLA |
| `signalized-junction-left-turn-enter-flow-005` | 66.01 | 100.00 | 35.55 | Base + SteerVLA |
| **family mean** | **58.46** | **80.00** | **63.89** | |

### `parked-obstacle` — 4/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `parked-obstacle-001` | 72.73 | 36.00 | 100.00 | Base + SimLingo |
| `parked-obstacle-002` | 77.73 | 60.00 | 100.00 | Base + SimLingo |
| `parked-obstacle-004` | 60.28 | 60.00 | 100.00 | Base + SimLingo |
| `parked-obstacle-005` | 73.87 | 60.00 | 100.00 | Base + SimLingo |
| **family mean** | **71.15** | **54.00** | **100.00** | |

### `non-signalized-junction-left-turn` — 3/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `non-signalized-junction-left-turn-002` | 54.36 | 100.00 | 60.00 | Base + SteerVLA |
| `non-signalized-junction-left-turn-003` | n/a | 42.00 | 60.00 | SimLingo + SteerVLA |
| `non-signalized-junction-left-turn-004` | 39.35 | 100.00 | 31.97 | Base + SteerVLA |
| **family mean** | **46.86** | **80.67** | **50.66** | |

### `accident-two-ways` — 3/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `accident-two-ways-002` | 28.50 | 100.00 | 13.79 | Base + SteerVLA |
| `accident-two-ways-003` | 22.63 | 70.00 | 100.00 | Base + SimLingo |
| `accident-two-ways-005` | 36.02 | 100.00 | 70.00 | Base + SteerVLA |
| **family mean** | **29.05** | **90.00** | **61.26** | |

### `opposite-vehicle-running-red-light` — 3/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `opposite-vehicle-running-red-light-002` | 44.42 | 80.00 | 100.00 | Base + SimLingo |
| `opposite-vehicle-running-red-light-003` | 32.34 | 100.00 | 28.46 | Base + SteerVLA |
| `opposite-vehicle-running-red-light-005` | 100.00 | 32.30 | 25.04 | SimLingo + SteerVLA |
| **family mean** | **58.92** | **70.77** | **51.17** | |

### `construction-obstacle` — 3/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `construction-obstacle-001` | 40.88 | 60.00 | 100.00 | Base + SimLingo |
| `construction-obstacle-003` | 80.78 | 39.00 | 100.00 | Base + SimLingo |
| `construction-obstacle-004` | 29.14 | 60.00 | 100.00 | Base + SimLingo |
| **family mean** | **50.26** | **53.00** | **100.00** | |

### `highway-exit` — 3/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `highway-exit-001` | 75.83 | 76.14 | 76.14 | Base + SimLingo + SteerVLA |
| `highway-exit-002` | 63.20 | 60.00 | 75.87 | Base + SimLingo + SteerVLA |
| `highway-exit-004` | 65.33 | 60.00 | 100.00 | Base + SimLingo |
| **family mean** | **68.12** | **65.38** | **84.00** | |

### `pedestrian-crossing` — 3/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `pedestrian-crossing-001` | 35.20 | 100.00 | 80.00 | Base + SteerVLA |
| `pedestrian-crossing-004` | n/a | 80.00 | 80.00 | SimLingo + SteerVLA |
| `pedestrian-crossing-005` | 71.83 | 70.00 | 100.00 | Base + SimLingo |
| **family mean** | **53.52** | **83.33** | **86.67** | |

### `interurban-actor-flow` — 3/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `interurban-actor-flow-002` | 92.50 | 100.00 | 92.17 | Base + SteerVLA |
| `interurban-actor-flow-003` | 60.00 | 92.77 | 100.00 | Base + SimLingo |
| `interurban-actor-flow-005` | 92.50 | 93.15 | 92.17 | Base + SimLingo + SteerVLA |
| **family mean** | **81.66** | **95.31** | **94.78** | |

### `opposite-vehicle-taking-priority` — 2/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `opposite-vehicle-taking-priority-003` | 44.09 | 60.00 | 31.92 | Base + SimLingo + SteerVLA |
| `opposite-vehicle-taking-priority-004` | 39.54 | 60.00 | 60.00 | Base + SimLingo + SteerVLA |
| **family mean** | **41.82** | **60.00** | **45.96** | |

### `parking-exit` — 2/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `parking-exit-002` | 1.52 | 42.00 | 100.00 | Base + SimLingo |
| `parking-exit-003` | 1.50 | 60.00 | 100.00 | Base + SimLingo |
| **family mean** | **1.51** | **51.00** | **100.00** | |

### `signalized-junction-right-turn` — 2/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `signalized-junction-right-turn-004` | 42.00 | 60.00 | 60.00 | Base + SimLingo + SteerVLA |
| `signalized-junction-right-turn-005` | 51.33 | 60.00 | 60.00 | Base + SimLingo + SteerVLA |
| **family mean** | **46.67** | **60.00** | **60.00** | |

### `static-cut-in` — 2/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `static-cut-in-001` | 52.00 | 33.69 | 100.00 | Base + SimLingo |
| `static-cut-in-004` | 60.00 | 60.00 | 100.00 | Base + SimLingo |
| **family mean** | **56.00** | **46.84** | **100.00** | |

### `vanilla-signalized-turn-encounter-red-light` — 2/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `vanilla-signalized-turn-encounter-red-light-003` | 37.51 | 70.00 | 100.00 | Base + SimLingo |
| `vanilla-signalized-turn-encounter-red-light-004` | 60.67 | 70.00 | 100.00 | Base + SimLingo |
| **family mean** | **49.09** | **70.00** | **100.00** | |

### `accident` — 1/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `accident-005` | 28.36 | 60.00 | 60.00 | Base + SimLingo + SteerVLA |
| **family mean** | **28.36** | **60.00** | **60.00** | |

### `merger-into-slow-traffic-v2` — 1/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `merger-into-slow-traffic-v2-005` | 100.00 | 6.40 | 45.50 | SimLingo + SteerVLA |
| **family mean** | **100.00** | **6.40** | **45.50** | |

### `vehicle-turning-route-pedestrian` — 1/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `vehicle-turning-route-pedestrian-005` | 29.29 | 100.00 | 25.00 | Base + SteerVLA |
| **family mean** | **29.29** | **100.00** | **25.00** | |

### `construction-obstacle-two-ways` — 1/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `construction-obstacle-two-ways-001` | 28.32 | 60.00 | 100.00 | Base + SimLingo |
| **family mean** | **28.32** | **60.00** | **100.00** | |

### `crossing-bicycle-flow` — 1/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `crossing-bicycle-flow-004` | 56.23 | 100.00 | 40.27 | Base + SteerVLA |
| **family mean** | **56.23** | **100.00** | **40.27** | |

### `vanilla-non-signalized-turn-encounter-stopsign` — 1/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `vanilla-non-signalized-turn-encounter-stopsign-005` | 61.79 | 60.00 | 100.00 | Base + SimLingo |
| **family mean** | **61.79** | **60.00** | **100.00** | |

### `vanilla-signalized-turn-encounter-green-light` — 1/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `vanilla-signalized-turn-encounter-green-light-004` | 37.68 | 87.81 | 100.00 | Base + SimLingo |
| **family mean** | **37.68** | **87.81** | **100.00** | |

### `blocked-intersection` — 1/5 flagged

| Route | Base | SimLingo | SteerVLA | Which fail |
|---|---:|---:|---:|---|
| `blocked-intersection-004` | 57.19 | 70.00 | 100.00 | Base + SimLingo |
| **family mean** | **57.19** | **70.00** | **100.00** | |

27 families cover the 69 flagged routes.
