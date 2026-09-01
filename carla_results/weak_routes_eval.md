# Weak-route evaluation — prior-14k, one episode per route

_Updated 2026-09-01 11:31:55 local._

Inference-only `main_carla` runs (`--eval-only true`, `--max-episodes 1`, `--online-steps 4000`, video saved even if the route never terminates) over the routes that either **disagree most across seeds** or score **below 100 in every seed** — filtered to mean DS < 95. Five workers, one per render-capable GPU (2/3/4/6/7).

## Progress

| | Routes |
|---|---:|
| Selected | 104 |
| **Completed (video written)** | **0** |
| In flight | 66 |
| Still queued | 99 |

## Results so far

> `seed 0/1/2` are the reference full-benchmark Driving Scores for the same route, for context on why it was selected. `steps` is how far this single episode ran (4000 = hit the cap without terminating).

| Route | seed 0 | seed 1 | seed 2 | spread | steps | video |
|---|---:|---:|---:|---:|---:|---|
| _(none finished yet)_ | | | | | | |

### In flight (66)

`accident-001` · `accident-003` · `accident-005` · `accident-two-ways-001` · `accident-two-ways-002` · `accident-two-ways-003` · `accident-two-ways-004` · `accident-two-ways-005` · `blocked-intersection-001` · `blocked-intersection-002` · `blocked-intersection-004` · `construction-obstacle-001` · `construction-obstacle-002` · `construction-obstacle-003` · `construction-obstacle-004` · `construction-obstacle-two-ways-001` · `construction-obstacle-two-ways-002` · `construction-obstacle-two-ways-003` · `construction-obstacle-two-ways-005` · `crossing-bicycle-flow-002` · `crossing-bicycle-flow-004` · `enter-actor-flow-003` · `enter-actor-flow-005` · `hard-break-route-002` · `hard-break-route-003` · `hazard-at-side-lane-003` · `hazard-at-side-lane-004` · `highway-cut-in-001` · `highway-exit-001` · `highway-exit-002` · `highway-exit-004` · `interurban-actor-flow-002` · `interurban-actor-flow-003` · `interurban-actor-flow-005` · `non-signalized-junction-left-turn-001` · `non-signalized-junction-left-turn-002` · `non-signalized-junction-left-turn-004` · `non-signalized-junction-left-turn-005` · `non-signalized-junction-left-turn-enter-flow-001` · `non-signalized-junction-left-turn-enter-flow-002` · `non-signalized-junction-left-turn-enter-flow-003` · `non-signalized-junction-left-turn-enter-flow-004` · `non-signalized-junction-left-turn-enter-flow-005` · `non-signalized-junction-right-turn-001` · `non-signalized-junction-right-turn-003` · `non-signalized-junction-right-turn-004` · `non-signalized-junction-right-turn-005` · `opposite-vehicle-running-red-light-002` · `opposite-vehicle-running-red-light-003` · `opposite-vehicle-taking-priority-002` · `opposite-vehicle-taking-priority-003` · `opposite-vehicle-taking-priority-004` · `parked-obstacle-001` · `parked-obstacle-002` · `parked-obstacle-003` · `parked-obstacle-004` · `parked-obstacle-005` · `parked-obstacle-two-ways-002` · `parked-obstacle-two-ways-003` · `parked-obstacle-two-ways-005` · `parking-exit-002` · `parking-exit-003` · `parking-exit-004` · `pedestrian-crossing-001` · `pedestrian-crossing-005` · `signalized-junction-left-turn-001`

### Queued (99)

`accident-two-ways-003` · `accident-two-ways-004` · `accident-two-ways-005` · `blocked-intersection-001` · `blocked-intersection-002` · `blocked-intersection-004` · `construction-obstacle-001` · `construction-obstacle-002` · `construction-obstacle-003` · `construction-obstacle-004` · `construction-obstacle-two-ways-001` · `construction-obstacle-two-ways-002` · `construction-obstacle-two-ways-003` · `construction-obstacle-two-ways-005` · `crossing-bicycle-flow-002` · `crossing-bicycle-flow-004` · `enter-actor-flow-003` · `enter-actor-flow-005` · `hard-break-route-002` · `hard-break-route-003` · `hazard-at-side-lane-003` · `hazard-at-side-lane-004` · `highway-cut-in-001` · `highway-exit-001` · `highway-exit-002` · `highway-exit-004` · `interurban-actor-flow-002` · `interurban-actor-flow-003` · `interurban-actor-flow-005` · `non-signalized-junction-left-turn-001` · `non-signalized-junction-left-turn-002` · `non-signalized-junction-left-turn-004` · `non-signalized-junction-left-turn-005` · `non-signalized-junction-left-turn-enter-flow-001` · `non-signalized-junction-left-turn-enter-flow-002` · `non-signalized-junction-left-turn-enter-flow-003` · `non-signalized-junction-left-turn-enter-flow-004` · `non-signalized-junction-left-turn-enter-flow-005` · `non-signalized-junction-right-turn-001` · `non-signalized-junction-right-turn-003` …

---

> Videos: `/raid/users/cglossop/carla_exps/OGBench-CARLA/weak_routes/<exp>/videos/epNNNN.mp4`
>
> Regenerate: `.venv/bin/python .run_carla/gen_weak_routes_report.py`
