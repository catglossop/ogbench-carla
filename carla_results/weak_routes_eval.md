# Weak-route evaluation — prior-14k, one episode per route

_Updated 2026-09-01 12:04:45 local._

Inference-only `main_carla` runs (`--eval-only true`, `--max-episodes 1`, `--online-steps 4000`, video saved even if the route never terminates) over the routes that either **disagree most across seeds** or score **below 100 in every seed** — filtered to mean DS < 95. Five workers, one per render-capable GPU (2/3/4/6/7).

## Progress

| | Routes |
|---|---:|
| Selected | 104 |
| **Completed (video written)** | **14** |
| Started, no video (rerun) | 0 |
| **To run on the next machine** | **94** |

## Results so far

> `seed 0/1/2` are the reference full-benchmark Driving Scores for the same route, for context on why it was selected. `steps` is how far this single episode ran (4000 = hit the cap without terminating).

| Route | seed 0 | seed 1 | seed 2 | spread | steps | video |
|---|---:|---:|---:|---:|---:|---|
| `accident-003` | 100.00 | 19.60 | 60.00 | 80.40 | 406 | `ep0001_incomplete.mp4` |
| `blocked-intersection-004` | 100.00 | 42.00 | 29.57 | 70.42 | 1420 | `ep0001.mp4` |
| `blocked-intersection-001` | 30.26 | 100.00 | 60.00 | 69.74 | 1126 | `ep0001_incomplete.mp4` |
| `blocked-intersection-002` | 34.50 | 89.56 | 100.00 | 65.50 | 520 | `ep0001_incomplete.mp4` |
| `construction-obstacle-003` | 100.00 | 100.00 | 42.33 | 57.67 | 691 | `ep0001_incomplete.mp4` |
| `construction-obstacle-002` | 12.96 | 65.00 | 36.00 | 52.04 | 674 | `ep0001_incomplete.mp4` |
| `accident-two-ways-002` | 15.38 | 56.54 | 13.58 | 42.97 | 264 | `ep0001_incomplete.mp4` |
| `accident-two-ways-004` | 65.00 | 77.70 | 39.64 | 38.06 | 2321 | `ep0001_incomplete.mp4` |
| `construction-obstacle-001` | 28.82 | 65.00 | 28.82 | 36.18 | 1584 | `ep0001.mp4` |
| `accident-two-ways-001` | 34.64 | 65.00 | 39.00 | 30.36 | 296 | `ep0001_incomplete.mp4` |
| `accident-two-ways-005` | 27.40 | 38.40 | 42.25 | 14.85 | 567 | `ep0001_incomplete.mp4` |
| `accident-two-ways-003` | 24.80 | 26.59 | 16.49 | 10.10 | 292 | `ep0001_incomplete.mp4` |
| `accident-005` | 24.85 | 30.11 | 30.11 | 5.26 | 4000 | `ep0001_incomplete.mp4` |
| `accident-001` | 49.48 | 50.62 | 49.48 | 1.14 | 4000 | `ep0001_incomplete.mp4` |

### To run next (94) — `.run_carla/jobs/weak_queue_RESUME.txt`

`blocked-intersection-004` · `construction-obstacle-001` · `construction-obstacle-002` · `construction-obstacle-003` · `construction-obstacle-004` · `construction-obstacle-two-ways-001` · `construction-obstacle-two-ways-002` · `construction-obstacle-two-ways-003` · `construction-obstacle-two-ways-005` · `crossing-bicycle-flow-002` · `crossing-bicycle-flow-004` · `enter-actor-flow-003` · `enter-actor-flow-005` · `hard-break-route-002` · `hard-break-route-003` · `hazard-at-side-lane-003` · `hazard-at-side-lane-004` · `highway-cut-in-001` · `highway-exit-001` · `highway-exit-002` · `highway-exit-004` · `interurban-actor-flow-002` · `interurban-actor-flow-003` · `interurban-actor-flow-005` · `non-signalized-junction-left-turn-001` · `non-signalized-junction-left-turn-002` · `non-signalized-junction-left-turn-004` · `non-signalized-junction-left-turn-005` · `non-signalized-junction-left-turn-enter-flow-001` · `non-signalized-junction-left-turn-enter-flow-002` · `non-signalized-junction-left-turn-enter-flow-003` · `non-signalized-junction-left-turn-enter-flow-004` · `non-signalized-junction-left-turn-enter-flow-005` · `non-signalized-junction-right-turn-001` · `non-signalized-junction-right-turn-003` · `non-signalized-junction-right-turn-004` · `non-signalized-junction-right-turn-005` · `opposite-vehicle-running-red-light-002` · `opposite-vehicle-running-red-light-003` · `opposite-vehicle-taking-priority-002` …

---

> Videos: `/raid/users/cglossop/carla_exps/OGBench-CARLA/weak_routes/<exp>/videos/epNNNN.mp4`
>
> Regenerate: `.venv/bin/python .run_carla/gen_weak_routes_report.py`
