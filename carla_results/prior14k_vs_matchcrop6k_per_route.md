# prior-14k (3 seeds) vs matched-crop ll_heavy 6000 — Bench2Drive, all 220 routes

| | |
|---|---|
| prior-14k | `gs://cat-logs/pi05_steervla_cot_simplified_reasoning_commentary/pi05_steervla_cot_simplfied_reasoning_commentary_0823/pi05_steervla_cot_simplfied_reasoning_commentary_0823_20260823_154520/14000` — seeds 0, 1, 2 |
| matched-crop 6000 | `gs://cat-logs/pi05_steervla_cot_simplified_reasoning_ll_heavy/ll_heavy_unnormed_matchcrop/ll_heavy_unnormed_matchcrop_20260904_152800/6000` — seed 0 |
| Common settings | frozen policy, `actions_per_model_query=3`, `actions_per_cot=5`, greedy CoT, all 220 Bench2Drive routes |

## ⚠ Read this before comparing the columns

**1. The benchmark's noise floor is larger than most differences you will want to claim.** The three prior-14k seeds differ only in `env.reset(seed)`, yet their overall DS spans **70.46 / 74.20 / 70.86 — a 3.74 DS spread**, and individual routes swing by up to **97 DS**. Only ~40% of the benchmark is fully deterministic. The matched-crop column is a **single seed**, so treat any gap under ~4 DS overall as indistinguishable from seed luck.

**2. Two things changed at once**, so this is not a clean ablation of the crop: the matched-crop run was trained with `SIMLINGO_FRAMING_CROP` applied to every corpus *and* is evaluated with `proprio_norm=False` (matching its TrainConfig), whereas prior-14k ran with the inherited `proprio_norm=True`. It is also a different training run entirely (unnormalized ll_heavy vs the commentary run), not just a later checkpoint of the same one.

**3. Per-route prior-14k coverage is partial.** `prior14k_per_route.md` tabulates 173 of 220 routes; the raw seed runs are not on this box, so the remaining 47 show `n/a` in the per-route table. **The overall prior-14k figures below are unaffected** — they are quoted from `full_leaderboard_prior14k.md`, which aggregated the complete runs.

## Overall

| Run | DS | RC | IP | SR (DS=100) | n |
|---|---:|---:|---:|---:|---:|
| prior-14k, seed 0 | 70.46 | 85.90 | 0.818 | 44.04% | 218 |
| prior-14k, seed 1 | 74.20 | 89.40 | 0.828 | 50.00% | 218 |
| prior-14k, seed 2 | 70.86 | 86.00 | 0.823 | 44.95% | 218 |
| **prior-14k, mean of 3 seeds** | **71.84** | 87.10 | 0.823 | 46.33% | 218 |
| matched-crop 6000, seed 0 | 73.85 | 91.38 | 0.811 | 46.12% | 219 |
| matched-crop 6000, seed 1  _(partial)_ | 41.84 | 79.73 | 0.595 | 12.50% | 8 |
| matched-crop 6000, seed 2 | — | — | — | — | 0 |

> Only seed 0 is complete so far; seeds 1 and 2 are queued/running on GPU 5 (~27 h each). Until at least two seeds finish, the matched-crop column cannot be compared to prior-14k's 3-seed mean — a single seed sits inside the 3.74 DS noise floor.

> Run complete: **219/220 scored**. 1 route(s) never produced a record and are excluded as harness failures, not driving outcomes — `enter-actor-flow-001`. `enter-actor-flow-001` segfaulted CARLA at tick ~1075 on every one of 3 attempts across two slot configurations, so it is not scoreable on this box. For reference the prior-14k seed runs also scored 218/220 each, so the columns are comparable.

> Against prior-14k's 3-seed mean, matched-crop **seed 0 alone** is **+2.01 DS** — inside the 3.74 DS seed spread, hence the extra seeds.

## Per-route Driving Score

| Route | Town | prior s0 | prior s1 | prior s2 | prior mean | prior spread | mc s0 | mc s1 | mc s2 | mc mean | Δ vs prior mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `accident-001` | Town10HD | 49.48 | 50.62 | 49.48 | **49.86** | 1.14 | 50.62 | 51.77 | — | **51.20** | +1.34 |
| `accident-002` | Town03 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | 39.26 | — | **69.63** | -30.37 |
| `accident-003` | Town12 | 100.00 | 19.60 | 60.00 | **59.87** | 80.40 | 21.60 | 28.92 | — | **25.26** | -34.60 |
| `accident-004` | Town12 | n/a | n/a | n/a | n/a | n/a | 16.84 | 28.07 | — | **22.46** | — |
| `accident-005` | Town13 | 24.85 | 30.11 | 30.11 | **28.36** | 5.26 | 36.00 | 21.60 | — | **28.80** | +0.44 |
| `accident-two-ways-001` | Town12 | 34.64 | 65.00 | 39.00 | **46.21** | 30.36 | 11.27 | 100.00 | — | **55.64** | +9.42 |
| `accident-two-ways-002` | Town15 | 15.38 | 56.54 | 13.58 | **28.50** | 42.96 | 20.77 | 29.09 | — | **24.93** | -3.57 |
| `accident-two-ways-003` | Town05 | 24.80 | 26.59 | 16.49 | **22.63** | 10.10 | 19.17 | 36.00 | — | **27.58** | +4.96 |
| `accident-two-ways-004` | Town12 | 65.00 | 77.70 | 39.64 | **60.78** | 38.06 | 57.09 | — | — | **57.09** | -3.69 |
| `accident-two-ways-005` | Town13 | 27.40 | 38.40 | 42.25 | **36.02** | 14.85 | 56.37 | — | — | **56.37** | +20.35 |
| `blocked-intersection-001` | Town12 | 30.26 | 100.00 | 60.00 | **63.42** | 69.74 | 100.00 | — | — | **100.00** | +36.58 |
| `blocked-intersection-002` | Town04 | 34.50 | 89.56 | 100.00 | **74.69** | 65.50 | 6.84 | — | — | **6.84** | -67.85 |
| `blocked-intersection-003` | Town11 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 60.00 | — | — | **60.00** | -40.00 |
| `blocked-intersection-004` | Town12 | 100.00 | 42.00 | 29.57 | **57.19** | 70.43 | 42.00 | — | — | **42.00** | -15.19 |
| `blocked-intersection-005` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 60.00 | — | — | **60.00** | -40.00 |
| `construction-obstacle-001` | Town06 | 28.82 | 65.00 | 28.82 | **40.88** | 36.18 | 21.60 | — | — | **21.60** | -19.28 |
| `construction-obstacle-002` | Town04 | 12.96 | 65.00 | 36.00 | **37.99** | 52.04 | 60.00 | — | — | **60.00** | +22.01 |
| `construction-obstacle-003` | Town04 | 100.00 | 100.00 | 42.33 | **80.78** | 57.67 | 100.00 | — | — | **100.00** | +19.22 |
| `construction-obstacle-004` | Town12 | 29.39 | 28.63 | 29.39 | **29.14** | 0.76 | 36.00 | — | — | **36.00** | +6.86 |
| `construction-obstacle-005` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 54.55 | — | — | **54.55** | -45.45 |
| `construction-obstacle-two-ways-001` | Town12 | 28.82 | 28.82 | 27.31 | **28.32** | 1.51 | 10.39 | — | — | **10.39** | -17.93 |
| `construction-obstacle-two-ways-002` | Town12 | 33.14 | 25.35 | 33.14 | **30.54** | 7.79 | 33.89 | — | — | **33.89** | +3.35 |
| `construction-obstacle-two-ways-003` | Town12 | 28.82 | 28.82 | 11.60 | **23.08** | 17.22 | 28.82 | — | — | **28.82** | +5.74 |
| `construction-obstacle-two-ways-004` | Town11 | n/a | n/a | n/a | n/a | n/a | 58.30 | — | — | **58.30** | — |
| `construction-obstacle-two-ways-005` | Town12 | 34.88 | 33.36 | 31.85 | **33.36** | 3.03 | 33.36 | — | — | **33.36** | -0.00 |
| `control-loss-001` | Town10HD | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `control-loss-002` | Town02 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `control-loss-003` | Town01 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `control-loss-004` | Town11 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `control-loss-005` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `crossing-bicycle-flow-001` | Town12 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `crossing-bicycle-flow-002` | Town12 | 70.00 | 70.00 | 70.00 | **70.00** | 0.00 | 100.00 | — | — | **100.00** | +30.00 |
| `crossing-bicycle-flow-003` | Town12 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `crossing-bicycle-flow-004` | Town12 | 100.00 | 18.51 | 50.18 | **56.23** | 81.49 | 100.00 | — | — | **100.00** | +43.77 |
| `crossing-bicycle-flow-005` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `dynamic-object-crossing-001` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `dynamic-object-crossing-002` | Town01 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `dynamic-object-crossing-003` | Town02 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `dynamic-object-crossing-004` | Town11 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `dynamic-object-crossing-005` | Town15 | n/a | n/a | n/a | n/a | n/a | 50.00 | — | — | **50.00** | — |
| `enter-actor-flow-001` | Town12 | n/a | n/a | n/a | n/a | n/a | 0.00 | — | — | **0.00** | — |
| `enter-actor-flow-002` | Town12 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `enter-actor-flow-003` | Town12 | 31.52 | 36.00 | 21.12 | **29.55** | 14.88 | 45.92 | — | — | **45.92** | +16.37 |
| `enter-actor-flow-004` | Town05 | n/a | n/a | n/a | n/a | n/a | 33.99 | — | — | **33.99** | — |
| `enter-actor-flow-005` | Town13 | 28.63 | 27.39 | 28.63 | **28.22** | 1.24 | 100.00 | — | — | **100.00** | +71.78 |
| `hard-break-route-001` | Town10HD | n/a | n/a | n/a | n/a | n/a | 60.00 | — | — | **60.00** | — |
| `hard-break-route-002` | Town01 | 3.07 | 100.00 | 3.98 | **35.68** | 96.93 | 29.82 | — | — | **29.82** | -5.87 |
| `hard-break-route-003` | Town04 | 7.59 | 100.00 | 21.11 | **42.90** | 92.41 | 60.00 | — | — | **60.00** | +17.10 |
| `hard-break-route-004` | Town03 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 88.58 | — | — | **88.58** | -11.42 |
| `hard-break-route-005` | Town13 | n/a | n/a | n/a | n/a | n/a | 60.00 | — | — | **60.00** | — |
| `hazard-at-side-lane-001` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `hazard-at-side-lane-002` | Town12 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `hazard-at-side-lane-003` | Town04 | 100.00 | 33.44 | 92.88 | **75.44** | 66.56 | 84.47 | — | — | **84.47** | +9.03 |
| `hazard-at-side-lane-004` | Town05 | 100.00 | 100.00 | 38.18 | **79.39** | 61.82 | 100.00 | — | — | **100.00** | +20.61 |
| `hazard-at-side-lane-005` | Town15 | n/a | n/a | n/a | n/a | n/a | 60.00 | — | — | **60.00** | — |
| `hazard-at-side-lane-two-ways-001` | Town07 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `hazard-at-side-lane-two-ways-002` | Town03 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `hazard-at-side-lane-two-ways-003` | Town04 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `hazard-at-side-lane-two-ways-004` | Town05 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `hazard-at-side-lane-two-ways-005` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 60.00 | — | — | **60.00** | -40.00 |
| `highway-cut-in-001` | Town12 | 60.00 | 60.00 | 60.00 | **60.00** | 0.00 | 100.00 | — | — | **100.00** | +40.00 |
| `highway-cut-in-002` | Town12 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `highway-cut-in-003` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `highway-cut-in-004` | Town12 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `highway-cut-in-005` | Town13 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `highway-exit-001` | Town12 | 79.77 | 76.14 | 71.59 | **75.83** | 8.18 | 91.82 | — | — | **91.82** | +15.99 |
| `highway-exit-002` | Town12 | 77.88 | 33.85 | 77.88 | **63.20** | 44.03 | 28.76 | — | — | **28.76** | -34.44 |
| `highway-exit-003` | Town13 | n/a | n/a | n/a | n/a | n/a | 36.00 | — | — | **36.00** | — |
| `highway-exit-004` | Town12 | 60.00 | 100.00 | 36.00 | **65.33** | 64.00 | 60.00 | — | — | **60.00** | -5.33 |
| `highway-exit-005` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 36.00 | — | — | **36.00** | -64.00 |
| `interurban-actor-flow-001` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `interurban-actor-flow-002` | Town12 | 92.17 | 92.17 | 93.15 | **92.50** | 0.98 | 92.17 | — | — | **92.17** | -0.33 |
| `interurban-actor-flow-003` | Town13 | 60.00 | 60.00 | 60.00 | **60.00** | 0.00 | 36.00 | — | — | **36.00** | -24.00 |
| `interurban-actor-flow-004` | Town12 | 97.24 | 97.24 | 96.33 | **96.94** | 0.91 | 97.24 | — | — | **97.24** | +0.30 |
| `interurban-actor-flow-005` | Town12 | 92.17 | 93.15 | 92.17 | **92.50** | 0.98 | 92.17 | — | — | **92.17** | -0.33 |
| `interurban-advanced-actor-flow-001` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `interurban-advanced-actor-flow-002` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `interurban-advanced-actor-flow-003` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `interurban-advanced-actor-flow-004` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `interurban-advanced-actor-flow-005` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `invading-turn-001` | Town12 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `invading-turn-002` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `invading-turn-003` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `invading-turn-004` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `invading-turn-005` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `merger-into-slow-traffic-001` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `merger-into-slow-traffic-002` | Town12 | n/a | n/a | n/a | n/a | n/a | 14.83 | — | — | **14.83** | — |
| `merger-into-slow-traffic-003` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 60.00 | — | — | **60.00** | -40.00 |
| `merger-into-slow-traffic-004` | Town13 | 97.77 | 98.66 | 98.22 | **98.22** | 0.89 | 98.66 | — | — | **98.66** | +0.44 |
| `merger-into-slow-traffic-005` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `merger-into-slow-traffic-v2-001` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 60.00 | — | — | **60.00** | -40.00 |
| `merger-into-slow-traffic-v2-002` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `merger-into-slow-traffic-v2-003` | Town06 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `merger-into-slow-traffic-v2-004` | Town06 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `merger-into-slow-traffic-v2-005` | Town06 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `non-signalized-junction-left-turn-001` | Town12 | 60.00 | 60.00 | 18.27 | **46.09** | 41.73 | 60.00 | — | — | **60.00** | +13.91 |
| `non-signalized-junction-left-turn-002` | Town12 | 60.00 | 50.73 | 52.35 | **54.36** | 9.27 | 52.35 | — | — | **52.35** | -2.01 |
| `non-signalized-junction-left-turn-003` | Town12 | n/a | n/a | n/a | n/a | n/a | 31.92 | — | — | **31.92** | — |
| `non-signalized-junction-left-turn-004` | Town03 | 32.80 | 31.97 | 53.29 | **39.35** | 21.32 | 60.00 | — | — | **60.00** | +20.65 |
| `non-signalized-junction-left-turn-005` | Town12 | 60.00 | 60.00 | 60.00 | **60.00** | 0.00 | 60.00 | — | — | **60.00** | +0.00 |
| `non-signalized-junction-left-turn-enter-flow-001` | Town04 | 48.00 | 10.95 | 48.00 | **35.65** | 37.05 | 80.00 | — | — | **80.00** | +44.35 |
| `non-signalized-junction-left-turn-enter-flow-002` | Town04 | 31.94 | 26.26 | 42.58 | **33.59** | 16.32 | 44.45 | — | — | **44.45** | +10.86 |
| `non-signalized-junction-left-turn-enter-flow-003` | Town12 | 60.00 | 60.00 | 60.00 | **60.00** | 0.00 | 100.00 | — | — | **100.00** | +40.00 |
| `non-signalized-junction-left-turn-enter-flow-004` | Town12 | 27.37 | 48.94 | 47.54 | **41.28** | 21.57 | 60.00 | — | — | **60.00** | +18.72 |
| `non-signalized-junction-left-turn-enter-flow-005` | Town12 | 70.00 | 22.21 | 22.21 | **38.14** | 47.79 | 100.00 | — | — | **100.00** | +61.86 |
| `non-signalized-junction-right-turn-001` | Town12 | 36.00 | 60.00 | 60.00 | **52.00** | 24.00 | 100.00 | — | — | **100.00** | +48.00 |
| `non-signalized-junction-right-turn-002` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `non-signalized-junction-right-turn-003` | Town13 | 36.00 | 7.36 | 24.48 | **22.61** | 28.64 | 36.02 | — | — | **36.02** | +13.40 |
| `non-signalized-junction-right-turn-004` | Town13 | 60.00 | 34.52 | 20.41 | **38.31** | 39.59 | 100.00 | — | — | **100.00** | +61.69 |
| `non-signalized-junction-right-turn-005` | Town13 | 36.00 | 35.50 | 100.00 | **57.17** | 64.50 | 100.00 | — | — | **100.00** | +42.83 |
| `opposite-vehicle-running-red-light-001` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 56.25 | — | — | **56.25** | -43.75 |
| `opposite-vehicle-running-red-light-002` | Town04 | 24.45 | 28.80 | 80.00 | **44.42** | 55.55 | 24.45 | — | — | **24.45** | -19.97 |
| `opposite-vehicle-running-red-light-003` | Town03 | 32.74 | 32.74 | 31.55 | **32.34** | 1.19 | 60.00 | — | — | **60.00** | +27.66 |
| `opposite-vehicle-running-red-light-004` | Town12 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `opposite-vehicle-running-red-light-005` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 25.56 | — | — | **25.56** | -74.44 |
| `opposite-vehicle-taking-priority-001` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `opposite-vehicle-taking-priority-002` | Town12 | 60.00 | 35.10 | 60.00 | **51.70** | 24.90 | 60.00 | — | — | **60.00** | +8.30 |
| `opposite-vehicle-taking-priority-003` | Town12 | 36.28 | 36.00 | 60.00 | **44.09** | 24.00 | 52.70 | — | — | **52.70** | +8.61 |
| `opposite-vehicle-taking-priority-004` | Town12 | 11.32 | 47.31 | 60.00 | **39.54** | 48.68 | 100.00 | — | — | **100.00** | +60.46 |
| `opposite-vehicle-taking-priority-005` | Town13 | n/a | n/a | n/a | n/a | n/a | 80.00 | — | — | **80.00** | — |
| `parked-obstacle-001` | Town12 | 100.00 | 18.20 | 100.00 | **72.73** | 81.80 | 100.00 | — | — | **100.00** | +27.27 |
| `parked-obstacle-002` | Town06 | 33.18 | 100.00 | 100.00 | **77.73** | 66.82 | 100.00 | — | — | **100.00** | +22.27 |
| `parked-obstacle-003` | Town05 | 20.49 | 100.00 | 100.00 | **73.50** | 79.51 | 60.00 | — | — | **60.00** | -13.50 |
| `parked-obstacle-004` | Town15 | 20.84 | 100.00 | 60.00 | **60.28** | 79.16 | 60.00 | — | — | **60.00** | -0.28 |
| `parked-obstacle-005` | Town12 | 100.00 | 100.00 | 21.60 | **73.87** | 78.40 | 60.00 | — | — | **60.00** | -13.87 |
| `parked-obstacle-two-ways-001` | Town11 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `parked-obstacle-two-ways-002` | Town03 | 60.00 | 60.00 | 60.00 | **60.00** | 0.00 | 98.47 | — | — | **98.47** | +38.47 |
| `parked-obstacle-two-ways-003` | Town12 | 65.00 | 39.00 | 38.14 | **47.38** | 26.86 | 100.00 | — | — | **100.00** | +52.62 |
| `parked-obstacle-two-ways-004` | Town12 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `parked-obstacle-two-ways-005` | Town13 | 100.00 | 36.00 | 14.19 | **50.06** | 85.81 | 95.46 | — | — | **95.46** | +45.39 |
| `parking-crossing-pedestrian-001` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `parking-crossing-pedestrian-002` | Town03 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `parking-crossing-pedestrian-003` | Town02 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `parking-crossing-pedestrian-004` | Town13 | n/a | n/a | n/a | n/a | n/a | 36.64 | — | — | **36.64** | — |
| `parking-crossing-pedestrian-005` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 50.00 | — | — | **50.00** | -50.00 |
| `parking-cut-in-001` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 60.00 | — | — | **60.00** | -40.00 |
| `parking-cut-in-002` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 19.64 | — | — | **19.64** | -80.36 |
| `parking-cut-in-003` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 19.64 | — | — | **19.64** | -80.36 |
| `parking-cut-in-004` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 24.56 | — | — | **24.56** | -75.44 |
| `parking-cut-in-005` | Town05 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 60.00 | — | — | **60.00** | -40.00 |
| `parking-exit-001` | Town12 | n/a | n/a | n/a | n/a | n/a | 14.04 | — | — | **14.04** | — |
| `parking-exit-002` | Town03 | 1.52 | 1.52 | 1.52 | **1.52** | 0.00 | 1.52 | — | — | **1.52** | -0.00 |
| `parking-exit-003` | Town03 | 1.50 | 1.50 | 1.50 | **1.50** | 0.00 | 60.00 | — | — | **60.00** | +58.50 |
| `parking-exit-004` | Town13 | 14.04 | 45.47 | 1.66 | **20.39** | 43.81 | 60.00 | — | — | **60.00** | +39.61 |
| `parking-exit-005` | Town13 | n/a | n/a | n/a | n/a | n/a | 87.88 | — | — | **87.88** | — |
| `pedestrian-crossing-001` | Town12 | 48.00 | 28.80 | 28.80 | **35.20** | 19.20 | 60.00 | — | — | **60.00** | +24.80 |
| `pedestrian-crossing-002` | Town07 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `pedestrian-crossing-003` | Town03 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `pedestrian-crossing-004` | Town04 | n/a | n/a | n/a | n/a | n/a | 80.00 | — | — | **80.00** | — |
| `pedestrian-crossing-005` | Town11 | 45.50 | 70.00 | 100.00 | **71.83** | 54.50 | 100.00 | — | — | **100.00** | +28.17 |
| `sequential-lane-change-001` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 60.00 | — | — | **60.00** | -40.00 |
| `sequential-lane-change-002` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `sequential-lane-change-003` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 60.00 | — | — | **60.00** | -40.00 |
| `sequential-lane-change-004` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `sequential-lane-change-005` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 60.00 | — | — | **60.00** | -40.00 |
| `signalized-junction-left-turn-001` | Town12 | 30.37 | 30.37 | 29.47 | **30.07** | 0.90 | 60.00 | — | — | **60.00** | +29.93 |
| `signalized-junction-left-turn-002` | Town12 | 29.60 | 60.00 | 33.80 | **41.13** | 30.40 | 29.60 | — | — | **29.60** | -11.53 |
| `signalized-junction-left-turn-003` | Town12 | 21.62 | 35.57 | 23.92 | **27.04** | 13.95 | 43.99 | — | — | **43.99** | +16.95 |
| `signalized-junction-left-turn-004` | Town12 | 100.00 | 36.34 | 22.88 | **53.07** | 77.12 | 53.03 | — | — | **53.03** | -0.04 |
| `signalized-junction-left-turn-005` | Town12 | 27.55 | 28.77 | 60.00 | **38.77** | 32.45 | 26.68 | — | — | **26.68** | -12.09 |
| `signalized-junction-left-turn-enter-flow-001` | Town04 | 60.00 | 60.00 | 60.00 | **60.00** | 0.00 | 60.00 | — | — | **60.00** | +0.00 |
| `signalized-junction-left-turn-enter-flow-002` | Town15 | 70.00 | 70.00 | 70.00 | **70.00** | 0.00 | 40.55 | — | — | **40.55** | -29.45 |
| `signalized-junction-left-turn-enter-flow-003` | Town12 | 36.00 | 33.49 | 100.00 | **56.50** | 66.51 | 100.00 | — | — | **100.00** | +43.50 |
| `signalized-junction-left-turn-enter-flow-004` | Town12 | 42.00 | 70.00 | 42.00 | **51.33** | 28.00 | 70.00 | — | — | **70.00** | +18.67 |
| `signalized-junction-left-turn-enter-flow-005` | Town12 | 62.02 | 70.00 | 66.01 | **66.01** | 7.98 | 100.00 | — | — | **100.00** | +33.99 |
| `signalized-junction-right-turn-001` | Town12 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `signalized-junction-right-turn-002` | Town12 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `signalized-junction-right-turn-003` | Town05 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `signalized-junction-right-turn-004` | Town05 | 42.00 | 42.00 | 42.00 | **42.00** | 0.00 | 70.00 | — | — | **70.00** | +28.00 |
| `signalized-junction-right-turn-005` | Town15 | 42.00 | 70.00 | 42.00 | **51.33** | 28.00 | 100.00 | — | — | **100.00** | +48.67 |
| `static-cut-in-001` | Town06 | 60.00 | 60.00 | 36.00 | **52.00** | 24.00 | 60.00 | — | — | **60.00** | +8.00 |
| `static-cut-in-002` | Town05 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `static-cut-in-003` | Town15 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 60.00 | — | — | **60.00** | -40.00 |
| `static-cut-in-004` | Town12 | 60.00 | 60.00 | 60.00 | **60.00** | 0.00 | 60.00 | — | — | **60.00** | +0.00 |
| `static-cut-in-005` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `t_-junction-001` | Town07 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 60.00 | — | — | **60.00** | -40.00 |
| `t_-junction-002` | Town01 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `t_-junction-003` | Town02 | 37.48 | 37.48 | 37.48 | **37.48** | 0.00 | 70.00 | — | — | **70.00** | +32.52 |
| `t_-junction-004` | Town04 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `t_-junction-005` | Town10HD | 25.51 | 96.25 | 25.09 | **48.95** | 71.16 | 100.00 | — | — | **100.00** | +51.05 |
| `vanilla-non-signalized-turn-001` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `vanilla-non-signalized-turn-002` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `vanilla-non-signalized-turn-003` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 63.24 | — | — | **63.24** | -36.76 |
| `vanilla-non-signalized-turn-004` | Town12 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 98.51 | — | — | **98.51** | -1.49 |
| `vanilla-non-signalized-turn-005` | Town12 | 60.00 | 26.63 | 100.00 | **62.21** | 73.37 | 100.00 | — | — | **100.00** | +37.79 |
| `vanilla-non-signalized-turn-encounter-stopsign-001` | Town13 | 47.35 | 13.71 | 36.00 | **32.35** | 33.64 | 100.00 | — | — | **100.00** | +67.65 |
| `vanilla-non-signalized-turn-encounter-stopsign-002` | Town13 | 8.55 | 18.46 | 17.41 | **14.81** | 9.91 | 60.00 | — | — | **60.00** | +45.19 |
| `vanilla-non-signalized-turn-encounter-stopsign-003` | Town12 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `vanilla-non-signalized-turn-encounter-stopsign-004` | Town12 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `vanilla-non-signalized-turn-encounter-stopsign-005` | Town12 | 48.00 | 77.36 | 60.00 | **61.79** | 29.36 | 100.00 | — | — | **100.00** | +38.21 |
| `vanilla-signalized-turn-encounter-green-light-001` | Town12 | 60.00 | 60.00 | 8.86 | **42.95** | 51.14 | 100.00 | — | — | **100.00** | +57.05 |
| `vanilla-signalized-turn-encounter-green-light-002` | Town12 | 60.00 | 8.29 | 13.14 | **27.14** | 51.71 | 3.43 | — | — | **3.43** | -23.71 |
| `vanilla-signalized-turn-encounter-green-light-003` | Town12 | 6.85 | 3.47 | 3.47 | **4.60** | 3.38 | 8.53 | — | — | **8.53** | +3.93 |
| `vanilla-signalized-turn-encounter-green-light-004` | Town07 | 9.22 | 94.59 | 9.22 | **37.68** | 85.37 | 46.28 | — | — | **46.28** | +8.61 |
| `vanilla-signalized-turn-encounter-green-light-005` | Town07 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `vanilla-signalized-turn-encounter-red-light-001` | Town12 | n/a | n/a | n/a | n/a | n/a | 44.01 | — | — | **44.01** | — |
| `vanilla-signalized-turn-encounter-red-light-002` | Town12 | n/a | n/a | n/a | n/a | n/a | 70.00 | — | — | **70.00** | — |
| `vanilla-signalized-turn-encounter-red-light-003` | Town13 | 6.27 | 100.00 | 6.27 | **37.51** | 93.73 | 100.00 | — | — | **100.00** | +62.49 |
| `vanilla-signalized-turn-encounter-red-light-004` | Town13 | 70.00 | 70.00 | 42.00 | **60.67** | 28.00 | 100.00 | — | — | **100.00** | +39.33 |
| `vanilla-signalized-turn-encounter-red-light-005` | Town13 | 31.84 | 7.42 | 31.51 | **23.59** | 24.42 | 100.00 | — | — | **100.00** | +76.41 |
| `vehicle-opens-door-two-ways-001` | Town11 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `vehicle-opens-door-two-ways-002` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 98.49 | — | — | **98.49** | -1.51 |
| `vehicle-opens-door-two-ways-003` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 33.27 | — | — | **33.27** | -66.73 |
| `vehicle-opens-door-two-ways-004` | Town13 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `vehicle-opens-door-two-ways-005` | Town13 | 100.00 | 100.00 | 100.00 | **100.00** | 0.00 | 100.00 | — | — | **100.00** | +0.00 |
| `vehicle-turning-route-001` | Town12 | n/a | n/a | n/a | n/a | n/a | 100.00 | — | — | **100.00** | — |
| `vehicle-turning-route-002` | Town12 | 60.00 | 100.00 | 35.00 | **65.00** | 65.00 | 62.41 | — | — | **62.41** | -2.59 |
| `vehicle-turning-route-003` | Town13 | 60.00 | 60.00 | 60.00 | **60.00** | 0.00 | 100.00 | — | — | **100.00** | +40.00 |
| `vehicle-turning-route-004` | Town13 | n/a | n/a | n/a | n/a | n/a | 29.44 | — | — | **29.44** | — |
| `vehicle-turning-route-005` | Town13 | 15.19 | 19.63 | 28.80 | **21.21** | 13.61 | 100.00 | — | — | **100.00** | +78.79 |
| `vehicle-turning-route-pedestrian-001` | Town12 | n/a | n/a | n/a | n/a | n/a | 30.27 | — | — | **30.27** | — |
| `vehicle-turning-route-pedestrian-002` | Town12 | 100.00 | 100.00 | 30.28 | **76.76** | 69.72 | 100.00 | — | — | **100.00** | +23.24 |
| `vehicle-turning-route-pedestrian-003` | Town12 | 15.01 | 21.25 | 100.00 | **45.42** | 84.99 | 6.52 | — | — | **6.52** | -38.90 |
| `vehicle-turning-route-pedestrian-004` | Town13 | 100.00 | 12.96 | 100.00 | **70.99** | 87.04 | 41.73 | — | — | **41.73** | -29.26 |
| `vehicle-turning-route-pedestrian-005` | Town13 | 12.78 | 38.23 | 36.87 | **29.29** | 25.45 | 100.00 | — | — | **100.00** | +70.71 |
| `yield-to-emergency-vehicle-001` | Town03 | 70.00 | 70.00 | 70.00 | **70.00** | 0.00 | 70.00 | — | — | **70.00** | +0.00 |
| `yield-to-emergency-vehicle-002` | Town13 | 70.00 | 70.00 | 70.00 | **70.00** | 0.00 | 70.00 | — | — | **70.00** | +0.00 |
| `yield-to-emergency-vehicle-003` | Town13 | 70.00 | 70.00 | 70.00 | **70.00** | 0.00 | 70.00 | — | — | **70.00** | +0.00 |
| `yield-to-emergency-vehicle-004` | Town13 | 70.00 | 70.00 | 70.00 | **70.00** | 0.00 | 70.00 | — | — | **70.00** | +0.00 |
| `yield-to-emergency-vehicle-005` | Town13 | 70.00 | 70.00 | 70.00 | **70.00** | 0.00 | 70.00 | — | — | **70.00** | +0.00 |

> `n/a` = not tabulated in the prior-14k markdown (see caveat 3). `—` = that matched-crop seed has not scored the route yet. **`mc mean` averages only the seeds present**, so while seeds 1/2 are still running it mixes 1-seed and 2-seed averages across rows and the `Δ` column is not yet like-for-like. Both become comparable once all three seeds finish.

## Per-route status — matched-crop 6000

| Route | status | DS | RC | IP |
|---|---|---:|---:|---:|
| `accident-001` | Failed - TickRuntime | 50.62 | 50.62 | 1.000 |
| `accident-002` | Completed | 100.00 | 100.00 | 1.000 |
| `accident-003` | Completed | 21.60 | 100.00 | 0.216 |
| `accident-004` | Failed - TickRuntime | 16.84 | 46.79 | 0.360 |
| `accident-005` | Completed | 36.00 | 100.00 | 0.360 |
| `accident-two-ways-001` | Failed - TickRuntime | 11.27 | 49.67 | 0.227 |
| `accident-two-ways-002` | Completed | 20.77 | 100.00 | 0.208 |
| `accident-two-ways-003` | Failed - TickRuntime | 19.17 | 54.74 | 0.350 |
| `accident-two-ways-004` | Failed - TickRuntime | 57.09 | 95.15 | 0.600 |
| `accident-two-ways-005` | Completed | 56.37 | 100.00 | 0.564 |
| `blocked-intersection-001` | Completed | 100.00 | 100.00 | 1.000 |
| `blocked-intersection-002` | Failed - TickRuntime | 6.84 | 6.84 | 1.000 |
| `blocked-intersection-003` | Completed | 60.00 | 100.00 | 0.600 |
| `blocked-intersection-004` | Completed | 42.00 | 100.00 | 0.420 |
| `blocked-intersection-005` | Completed | 60.00 | 100.00 | 0.600 |
| `construction-obstacle-001` | Completed | 21.60 | 100.00 | 0.216 |
| `construction-obstacle-002` | Completed | 60.00 | 100.00 | 0.600 |
| `construction-obstacle-003` | Completed | 100.00 | 100.00 | 1.000 |
| `construction-obstacle-004` | Completed | 36.00 | 100.00 | 0.360 |
| `construction-obstacle-005` | Completed | 54.55 | 100.00 | 0.545 |
| `construction-obstacle-two-ways-001` | Completed | 10.39 | 100.00 | 0.104 |
| `construction-obstacle-two-ways-002` | Failed - TickRuntime | 33.89 | 33.89 | 1.000 |
| `construction-obstacle-two-ways-003` | Failed - TickRuntime | 28.82 | 28.82 | 1.000 |
| `construction-obstacle-two-ways-004` | Completed | 58.30 | 100.00 | 0.583 |
| `construction-obstacle-two-ways-005` | Failed - TickRuntime | 33.36 | 33.36 | 1.000 |
| `control-loss-001` | Completed | 100.00 | 100.00 | 1.000 |
| `control-loss-002` | Completed | 100.00 | 100.00 | 1.000 |
| `control-loss-003` | Completed | 100.00 | 100.00 | 1.000 |
| `control-loss-004` | Completed | 100.00 | 100.00 | 1.000 |
| `control-loss-005` | Completed | 100.00 | 100.00 | 1.000 |
| `crossing-bicycle-flow-001` | Completed | 100.00 | 100.00 | 1.000 |
| `crossing-bicycle-flow-002` | Completed | 100.00 | 100.00 | 1.000 |
| `crossing-bicycle-flow-003` | Completed | 100.00 | 100.00 | 1.000 |
| `crossing-bicycle-flow-004` | Completed | 100.00 | 100.00 | 1.000 |
| `crossing-bicycle-flow-005` | Completed | 100.00 | 100.00 | 1.000 |
| `dynamic-object-crossing-001` | Completed | 100.00 | 100.00 | 1.000 |
| `dynamic-object-crossing-002` | Completed | 100.00 | 100.00 | 1.000 |
| `dynamic-object-crossing-003` | Completed | 100.00 | 100.00 | 1.000 |
| `dynamic-object-crossing-004` | Completed | 100.00 | 100.00 | 1.000 |
| `dynamic-object-crossing-005` | Completed | 50.00 | 100.00 | 0.500 |
| `enter-actor-flow-001` | NoRecord (rc=-9) | 0.00 | 0.00 | 0.000 |
| `enter-actor-flow-002` | Completed | 100.00 | 100.00 | 1.000 |
| `enter-actor-flow-003` | Failed - Agent deviated from the route | 45.92 | 45.92 | 1.000 |
| `enter-actor-flow-004` | Completed | 33.99 | 100.00 | 0.340 |
| `enter-actor-flow-005` | Completed | 100.00 | 100.00 | 1.000 |
| `hard-break-route-001` | Completed | 60.00 | 100.00 | 0.600 |
| `hard-break-route-002` | Completed | 29.82 | 100.00 | 0.298 |
| `hard-break-route-003` | Completed | 60.00 | 100.00 | 0.600 |
| `hard-break-route-004` | Failed | 88.58 | 88.58 | 1.000 |
| `hard-break-route-005` | Completed | 60.00 | 100.00 | 0.600 |
| `hazard-at-side-lane-001` | Completed | 100.00 | 100.00 | 1.000 |
| `hazard-at-side-lane-002` | Completed | 100.00 | 100.00 | 1.000 |
| `hazard-at-side-lane-003` | Failed - TickRuntime | 84.47 | 84.47 | 1.000 |
| `hazard-at-side-lane-004` | Completed | 100.00 | 100.00 | 1.000 |
| `hazard-at-side-lane-005` | Completed | 60.00 | 100.00 | 0.600 |
| `hazard-at-side-lane-two-ways-001` | Completed | 100.00 | 100.00 | 1.000 |
| `hazard-at-side-lane-two-ways-002` | Completed | 100.00 | 100.00 | 1.000 |
| `hazard-at-side-lane-two-ways-003` | Completed | 100.00 | 100.00 | 1.000 |
| `hazard-at-side-lane-two-ways-004` | Completed | 100.00 | 100.00 | 1.000 |
| `hazard-at-side-lane-two-ways-005` | Completed | 60.00 | 100.00 | 0.600 |
| `highway-cut-in-001` | Completed | 100.00 | 100.00 | 1.000 |
| `highway-cut-in-002` | Completed | 100.00 | 100.00 | 1.000 |
| `highway-cut-in-003` | Completed | 100.00 | 100.00 | 1.000 |
| `highway-cut-in-004` | Completed | 100.00 | 100.00 | 1.000 |
| `highway-cut-in-005` | Completed | 100.00 | 100.00 | 1.000 |
| `highway-exit-001` | Completed | 91.82 | 100.00 | 0.918 |
| `highway-exit-002` | Failed - Agent deviated from the route | 28.76 | 79.89 | 0.360 |
| `highway-exit-003` | Completed | 36.00 | 100.00 | 0.360 |
| `highway-exit-004` | Completed | 60.00 | 100.00 | 0.600 |
| `highway-exit-005` | Completed | 36.00 | 100.00 | 0.360 |
| `interurban-actor-flow-001` | Completed | 100.00 | 100.00 | 1.000 |
| `interurban-actor-flow-002` | Failed - Agent deviated from the route | 92.17 | 92.17 | 1.000 |
| `interurban-actor-flow-003` | Completed | 36.00 | 100.00 | 0.360 |
| `interurban-actor-flow-004` | Failed - Agent deviated from the route | 97.24 | 97.24 | 1.000 |
| `interurban-actor-flow-005` | Failed - Agent deviated from the route | 92.17 | 92.17 | 1.000 |
| `interurban-advanced-actor-flow-001` | Completed | 100.00 | 100.00 | 1.000 |
| `interurban-advanced-actor-flow-002` | Completed | 100.00 | 100.00 | 1.000 |
| `interurban-advanced-actor-flow-003` | Completed | 100.00 | 100.00 | 1.000 |
| `interurban-advanced-actor-flow-004` | Completed | 100.00 | 100.00 | 1.000 |
| `interurban-advanced-actor-flow-005` | Completed | 100.00 | 100.00 | 1.000 |
| `invading-turn-001` | Completed | 100.00 | 100.00 | 1.000 |
| `invading-turn-002` | Completed | 100.00 | 100.00 | 1.000 |
| `invading-turn-003` | Completed | 100.00 | 100.00 | 1.000 |
| `invading-turn-004` | Completed | 100.00 | 100.00 | 1.000 |
| `invading-turn-005` | Completed | 100.00 | 100.00 | 1.000 |
| `merger-into-slow-traffic-001` | Completed | 100.00 | 100.00 | 1.000 |
| `merger-into-slow-traffic-002` | Failed - TickRuntime | 14.83 | 76.93 | 0.193 |
| `merger-into-slow-traffic-003` | Completed | 60.00 | 100.00 | 0.600 |
| `merger-into-slow-traffic-004` | Failed - Agent deviated from the route | 98.66 | 98.66 | 1.000 |
| `merger-into-slow-traffic-005` | Completed | 100.00 | 100.00 | 1.000 |
| `merger-into-slow-traffic-v2-001` | Completed | 60.00 | 100.00 | 0.600 |
| `merger-into-slow-traffic-v2-002` | Completed | 100.00 | 100.00 | 1.000 |
| `merger-into-slow-traffic-v2-003` | Completed | 100.00 | 100.00 | 1.000 |
| `merger-into-slow-traffic-v2-004` | Completed | 100.00 | 100.00 | 1.000 |
| `merger-into-slow-traffic-v2-005` | Completed | 100.00 | 100.00 | 1.000 |
| `non-signalized-junction-left-turn-001` | Completed | 60.00 | 100.00 | 0.600 |
| `non-signalized-junction-left-turn-002` | Failed - Agent deviated from the route | 52.35 | 52.35 | 1.000 |
| `non-signalized-junction-left-turn-003` | Failed - Agent deviated from the route | 31.92 | 53.20 | 0.600 |
| `non-signalized-junction-left-turn-004` | Completed | 60.00 | 100.00 | 0.600 |
| `non-signalized-junction-left-turn-005` | Completed | 60.00 | 100.00 | 0.600 |
| `non-signalized-junction-left-turn-enter-flow-001` | Completed | 80.00 | 100.00 | 0.800 |
| `non-signalized-junction-left-turn-enter-flow-002` | Completed | 44.45 | 100.00 | 0.445 |
| `non-signalized-junction-left-turn-enter-flow-003` | Completed | 100.00 | 100.00 | 1.000 |
| `non-signalized-junction-left-turn-enter-flow-004` | Completed | 60.00 | 100.00 | 0.600 |
| `non-signalized-junction-left-turn-enter-flow-005` | Completed | 100.00 | 100.00 | 1.000 |
| `non-signalized-junction-right-turn-001` | Completed | 100.00 | 100.00 | 1.000 |
| `non-signalized-junction-right-turn-002` | Completed | 100.00 | 100.00 | 1.000 |
| `non-signalized-junction-right-turn-003` | Completed | 36.02 | 100.00 | 0.360 |
| `non-signalized-junction-right-turn-004` | Completed | 100.00 | 100.00 | 1.000 |
| `non-signalized-junction-right-turn-005` | Completed | 100.00 | 100.00 | 1.000 |
| `opposite-vehicle-running-red-light-001` | Failed - Agent deviated from the route | 56.25 | 56.25 | 1.000 |
| `opposite-vehicle-running-red-light-002` | Failed - Agent deviated from the route | 24.45 | 50.93 | 0.480 |
| `opposite-vehicle-running-red-light-003` | Completed | 60.00 | 100.00 | 0.600 |
| `opposite-vehicle-running-red-light-004` | Completed | 100.00 | 100.00 | 1.000 |
| `opposite-vehicle-running-red-light-005` | Failed - TickRuntime | 25.56 | 25.56 | 1.000 |
| `opposite-vehicle-taking-priority-001` | Completed | 100.00 | 100.00 | 1.000 |
| `opposite-vehicle-taking-priority-002` | Completed | 60.00 | 100.00 | 0.600 |
| `opposite-vehicle-taking-priority-003` | Failed - Agent deviated from the route | 52.70 | 52.70 | 1.000 |
| `opposite-vehicle-taking-priority-004` | Completed | 100.00 | 100.00 | 1.000 |
| `opposite-vehicle-taking-priority-005` | Completed | 80.00 | 100.00 | 0.800 |
| `parked-obstacle-001` | Completed | 100.00 | 100.00 | 1.000 |
| `parked-obstacle-002` | Completed | 100.00 | 100.00 | 1.000 |
| `parked-obstacle-003` | Completed | 60.00 | 100.00 | 0.600 |
| `parked-obstacle-004` | Completed | 60.00 | 100.00 | 0.600 |
| `parked-obstacle-005` | Completed | 60.00 | 100.00 | 0.600 |
| `parked-obstacle-two-ways-001` | Completed | 100.00 | 100.00 | 1.000 |
| `parked-obstacle-two-ways-002` | Completed | 98.47 | 100.00 | 0.985 |
| `parked-obstacle-two-ways-003` | Completed | 100.00 | 100.00 | 1.000 |
| `parked-obstacle-two-ways-004` | Completed | 100.00 | 100.00 | 1.000 |
| `parked-obstacle-two-ways-005` | Completed | 95.46 | 100.00 | 0.955 |
| `parking-crossing-pedestrian-001` | Completed | 100.00 | 100.00 | 1.000 |
| `parking-crossing-pedestrian-002` | Completed | 100.00 | 100.00 | 1.000 |
| `parking-crossing-pedestrian-003` | Completed | 100.00 | 100.00 | 1.000 |
| `parking-crossing-pedestrian-004` | Completed | 36.64 | 100.00 | 0.366 |
| `parking-crossing-pedestrian-005` | Completed | 50.00 | 100.00 | 0.500 |
| `parking-cut-in-001` | Completed | 60.00 | 100.00 | 0.600 |
| `parking-cut-in-002` | Completed | 19.64 | 100.00 | 0.196 |
| `parking-cut-in-003` | Completed | 19.64 | 100.00 | 0.196 |
| `parking-cut-in-004` | Failed - Agent got blocked | 24.56 | 43.97 | 0.559 |
| `parking-cut-in-005` | Completed | 60.00 | 100.00 | 0.600 |
| `parking-exit-001` | Completed | 14.04 | 100.00 | 0.140 |
| `parking-exit-002` | Failed - TickRuntime | 1.52 | 1.52 | 1.000 |
| `parking-exit-003` | Completed | 60.00 | 100.00 | 0.600 |
| `parking-exit-004` | Completed | 60.00 | 100.00 | 0.600 |
| `parking-exit-005` | Completed | 87.88 | 100.00 | 0.879 |
| `pedestrian-crossing-001` | Completed | 60.00 | 100.00 | 0.600 |
| `pedestrian-crossing-002` | Completed | 100.00 | 100.00 | 1.000 |
| `pedestrian-crossing-003` | Completed | 100.00 | 100.00 | 1.000 |
| `pedestrian-crossing-004` | Completed | 80.00 | 100.00 | 0.800 |
| `pedestrian-crossing-005` | Completed | 100.00 | 100.00 | 1.000 |
| `sequential-lane-change-001` | Completed | 60.00 | 100.00 | 0.600 |
| `sequential-lane-change-002` | Completed | 100.00 | 100.00 | 1.000 |
| `sequential-lane-change-003` | Completed | 60.00 | 100.00 | 0.600 |
| `sequential-lane-change-004` | Completed | 100.00 | 100.00 | 1.000 |
| `sequential-lane-change-005` | Completed | 60.00 | 100.00 | 0.600 |
| `signalized-junction-left-turn-001` | Completed | 60.00 | 100.00 | 0.600 |
| `signalized-junction-left-turn-002` | Failed - TickRuntime | 29.60 | 29.60 | 1.000 |
| `signalized-junction-left-turn-003` | Failed - Agent deviated from the route | 43.99 | 43.99 | 1.000 |
| `signalized-junction-left-turn-004` | Failed - Agent deviated from the route | 53.03 | 53.03 | 1.000 |
| `signalized-junction-left-turn-005` | Failed - TickRuntime | 26.68 | 44.47 | 0.600 |
| `signalized-junction-left-turn-enter-flow-001` | Completed | 60.00 | 100.00 | 0.600 |
| `signalized-junction-left-turn-enter-flow-002` | Failed - Agent deviated from the route | 40.55 | 57.93 | 0.700 |
| `signalized-junction-left-turn-enter-flow-003` | Completed | 100.00 | 100.00 | 1.000 |
| `signalized-junction-left-turn-enter-flow-004` | Completed | 70.00 | 100.00 | 0.700 |
| `signalized-junction-left-turn-enter-flow-005` | Completed | 100.00 | 100.00 | 1.000 |
| `signalized-junction-right-turn-001` | Completed | 100.00 | 100.00 | 1.000 |
| `signalized-junction-right-turn-002` | Completed | 100.00 | 100.00 | 1.000 |
| `signalized-junction-right-turn-003` | Completed | 100.00 | 100.00 | 1.000 |
| `signalized-junction-right-turn-004` | Completed | 70.00 | 100.00 | 0.700 |
| `signalized-junction-right-turn-005` | Completed | 100.00 | 100.00 | 1.000 |
| `static-cut-in-001` | Completed | 60.00 | 100.00 | 0.600 |
| `static-cut-in-002` | Completed | 100.00 | 100.00 | 1.000 |
| `static-cut-in-003` | Completed | 60.00 | 100.00 | 0.600 |
| `static-cut-in-004` | Completed | 60.00 | 100.00 | 0.600 |
| `static-cut-in-005` | Completed | 100.00 | 100.00 | 1.000 |
| `t_-junction-001` | Completed | 60.00 | 100.00 | 0.600 |
| `t_-junction-002` | Completed | 100.00 | 100.00 | 1.000 |
| `t_-junction-003` | Completed | 70.00 | 100.00 | 0.700 |
| `t_-junction-004` | Completed | 100.00 | 100.00 | 1.000 |
| `t_-junction-005` | Completed | 100.00 | 100.00 | 1.000 |
| `vanilla-non-signalized-turn-001` | Completed | 100.00 | 100.00 | 1.000 |
| `vanilla-non-signalized-turn-002` | Completed | 100.00 | 100.00 | 1.000 |
| `vanilla-non-signalized-turn-003` | Completed | 63.24 | 100.00 | 0.632 |
| `vanilla-non-signalized-turn-004` | Completed | 98.51 | 100.00 | 0.985 |
| `vanilla-non-signalized-turn-005` | Completed | 100.00 | 100.00 | 1.000 |
| `vanilla-non-signalized-turn-encounter-stopsign-001` | Completed | 100.00 | 100.00 | 1.000 |
| `vanilla-non-signalized-turn-encounter-stopsign-002` | Completed | 60.00 | 100.00 | 0.600 |
| `vanilla-non-signalized-turn-encounter-stopsign-003` | Completed | 100.00 | 100.00 | 1.000 |
| `vanilla-non-signalized-turn-encounter-stopsign-004` | Completed | 100.00 | 100.00 | 1.000 |
| `vanilla-non-signalized-turn-encounter-stopsign-005` | Completed | 100.00 | 100.00 | 1.000 |
| `vanilla-signalized-turn-encounter-green-light-001` | Completed | 100.00 | 100.00 | 1.000 |
| `vanilla-signalized-turn-encounter-green-light-002` | Failed - TickRuntime | 3.43 | 3.43 | 1.000 |
| `vanilla-signalized-turn-encounter-green-light-003` | Failed - TickRuntime | 8.53 | 8.53 | 1.000 |
| `vanilla-signalized-turn-encounter-green-light-004` | Completed | 46.28 | 100.00 | 0.463 |
| `vanilla-signalized-turn-encounter-green-light-005` | Completed | 100.00 | 100.00 | 1.000 |
| `vanilla-signalized-turn-encounter-red-light-001` | Failed - Agent deviated from the route | 44.01 | 44.01 | 1.000 |
| `vanilla-signalized-turn-encounter-red-light-002` | Completed | 70.00 | 100.00 | 0.700 |
| `vanilla-signalized-turn-encounter-red-light-003` | Completed | 100.00 | 100.00 | 1.000 |
| `vanilla-signalized-turn-encounter-red-light-004` | Completed | 100.00 | 100.00 | 1.000 |
| `vanilla-signalized-turn-encounter-red-light-005` | Completed | 100.00 | 100.00 | 1.000 |
| `vehicle-opens-door-two-ways-001` | Completed | 100.00 | 100.00 | 1.000 |
| `vehicle-opens-door-two-ways-002` | Completed | 98.49 | 100.00 | 0.985 |
| `vehicle-opens-door-two-ways-003` | Completed | 33.27 | 100.00 | 0.333 |
| `vehicle-opens-door-two-ways-004` | Completed | 100.00 | 100.00 | 1.000 |
| `vehicle-opens-door-two-ways-005` | Completed | 100.00 | 100.00 | 1.000 |
| `vehicle-turning-route-001` | Completed | 100.00 | 100.00 | 1.000 |
| `vehicle-turning-route-002` | Failed - TickRuntime | 62.41 | 62.41 | 1.000 |
| `vehicle-turning-route-003` | Completed | 100.00 | 100.00 | 1.000 |
| `vehicle-turning-route-004` | Failed - Agent got blocked | 29.44 | 49.07 | 0.600 |
| `vehicle-turning-route-005` | Completed | 100.00 | 100.00 | 1.000 |
| `vehicle-turning-route-pedestrian-001` | Failed - TickRuntime | 30.27 | 30.27 | 1.000 |
| `vehicle-turning-route-pedestrian-002` | Completed | 100.00 | 100.00 | 1.000 |
| `vehicle-turning-route-pedestrian-003` | Failed - Agent deviated from the route | 6.52 | 50.30 | 0.130 |
| `vehicle-turning-route-pedestrian-004` | Failed - Agent deviated from the route | 41.73 | 41.73 | 1.000 |
| `vehicle-turning-route-pedestrian-005` | Completed | 100.00 | 100.00 | 1.000 |
| `yield-to-emergency-vehicle-001` | Completed | 70.00 | 100.00 | 0.700 |
| `yield-to-emergency-vehicle-002` | Completed | 70.00 | 100.00 | 0.700 |
| `yield-to-emergency-vehicle-003` | Completed | 70.00 | 100.00 | 0.700 |
| `yield-to-emergency-vehicle-004` | Completed | 70.00 | 100.00 | 0.700 |
| `yield-to-emergency-vehicle-005` | Completed | 70.00 | 100.00 | 0.700 |

