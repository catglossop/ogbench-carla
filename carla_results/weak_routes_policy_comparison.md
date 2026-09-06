# Weak-route eval — base policy vs. SimLingo vs. SteerVLA

*Generated 2026-09-01 14:35 local.*

The **94 routes currently being evaluated** (the weak-route queue), scored against the two
baselines. Base-policy numbers are **prior-14k**, mean of seeds 0/1/2, from `carla_results/prior14k_per_route.md`.
Baselines are the 220-route Bench2Drive runs in `~/carla_results/` (`merged.json` = SimLingo,
`steervla (3).json` = SteerVLA), joined on leaderboard `route_id` via `ogbench.carla.route_registry`.
All 94 routes matched in all three sources. "Poor" = DS < 80.

## Summary


| Bucket                                         | Routes | Reading                                                              |
| ---------------------------------------------- | ------ | -------------------------------------------------------------------- |
| **Hard for everyone** — both baselines poor    | **18** | Genuinely hard scenarios, not base-policy artefacts                  |
| Exactly one baseline poor                      | 34     | Mixed / method-specific                                              |
| **Base-policy-specific** — both baselines fine | **42** | Where our policy uniquely fails (35 have *both* baselines at DS 100) |


So roughly **45%** of the weak routes are weaknesses of the base policy alone, while **19%** defeat all three policies.

## Hard for all three (18)

> Both baselines below 80 DS. These are the routes where a better checkpoint is unlikely to be the whole answer.


| Route                                              | Base (prior-14k) | SimLingo | SteerVLA |
| -------------------------------------------------- | ---------------- | -------- | -------- |
| `signalized-junction-left-turn-001`                | 30.07            | 23.15    | 60       |
| `opposite-vehicle-taking-priority-003`             | 44.09            | 60       | 31.92    |
| `non-signalized-junction-right-turn-003`           | 22.61            | 36       | 60       |
| `non-signalized-junction-right-turn-005`           | 57.17            | 60       | 36       |
| `signalized-junction-left-turn-002`                | 41.13            | 36       | 60       |
| `signalized-junction-left-turn-003`                | 27.04            | 42       | 60       |
| `non-signalized-junction-left-turn-enter-flow-002` | 33.59            | 48       | 60       |
| `opposite-vehicle-taking-priority-004`             | 39.54            | 60       | 60       |
| `signalized-junction-left-turn-005`                | 38.77            | 60       | 60       |
| `signalized-junction-left-turn-enter-flow-001`     | 60               | 60       | 60       |
| `signalized-junction-right-turn-004`               | 42               | 60       | 60       |
| `signalized-junction-right-turn-005`               | 51.33            | 60       | 60       |
| `yield-to-emergency-vehicle-005`                   | 70               | 70       | 60       |
| `highway-exit-002`                                 | 63.2             | 60       | 75.87    |
| `yield-to-emergency-vehicle-001`                   | 70               | 70       | 70       |
| `yield-to-emergency-vehicle-003`                   | 70               | 70       | 70       |
| `yield-to-emergency-vehicle-004`                   | 70               | 70       | 70       |
| `highway-exit-001`                                 | 75.83            | 76.14    | 76.14    |




## Base-policy-specific failures (42)

> Both baselines at or above 80 DS while the base policy struggles — the clearest headroom. Worst base score first.


| Route                                                | Base (prior-14k) | SimLingo | SteerVLA | Gap to best baseline |
| ---------------------------------------------------- | ---------------- | -------- | -------- | -------------------- |
| `vanilla-signalized-turn-encounter-green-light-003`  | 4.6              | 100      | 100      | +95.4                |
| `vanilla-non-signalized-turn-encounter-stopsign-002` | 14.81            | 100      | 100      | +85.19               |
| `parking-exit-004`                                   | 20.39            | 100      | 100      | +79.61               |
| `vehicle-turning-route-005`                          | 21.21            | 100      | 100      | +78.79               |
| `construction-obstacle-two-ways-003`                 | 23.08            | 100      | 100      | +76.92               |
| `vanilla-signalized-turn-encounter-red-light-005`    | 23.59            | 100      | 100      | +76.41               |
| `vanilla-signalized-turn-encounter-green-light-002`  | 27.14            | 100      | 100      | +72.86               |
| `construction-obstacle-two-ways-002`                 | 30.54            | 100      | 100      | +69.46               |
| `vanilla-non-signalized-turn-encounter-stopsign-001` | 32.35            | 100      | 100      | +67.65               |
| `construction-obstacle-two-ways-005`                 | 33.36            | 100      | 100      | +66.64               |
| `pedestrian-crossing-001`                            | 35.2             | 100      | 80       | +64.8                |
| `non-signalized-junction-left-turn-enter-flow-001`   | 35.65            | 80       | 100      | +64.35               |
| `hard-break-route-002`                               | 35.68            | 100      | 100      | +64.32               |
| `t_-junction-003`                                    | 37.48            | 100      | 100      | +62.52               |
| `vanilla-signalized-turn-encounter-green-light-004`  | 37.68            | 87.81    | 100      | +62.32               |
| `construction-obstacle-002`                          | 37.99            | 100      | 100      | +62.01               |
| `hard-break-route-003`                               | 42.9             | 100      | 100      | +57.1                |
| `vanilla-signalized-turn-encounter-green-light-001`  | 42.95            | 100      | 100      | +57.05               |
| `opposite-vehicle-running-red-light-002`             | 44.42            | 80       | 100      | +55.58               |
| `vehicle-turning-route-pedestrian-003`               | 45.42            | 100      | 100      | +54.58               |
| `non-signalized-junction-left-turn-001`              | 46.09            | 100      | 100      | +53.91               |
| `parked-obstacle-two-ways-003`                       | 47.38            | 100      | 100      | +52.62               |
| `t_-junction-005`                                    | 48.95            | 100      | 100      | +51.05               |
| `parked-obstacle-two-ways-005`                       | 50.06            | 100      | 100      | +49.94               |
| `opposite-vehicle-taking-priority-002`               | 51.7             | 100      | 100      | +48.3                |
| `highway-cut-in-001`                                 | 60               | 100      | 100      | +40                  |
| `interurban-actor-flow-003`                          | 60               | 92.77    | 100      | +40                  |
| `non-signalized-junction-left-turn-005`              | 60               | 100      | 100      | +40                  |
| `non-signalized-junction-left-turn-enter-flow-003`   | 60               | 100      | 100      | +40                  |
| `parked-obstacle-two-ways-002`                       | 60               | 100      | 100      | +40                  |
| `vehicle-turning-route-003`                          | 60               | 100      | 100      | +40                  |
| `vanilla-non-signalized-turn-005`                    | 62.21            | 100      | 100      | +37.79               |
| `vehicle-turning-route-002`                          | 65               | 100      | 100      | +35                  |
| `crossing-bicycle-flow-002`                          | 70               | 100      | 100      | +30                  |
| `signalized-junction-left-turn-enter-flow-002`       | 70               | 100      | 100      | +30                  |
| `vehicle-turning-route-pedestrian-004`               | 70.99            | 100      | 100      | +29.01               |
| `parked-obstacle-003`                                | 73.5             | 100      | 100      | +26.5                |
| `hazard-at-side-lane-003`                            | 75.44            | 100      | 100      | +24.56               |
| `vehicle-turning-route-pedestrian-002`               | 76.76            | 100      | 100      | +23.24               |
| `hazard-at-side-lane-004`                            | 79.39            | 100      | 100      | +20.61               |
| `interurban-actor-flow-002`                          | 92.5             | 100      | 92.17    | +7.5                 |
| `interurban-actor-flow-005`                          | 92.5             | 93.15    | 92.17    | +0.65                |




## By scenario family

> Mean DS across the weak routes in each family. `gap` = mean baseline minus base policy; a large gap means the base policy is the outlier.


| Family                                           | n   | Base  | SimLingo | SteerVLA | gap    |
| ------------------------------------------------ | --- | ----- | -------- | -------- | ------ |
| `signalized-junction-left-turn`                  | 5   | 38.02 | 52.23    | 60       | +18.1  |
| `signalized-junction-right-turn`                 | 2   | 46.67 | 60       | 60       | +13.33 |
| `non-signalized-junction-right-turn`             | 4   | 42.52 | 54       | 74       | +21.48 |
| `opposite-vehicle-taking-priority`               | 3   | 45.11 | 73.33    | 63.97    | +23.54 |
| `yield-to-emergency-vehicle`                     | 5   | 70    | 70       | 74       | +2     |
| `static-cut-in`                                  | 2   | 56    | 46.84    | 100      | +17.42 |
| `highway-exit`                                   | 3   | 68.12 | 65.38    | 84       | +6.57  |
| `enter-actor-flow`                               | 2   | 28.88 | 80       | 73.28    | +47.76 |
| `opposite-vehicle-running-red-light`             | 2   | 38.38 | 90       | 64.23    | +38.73 |
| `signalized-junction-left-turn-enter-flow`       | 5   | 60.77 | 84       | 71.11    | +16.79 |
| `non-signalized-junction-left-turn-enter-flow`   | 5   | 41.73 | 69.6     | 92       | +39.07 |
| `parked-obstacle`                                | 5   | 71.62 | 63.2     | 100      | +9.98  |
| `construction-obstacle`                          | 4   | 47.2  | 64.75    | 100      | +35.18 |
| `parking-exit`                                   | 3   | 7.8   | 67.33    | 100      | +75.86 |
| `blocked-intersection`                           | 1   | 57.19 | 70       | 100      | +27.81 |
| `crossing-bicycle-flow`                          | 2   | 63.12 | 100      | 70.14    | +21.95 |
| `non-signalized-junction-left-turn`              | 4   | 49.95 | 100      | 72.99    | +36.55 |
| `pedestrian-crossing`                            | 2   | 53.52 | 85       | 90       | +33.98 |
| `vanilla-signalized-turn-encounter-red-light`    | 3   | 40.59 | 80       | 100      | +49.41 |
| `vehicle-turning-route-pedestrian`               | 4   | 55.62 | 100      | 81.25    | +35.01 |
| `vanilla-non-signalized-turn-encounter-stopsign` | 3   | 36.32 | 86.67    | 100      | +57.02 |
| `construction-obstacle-two-ways`                 | 4   | 28.83 | 90       | 100      | +66.17 |
| `interurban-actor-flow`                          | 3   | 81.66 | 95.31    | 94.78    | +13.38 |
| `vanilla-signalized-turn-encounter-green-light`  | 4   | 28.09 | 96.95    | 100      | +70.38 |
| `hard-break-route`                               | 2   | 39.29 | 100      | 100      | +60.71 |
| `hazard-at-side-lane`                            | 2   | 77.42 | 100      | 100      | +22.58 |
| `highway-cut-in`                                 | 1   | 60    | 100      | 100      | +40    |
| `parked-obstacle-two-ways`                       | 3   | 52.48 | 100      | 100      | +47.52 |
| `t_-junction`                                    | 2   | 43.22 | 100      | 100      | +56.78 |
| `vanilla-non-signalized-turn`                    | 1   | 62.21 | 100      | 100      | +37.79 |
| `vehicle-turning-route`                          | 3   | 48.74 | 100      | 100      | +51.26 |




## All 94 routes


| Route                                                | Base mean | seed 0 | seed 1 | seed 2 | SimLingo | SteerVLA | Bucket        |
| ---------------------------------------------------- | --------- | ------ | ------ | ------ | -------- | -------- | ------------- |
| `blocked-intersection-004`                           | 57.19     | 100    | 42     | 29.57  | 70       | 100      | one poor      |
| `construction-obstacle-001`                          | 40.88     | 28.82  | 65     | 28.82  | 60       | 100      | one poor      |
| `construction-obstacle-002`                          | 37.99     | 12.96  | 65     | 36     | 100      | 100      | base-specific |
| `construction-obstacle-003`                          | 80.78     | 100    | 100    | 42.33  | 39       | 100      | one poor      |
| `construction-obstacle-004`                          | 29.14     | 29.39  | 28.63  | 29.39  | 60       | 100      | one poor      |
| `construction-obstacle-two-ways-001`                 | 28.32     | 28.82  | 28.82  | 27.31  | 60       | 100      | one poor      |
| `construction-obstacle-two-ways-002`                 | 30.54     | 33.14  | 25.35  | 33.14  | 100      | 100      | base-specific |
| `construction-obstacle-two-ways-003`                 | 23.08     | 28.82  | 28.82  | 11.6   | 100      | 100      | base-specific |
| `construction-obstacle-two-ways-005`                 | 33.36     | 34.88  | 33.36  | 31.85  | 100      | 100      | base-specific |
| `crossing-bicycle-flow-002`                          | 70        | 70     | 70     | 70     | 100      | 100      | base-specific |
| `crossing-bicycle-flow-004`                          | 56.23     | 100    | 18.51  | 50.18  | 100      | 40.27    | one poor      |
| `enter-actor-flow-003`                               | 29.55     | 31.52  | 36     | 21.12  | 60       | 100      | one poor      |
| `enter-actor-flow-005`                               | 28.22     | 28.63  | 27.39  | 28.63  | 100      | 46.57    | one poor      |
| `hard-break-route-002`                               | 35.68     | 3.07   | 100    | 3.98   | 100      | 100      | base-specific |
| `hard-break-route-003`                               | 42.9      | 7.59   | 100    | 21.11  | 100      | 100      | base-specific |
| `hazard-at-side-lane-003`                            | 75.44     | 100    | 33.44  | 92.88  | 100      | 100      | base-specific |
| `hazard-at-side-lane-004`                            | 79.39     | 100    | 100    | 38.18  | 100      | 100      | base-specific |
| `highway-cut-in-001`                                 | 60        | 60     | 60     | 60     | 100      | 100      | base-specific |
| `highway-exit-001`                                   | 75.83     | 79.77  | 76.14  | 71.59  | 76.14    | 76.14    | hard for all  |
| `highway-exit-002`                                   | 63.2      | 77.88  | 33.85  | 77.88  | 60       | 75.87    | hard for all  |
| `highway-exit-004`                                   | 65.33     | 60     | 100    | 36     | 60       | 100      | one poor      |
| `interurban-actor-flow-002`                          | 92.5      | 92.17  | 92.17  | 93.15  | 100      | 92.17    | base-specific |
| `interurban-actor-flow-003`                          | 60        | 60     | 60     | 60     | 92.77    | 100      | base-specific |
| `interurban-actor-flow-005`                          | 92.5      | 92.17  | 93.15  | 92.17  | 93.15    | 92.17    | base-specific |
| `non-signalized-junction-left-turn-001`              | 46.09     | 60     | 60     | 18.27  | 100      | 100      | base-specific |
| `non-signalized-junction-left-turn-002`              | 54.36     | 60     | 50.73  | 52.35  | 100      | 60       | one poor      |
| `non-signalized-junction-left-turn-004`              | 39.35     | 32.8   | 31.97  | 53.29  | 100      | 31.97    | one poor      |
| `non-signalized-junction-left-turn-005`              | 60        | 60     | 60     | 60     | 100      | 100      | base-specific |
| `non-signalized-junction-left-turn-enter-flow-001`   | 35.65     | 48     | 10.95  | 48     | 80       | 100      | base-specific |
| `non-signalized-junction-left-turn-enter-flow-002`   | 33.59     | 31.94  | 26.26  | 42.58  | 48       | 60       | hard for all  |
| `non-signalized-junction-left-turn-enter-flow-003`   | 60        | 60     | 60     | 60     | 100      | 100      | base-specific |
| `non-signalized-junction-left-turn-enter-flow-004`   | 41.28     | 27.37  | 48.94  | 47.54  | 60       | 100      | one poor      |
| `non-signalized-junction-left-turn-enter-flow-005`   | 38.14     | 70     | 22.21  | 22.21  | 60       | 100      | one poor      |
| `non-signalized-junction-right-turn-001`             | 52        | 36     | 60     | 60     | 60       | 100      | one poor      |
| `non-signalized-junction-right-turn-003`             | 22.61     | 36     | 7.36   | 24.48  | 36       | 60       | hard for all  |
| `non-signalized-junction-right-turn-004`             | 38.31     | 60     | 34.52  | 20.41  | 60       | 100      | one poor      |
| `non-signalized-junction-right-turn-005`             | 57.17     | 36     | 35.5   | 100    | 60       | 36       | hard for all  |
| `opposite-vehicle-running-red-light-002`             | 44.42     | 24.45  | 28.8   | 80     | 80       | 100      | base-specific |
| `opposite-vehicle-running-red-light-003`             | 32.34     | 32.74  | 32.74  | 31.55  | 100      | 28.46    | one poor      |
| `opposite-vehicle-taking-priority-002`               | 51.7      | 60     | 35.1   | 60     | 100      | 100      | base-specific |
| `opposite-vehicle-taking-priority-003`               | 44.09     | 36.28  | 36     | 60     | 60       | 31.92    | hard for all  |
| `opposite-vehicle-taking-priority-004`               | 39.54     | 11.32  | 47.31  | 60     | 60       | 60       | hard for all  |
| `parked-obstacle-001`                                | 72.73     | 100    | 18.2   | 100    | 36       | 100      | one poor      |
| `parked-obstacle-002`                                | 77.73     | 33.18  | 100    | 100    | 60       | 100      | one poor      |
| `parked-obstacle-003`                                | 73.5      | 20.49  | 100    | 100    | 100      | 100      | base-specific |
| `parked-obstacle-004`                                | 60.28     | 20.84  | 100    | 60     | 60       | 100      | one poor      |
| `parked-obstacle-005`                                | 73.87     | 100    | 100    | 21.6   | 60       | 100      | one poor      |
| `parked-obstacle-two-ways-002`                       | 60        | 60     | 60     | 60     | 100      | 100      | base-specific |
| `parked-obstacle-two-ways-003`                       | 47.38     | 65     | 39     | 38.14  | 100      | 100      | base-specific |
| `parked-obstacle-two-ways-005`                       | 50.06     | 100    | 36     | 14.19  | 100      | 100      | base-specific |
| `parking-exit-002`                                   | 1.52      | 1.52   | 1.52   | 1.52   | 42       | 100      | one poor      |
| `parking-exit-003`                                   | 1.5       | 1.5    | 1.5    | 1.5    | 60       | 100      | one poor      |
| `parking-exit-004`                                   | 20.39     | 14.04  | 45.47  | 1.66   | 100      | 100      | base-specific |
| `pedestrian-crossing-001`                            | 35.2      | 48     | 28.8   | 28.8   | 100      | 80       | base-specific |
| `pedestrian-crossing-005`                            | 71.83     | 45.5   | 70     | 100    | 70       | 100      | one poor      |
| `signalized-junction-left-turn-001`                  | 30.07     | 30.37  | 30.37  | 29.47  | 23.15    | 60       | hard for all  |
| `signalized-junction-left-turn-002`                  | 41.13     | 29.6   | 60     | 33.8   | 36       | 60       | hard for all  |
| `signalized-junction-left-turn-003`                  | 27.04     | 21.62  | 35.57  | 23.92  | 42       | 60       | hard for all  |
| `signalized-junction-left-turn-004`                  | 53.07     | 100    | 36.34  | 22.88  | 100      | 60       | one poor      |
| `signalized-junction-left-turn-005`                  | 38.77     | 27.55  | 28.77  | 60     | 60       | 60       | hard for all  |
| `signalized-junction-left-turn-enter-flow-001`       | 60        | 60     | 60     | 60     | 60       | 60       | hard for all  |
| `signalized-junction-left-turn-enter-flow-002`       | 70        | 70     | 70     | 70     | 100      | 100      | base-specific |
| `signalized-junction-left-turn-enter-flow-003`       | 56.5      | 36     | 33.49  | 100    | 60       | 100      | one poor      |
| `signalized-junction-left-turn-enter-flow-004`       | 51.33     | 42     | 70     | 42     | 100      | 60       | one poor      |
| `signalized-junction-left-turn-enter-flow-005`       | 66.01     | 62.02  | 70     | 66.01  | 100      | 35.55    | one poor      |
| `signalized-junction-right-turn-004`                 | 42        | 42     | 42     | 42     | 60       | 60       | hard for all  |
| `signalized-junction-right-turn-005`                 | 51.33     | 42     | 70     | 42     | 60       | 60       | hard for all  |
| `static-cut-in-001`                                  | 52        | 60     | 60     | 36     | 33.69    | 100      | one poor      |
| `static-cut-in-004`                                  | 60        | 60     | 60     | 60     | 60       | 100      | one poor      |
| `t_-junction-003`                                    | 37.48     | 37.48  | 37.48  | 37.48  | 100      | 100      | base-specific |
| `t_-junction-005`                                    | 48.95     | 25.51  | 96.25  | 25.09  | 100      | 100      | base-specific |
| `vanilla-non-signalized-turn-005`                    | 62.21     | 60     | 26.63  | 100    | 100      | 100      | base-specific |
| `vanilla-non-signalized-turn-encounter-stopsign-001` | 32.35     | 47.35  | 13.71  | 36     | 100      | 100      | base-specific |
| `vanilla-non-signalized-turn-encounter-stopsign-002` | 14.81     | 8.55   | 18.46  | 17.41  | 100      | 100      | base-specific |
| `vanilla-non-signalized-turn-encounter-stopsign-005` | 61.79     | 48     | 77.36  | 60     | 60       | 100      | one poor      |
| `vanilla-signalized-turn-encounter-green-light-001`  | 42.95     | 60     | 60     | 8.86   | 100      | 100      | base-specific |
| `vanilla-signalized-turn-encounter-green-light-002`  | 27.14     | 60     | 8.29   | 13.14  | 100      | 100      | base-specific |
| `vanilla-signalized-turn-encounter-green-light-003`  | 4.6       | 6.85   | 3.47   | 3.47   | 100      | 100      | base-specific |
| `vanilla-signalized-turn-encounter-green-light-004`  | 37.68     | 9.22   | 94.59  | 9.22   | 87.81    | 100      | base-specific |
| `vanilla-signalized-turn-encounter-red-light-003`    | 37.51     | 6.27   | 100    | 6.27   | 70       | 100      | one poor      |
| `vanilla-signalized-turn-encounter-red-light-004`    | 60.67     | 70     | 70     | 42     | 70       | 100      | one poor      |
| `vanilla-signalized-turn-encounter-red-light-005`    | 23.59     | 31.84  | 7.42   | 31.51  | 100      | 100      | base-specific |
| `vehicle-turning-route-002`                          | 65        | 60     | 100    | 35     | 100      | 100      | base-specific |
| `vehicle-turning-route-003`                          | 60        | 60     | 60     | 60     | 100      | 100      | base-specific |
| `vehicle-turning-route-005`                          | 21.21     | 15.19  | 19.63  | 28.8   | 100      | 100      | base-specific |
| `vehicle-turning-route-pedestrian-002`               | 76.76     | 100    | 100    | 30.28  | 100      | 100      | base-specific |
| `vehicle-turning-route-pedestrian-003`               | 45.42     | 15.01  | 21.25  | 100    | 100      | 100      | base-specific |
| `vehicle-turning-route-pedestrian-004`               | 70.99     | 100    | 12.96  | 100    | 100      | 100      | base-specific |
| `vehicle-turning-route-pedestrian-005`               | 29.29     | 12.78  | 38.23  | 36.87  | 100      | 25       | one poor      |
| `yield-to-emergency-vehicle-001`                     | 70        | 70     | 70     | 70     | 70       | 70       | hard for all  |
| `yield-to-emergency-vehicle-002`                     | 70        | 70     | 70     | 70     | 70       | 100      | one poor      |
| `yield-to-emergency-vehicle-003`                     | 70        | 70     | 70     | 70     | 70       | 70       | hard for all  |
| `yield-to-emergency-vehicle-004`                     | 70        | 70     | 70     | 70     | 70       | 70       | hard for all  |
| `yield-to-emergency-vehicle-005`                     | 70        | 70     | 70     | 70     | 70       | 60       | hard for all  |




## Caveats

- **Baseline scores cluster on penalty multiples.** Across the 188 baseline scores here, DS=100 appears 110x, DS=60 38x and DS=70 12x. A 60 or 70 is usually route completion at 100% multiplied by a fixed infraction penalty, not partial progress — so treat them as *one specific infraction*, not general degradation.
- **Base numbers are 3-seed means; baselines are single runs.** `prior14k_per_route.md` shows seed spread up to 97 DS on these routes, so a single-run baseline score carries the same uncertainty and is not directly comparable at fine granularity. Bucket boundaries near DS 80 are therefore soft.
- **The** `success rate` **field inside** `steervla (3).json` **(73.18%) is stale** and disagrees with that file's own per-route records (77.27% by DS=100); `~/carla_results/comparison.md` notes the same. Per-route `score_composed` values used here are unaffected.
- `steervla (3).json` is not strict-JSON valid (raw control character ~byte 525306); parsed with `strict=False`.

