# Handover — weak-route evaluation queue

## What this is

Inference-only single-episode `main_carla` runs over 104 Bench2Drive routes selected from the
three-seed sweep of **prior-14k**: the 30 routes where seeds disagree most, plus every route
scoring below 100 DS in all three seeds, filtered to **mean DS < 95**.

Purpose: look at *why* these routes fail, with a saved video per route, rather than only a score.

## Current state

Live status, refreshed every 60 s while the machine is up:

    carla_results/weak_routes_eval.md

Queue file (routes not yet started) — **this is the resume point**:

    .run_carla/jobs/weak_queue.txt

Per-route stdout: `.run_carla/jobs/weak_<route>.log`
Runs + videos: `/raid/users/cglossop/carla_exps/OGBench-CARLA/weak_routes/<exp>/videos/epNNNN.mp4`

## Resuming on another machine

1. Copy `.run_carla/weak_routes.txt` (the full selection) and
   `.run_carla/jobs/weak_queue.txt` (what is left).
2. Make the checkpoint available at the same path, or edit `CKPT` in the runner:
   `/home/cglossop/steervla_pi_ckpts/pi05_cot_simplified_0823_154520/14000`
   (GCS: `gs://cat-logs/pi05_steervla_cot_simplified_reasoning_commentary/pi05_steervla_cot_simplfied_reasoning_commentary_0823/pi05_steervla_cot_simplfied_reasoning_commentary_0823_20260823_154520/14000`)
3. Edit the `GPUS=(...)` line in `.run_carla/weak_route_queue.sh` to the render-capable GPUs on
   that box. **On this box 0/1/5 cannot render CARLA** — verify before assuming.
4. Run it. It copies `weak_routes.txt` over the queue on start, so to resume only the remainder:

       cp .run_carla/jobs/weak_queue.txt .run_carla/weak_routes.txt
       nohup .run_carla/weak_route_queue.sh > .run_carla/jobs/weak_queue_runner.log 2>&1 &

## Per-route command (what the queue runs)

    CARLA_ROOT=/home/cglossop/carla \
    CARLA_RPC_BASE=<20000+100*gpu> CARLA_TM_BASE=<26000+100*gpu> CARLA_DISPLAY_BASE=<110+gpu> \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    bash run_carla.sh --route <ROUTE> --train-gpu <G> --render-adapter <G> \
      --agent-config .run_carla/rollout_infer_apmq3_apc5.py \
      --steervla-checkpoint <CKPT> \
      --eval-only true --train-mode rl --critic-mode none \
      --max-episodes 1 --online-steps 4000 --save-buffer false \
      --save-video-local true --run-group weak_routes --wandb-mode online

`--train-mode rl` is not optional: `run_carla.sh` defaults it to `dagger`, which changes the
update dispatch. `--max-episodes 1` is what stops a short episode being followed by a second one.

## Gotchas carried over from this session

- **`enter-actor-flow-002` reproducibly crashes CARLA at tick ~1894** — seven times across three
  checkpoints and three seeds. It will hang a worker; the harness only recovers after the
  90-minute route timeout. `enter-actor-flow-001` is flaky at varying ticks.
- A hung worker looks like: per-route log stale for minutes, its CARLA rpc port not listening.
  Clear it by SIGINT→SIGTERM→SIGKILL on the worker's **process group**, never the orchestrator.
- `/tmp/carla_rpc<PORT>.log` is shared across users; a port whose log is owned by someone else
  fails with `PermissionError`. Check `stat -c %U /tmp/carla_rpc<PORT>.log` before picking ports.
- JAX puts a ~528 MB context on **every visible GPU** (the harness unsets `CUDA_VISIBLE_DEVICES`
  so `training_gpu_rank` indexes correctly). Workers therefore appear on all GPUs in nvtop;
  that is not a leak.

## Uncommitted changes on `dev` this depends on

`run_leaderboard.py` per-GPU JAX cache · `main_carla.py` fixed-height video panel,
`warmup_episodes`, raw-frame capture for CAST · `cast_relabel.py` async review + dual frame
streams · `vlm_feedback.py` merged prompts + traffic-flow inference.
Commit before moving machines, or the queue will not reproduce.
