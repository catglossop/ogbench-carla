# Handover — SteerVLA checkpoint comparison on Bench2Drive (20-route subset)

**Status:** paused, ~35% complete. Stopped on `bellman` 2026-08-24 18:0x because the box
could no longer boot CARLA (see §7). Nothing is running there; no scheduled job will
resume it. This doc is self-contained — everything needed to finish the job on a new
machine is either inline or reachable from `gs://`.

**Goal:** evaluate 4 training checkpoints (steps 4000 / 6000 / 8000 / 10000) of the
SteerVLA CoT run on the same 20 Bench2Drive routes, and compare them against an already
complete base-policy baseline (§8).

---

## 1. What is already banked — do NOT re-run these

These routes have real, valid records. `--resume` keeps them automatically. 12 of the
80 (4 checkpoints × 20 routes) cells are done.

| Checkpoint | Route | DS | RC | Penalty | Status |
|---|---|---:|---:|---:|---|
| 4000 | invading-turn-002 | 94.64 | 94.64 | 1.00 | Failed - TickRuntime |
| 4000 | parking-crossing-pedestrian-005 | 100.00 | 100.00 | 1.00 | Completed |
| 6000 | invading-turn-002 | 100.00 | 100.00 | 1.00 | Completed |
| 6000 | parking-crossing-pedestrian-005 | 50.00 | 100.00 | 0.50 | Completed |
| 8000 | invading-turn-002 | 100.00 | 100.00 | 1.00 | Completed |
| 8000 | parking-crossing-pedestrian-005 | 100.00 | 100.00 | 1.00 | Completed |
| 8000 | vehicle-opens-door-two-ways-001 | 100.00 | 100.00 | 1.00 | Completed |
| 8000 | vehicle-turning-route-pedestrian-005 | 100.00 | 100.00 | 1.00 | Completed |
| 10000 | hazard-at-side-lane-004 | 100.00 | 100.00 | 1.00 | Completed |
| 10000 | invading-turn-002 | 60.00 | 100.00 | 0.60 | Completed |
| 10000 | parking-crossing-pedestrian-005 | 100.00 | 100.00 | 1.00 | Completed |
| 10000 | vehicle-turning-route-pedestrian-005 | 100.00 | 100.00 | 1.00 | Completed |

Remaining: **18** routes for step 4000, **18** for 6000, **16** for 8000, **16** for 10000.

> These banked records live in `leaderboard_runs/ckptcmp_step<STEP>/records/*.json` on
> `bellman`, which is **gitignored** (§3), so they are NOT in the repo. They are bundled
> alongside this doc as **`ckptcmp_banked_records.tar.gz`** (3.3 KB) — copy it across and
> `tar xzf` it from the repo root to restore all four `leaderboard_runs/ckptcmp_step*/records/`
> directories, then launch with `--resume`. That saves ~6 GPU-hours. If you skip it, just
> re-run all 20 routes per checkpoint; the table above then serves as a cross-check
> rather than an input.
>
> The archive holds 15 files, of which 12 carry real records — 3 (step 4000's
> `hazard-at-side-lane-004`, `merger-into-slow-traffic-004`,
> `non-signalized-junction-right-turn-003`) are empty shells that `--resume` correctly
> treats as unscored and re-runs.

---

## 2. The 20 routes

Chosen so the base policy fails 10 and passes 10, one route per scenario type. Order
matters only for resumability, not correctness.

**Base policy FAILED these 10** (the interesting half — is the new checkpoint better?):
```
parking-crossing-pedestrian-005
invading-turn-002
vehicle-opens-door-two-ways-001
accident-two-ways-004
vehicle-turning-route-pedestrian-005
t_-junction-002
hazard-at-side-lane-004
non-signalized-junction-right-turn-003
merger-into-slow-traffic-004
interurban-actor-flow-004
```

**Base policy PASSED these 10** (regression check):
```
blocked-intersection-005
construction-obstacle-003
control-loss-001
crossing-bicycle-flow-001
dynamic-object-crossing-001
hard-break-route-003
hazard-at-side-lane-two-ways-001
highway-cut-in-001
highway-exit-004
interurban-advanced-actor-flow-001
```

Write all 20 (in the order above) to `.run_carla/ckpt_cmp_routes.txt`, one per line.

---

## 3. Repo — and the gitignore trap

```
git@github.com:catglossop/ogbench-carla.git
branch dev, commit 2955195
```

**A fresh clone does NOT give you a runnable setup.** `.gitignore` excludes both
`.run_carla/` (line 21) and `leaderboard_runs/` (line 19). That means the route list,
the agent config, the run scripts, and every result are absent from the repo. You must
recreate `.run_carla/ckpt_cmp_routes.txt` (§2) and `.run_carla/rollout_infer_apmq3_apc5.py`
(§5) by hand. Everything else the run touches *is* tracked:
`run_leaderboard.py`, `impls/configs/steervla_rollout_job2_eaf004_no_ego_history.py`,
`impls/configs/steervla_dsrl_config.py`, `LEADERBOARD_EVAL.md`.

---

## 4. Checkpoints

Canonical source (each ~46–47 GB, Orbax/OCDBT):

```
gs://cat-logs/pi05_steervla_cot_simplified_reasoning_commentary/pi05_steervla_cot_simplfied_reasoning_commentary_0823/pi05_steervla_cot_simplfied_reasoning_commentary_0823_20260823_154520/<STEP>
```
for `<STEP>` in `4000 6000 8000 10000`. Note the misspelling `simplfied` in the inner
path segments — it is real, keep it.

```bash
SRC=gs://cat-logs/pi05_steervla_cot_simplified_reasoning_commentary/pi05_steervla_cot_simplfied_reasoning_commentary_0823/pi05_steervla_cot_simplfied_reasoning_commentary_0823_20260823_154520
DEST=/path/on/new/box/steervla_pi_ckpts/pi05_cot_simplified_0823_154520
for s in 4000 6000 8000 10000; do
  mkdir -p "$DEST/$s"
  gsutil -q -m cp -r "$SRC/$s/*" "$DEST/$s/"
done
```
Verify each with `test -f $DEST/$s/params/commit_success.txt` before use.

**Already verified, do not redo:** all four are architecturally identical to the base
policy checkpoint — 51/51 parameters, zero shape mismatches, compared via
`params/array_metadatas/process_0`. This is why the existing agent config loads them
unmodified and the comparison is like-for-like. (No `commentary` train config exists in
either local openpi, and the checkpoints record no config, so the param-tree diff was
the only way to establish compatibility.)

---

## 5. Agent config

Create `.run_carla/rollout_infer_apmq3_apc5.py`. **`_BASE` is an absolute path — change
it to the new machine's checkout root.**

```python
from pathlib import Path
import runpy

_BASE = Path("/CHANGE/ME/ogbench-carla/impls/configs/steervla_rollout_job2_eaf004_no_ego_history.py")
_BASE_GET_CONFIG = runpy.run_path(str(_BASE))["get_config"]


def get_config():
    config = _BASE_GET_CONFIG()
    if "debug_noise" in config:
        config.debug_noise = False
    config.steervla.actions_per_model_query = 3
    config.steervla.actions_per_cot = 5
    return config
```

The base config already forces inference-only (all four update switches and
`load_trainable_params` are False). `debug_noise=False` mirrors what `run_leaderboard.py`
forces anyway, and is the suspect in the intermittent openpi RMSNorm crash documented in
`LEADERBOARD_EVAL.md` §6.2. Route re-anchoring (`reanchor_cached_chunk=True`, from
`impls/configs/steervla_dsrl_config.py:177`) is active and matters because
`actions_per_model_query=3 > 1`.

---

## 6. Launch command

One checkpoint per GPU, routes sequential within each. Needs ~50 GB free VRAM per GPU
(48 GB for the JAX worker at `--xla-mem-fraction 0.30`, plus ~7 GB for CARLA).

```bash
cd /path/to/ogbench-carla
CKPTS=/path/on/new/box/steervla_pi_ckpts/pi05_cot_simplified_0823_154520
i=0
for c in 4000 6000 8000 10000; do
  g=<gpu index for this checkpoint>
  rpc=$((13000 + 100*i)); tm=$((19000 + 100*i)); disp=$((40 + i))
  sess="lb-ckpt${c}"
  tmux kill-session -t "$sess" 2>/dev/null
  tmux new-session -d -s "$sess" -x 220 -y 55
  tmux send-keys -t "$sess" \
    "CARLA_ROOT=/path/to/carla CARLA_RPC_BASE=$rpc CARLA_TM_BASE=$tm CARLA_DISPLAY_BASE=$disp \
     .venv/bin/python run_leaderboard.py --slots ${g}:${g} \
     --routes @.run_carla/ckpt_cmp_routes.txt \
     --agent-config .run_carla/rollout_infer_apmq3_apc5.py \
     --steervla-checkpoint \$CKPTS/$c --xla-mem-fraction 0.30 \
     --jax-cache-dir ~/.cache/jax_ckptcmp_gpu\${g} --resume \
     --out-dir leaderboard_runs/ckptcmp_step${c} 2>&1 \
     | tee -a leaderboard_runs/ckptcmp_step${c}.console.log" C-m
  i=$((i+1))
done
```

Drop `--resume` only if you are starting with an empty `leaderboard_runs/`. `tmux` is
required — the rich dashboard needs a TTY (use `--no-ui` otherwise).

**Runtime:** routes ran 5–35 min each (median ~15). Budget **8–10 h wall-clock per GPU**
for a full 16–18 route resume. All four run concurrently, so ~8–10 h total.

---

## 7. Why this stalled — read before launching

From ~11:45 on 2026-08-24, every route on `bellman` died the same way: CARLA's RPC port
came up fine (~28 s), then `load_world` timed out 20 consecutive times and the simulator
segfaulted with

```
LowLevelFatalError [File:Unknown] [Line: 1214]
GameThread timed out waiting for RenderThread after 60.00 secs
Signal 11 caught. → Segmentation fault (core dumped)
```

This is **UE4's render thread being starved**, caused by machine-wide GPU contention from
other users' jobs (all 8 GPUs holding 85–122 GB of resident CUDA contexts). Ruled out:
it reproduced on a completely idle GPU with 77 GB free, and CPU (224 cores, load 34), RAM
(1.7 TB free), disk (idle), `/dev/shm` (1%), and ECC/Xid were all clean. Other users'
CARLA instances ran fine throughout — this is contention, not a broken install.

**Gate every launch on this preflight.** It boots a throwaway CARLA on the emptiest GPU
and proves `load_world` works before you commit 8–10 GPU-hours. It cost a full day to
learn this lesson; do not skip it.

```bash
#!/usr/bin/env bash
set -u
PORT=14600
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
LOG=$(mktemp)
cleanup() { pkill -u "$USER" -f "carla-rpc-port=${PORT}" 2>/dev/null; }
trap cleanup EXIT
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  /path/to/carla/CarlaUE4.sh -RenderOffScreen -nosound \
  -carla-rpc-port=$PORT -graphicsadapter="$GPU" -carla-streaming-port=$((PORT+1)) \
  > "$LOG" 2>&1 &
for _ in $(seq 1 12); do sleep 10; ss -ltn | grep -q ":${PORT} " && break; done
ss -ltn | grep -q ":${PORT} " || { echo "PREFLIGHT FAIL: RPC never came up"; tail -4 "$LOG"; exit 1; }
.venv/bin/python - "$PORT" <<'PY'
import carla, sys, time
c = carla.Client('localhost', int(sys.argv[1])); c.set_timeout(180.0)
t = time.time()
try:
    c.load_world('Town12')
    print("PREFLIGHT OK: load_world('Town12') in %.0fs" % (time.time() - t)); sys.exit(0)
except Exception as e:
    print("PREFLIGHT FAIL after %.0fs: %s" % (time.time() - t, e)); sys.exit(1)
PY
```

---

## 8. Baseline for comparison

Base policy = `..._no_ego_history` @ step 6000, **same agent config, same 20 routes**,
from `leaderboard_runs/b2d_apmq3_apc5_seed0`. Aggregate over these 20:
**DS 86.03 · RC 95.15 · IP 0.894 · SR 50.00%**.

| Route | DS | RC | P | Status |
|---|---:|---:|---:|---|
| parking-crossing-pedestrian-005 | 58.46 | 97.44 | 0.60 | Failed - TickRuntime |
| invading-turn-002 | 96.43 | 96.43 | 1.00 | Failed - TickRuntime |
| vehicle-opens-door-two-ways-001 | 94.00 | 94.00 | 1.00 | Failed - TickRuntime |
| accident-two-ways-004 | 59.96 | 92.24 | 0.65 | Failed - TickRuntime |
| vehicle-turning-route-pedestrian-005 | 10.65 | 90.40 | 0.12 | Failed - TickRuntime |
| t_-junction-002 | 87.58 | 87.58 | 1.00 | Failed - TickRuntime |
| hazard-at-side-lane-004 | 85.20 | 85.20 | 1.00 | Failed - TickRuntime |
| non-signalized-junction-right-turn-003 | 33.83 | 65.06 | 0.52 | Failed - TickRuntime |
| merger-into-slow-traffic-004 | 98.22 | 98.22 | 1.00 | Failed - Agent deviated from the route |
| interurban-actor-flow-004 | 96.33 | 96.33 | 1.00 | Failed - Agent deviated from the route |
| blocked-intersection-005 | 100.00 | 100.00 | 1.00 | Completed |
| construction-obstacle-003 | 100.00 | 100.00 | 1.00 | Completed |
| control-loss-001 | 100.00 | 100.00 | 1.00 | Completed |
| crossing-bicycle-flow-001 | 100.00 | 100.00 | 1.00 | Completed |
| dynamic-object-crossing-001 | 100.00 | 100.00 | 1.00 | Completed |
| hard-break-route-003 | 100.00 | 100.00 | 1.00 | Completed |
| hazard-at-side-lane-two-ways-001 | 100.00 | 100.00 | 1.00 | Completed |
| highway-cut-in-001 | 100.00 | 100.00 | 1.00 | Completed |
| highway-exit-004 | 100.00 | 100.00 | 1.00 | Completed |
| interurban-advanced-actor-flow-001 | 100.00 | 100.00 | 1.00 | Completed |

> **Selection bias — do not read 86.03 / 50% as the base policy's overall Bench2Drive
> score.** These 20 routes were deliberately picked 10-fail / 10-pass. Across all 220
> routes the base policy scores DS 64.24 and SR 37.73%. This column is only valid as a
> per-route comparator against the checkpoint columns.

---

## 9. Scoring — two definitions, don't mix them

- **Bench2Drive success = Driving Score == 100.** Use this. It reproduces `merged.json`'s
  stored success rate exactly.
- `run_leaderboard.py` separately reports `STATUS_SUCCESS = ("Completed", "Perfect")`,
  which is **not** the same thing and reads ~27 points higher on the full 220-route set
  (64.55% vs 37.73%). Both appear in the tooling; label whichever you report.
- `Driving Score = Route Completion × Infraction Penalty`.

**`Failed - TickRuntime`** means the route hit the hard 4000-simulator-tick cap
(= 200.05 s of game time at 20 Hz), raised at `ogbench/carla/carla_utils.py:1055` and
caught at `:3266`. It is a harness timeout, not a crash. It affected 28% of base-policy
routes. The SimLingo/SteerVLA reference runs had max durations of 355.9 s / 377.0 s, so
that cap was **not** active for them — a partial confound when comparing against those
references (though it would only have touched ~4% of their routes).

---

## 10. Pitfalls that already cost time

1. **`NoRecord (rc=1)` rows are DS=0 placeholders, not driving failures.** When a route
   exhausts its retries without CARLA ever writing a record, `run_leaderboard.py:1007`
   constructs `RouteResult(route=..., status="NoRecord (rc=1)")`, and the dataclass
   (line 475) defaults `score_composed`/`score_route`/`score_penalty` to `0.0`. **Exclude
   these from every average** — they dragged one checkpoint's DS from 90 → 72 and made
   the live dashboards read 15–33 instead of 75–100. Filter on
   `status.startswith(("NoRecord", "Timeout"))`.
2. **Port collisions with concurrent orchestrators.** Slot ports are
   `rpc = RPC_BASE + 100*index`, `tm = TM_BASE + 100*index`, `display = DISPLAY_BASE + index`,
   where `index` comes from `enumerate` — so four separate orchestrators all start at
   index 0 and collide. Override per run with `CARLA_RPC_BASE` / `CARLA_TM_BASE` /
   `CARLA_DISPLAY_BASE` (`run_leaderboard.py:89-91`). The §6 command does this.
3. **JAX cache must be GPU-scoped.** Cached executables carry their device assignment.
   Use a distinct `--jax-cache-dir` per GPU, as in §6.
4. **Never `pkill -f <pattern>` from a shell whose own command line contains that
   pattern** — it matches itself and kills the shell. Bracket it: `run_leaderboar[d]\.py`.
   Documented in `LEADERBOARD_EVAL.md` §8; it bit us anyway.
5. **`--resume` re-runs any route lacking a record file**, and a clean single `SIGINT`
   parks in-flight routes as `pending` rather than scoring them 0. So an interrupted run
   loses nothing — always stop with one `SIGINT`, never `SIGKILL`.
6. On a **shared box, only touch your own processes.** Scope every `pkill` by port/display;
   never use a blanket `reset_carla.sh`-style kill that takes out every CARLA on the host.
7. `steervla (3).json`-style result files may contain invalid control characters —
   `json.load(..., strict=False)`. Join reference results on `route_id`
   (`RouteScenario_<id>_rep0`) via `ogbench.carla.route_registry`, **not** on
   `scenario_name`, which has only 44 unique values across 220 routes.

---

## 11. Definition of done

All 80 cells (4 checkpoints × 20 routes) have real records, then regenerate the report.
The generator lives on `bellman` at `.run_carla/gen_ckpt_report.py` (gitignored — copy it
across, or rewrite it) and emits `carla_results/checkpoint_comparison.md` with: progress,
summary over all 20, the base-FAILED subset, the base-PASSED subset, per-route Driving
Score, per-route status, and the selection-bias caveat. It already excludes harness
failures from all aggregates.

The question to answer: **does any checkpoint beat the base policy on the 10 routes it
failed, without regressing the 10 it passed?**
