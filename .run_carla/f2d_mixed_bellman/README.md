# f2d mixed CAST-relabel sweep — bellman half

Run part of the mixed-stack CAST-relabel sweep on **bellman** (the `/raid` machine) while dgx6 runs
the rest. Both halves use the same `SWEEP_NAME`, the same HL checkpoint and the same recipe, so the
results merge into one table.

The sweep trains the **original SimLingo InternVL2 high level** against a **frozen pi05
no-ego-history low level** — the mixed stack. The HL optimizer lives in the InternVL2 worker process
(`impls/vlas/internvl2_hl_worker.py --trainable`); pi05 is never loaded trainable. Everything here
travels on branch `f2d-mixed-cast-bellman`.

| File | What it is |
|---|---|
| `launch_bellman.sh` | preflight + launch, with bellman's `/raid` paths |
| `handoff_routes.py` | moves routes out of dgx6's live queue and writes this machine's route file |
| `hl_python.sh.template` | only if bellman's `.venv` lacks the simlingo extra |
| `impls/configs/mixed_steervla_cast_relabel_train_bellman_config.py` | the config: same recipe, `/raid` paths |

---

## 1. What differs from dgx6 (and what must not)

Only paths differ. The config inherits every training knob from
`mixed_steervla_cast_relabel_train_config.py`:

| | dgx6 | bellman |
|---|---|---|
| HL checkpoint | `/data/local/cglossop/2026_05_24_06_52_33_simlingo_seed1_bellman/…/epoch=019.ckpt` | `/raid/users/celine/steervla-ckpts/2026_05_24_06_52_33_simlingo_seed1_bellman/…/epoch=019.ckpt` |
| pi05 LL | `/data/local/cglossop/f2d_bon/ll/6000` | `/raid/users/cglossop/openpi/cat-logs/…/6000` |
| HL replay pool | `simlingo_hl_simplified_img` under `/data/local/cglossop` (frames bundled as `.npz`) | `simlingo_hl_simplified` under `/raid/users/cglossop/simlingo_hl_pools` (frames referenced in the SimLingo database) |
| caches | overridden to `/data/local` (dgx6 has no `/raid`) | `run_carla.sh` defaults are already `/raid`; nothing to set |
| outputs | `/data/local/cglossop/f2d_mixed_cast/…` | `/raid/users/cglossop/{sweeps,sweep_results}/<SWEEP_NAME>` |

**Identical on both, do not change:** `SWEEP_NAME`, seed 0, `online_steps=10000`,
`max_hl_updates=500`, stop on driving score 95 with streak 3, `--cot-temperature 0.1`,
`--hl-kl-coef 0.05`, HL batch 64 / micro 8 / 10 grad steps per 200 env steps, replay weight 0.5,
online bad fraction 0.9, adaptive sampling off, **no periodic evals**, and a **3-seed frozen eval at
the end** of each route.

---

## 2. Two GPUs per route

The InternVL2 HL is a separate process and gets its own card:

```
SWEEP_GPUS="3 5"        # policy + CARLA, one entry per parallel route
SWEEP_HL_GPUS="6 7"     # the HL worker for that route (entry i pairs with entry i)
```

The driver waits for **both** cards of a pair to fall below `GPU_FREE_MIB` (default 20 GB) before
starting a route. Measured on dgx6: ~30 GB on the policy card, ~45–51 GB on the HL card (peak 41 GB
during an update), so a pair of 80 GB cards is comfortable.

---

## 3. Splitting the routes (do this first)

The machines share no disk, so nothing stops both from running the same route. The split is by
route list, and it must be made by **removing** routes from dgx6's live queue:

```bash
# on dgx6, in the f2d-mixed-cast worktree
JOBS=.run_carla/jobs/f2dmixed_internvl2hl_pi05ll_fixedcarla_kl005_seed0
python3 .run_carla/f2d_mixed_bellman/handoff_routes.py --jobs-dir $JOBS \
  --out routes_bellman.txt --tail 4 --dry-run      # or --routes <name> <name> ...
# drop --dry-run to apply, then copy routes_bellman.txt to bellman
```

It holds the driver's own `flock`, so a worker cannot pop a route mid-edit, and it refuses routes
that are already running or finished — those are paid for already.

Prefer taking routes from the **tail** of the queue: the head is what dgx6 starts next, and a route
it is already running cannot be handed over without throwing away its progress.

---

## 4. Setup on bellman

1. **Code**

   ```bash
   cd ~/ogbench-carla && git fetch && git checkout f2d-mixed-cast-bellman
   GIT_LFS_SKIP_SMUDGE=1 uv sync --extra all-gpu --extra simlingo
   ```

   The `simlingo` extra is what lets the HL worker import peft/hydra/lightning/timm. If you would
   rather not touch bellman's shared `.venv`, copy `hl_python.sh.template` to `hl_python.sh`, point
   it at `/raid/users/cglossop/ogbench-simlingo-deps`, `chmod +x`, and set `MIXED_HL_PYTHON` to it.

2. **Check the paths the preflight cannot guess.** `simlingo_source_root` is machine-specific and
   has already changed once on `dev`; confirm which checkout bellman should use before launching.

3. **CARLA.** 0.9.16 + the f2d content pack at `CARLA_ROOT` for 15 routes, and 0.9.15 at
   `F2D_CARLA_0915_ROOT` with its py3.10 env for `generalization-animals-1076` only.

4. **Launch**

   ```bash
   cd ~/ogbench-carla
   ROUTES_FILE=routes_bellman.txt SWEEP_GPUS="3 5" SWEEP_HL_GPUS="6 7" \
     ./.run_carla/f2d_mixed_bellman/launch_bellman.sh --dry-run    # every line must say ok
   tmux new -s f2d-mixed-bellman
   ROUTES_FILE=routes_bellman.txt SWEEP_GPUS="3 5" SWEEP_HL_GPUS="6 7" \
     ./.run_carla/f2d_mixed_bellman/launch_bellman.sh --arm
   ```

   Don't launch from a shell that will close: the sweep dies with it. `--status` and `--stop` on
   `./.run_carla/b2d_subset_sweep.sh` work as usual.

---

## 5. Checks worth making in the first hour

- `[eval_run] two-GPU split: policy/sim gpu=X, HL gpu=Y` — the pair is what you intended.
- `[internvl2-hl] trainable: 682 tensors / 326.1M params` — the HL really is training. Without
  `--trainable` the worker would serve inference silently and the route would waste its budget.
- The first `[mixed-steervla.update_hl] loss=…` at about env step 200–260, then every 200 steps.
- `HL batch mix (pool -> count)` should reach `{'simlingo_hl_simplified': 32, 'online': 32}`. Early
  batches are online-short and backfilled from replay while the CAST pool fills — that is expected,
  but if it never reaches ~32 online, CAST is not producing samples.
- W&B runs must appear under **`catherine_glossop`** / entity `catherineglossop`.
- On `animals-1076`: `[fail2drive animal] requested=walker.animal.1007 spawned=[…] matched=True`.
  Without it the animal never spawns and the route scores the wrong scenario.

---

## 6. Merging back

When both halves are done, copy bellman's run dirs into whichever machine will write the table, then
regenerate it:

```bash
rsync -a --ignore-existing \
  /raid/users/cglossop/sweeps/<SWEEP_NAME>/ <other>:/data/local/cglossop/f2d_mixed_cast/sweeps/<SWEEP_NAME>/

OGBENCH_SAVE_DIR=<sweeps>/<SWEEP_NAME> RESULTS_DIR=<results>/<SWEEP_NAME> SWEEP=<SWEEP_NAME> \
  ROUTES_FILE=f2d_steervla_subset.txt RESULT_TRAIN_STEP=10000 ./.run_carla/sweep_results.sh
```

`--ignore-existing` keeps the first copy of any route that (despite the split) ran twice.

---

## 7. Gotchas already paid for

- **Killing a route by hand leaves CARLA behind.** The driver only cleans up after a *watchdog*
  abort. If you kill a route yourself, also `pkill -f "carla-rpc-port=<port>"` and remove
  `/tmp/.X<display>-lock`, or the next route on that worker starts into a held GPU. Use a bracketed
  pattern (`[c]arla-rpc-port=`) so the pattern does not match your own command line.
- **A route that exits is not requeued.** The worker logs `DONE (exit N)` and moves on, so a
  hand-stopped route simply has no result and will be picked up again only on a later re-arm.
- **The W&B key path is hardcoded.** `eval_subset_run.sh` reads `/home/cglossop/.wandb_school_key`
  unconditionally; if it is missing the key comes back empty and runs log under whatever account
  `~/.netrc` holds, with no error.
- **Inference cost is the HL decode loop**, ~63 ms/token for the 0.65B model (measured), so ~4 s per
  HL query of ~68 tokens, every ~6 env steps. KV caching is already on and buys only ~19%.
- **Each HL update re-tiles its 64 images on every one of its 10 gradient steps** — `pixel_tiles` is
  225 ms/record, so ~144 s of each ~270 s update is redundant. Hoisting the tiles out of the step
  loop is bit-identical and would cut roughly 22% off total wall time. Not done yet; if it lands,
  apply it on both machines so the two halves keep the same throughput profile.
