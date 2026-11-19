# f2d mixed-policy Qwen BoN sweep — second-machine handover

Run the rest of the f2d zero-shot Qwen Best-of-N sweep on a second machine, **routes in reverse
order**, while bellman keeps going forwards. The two meet in the middle and the results merge into
one table.

Everything in this directory travels with branch `qwen-zs-bon-sweep`:

| File | What it's for |
|---|---|
| `launch_reverse.sh` | preflight checks, then arms the sweep in reverse route order |
| `routes_f2d_reverse.txt` | the 11 routes with work left, last route first |
| `build_ckpt_overrides.py` | picks each route's last HL checkpoint on this machine (called by the launcher) |
| `hl_python.sh.template` | the Python the InternVL2 HL worker runs under |
| `cells_done.py` / `prune_queue.py` | the meet-in-the-middle sync |

---

## Where things stand (snapshot 2026-09-21 16:11, bellman)

**17 / 48 cells done.** The first five routes are finished for all three seeds and are **not** in
this machine's list: construction-permutations-1019, custom-obstacles-1020, pedestrians-on-road-1085,
construction-pedestrian-1011, pedestrian-crowd-1069.

The 11 routes left, in the order **this** machine runs them (bellman works the same list from the
bottom up):

| # | Route | Last checkpoint | Done on bellman |
|--:|---|--:|---|
| 1 | generalization-wall-1097 | 10000 | — |
| 2 | generalization-animals-1076 | 10000 | — |
| 3 | generalization-bad-parking-1004 | 10000 | — |
| 4 | generalization-obscured-stop-1046 | 2276 | — |
| 5 | generalization-image-on-object-1041 | 1669 | — |
| 6 | generalization-right-of-way-1056 | 10000 | — |
| 7 | generalization-right-construction-1094 | 9141 | — |
| 8 | generalization-pedestrian-other-blocker-1072 | 10000 | — |
| 9 | generalization-hard-brake-1036 | 10000 | seed 0 |
| 10 | generalization-fully-blocked-1032 | 10000 | seed 2 |
| 11 | generalization-custom-obstacles-1024 | 10000 | — (bellman is on it) |

That is **31 cells** to split between the two machines. Checkpoints are each route's **last**
one, not a fixed 10k; the uneven steps (2276, 1669, 9141) are training runs that ended early. The
table is what `build_ckpt_overrides.py` resolves on bellman, and it matches bellman's own sweep.

---

## Protocol — must match bellman exactly

`launch_reverse.sh` sets all of this; listed so you can check it.

- **Actor:** mixed SteerVLA (`impls/configs/steervla_mixed_eval_config.py`). The route's InternVL2 HL
  writes reasoning + subtask; the pi05 LL is `pi05_steervla_simplified_reasoning_no_ego_history_v1/6000`
  with actor config `pi05_steervla_cot_simplified_reasoning_no_ego_history`.
- **HL sampled at temperature 1.0**, seeded per request, so the 4 candidates get different subtasks.
- **Best-of-N:** 4 candidates, sampled **sequentially**, re-query every 3 env steps, no brake candidate.
- **Critic:** zero-shot `Qwen/Qwen3.8-27B`, no adapter, `scene_criteria_v2`, BF16 eager, one critic
  per worker on that worker's own GPU.
- **Episodes end leaderboard-style:** no step cap, no stuck cutoff; only AgentBlockedTest (60 s under
  0.1 m/s), InRouteTest, or route completion end an episode.
- **CARLA seeds 0, 1, 2**, one episode each (`EVAL_SEED_OFFSET=0`, `N_EVAL=1`).
- **`HL_KV_CACHE` must be unset.** The cached decode path is ~10% faster but not bit-reproducible
  against bellman's cells. The launcher unsets it.

---

## Setup

### 1. Code

```bash
git clone git@github.com:catglossop/ogbench-carla.git && cd ogbench-carla
git checkout qwen-zs-bon-sweep
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra all-gpu --extra simlingo
```

### 2. The Python 3.10 env for CARLA 0.9.15

Fail2Drive runs CARLA in a separate py3.10 subprocess. Recipe from `ogbench/carla/README.md` §2b:

```bash
uv venv --python 3.10 .venv-carla-0915
uv pip install --python .venv-carla-0915/bin/python \
  "carla==0.9.15" "numpy<2" "py-trees==0.8.3" \
  absl-py pyyaml gymnasium networkx shapely tabulate xmlschema \
  opencv-python-headless matplotlib imageio scipy pillow tqdm \
  "bench2drive @ git+https://github.com/catglossop/Bench2Drive.git" \
  "fail2drive @ git+https://github.com/catglossop/fail2drive.git"
.venv-carla-0915/bin/python -c "import carla, srunner, fail2drive; print('ok')"
```

It can live elsewhere if you set `CARLA_0915_PYTHON`.

### 3. CARLA 0.9.15 with the animal assets

`generalization-animals-1076` needs Fail2Drive's animal walkers. Without them the animal never spawns
and the episode still runs to completion, so the run **looks** fine and scores the wrong scenario.

```bash
./install_f2d_content.sh <F2D_CARLA_0915_ROOT> <fail2drive-content-zip>
ls <F2D_CARLA_0915_ROOT>/CarlaUE4/Content/AnimalVarietyPack   # must exist
```

On bellman, the animals route spawns `walker.animal.1007` and the log shows `matched=True`. Check for
that line in the first animals job (step 7).

### 4. Model assets

| Asset | Size | Source → set this variable |
|---|--:|---|
| f2d HL checkpoints (11 routes) | 27.9 GB | copied to `/home/celinet/steervla_harp_ckpts/f2dsteervla_simlingo_fixedcarla_kl005_seed0` → `HL_CKPT_ROOT` |
| pi05 `no_ego_history` LL, step 6000 | 46 GB | GCS, below → `MIXED_LL_CHECKPOINT` |
| `Qwen/Qwen3.8-27B` HF cache | 57 GB | copy bellman's `/raid/users/celine/qwen-critic/huggingface` or re-download → `QWEN_HF_HOME` |
| `qwen-critic` checkout | — | `QWEN_ROOT` |
| `simlingo-steervla` checkout | — | `SIMLINGO_SOURCE_ROOT` |
| SimLingo deps for the HL worker | 41 MB | copy bellman's `/raid/users/cglossop/ogbench-simlingo-deps` |

**HL checkpoints:** copy only each route's final checkpoint dir, keeping `pytorch_model.bin` **and
`.hydra/`** together; the loader looks upward for `.hydra/config.yaml`. The run-dir names only need
to keep `…<route>_seed_0_<timestamp>/checkpoints/<step>/`. Do **not** copy
`generalization-animals-1076_seed_0_20260917_234028.INVALID_ran_on_0916_no_animal`: it trained
before the animal assets existed. The resolver skips anything marked `INVALID`, but it's simplest
to leave it out.

> **Watch the sort order when picking "the last" checkpoint by hand.** `ls | tail -1` gives
> `8000` for wall-1097 because `"8000" > "10000"` as text. The last one is 10000. Use `sort -V`,
> or just rely on `build_ckpt_overrides.py`, which sorts numerically.

**LL checkpoint from GCS.** Local paths on bellman mirror `gs://cat-logs/...`, so the checkpoint
should be at the path below. **I could not verify it** (gcloud on bellman needed reauth), so list it
first:

```bash
LL=pi05_steervla_cot_simplified_reasoning_no_ego_history/pi05_steervla_simplified_reasoning_no_ego_history_v1/pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000
gsutil ls gs://cat-logs/$LL/                    # expect params/ among the entries
gsutil -m cp -r gs://cat-logs/$LL <local-dir>/  # then MIXED_LL_CHECKPOINT=<local-dir>/6000
```

**HL worker Python:** `cp hl_python.sh.template hl_python.sh`, fill in the SimLingo-deps path and the
repo path, then `chmod +x`. That's `MIXED_HL_PYTHON`.

### 5. Seed bellman's finished cells

Two cells in this machine's routes are already done on bellman. Copy their summaries in so the sweep
skips them:

```bash
# on bellman
cd /raid/users/cglossop/sweep_results/f2dsteervla_kl005_seed0_qwenzs_bon_mixed
tar czf /tmp/bellman_done.tgz */carla_seed_[0-9]/run_summary_frozen_eval.json
# on this machine, after copying the tarball over
mkdir -p $WORK_ROOT/sweep_results/f2dsteervla_kl005_seed0_qwenzs_bon_mixed
tar xzf bellman_done.tgz -C $WORK_ROOT/sweep_results/f2dsteervla_kl005_seed0_qwenzs_bon_mixed
```

Copy all of them, not only these two routes. Anything outside `routes_f2d_reverse.txt` is ignored,
and the full set is what the sync below compares against.

### 6. W&B: `catherine_glossop` only

Every run must log as **`catherine_glossop`**, entity `catherineglossop`, using the school key. Copy
bellman's `/home/cglossop/.wandb_school_key` to this machine (keep it `chmod 600`, never echo it) and
set `WANDB_KEY_FILE` to it.

This matters more here than on bellman. The runner used to read `/home/cglossop/.wandb_school_key`
unconditionally, and on a machine without that file the key would silently come back empty, so W&B
logged under whatever account that machine's `~/.netrc` held. That's fixed on this branch: a missing
key now stops the run. The launcher also checks `wandb.Api().viewer.username` and refuses to arm
unless it is `catherine_glossop`.

### 7. Configure, dry-run, arm

```bash
cd .run_carla/f2d_handover
export HL_CKPT_ROOT=/home/celinet/steervla_harp_ckpts/f2dsteervla_simlingo_fixedcarla_kl005_seed0
export MIXED_LL_CHECKPOINT=<local-dir>/6000
export F2D_CARLA_0915_ROOT=<carla 0.9.15 install>
export SIMLINGO_SOURCE_ROOT=<simlingo-steervla checkout>
export MIXED_HL_PYTHON=$PWD/hl_python.sh
export QWEN_ROOT=<qwen-critic checkout>
export QWEN_HF_HOME=<hf cache with Qwen3.8-27B>
export WANDB_KEY_FILE=<copy of the school key>
export WORK_ROOT=<writable dir, e.g. /home/celinet/f2d_bon>
export SWEEP_GPUS="0 1"            # one route per GPU; each needs ~110 GB free (critic + policy + CARLA)

./launch_reverse.sh --dry-run      # every preflight line must say ok; prints the first job
systemd-run --user --unit=f2d-bon-reverse --collect \
  --working-directory=$PWD bash ./launch_reverse.sh --arm
```

Using tmux instead of systemd is fine. Don't launch it from a shell that will close: the sweep dies
with it.

Default critic ports are **18870 / 18871**, deliberately **not** the scripts' 18850 / 18851 (see
Gotchas). Set `QWEN_PORTS` if those are taken.

---

## Meeting in the middle

The two machines share no disk, so each has to be told what the other finished. Otherwise both
would eventually run the whole middle of the list. Sync every few hours, and once more when they get
close:

```bash
# on each machine: export what it has done
python3 .run_carla/f2d_handover/cells_done.py <RESULTS_DIR> > done_<machine>.tsv

# copy each file to the OTHER machine, then there:
python3 .run_carla/f2d_handover/prune_queue.py <JOBS_DIR> done_<other>.tsv
```

| | `RESULTS_DIR` | `JOBS_DIR` |
|---|---|---|
| bellman | `/raid/users/cglossop/sweep_results/f2dsteervla_kl005_seed0_qwenzs_bon_mixed` | `~/ogbench-carla/.claude/worktrees/qwen-zs-bon-sweep/.run_carla/jobs/f2dsteervla_kl005_seed0_qwenzs_bon_mixed` |
| this machine | `$WORK_ROOT/sweep_results/f2dsteervla_kl005_seed0_qwenzs_bon_mixed` | `<repo>/.run_carla/jobs/f2dsteervla_kl005_seed0_qwenzs_bon_mixed` |

`prune_queue.py` only removes **queued** cells, holds the driver's queue lock, and is safe while the
sweep runs. When a machine's queue empties, its workers log `queue empty` and stop. The cells running
at the moment they meet may overlap by one or two. That's harmless: the protocol is identical, so
keep whichever finished first.

---

## Merging back

When both have stopped, copy this machine's results into bellman's and regenerate the table there
(bellman has the training summaries the baseline column needs):

```bash
# from this machine -> bellman; --ignore-existing keeps bellman's copy of any overlapping cell
rsync -a --ignore-existing \
  $WORK_ROOT/sweep_results/f2dsteervla_kl005_seed0_qwenzs_bon_mixed/ \
  bellman:/raid/users/cglossop/sweep_results/f2dsteervla_kl005_seed0_qwenzs_bon_mixed/

# on bellman
python3 .run_carla/qwen_zs_bon_results_md.py --actor mixed \
  --source-sweep f2dsteervla_simlingo_fixedcarla_kl005_seed0 \
  --run-group f2dsteervla_kl005_seed0_qwenzs_bon_mixed \
  --routes-file /raid/users/cglossop/sweep_results/qwenzs_mixed_b2d_sources/routes_f2d.txt \
  --carla-seeds "0 1 2"
```

That table's timing section reads bellman's job logs only, so its per-GPU numbers cover bellman's
cells.

---

## Gotchas (each one cost time on bellman)

- **Two sessions on one critic port.** The critic script defaults to port 18850. On 2026-09-21
  another session started its own critic on bellman's 18850 / GPU 5, which killed the sweep's critic
  and a running episode with it. Keep this machine's ports distinct and don't reuse them elsewhere.
- **The critic dies when its parent unit exits.** The critic runs inside whichever unit started it.
  If that unit finishes (e.g. a sweep driver), systemd collects the critic with it, crashing any
  episode that's still running (exit 134), and the next driver polls `critic unhealthy` forever. If
  you start a critic by hand, give it its own unit that stays in the foreground.
- **A crashed f2d run can block its slot.** On f2d, Xvfb belongs to the `.venv-carla-0915` env
  server, not `main_carla`, and the driver's cleanup can't see that server. Signs: every new job on
  one worker dies within a second and the log fills with `SLOT-ERROR`. Fix: find
  `.venv-carla-0915/bin/python` processes that aren't in a live `run_carla.sh` session and
  `kill -9` them. No cell is lost, since SLOT-ERROR requeues without spending a retry.
- **CARLA's GPU numbering can differ from `nvidia-smi`.** `--render-adapter` is passed straight to
  CARLA's `-graphicsadapter`, and CARLA's adapter order doesn't have to match `nvidia-smi`. On
  bellman it's swapped. Check the first job's CARLA actually renders on the GPU you intended.
- **A route can stall without ending.** Leaderboard-faithful endings mean a creeping ego can run for
  hours (AgentBlocked resets whenever it tops 0.1 m/s). pedestrian-crowd-1069 ran ~5 h per seed;
  fully-blocked-1032 legitimately stops and takes ~70 min to time out. To cut one short, bellman's
  practice has been a hand score of `route_completion × collision multiplier` (vehicle 0.60,
  static 0.65, pedestrian 0.50), confirmed by you, written **before** killing so the driver logs
  DONE instead of retrying. See the `manual_score` block in
  `generalization-pedestrian-crowd-1069/carla_seed_0/run_summary_frozen_eval.json` for the format.
