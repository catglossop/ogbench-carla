"""Offline pooled HL training for the SimLingo SteerVLA (no CARLA).

Takes the CAST-relabel HL samples that several finished per-route HARP runs wrote
(``<run>/cast_relabel_hl_dataset``), pools them, and trains ONE high-level policy from the same
base HL checkpoint the source runs started from, with their exact HL update recipe (the
``agent.steervla`` block of a source run's ``flags.json``: lr, batch 64 / micro 8, 10 grad steps per
update, KL 0.05 to the base, 50/50 online/SimLingo-replay mix, BAD/precursor bucketing). Nothing is
rolled out: the online pool is fixed.

``--balance_routes`` (default on) gives every source run an equal share of the online half of each
batch (``hl_online_balance_subdirs`` in :mod:`vlas.hl_pool_sampling`), so one route with 10x the
samples does not dominate.

Layout written under ``--out_dir``::

    pool/<route> -> <source run>/cast_relabel_hl_dataset   (symlinks; the pooled online root)
    simlingo_hl_batches/update_*.json                        (text dump of every batch)
    checkpoints/<grad_step>/pytorch_model.bin + .hydra/      (loadable as steervla.hl_checkpoint)
    train_manifest.json, train_log.jsonl

Usage::

    CUDA_VISIBLE_DEVICES=2 .venv/bin/python impls/train_hl_offline_pooled.py \
        --source_runs=<run1>,<run2>,... --out_dir=/raid/.../pooled_run \
        --total_grad_steps=500 --checkpoint_every=150 --wandb_group=<group>
"""

import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_list("source_runs", [], "Finished HARP run dirs whose cast_relabel_hl_dataset to pool.")
flags.DEFINE_string("out_dir", "", "Output run dir (pool symlinks, checkpoints, logs).")
flags.DEFINE_integer("total_grad_steps", 500, "Total HL gradient steps.")
flags.DEFINE_integer("checkpoint_every", 150, "Save a checkpoint every N grad steps (plus the final step).")
flags.DEFINE_integer("steps_per_update", 0, "Grad steps per sampled batch (0 = source hl_update_num_steps).")
flags.DEFINE_bool("balance_routes", True, "Equal online share per source run.")
flags.DEFINE_integer("seed", 0, "numpy/torch/random seed for batch sampling.")
flags.DEFINE_string("wandb_project", "OGBench-CARLA", "wandb project.")
flags.DEFINE_string("wandb_group", "pooled_hl", "wandb group (<=64 chars).")
flags.DEFINE_string("run_name", "", "wandb run name (default: out_dir basename).")


def _route_of(run_dir: Path) -> str:
    f = json.loads((run_dir / "flags.json").read_text())
    for key in ("route", "carla_route", "env_name"):
        if f.get(key):
            return str(f[key])
    # Fall back to the dir name: dsrl_..._upd-hl_<route>_seed_<n>_<ts>
    name = run_dir.name
    return name.split("_upd-hl_", 1)[-1].rsplit("_seed_", 1)[0]


def main(_):
    import numpy as np
    import torch

    import wandb

    if not FLAGS.source_runs or not FLAGS.out_dir:
        raise SystemExit("--source_runs and --out_dir are required")
    out = Path(FLAGS.out_dir)
    pool = out / "pool"
    ckpt_dir = out / "checkpoints"
    pool.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    runs = [Path(r) for r in FLAGS.source_runs]
    steervla_cfgs = []
    sources = []
    for run in runs:
        ds = run / "cast_relabel_hl_dataset"
        if not ds.is_dir():
            raise SystemExit(f"missing {ds}")
        route = _route_of(run)
        link = pool / route
        if link.is_symlink() or link.exists():
            if link.resolve() != ds.resolve():
                raise SystemExit(f"{link} already points at {link.resolve()}, not {ds}")
        else:
            link.symlink_to(ds, target_is_directory=True)
        n = sum(len(json.loads(m.read_text()).get("samples", [])) for m in ds.glob("*/hl_samples.json"))
        sources.append({"route": route, "run_dir": str(run), "dataset": str(ds), "num_samples": n})
        steervla_cfgs.append(json.loads((run / "flags.json").read_text())["agent"]["steervla"])

    base = steervla_cfgs[0]
    for run, cfg in zip(runs[1:], steervla_cfgs[1:]):
        diff = sorted(k for k in set(base) | set(cfg) if base.get(k) != cfg.get(k))
        if diff:
            raise SystemExit(f"steervla config of {run} differs from {runs[0]} in {diff}; refusing to pool")

    cfg = dict(base)
    cfg.update(
        load_trainable_params=True,
        hl_dataset_dir=str(pool),
        hl_update_every=1,
        training_gpu_rank=0,
        hl_training_gpu_rank=0,
    )
    steps_per_update = int(FLAGS.steps_per_update or cfg.get("hl_update_num_steps", 10))
    n_updates = -(-FLAGS.total_grad_steps // steps_per_update)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    manifest = {
        "sources": sources,
        "base_hl_checkpoint": cfg["hl_checkpoint"],
        "total_grad_steps": FLAGS.total_grad_steps,
        "steps_per_update": steps_per_update,
        "n_updates": n_updates,
        "checkpoint_every": FLAGS.checkpoint_every,
        "balance_routes": FLAGS.balance_routes,
        "seed": FLAGS.seed,
        "steervla": cfg,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "train_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k != "steervla"}, indent=2), flush=True)

    wandb.init(
        project=FLAGS.wandb_project,
        group=FLAGS.wandb_group,
        name=FLAGS.run_name or out.name,
        config={k: v for k, v in manifest.items()},
        mode=os.environ.get("WANDB_MODE", "online"),
    )

    from vlas.simlingo_steervla import create_simlingo_steervla_sample_fn

    _, actor = create_simlingo_steervla_sample_fn(cfg, {}, training_gpu_rank=0)
    actor.hl_online_balance_subdirs = bool(FLAGS.balance_routes)
    pools = actor._hl_pools()
    print("[offline_pooled] HL pools:", [(str(p.get("source") or p["name"]), round(p["weight"], 4)) for p in pools], flush=True)

    log_f = open(out / "train_log.jsonl", "a")
    grad_steps = 0
    saved = set()
    for u in range(1, n_updates + 1):
        ns = min(steps_per_update, FLAGS.total_grad_steps - grad_steps)
        info = actor.update_hl(batch_size=int(cfg["hl_update_batch_size"]), num_steps=ns, global_step=grad_steps + ns)
        if not info or "loss" not in info:
            raise SystemExit(f"update_hl declined at update {u}: {info} ({actor._last_hl_skip_reason})")
        grad_steps += ns
        # Per-route composition of this batch's online rows (from the batch dump just written).
        rec = {"update": u, "grad_steps": grad_steps, **{k: float(v) for k, v in info.items()}}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        wandb.log({f"hl/{k}": v for k, v in rec.items()}, step=grad_steps)
        if (grad_steps % FLAGS.checkpoint_every == 0 or grad_steps == FLAGS.total_grad_steps) and grad_steps not in saved:
            if actor.save_checkpoint(ckpt_dir, step=grad_steps, keep_last=0) is None:
                raise SystemExit(f"checkpoint export failed at {grad_steps}")
            saved.add(grad_steps)
    log_f.close()
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["checkpoints"] = sorted(saved)
    (out / "train_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[offline_pooled] done: {grad_steps} grad steps, checkpoints {sorted(saved)}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
