"""Central trainer for a **pooled CAST-relabel run**.

Runs *without* CARLA. It owns the only trainable OpenPI ``TrainState`` in the pool and does one
thing in a loop:

1. Watch the shared HL pool (``cast_pool.count_pool_samples``) written by N rollout workers.
2. Once ``round_new_samples`` fresh samples have landed, take one large ``update_hl`` round over
   the pooled corpus -- every route's corrections in the same batch.
3. Export a params-only checkpoint and **publish** it (``cast_pool.publish_checkpoint``).
4. Workers hot-reload it mid-episode and keep driving; go back to 1.

There is no barrier: workers never wait for the trainer and the trainer never waits for a
specific worker, so a CARLA crash on one route costs that route's share of the next round and
nothing else. Coordination is entirely through the filesystem -- see :mod:`cast_pool` for the
publication invariants that keep readers from seeing half-written state.

Usage (normally launched by ``run_cast_pool.sh``, which starts the workers alongside it)::

    uv run python impls/train_hl_pooled.py \
        --agent=impls/configs/steervla_cast_pooled_config.py \
        --pool_root=/raid/users/cglossop/cast_pool/<run>/pool \
        --checkpoint_dir=/raid/users/cglossop/cast_pool/<run>/checkpoints \
        --training_gpu=7
"""

import os
import sys
import time
from pathlib import Path

# Match main_carla.py: put ``impls/`` on sys.path so intra-impls imports are top-level.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from absl import app, flags
from cast_pool import count_pool_samples, discover_latest_version, pool_worker_names, publish_checkpoint
from ml_collections import config_flags

import wandb

FLAGS = flags.FLAGS

flags.DEFINE_string("pool_root", "", "Shared HL sample pool written by the rollout workers (required).")
flags.DEFINE_string("checkpoint_dir", "", "Where to publish params-only policy versions (required).")
flags.DEFINE_integer("training_gpu", -1, "JAX GPU index for the trainable state (-1 = default device).")
flags.DEFINE_integer("max_rounds", 0, "Stop after N rounds (0 = run until killed).")
flags.DEFINE_string("run_name", "", "wandb run name; defaults to the pool dir name.")
flags.DEFINE_string("wandb_project", "OGBench-CARLA", "wandb project.")
flags.DEFINE_string("wandb_group", "cast_pool", "wandb group.")

config_flags.DEFINE_config_file(
    "agent",
    "impls/configs/steervla_cast_pooled_config.py",
    lock_config=False,
)


def _cfg(config, path, default=None):
    """Read a possibly-absent nested ConfigDict key without raising."""
    node = config
    for part in path.split("."):
        if node is None or part not in node:
            return default
        node = node[part]
    return node


def main(_):
    config = FLAGS.agent
    pool_cfg = config.get("cast_pool") or {}

    pool_root = Path(FLAGS.pool_root or _cfg(config, "cast_pool.pool_root", "")).expanduser()
    ckpt_dir = Path(FLAGS.checkpoint_dir or _cfg(config, "cast_pool.checkpoint_dir", "")).expanduser()
    if not str(pool_root) or not str(ckpt_dir):
        raise ValueError("--pool_root and --checkpoint_dir are required (or set them in cast_pool.*).")
    pool_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    round_new_samples = int(pool_cfg.get("round_new_samples", 90))
    round_batch_size = int(pool_cfg.get("round_batch_size", 128))
    round_num_steps = int(pool_cfg.get("round_num_steps", 32))
    poll_interval = float(pool_cfg.get("poll_interval_sec", 15.0))
    keep_last_ckpts = int(pool_cfg.get("checkpoint_keep_last", 5))
    max_rounds = int(FLAGS.max_rounds or pool_cfg.get("max_rounds", 0))

    training_gpu = int(FLAGS.training_gpu if FLAGS.training_gpu >= 0 else config.get("training_gpu_rank", -1))

    run_name = FLAGS.run_name or f"cast_pool_trainer_{pool_root.parent.name}"
    wandb.init(
        project=FLAGS.wandb_project,
        group=FLAGS.wandb_group,
        name=run_name,
        config={
            "role": "trainer",
            "pool_root": str(pool_root),
            "checkpoint_dir": str(ckpt_dir),
            "round_new_samples": round_new_samples,
            "round_batch_size": round_batch_size,
            "round_num_steps": round_num_steps,
            "keep_last_rounds": _cfg(config, "steervla.hl_keep_last_rounds", 0),
        },
        mode=os.environ.get("WANDB_MODE", "online"),
    )

    # Build the trainable actor. Imported late so the flag parsing above (and any JAX device pinning
    # it implies) happens before JAX initializes.
    from vlas.steervla import create_steervla_pi0_cot_sample_fn

    steervla_cfg = dict(config.get("steervla") or {})
    steervla_cfg["load_trainable_params"] = True  # the trainer is the only trainable process.
    steervla_cfg["hl_training_gpu_rank"] = training_gpu
    steervla_cfg["training_gpu_rank"] = training_gpu

    print(f"[cast_pool.trainer] building trainable SteerVLA on GPU {training_gpu}...", flush=True)
    # No env here, so the raw-observation holder the rollout sample_fn would read stays empty --
    # the trainer only ever calls update_hl, which reads its batches off the pool.
    _sample_fn, actor = create_steervla_pi0_cot_sample_fn(
        steervla_cfg, {}, training_gpu_rank=training_gpu
    )
    actor.hl_dataset_dir = pool_root
    # The trainer is driven by rounds, not by a per-env-step throttle: every call must do work.
    actor.hl_update_every = 1
    print(
        f"[cast_pool.trainer] watching pool {pool_root} — a round fires every "
        f"{round_new_samples} new samples (batch {round_batch_size} x {round_num_steps} steps).",
        flush=True,
    )

    # Resume: if versions were already published (a restarted trainer), continue numbering after
    # the newest rather than overwriting a checkpoint some worker may already be running.
    latest = discover_latest_version(ckpt_dir)
    version = latest[0] if latest else 0
    if latest:
        print(f"[cast_pool.trainer] resuming after published version {version}.", flush=True)

    seen_samples, _ = count_pool_samples(pool_root)
    print(f"[cast_pool.trainer] pool starts at {seen_samples} samples.", flush=True)
    rounds_done = 0
    last_log = 0.0

    while True:
        total, windows = count_pool_samples(pool_root)
        new = total - seen_samples
        if new < round_new_samples:
            if time.time() - last_log > 60.0:
                workers = pool_worker_names(pool_root)
                print(
                    f"[cast_pool.trainer] waiting: {new}/{round_new_samples} new samples "
                    f"(pool {total} in {windows} windows from {len(workers)} workers).",
                    flush=True,
                )
                last_log = time.time()
            time.sleep(poll_interval)
            continue

        version += 1
        t0 = time.time()
        print(
            f"[cast_pool.trainer] round {rounds_done + 1} -> version {version}: "
            f"{new} new samples (pool {total}); running {round_num_steps} steps at batch {round_batch_size}.",
            flush=True,
        )
        info = actor.update_hl(
            batch_size=round_batch_size,
            num_steps=round_num_steps,
            global_step=version,
        )
        if not info:
            # update_hl declined (pool below hl_min_online_samples, or nothing readable). Don't burn
            # the version number or the new-sample credit -- retry on the next poll.
            version -= 1
            print("[cast_pool.trainer] update_hl returned nothing this round; retrying after poll.", flush=True)
            time.sleep(poll_interval)
            continue
        train_secs = time.time() - t0

        # Export, then publish. The sentinel goes last so no worker can load a partial checkpoint.
        saved = actor.save_checkpoint(ckpt_dir, step=version, keep_last=keep_last_ckpts)
        if saved is None:
            print("[cast_pool.trainer] checkpoint export failed; not publishing this version.", flush=True)
            version -= 1
            time.sleep(poll_interval)
            continue
        publish_checkpoint(
            ckpt_dir,
            version,
            meta={
                "pool_samples": total,
                "new_samples": new,
                "round": rounds_done + 1,
                "train_seconds": train_secs,
                "workers": pool_worker_names(pool_root),
            },
        )
        seen_samples = total
        rounds_done += 1
        total_secs = time.time() - t0
        print(
            f"[cast_pool.trainer] published version {version} in {total_secs:.0f}s "
            f"(train {train_secs:.0f}s, export {total_secs - train_secs:.0f}s).",
            flush=True,
        )
        metrics = {f"cast_pool/{k}": float(v) for k, v in info.items()}
        metrics.update({
            "cast_pool/version": version,
            "cast_pool/pool_samples": total,
            "cast_pool/new_samples": new,
            "cast_pool/pool_windows": windows,
            "cast_pool/num_workers": len(pool_worker_names(pool_root)),
            "cast_pool/train_seconds": train_secs,
            "cast_pool/round_seconds": total_secs,
        })
        wandb.log(metrics, step=version)

        if max_rounds and rounds_done >= max_rounds:
            print(f"[cast_pool.trainer] reached max_rounds={max_rounds}; exiting.", flush=True)
            break

    wandb.finish()


if __name__ == "__main__":
    app.run(main)
