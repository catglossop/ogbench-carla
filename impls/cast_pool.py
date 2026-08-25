"""Shared filesystem protocol for a **pooled CAST-relabel run**.

A pooled run is N rollout workers (one CARLA job per route) plus **one** trainer process:

* Each worker runs the ordinary ``cast_relabel`` observer, but writes its HL samples into a
  *shared* pool (``cast_relabel.hl_dataset_root``), so the pool accumulates corrected subtasks
  from every route at once. Workers are **inference-only** (``load_trainable_params=False``) and
  take no gradient steps -- see ``impls/main_carla.py``'s ``cast_pool`` wiring.
* The trainer (``impls/train_hl_pooled.py``) owns the only trainable OpenPI ``TrainState``. It
  watches the pool, and once ``round_new_samples`` fresh samples have landed it runs one large
  ``update_hl`` over the pooled corpus, exports a params-only checkpoint, and **publishes** it.
* Workers watch the checkpoint dir and hot-reload the newest published version mid-episode, so
  every route keeps driving the same (latest) policy without any barrier or restart.

Everything is coordinated through the filesystem -- there is no RPC and no shared memory, which
keeps a worker crash (routine, with CARLA) from taking the round down. Two invariants make that
safe against readers seeing half-written state:

**Checkpoints are published, not just written.** ``save_checkpoint`` writes ``<root>/<v>/params``
over many seconds; a worker scanning for new versions must not pick that up mid-write. So the
trainer writes :data:`PUBLISH_SENTINEL` into the version dir *after* the params are durable, and
:func:`discover_latest_version` only ever returns versions carrying that sentinel.

**Sample dirs are renamed into place.** A window dir is built under a ``.tmp-`` prefix and
``os.replace``d to its final name, so the trainer's glob never matches a manifest whose ``.npz``
files are still being written. The prefix is skipped by :func:`iter_pool_manifests`.

Layout::

    <pool_root>/                         # cast_relabel.hl_dataset_root, shared by all workers
      <worker_run_tag>/                  # one per worker (route+seed+timestamp); see run_tag
        ep0001_win0001/hl_samples.json   # + sample_XXXX.npz
        .tmp-ep0001_win0002/             # in flight, ignored by readers
    <checkpoint_root>/                   # cast_pool.checkpoint_dir
      1/params/...                       # version 1, published
      1/PUBLISHED.json
      2/params/...                       # version 2, still writing -- no sentinel, ignored
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Written into a checkpoint version dir *after* its params are durable. A version without this
# file is invisible to workers (see the module docstring).
PUBLISH_SENTINEL = "PUBLISHED.json"

# Prefix for a sample/window dir that is still being written. Readers skip these.
TMP_PREFIX = ".tmp-"

# Per-sample manifest key holding the policy version that *produced* the sample. Stamped at chunk
# capture time (not at window-flush time) because workers hot-reload mid-episode, so a single
# window can straddle two policy versions.
POLICY_VERSION_KEY = "policy_version"


# ── checkpoint publication ───────────────────────────────────────────────────────────


def publish_checkpoint(ckpt_root: str | Path, version: int, meta: dict[str, Any] | None = None) -> Path:
    """Mark ``<ckpt_root>/<version>`` complete so workers may load it.

    Call **after** :meth:`vlas.steervla.SteerVLAActor.save_checkpoint` has returned, never before:
    the sentinel is the only thing distinguishing a finished checkpoint from one mid-write.
    """
    version_dir = Path(ckpt_root) / str(int(version))
    version_dir.mkdir(parents=True, exist_ok=True)
    payload = {"version": int(version), "published_at": time.time(), **(meta or {})}
    sentinel = version_dir / PUBLISH_SENTINEL
    # Write-then-rename so a worker can never read a half-serialized sentinel.
    tmp = version_dir / (TMP_PREFIX + PUBLISH_SENTINEL)
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, sentinel)
    return version_dir


def discover_latest_version(
    ckpt_root: str | Path, *, min_version: int = 0
) -> tuple[int, Path] | None:
    """Newest *published* version strictly greater than ``min_version``, or ``None``.

    Returns ``(version, version_dir)``. Unpublished, malformed, and non-numeric dirs are skipped,
    so this is safe to poll against a trainer that is mid-export.
    """
    root = Path(ckpt_root)
    if not root.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for child in root.iterdir():
        if not child.is_dir() or child.name.startswith(TMP_PREFIX):
            continue
        try:
            version = int(child.name)
        except ValueError:
            continue
        if version <= int(min_version):
            continue
        if not (child / PUBLISH_SENTINEL).is_file():
            continue  # still being written by the trainer.
        if not (child / "params").is_dir():
            continue
        if best is None or version > best[0]:
            best = (version, child)
    return best


# ── pool inspection ──────────────────────────────────────────────────────────────────


def iter_pool_manifests(pool_root: str | Path) -> Iterator[Path]:
    """Yield every finished ``hl_samples.json`` under a pooled root.

    Handles both depths: ``<root>/<window>/hl_samples.json`` (a single-run dir, as a non-pooled
    cast_relabel run writes) and ``<root>/<worker>/<window>/hl_samples.json`` (pooled). Anything
    under a :data:`TMP_PREFIX` dir is in flight and skipped.
    """
    root = Path(pool_root)
    if not root.is_dir():
        return
    for manifest in root.glob("*/hl_samples.json"):
        if not any(p.name.startswith(TMP_PREFIX) for p in manifest.relative_to(root).parents):
            yield manifest
    for manifest in root.glob("*/*/hl_samples.json"):
        if not any(p.name.startswith(TMP_PREFIX) for p in manifest.relative_to(root).parents):
            yield manifest


def count_pool_samples(pool_root: str | Path) -> tuple[int, int]:
    """``(total_samples, num_windows)`` across every worker in the pool.

    Reads only the manifests' ``num_samples`` field -- it never opens the ``.npz`` payloads, so
    this stays cheap enough to poll on a short interval as the pool grows.
    """
    total = 0
    windows = 0
    for manifest in iter_pool_manifests(pool_root):
        try:
            data = json.loads(manifest.read_text())
        except Exception:
            continue  # torn or malformed read; it will be counted on a later poll.
        total += int(data.get("num_samples", 0) or 0)
        windows += 1
    return total, windows


def pool_worker_names(pool_root: str | Path) -> list[str]:
    """Worker run-tags that have contributed at least one finished window, for logging."""
    root = Path(pool_root)
    names: set[str] = set()
    for manifest in iter_pool_manifests(pool_root):
        rel = manifest.relative_to(root).parts
        if len(rel) >= 3:  # <worker>/<window>/hl_samples.json
            names.add(rel[0])
    return sorted(names)


# ── atomic sample-dir publication (worker side) ──────────────────────────────────────


def staging_dir_for(final_dir: str | Path) -> Path:
    """Sibling ``.tmp-<name>`` path to build ``final_dir`` in before renaming it into place."""
    final_dir = Path(final_dir)
    return final_dir.parent / (TMP_PREFIX + final_dir.name)


def commit_staged_dir(staged: str | Path, final_dir: str | Path) -> Path:
    """Atomically move a fully-written staging dir to its final name.

    ``os.replace`` on a directory is atomic within a filesystem, which is what makes the trainer's
    glob safe: a window is either absent or complete, never partially visible.

    A pre-existing target is **never** deleted. Window tags (``ep0004_win0012``) are unique within a
    process, so the only way to collide is across a crash-restart: ``run_carla.sh`` keeps
    ``--exp_name`` stable across retries (so the W&B run resumes), but ``run_online_carla`` restarts
    ``episode_count``/``window_count`` at 0, so a restarted worker regenerates tags it already used
    under the same ``run_tag``. Overwriting there would destroy complete, already-VLM-reviewed
    windows -- and because the trainer gates rounds on ``count_pool_samples`` growing past a
    high-water mark it never lowers, a shrinking pool stalls training outright. So a colliding
    window is parked at ``<tag>__r2``, ``<tag>__r3``, ... instead; those still match the reader
    globs, so the samples stay in the pool.
    """
    staged, final_dir = Path(staged), Path(final_dir)
    target = final_dir
    attempt = 2
    while target.exists():
        target = final_dir.parent / f"{final_dir.name}__r{attempt}"
        attempt += 1
    os.replace(staged, target)
    return target
