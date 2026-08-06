"""Offline converter: collected CAST-relabel HL samples -> an RLDS/TFDS SteerVLA dataset.

The collection half of the "collect-then-finetune" CAST variant
(``impls/configs/steervla_cast_collect_config.py``) rolls the frozen policy out for a large step
budget and writes every relabeled action chunk to disk as a ``steervla_hl_dataset_format`` sample::

    <hl-root>/<run_tag>/<window tag>/sample_0000.npz   # image, state, ego_hist, current_speed, actions
    <hl-root>/<run_tag>/<window tag>/hl_samples.json   # per-sample subtask/reasoning targets + provenance
    <hl-root>/<run_tag>/windows.jsonl                  # running per-window index (optional)

That layout is what the *online* ``SteerVLAActor.update_hl`` reads. This script converts the same
corpus into the *offline* form: a TFDS dataset laid out like the SimLingo RLDS datasets that
``openpi.training.steervla_rlds_dataset`` consumes, so steervla-pi can train on it at a large batch
size with no changes to the loader.

Target schema
-------------
Fields are exactly the ones ``_build_simlingo_restructure`` reads
(``DatasetFormat.SIMLINGO``), as configured by ``pi05_steervla_cot_simplified_reasoning_no_ego_history``::

    steps/observation/image                        jpeg, (image_size, image_size, 3)
    steps/observation/ego_hist                     (ego_history_len, 2) f32  raw [speed m/s, yaw deg]
    steps/speed                                    f32  current speed, m/s
    steps/routing_command                          text  bare instruction; the loader prefixes
                                                         "The current speed is X m/s. " itself
    steps/action/future_10_xy_delta_t              (10, 2) f32  meters (loader divides by 7.0)
    steps/action/future_10_xy_delta_space          (10, 2) f32  (loader uses as-is)
    steps/action/future_10_speed_course_delta_t    (10, 2) f32  PLACEHOLDER zeros -- see below
    steps/action/future_10_course_delta_space      (10,)   f32  PLACEHOLDER zeros -- see below
    steps/<subtask field>                          text  CoT subtask target   (default "prompt")
    steps/<reasoning field>                        text  CoT reasoning target (default "gemini_refined_label")

The two CoT field *names* come from the OpenPI train config (``hl_cot_subtask_key`` /
``hl_cot_reasoning_key``) when ``--actor-config`` is given, so the dataset is written under whatever
names the loader will read. For ``pi05_steervla_cot_simplified_reasoning_no_ego_history`` those are
``prompt`` (subtask) and ``gemini_refined_label`` (reasoning) -- note this is *not* the model's input
prompt, which the loader builds from ``routing_command`` + ``speed``.

Action round-trip
-----------------
The stored chunk is the executed OpenPI chunk **already denormalized to physical units** by
``SteerVLAActor._postprocess_action_trajectory`` (``denormalize_actions``: ``[:, :2] * 7``, ``[:, 2:]``
unchanged, for ``DELTA_XY_T_DELTA_XY_SPACE``). That is precisely the raw RLDS action, so the split is
an identity: columns 0:2 -> ``future_10_xy_delta_t``, columns 2:4 -> ``future_10_xy_delta_space``.
Feeding this dataset back through the loader reproduces the model-space action bit for bit.

``future_10_speed_course_delta_t`` / ``future_10_course_delta_space`` are written as **zeros**: the
rollout only ever produced the xy formulation, and under ``DELTA_XY_T_DELTA_XY_SPACE`` the loader
computes then discards them. Training with any other ``output_action_format`` on this dataset would
silently learn zeros, so the converter refuses unless ``--allow-placeholder-actions`` is passed.

The action fields are written regardless of ``action_supervision`` because the loader always reads
them; whether they are *supervised* is decided at training time by which mapping the dataset is
registered in (``hl_dataset_name_weight_mappings`` -> masked, ``dataset_name_weight_mappings`` ->
supervised). ``--split supervision`` emits two datasets so that choice can be made per half: the
corrective samples (subtask replaced, executed action no longer matches it) and the reinforcing ones
(original subtask kept, action still consistent -> safe to action-supervise).

Running it
----------
Needs ``tensorflow`` + ``tensorflow_datasets``, which the CARLA venv deliberately does not carry
(same constraint as ``impls/vlas/extract_hl_replay.py``). Run it in a TF-equipped env::

    uv run --with tensorflow-cpu --with tensorflow_datasets \\
        python impls/vlas/cast_hl_to_rlds.py \\
        --hl-root /raid/users/cglossop/cast_collect \\
        --out-dir /raid/datasets/steervla \\
        --dataset-name cast_relabel_hl_v1 \\
        --actor-config pi05_steervla_cot_simplified_reasoning_no_ego_history

``--dry-run`` needs neither TF nor openpi and just reports what would be written (sample counts,
GOOD/BAD/precursor split, per-route breakdown) -- use it to sanity-check a collection run in
progress.

The result is a self-describing TFDS directory (``<out-dir>/<dataset-name>/1.0.0/``). No builder
class has to be importable at training time: ``tfds.builder(name, data_dir=...)`` -- what the OpenPI
loader calls -- falls back to reading ``features.json`` from disk. Register it in the train config::

    hl_dataset_name_weight_mappings={"simplified_reasoning_dataset": 1.85, "cast_relabel_hl_v1": 1.0}
"""

from __future__ import annotations

import argparse
import json
import re
import types
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# RLDS action-chunk length baked into the SimLingo field names (``future_10_*``). The loader slices
# to the model's action_horizon, so this is the stored width, not the trained one.
RLDS_CHUNK_LEN = 10

# The only rollout action layout the collected chunks can be in; see "Action round-trip" above.
SUPPORTED_OUTPUT_ACTION_FORMAT = "DELTA_XY_T_DELTA_XY_SPACE"

# Defaults matching pi05_steervla_cot_simplified_reasoning_no_ego_history's RLDSSteerVLACoTDataConfig.
DEFAULT_SUBTASK_FIELD = "prompt"
DEFAULT_REASONING_FIELD = "gemini_refined_label"

# Square stretch applied to the stored 144x256 CARLA frame. 224 is what the model actually consumes
# (``ResizeImages(224, 224)``) and what the online HL update feeds it, so storing at 224 makes the
# offline pipeline pixel-identical to the online one instead of round-tripping through 512.
DEFAULT_IMAGE_SIZE = 224

# Indices into the raw CARLA ego-state vector; mirrors coaches.cast_relabel.
EGO_STATE_IDX_YAW = 5
EGO_STATE_IDX_SPEED = 15

# PaliGemma CoT segment sentinels, and the speed prefix the loader re-adds itself.
_LOC_SENTINEL_RE = re.compile(r"<loc\d+>")
_SPEED_PREFIX_RE = re.compile(r"^\s*The current speed is\s*-?[\d.]+\s*m/s\.\s*")


# ── loading the collected corpus (pure numpy; no TF, no openpi) ───────────────────────────


@dataclass
class CastSample:
    """One collected HL sample, resolved from its manifest entry + ``.npz``."""

    npz_path: Path
    # Manifest provenance.
    run_tag: str
    route: str
    episode: int
    window_index: int
    chunk_index: int
    episode_step: int
    global_step: int
    label: str | None
    credit_source: str
    action_matches_subtask: bool
    # Text targets.
    subtask: str
    reasoning: str
    routing_command: str
    prompt: str
    original_subtask: str
    original_reasoning: str
    current_speed: float

    @property
    def is_corrective(self) -> bool:
        """Subtask was replaced by the VLM, so the stored action no longer matches it."""
        return not self.action_matches_subtask

    def arrays(self) -> dict[str, np.ndarray]:
        with np.load(self.npz_path, allow_pickle=False) as data:
            return {k: np.asarray(data[k]) for k in data.files}


@dataclass
class CorpusStats:
    num_manifests: int = 0
    num_samples: int = 0
    skipped: Counter = field(default_factory=Counter)
    labels: Counter = field(default_factory=Counter)
    credit_sources: Counter = field(default_factory=Counter)
    routes: Counter = field(default_factory=Counter)
    runs: Counter = field(default_factory=Counter)
    corrective: int = 0
    reinforcing: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "num_manifests": self.num_manifests,
            "num_samples": self.num_samples,
            "num_corrective": self.corrective,
            "num_reinforcing": self.reinforcing,
            "skipped": dict(self.skipped),
            "labels": {str(k): v for k, v in self.labels.items()},
            "credit_sources": dict(self.credit_sources),
            "routes": dict(self.routes),
            "runs": dict(self.runs),
        }


def _strip_speed_prefix(prompt: str) -> str:
    """Recover the bare routing instruction from ``"The current speed is X m/s. <rc>"``.

    Only used for corpora written before ``routing_command`` was stored per sample; the loader
    re-adds the prefix from the ``speed`` field, so leaving it in would double it.
    """
    return _SPEED_PREFIX_RE.sub("", str(prompt or "")).strip()


def strip_cot_sentinels(text: Any) -> str:
    """Strip PaliGemma ``<locNNNN>`` sentinels from a stored CoT target.

    ``'<loc1022>The vehicle accelerates normally.;<loc1021>'`` -> ``'The vehicle accelerates normally.'``

    The rollout decode emits these around every CoT segment and ``cast_relabel`` stores the raw
    decoded string, so **every** reinforced sample on disk carries them (corrective samples get the
    VLM's clean text instead). The online HL reader strips them at read time
    (``vlas.steervla._read_hl_record``) precisely because training the backbone on them teaches it
    to emit sentinels as prose; the offline path has to do exactly the same. Duplicated from
    ``vlas.steervla.strip_cot_sentinels`` rather than imported so this script stays runnable in a
    bare TF env (importing steervla pulls in jax + openpi).
    """
    s = _LOC_SENTINEL_RE.sub(" ", str(text or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(" ;").strip()


def discover_manifests(roots: list[Path]) -> list[Path]:
    """Every ``hl_samples.json`` under the given roots, at any depth, deduplicated and sorted."""
    found: set[Path] = set()
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"--hl-root does not exist: {root}")
        if root.is_file() and root.name == "hl_samples.json":
            found.add(root.resolve())
            continue
        found.update(p.resolve() for p in root.rglob("hl_samples.json"))
    return sorted(found)


def load_corpus(
    roots: list[Path],
    *,
    keep: str = "all",
    min_subtask_chars: int = 1,
    limit: int | None = None,
) -> tuple[list[CastSample], CorpusStats]:
    """Read every manifest under ``roots`` into ``CastSample``s, filtering as requested.

    ``keep`` selects which half of the corpus to load: ``all``, ``corrective`` (BAD chunks whose
    subtask the VLM replaced) or ``reinforcing`` (GOOD/unlabeled chunks stored with the model's
    original subtask). Samples whose ``.npz`` is missing or whose subtask target is empty are
    skipped and counted in ``stats.skipped`` rather than failing the run -- a collection run that
    was killed mid-window can leave a half-written pair behind.
    """
    stats = CorpusStats()
    samples: list[CastSample] = []
    for manifest_path in discover_manifests(roots):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a half-written manifest must not abort the pass.
            stats.skipped["unreadable_manifest"] += 1
            continue
        stats.num_manifests += 1
        work_dir = manifest_path.parent
        # <hl-root>/<run_tag>/<window tag>/hl_samples.json -> run_tag is the grandparent's name.
        run_tag_fallback = work_dir.parent.name

        for entry in manifest.get("samples", []):
            sample_file = entry.get("sample_file")
            if not sample_file:
                stats.skipped["no_sample_file"] += 1
                continue
            npz_path = work_dir / str(sample_file)
            if not npz_path.is_file():
                stats.skipped["missing_npz"] += 1
                continue
            # Strip <loc> sentinels BEFORE the length gate: a reinforced sample's stored subtask is
            # the raw rollout decode, so an otherwise-empty one can still be several sentinel
            # characters long and would sneak past an un-stripped check.
            subtask = strip_cot_sentinels(entry.get("subtask"))
            if len(subtask) < max(1, int(min_subtask_chars)):
                stats.skipped["empty_subtask"] += 1
                continue

            prompt = str(entry.get("prompt") or "")
            routing_command = str(entry.get("routing_command") or "").strip()
            if not routing_command:
                routing_command = _strip_speed_prefix(prompt)

            sample = CastSample(
                npz_path=npz_path,
                run_tag=str(entry.get("run_tag") or run_tag_fallback),
                route=str(entry.get("route") or "unknown"),
                episode=int(entry.get("episode", -1)),
                window_index=int(entry.get("window_index", -1)),
                chunk_index=int(entry.get("chunk_index", -1)),
                episode_step=int(entry.get("episode_step", -1)),
                global_step=int(entry.get("global_step", -1)),
                label=(str(entry["label"]) if entry.get("label") else None),
                credit_source=str(entry.get("credit_source") or ""),
                action_matches_subtask=bool(entry.get("action_matches_subtask", False)),
                subtask=subtask,
                reasoning=strip_cot_sentinels(entry.get("reasoning")),
                routing_command=routing_command,
                prompt=prompt,
                original_subtask=strip_cot_sentinels(entry.get("original_subtask")),
                original_reasoning=str(entry.get("original_reasoning") or ""),
                current_speed=float(entry.get("current_speed", 0.0)),
            )
            if keep == "corrective" and not sample.is_corrective:
                stats.skipped["filtered_out"] += 1
                continue
            if keep == "reinforcing" and sample.is_corrective:
                stats.skipped["filtered_out"] += 1
                continue

            samples.append(sample)
            stats.num_samples += 1
            stats.labels[sample.label or "none"] += 1
            stats.credit_sources[sample.credit_source or "none"] += 1
            stats.routes[sample.route] += 1
            stats.runs[sample.run_tag] += 1
            if sample.is_corrective:
                stats.corrective += 1
            else:
                stats.reinforcing += 1
            if limit is not None and len(samples) >= limit:
                return samples, stats
    return samples, stats


def group_into_episodes(samples: list[CastSample], *, episode_key: str) -> list[list[CastSample]]:
    """Group samples into RLDS episodes, ordered within an episode by time.

    ``episode`` groups a whole CARLA episode (all its relabel windows) into one RLDS episode;
    ``window`` makes each relabel window its own. Both are only containers -- the loader flattens
    to frames and shuffles -- so the choice affects tfrecord example size and read granularity, not
    the training distribution.
    """
    buckets: dict[tuple, list[CastSample]] = {}
    for s in samples:
        if episode_key == "window":
            key = (s.run_tag, s.route, s.episode, s.window_index)
        else:
            key = (s.run_tag, s.route, s.episode)
        buckets.setdefault(key, []).append(s)
    out = []
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        out.append(sorted(buckets[key], key=lambda s: (s.window_index, s.episode_step, s.chunk_index)))
    return out


# ── array -> RLDS frame ───────────────────────────────────────────────────────────────────


def resize_stretch(image: np.ndarray, size: int) -> np.ndarray:
    """Distorting resize to ``(size, size)`` -- matches ``vlas.steervla.resize_stretch_np``.

    The SteerVLA pretraining frames were stretched to square, so a letterbox pad here would feed
    the backbone black bars it never saw. Kept identical to the online HL path on purpose.
    """
    img = np.asarray(image, dtype=np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[:2] == (size, size):
        return np.ascontiguousarray(img[..., :3])
    try:
        import cv2  # type: ignore

        resized = cv2.resize(img[..., :3], (size, size), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image  # type: ignore

        resized = np.asarray(Image.fromarray(img[..., :3]).resize((size, size), Image.BILINEAR))
    return np.ascontiguousarray(resized, dtype=np.uint8)


def _fit_ego_hist(ego_hist: Any, state: Any, current_speed: float, ego_history_len: int) -> np.ndarray:
    """``(ego_history_len, 2)`` of raw ``[speed, yaw]``, padded/truncated as needed.

    Falls back to tiling the current pair (read out of the raw CARLA state vector) for corpora
    written before ``ego_hist`` was stored.
    """
    if ego_hist is not None:
        arr = np.asarray(ego_hist, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] >= ego_history_len:
            return np.ascontiguousarray(arr[-ego_history_len:])
        pad = np.repeat(arr[:1], ego_history_len - arr.shape[0], axis=0)
        return np.ascontiguousarray(np.concatenate([pad, arr], axis=0))

    speed = float(current_speed)
    yaw = 0.0
    if state is not None:
        flat = np.asarray(state, dtype=np.float32).reshape(-1)
        if flat.size > EGO_STATE_IDX_SPEED:
            speed = float(flat[EGO_STATE_IDX_SPEED])
        if flat.size > EGO_STATE_IDX_YAW:
            yaw = float(flat[EGO_STATE_IDX_YAW])
    return np.tile(np.array([speed, yaw], dtype=np.float32), (ego_history_len, 1))


def _fit_action_chunk(actions: Any, action_dim: int = 4) -> np.ndarray:
    """``(RLDS_CHUNK_LEN, action_dim)`` physical chunk; pads short chunks with their last row."""
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, action_dim) if arr.size % action_dim == 0 else arr.reshape(1, -1)
    arr = arr[:, :action_dim]
    if arr.shape[1] < action_dim:
        arr = np.pad(arr, ((0, 0), (0, action_dim - arr.shape[1])))
    if arr.shape[0] >= RLDS_CHUNK_LEN:
        return np.ascontiguousarray(arr[:RLDS_CHUNK_LEN])
    pad_rows = RLDS_CHUNK_LEN - arr.shape[0]
    last = arr[-1:] if arr.shape[0] else np.zeros((1, action_dim), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate([arr, np.repeat(last, pad_rows, axis=0)], axis=0))


def build_frame(
    sample: CastSample,
    *,
    image_size: int,
    ego_history_len: int,
    subtask_field: str,
    reasoning_field: str,
) -> dict[str, Any]:
    """One RLDS step dict in the SimLingo layout the OpenPI loader reads."""
    arrays = sample.arrays()
    image = resize_stretch(arrays["image"], image_size)
    ego_hist = _fit_ego_hist(
        arrays.get("ego_hist"), arrays.get("state"), sample.current_speed, ego_history_len
    )
    chunk = _fit_action_chunk(arrays.get("actions"))
    speed = float(arrays["current_speed"]) if "current_speed" in arrays else sample.current_speed

    frame: dict[str, Any] = {
        "observation": {"image": image, "ego_hist": ego_hist},
        "action": {
            # Identity split of the physical chunk -- see the "Action round-trip" note above.
            "future_10_xy_delta_t": np.ascontiguousarray(chunk[:, 0:2]),
            "future_10_xy_delta_space": np.ascontiguousarray(chunk[:, 2:4]),
            # Placeholders: never produced by the xy rollout, discarded by the loader under
            # DELTA_XY_T_DELTA_XY_SPACE.
            "future_10_speed_course_delta_t": np.zeros((RLDS_CHUNK_LEN, 2), dtype=np.float32),
            "future_10_course_delta_space": np.zeros((RLDS_CHUNK_LEN,), dtype=np.float32),
        },
        "speed": np.float32(speed),
        "routing_command": sample.routing_command,
        # Provenance carried per frame so a trained-on sample can be traced back to its window and
        # its relabel verdict inspected without re-reading the corpus.
        "cast_label": sample.label or "none",
        "cast_credit_source": sample.credit_source or "none",
        "cast_action_matches_subtask": bool(sample.action_matches_subtask),
        "cast_original_subtask": sample.original_subtask,
        "cast_episode_step": np.int32(sample.episode_step),
        "cast_chunk_index": np.int32(sample.chunk_index),
    }
    # CoT targets under whatever names the OpenPI config will read them by.
    frame[subtask_field] = sample.subtask
    frame[reasoning_field] = sample.reasoning
    return frame


# ── OpenPI train-config resolution (optional) ─────────────────────────────────────────────


def resolve_openpi_params(actor_config: str) -> dict[str, Any]:
    """Pull the field names / formats this dataset must match out of an OpenPI ``TrainConfig``.

    Same trick as ``extract_hl_replay.py``: the source of truth for how the checkpoint was trained
    lives in steervla-pi, not here, so read it rather than restate it.
    """
    from openpi.training import config as openpi_train_config

    cfg = openpi_train_config.get_config(actor_config)
    data = cfg.data
    oaf = getattr(data, "output_action_format", None)
    return {
        "subtask_field": str(getattr(data, "hl_cot_subtask_key", DEFAULT_SUBTASK_FIELD)),
        "reasoning_field": str(getattr(data, "hl_cot_reasoning_key", DEFAULT_REASONING_FIELD)),
        "output_action_format": getattr(oaf, "name", str(oaf)),
        "include_ego_history": bool(getattr(data, "include_ego_history", False)),
        "action_chunk_size": int(getattr(cfg.model, "action_horizon", RLDS_CHUNK_LEN)),
    }


# ── TFDS writing ──────────────────────────────────────────────────────────────────────────


def _features_dict(tfds, *, image_size: int, ego_history_len: int, subtask_field: str, reasoning_field: str):
    step_features = {
        "observation": tfds.features.FeaturesDict(
            {
                # Encoded on disk: dlimp reads steps with decoding skipped and the OpenPI loader
                # calls tf.io.decode_image itself, so this MUST stay an encoded Image feature.
                "image": tfds.features.Image(
                    shape=(image_size, image_size, 3), dtype=np.uint8, encoding_format="jpeg"
                ),
                "ego_hist": tfds.features.Tensor(shape=(ego_history_len, 2), dtype=np.float32),
            }
        ),
        "action": tfds.features.FeaturesDict(
            {
                "future_10_xy_delta_t": tfds.features.Tensor(shape=(RLDS_CHUNK_LEN, 2), dtype=np.float32),
                "future_10_xy_delta_space": tfds.features.Tensor(shape=(RLDS_CHUNK_LEN, 2), dtype=np.float32),
                "future_10_speed_course_delta_t": tfds.features.Tensor(
                    shape=(RLDS_CHUNK_LEN, 2), dtype=np.float32
                ),
                "future_10_course_delta_space": tfds.features.Tensor(shape=(RLDS_CHUNK_LEN,), dtype=np.float32),
            }
        ),
        "speed": tfds.features.Scalar(dtype=np.float32),
        "routing_command": tfds.features.Text(),
        "cast_label": tfds.features.Text(),
        "cast_credit_source": tfds.features.Text(),
        "cast_action_matches_subtask": tfds.features.Scalar(dtype=np.bool_),
        "cast_original_subtask": tfds.features.Text(),
        "cast_episode_step": tfds.features.Scalar(dtype=np.int32),
        "cast_chunk_index": tfds.features.Scalar(dtype=np.int32),
        subtask_field: tfds.features.Text(),
        reasoning_field: tfds.features.Text(),
    }
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(step_features),
            "episode_metadata": tfds.features.FeaturesDict(
                {
                    "run_tag": tfds.features.Text(),
                    "route": tfds.features.Text(),
                    "episode": tfds.features.Scalar(dtype=np.int32),
                    "window_index": tfds.features.Scalar(dtype=np.int32),
                    "num_steps": tfds.features.Scalar(dtype=np.int32),
                }
            ),
        }
    )


def write_tfds_dataset(
    episodes: list[list[CastSample]],
    *,
    out_dir: Path,
    dataset_name: str,
    image_size: int,
    ego_history_len: int,
    subtask_field: str,
    reasoning_field: str,
    version: str,
    description: str,
) -> Path:
    """Build ``<out_dir>/<dataset_name>/<version>/`` as a TFDS dataset and return that path.

    The builder class is created on the fly so ``--dataset-name`` is free-form. Nothing needs to
    import it later: ``tfds.builder(name, data_dir=...)`` (what the OpenPI loader calls) falls back
    to ``builder_from_files``, which reconstructs the dataset from the ``features.json`` written here.
    """
    import tensorflow_datasets as tfds

    features = _features_dict(
        tfds,
        image_size=image_size,
        ego_history_len=ego_history_len,
        subtask_field=subtask_field,
        reasoning_field=reasoning_field,
    )

    def _generate() -> Iterator[tuple[str, dict[str, Any]]]:
        for i, episode in enumerate(episodes):
            head = episode[0]
            steps = [
                build_frame(
                    s,
                    image_size=image_size,
                    ego_history_len=ego_history_len,
                    subtask_field=subtask_field,
                    reasoning_field=reasoning_field,
                )
                for s in episode
            ]
            key = f"{head.run_tag}|{head.route}|ep{head.episode:05d}|win{head.window_index:05d}|{i:06d}"
            yield key, {
                "steps": steps,
                "episode_metadata": {
                    "run_tag": head.run_tag,
                    "route": head.route,
                    "episode": np.int32(head.episode),
                    "window_index": np.int32(head.window_index),
                    "num_steps": np.int32(len(steps)),
                },
            }

    # ``skip_registration``: TFDS registers every builder subclass by name in a process-global
    # table and rejects duplicates, so without this a second call in the same process (``--split
    # supervision`` writes two datasets) dies on "already registered". Registration only matters
    # for ``tfds.builder(name)`` name lookup, which nothing here needs: we instantiate the class
    # directly, and at *training* time the loader resolves the dataset from its on-disk
    # ``features.json`` via the read-only-builder fallback.
    class _Builder(tfds.core.GeneratorBasedBuilder, skip_registration=True):
        VERSION = tfds.core.Version(version)
        RELEASE_NOTES = {version: "CAST-relabel offline collection."}  # noqa: RUF012 - TFDS API.

        def _info(self) -> tfds.core.DatasetInfo:
            # Constructed directly rather than via ``dataset_info_from_configs``: that helper
            # resolves the builder's *package directory* to look for a bundled README, which a
            # dynamically-created class has no meaningful ``__module__`` for.
            return tfds.core.DatasetInfo(builder=self, description=description, features=features)

        def _split_generators(self, dl_manager):  # nothing to download.
            return {"train": _generate()}

        def _generate_examples(self):  # pragma: no cover - unused; _split_generators yields directly.
            yield from _generate()

    # TFDS derives the dataset name from the class name unless the class sets ``name`` itself.
    def _exec_body(ns: dict[str, Any]) -> None:
        ns["name"] = dataset_name
        ns["__module__"] = __name__

    builder_cls = types.new_class(
        "".join(part.capitalize() for part in dataset_name.split("_")) or "CastRelabelHlDataset",
        (_Builder,),
        {"skip_registration": True},
        _exec_body,
    )

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    builder = builder_cls(data_dir=str(out_dir))
    builder.download_and_prepare()
    return out_dir / dataset_name / version


# ── CLI ───────────────────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert collected CAST-relabel HL samples into an RLDS/TFDS SteerVLA dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--hl-root",
        action="append",
        required=True,
        help="Root of a collected corpus (cast_relabel.hl_dataset_root). Repeat to merge several.",
    )
    p.add_argument("--out-dir", default=None, help="TFDS data_dir to write into. Required unless --dry-run.")
    p.add_argument("--dataset-name", default="cast_relabel_hl_v1", help="TFDS dataset name (snake_case).")
    p.add_argument("--version", default="1.0.0", help="TFDS dataset version.")
    p.add_argument(
        "--actor-config",
        default=None,
        help=(
            "OpenPI TrainConfig to match (e.g. pi05_steervla_cot_simplified_reasoning_no_ego_history). "
            "Resolves the CoT field names and validates the action format. Needs openpi importable."
        ),
    )
    p.add_argument("--subtask-field", default=DEFAULT_SUBTASK_FIELD, help="Overridden by --actor-config.")
    p.add_argument("--reasoning-field", default=DEFAULT_REASONING_FIELD, help="Overridden by --actor-config.")
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Square stretch size.")
    p.add_argument("--ego-history-len", type=int, default=4, help="Pairs stored in observation/ego_hist.")
    p.add_argument(
        "--episode-key",
        choices=("episode", "window"),
        default="episode",
        help="RLDS episode granularity: one CARLA episode, or one relabel window.",
    )
    p.add_argument(
        "--split",
        choices=("none", "supervision"),
        default="none",
        help=(
            "'supervision' emits TWO datasets -- <name>_corrective (subtask replaced; register under "
            "hl_dataset_name_weight_mappings so the action stays masked) and <name>_reinforce "
            "(original subtask; may be action-supervised) -- instead of one combined dataset."
        ),
    )
    p.add_argument(
        "--keep",
        choices=("all", "corrective", "reinforcing"),
        default="all",
        help="Load only one half of the corpus. Ignored when --split supervision.",
    )
    p.add_argument("--limit", type=int, default=None, help="Stop after N samples (smoke tests).")
    p.add_argument("--min-subtask-chars", type=int, default=1, help="Drop samples with a shorter subtask.")
    p.add_argument(
        "--allow-placeholder-actions",
        action="store_true",
        help=(
            f"Proceed even when the resolved output_action_format is not {SUPPORTED_OUTPUT_ACTION_FORMAT}. "
            "The speed/course action fields are zeros, so any other format trains on placeholders."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Report what would be written; write nothing.")
    return p.parse_args(argv)


def _print_stats(stats: CorpusStats, episodes: list[list[CastSample]]) -> None:
    print(f"[cast_hl_to_rlds] manifests={stats.num_manifests} samples={stats.num_samples}")
    print(f"[cast_hl_to_rlds]   corrective={stats.corrective} reinforcing={stats.reinforcing}")
    print(f"[cast_hl_to_rlds]   labels={dict(stats.labels)} credit={dict(stats.credit_sources)}")
    print(f"[cast_hl_to_rlds]   runs={len(stats.runs)} routes={dict(stats.routes)}")
    if stats.skipped:
        print(f"[cast_hl_to_rlds]   skipped={dict(stats.skipped)}")
    if episodes:
        sizes = [len(e) for e in episodes]
        print(
            f"[cast_hl_to_rlds]   episodes={len(episodes)} "
            f"steps/episode min={min(sizes)} median={int(np.median(sizes))} max={max(sizes)}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    subtask_field, reasoning_field = args.subtask_field, args.reasoning_field
    if args.actor_config:
        params = resolve_openpi_params(args.actor_config)
        subtask_field = params["subtask_field"]
        reasoning_field = params["reasoning_field"]
        print(
            f"[cast_hl_to_rlds] {args.actor_config}: subtask_field={subtask_field!r} "
            f"reasoning_field={reasoning_field!r} output_action_format={params['output_action_format']} "
            f"include_ego_history={params['include_ego_history']}",
        )
        fmt = str(params["output_action_format"]).upper()
        if fmt != SUPPORTED_OUTPUT_ACTION_FORMAT and not args.allow_placeholder_actions:
            raise SystemExit(
                f"[cast_hl_to_rlds] {args.actor_config} trains with output_action_format={fmt}, but the "
                f"collected chunks are only {SUPPORTED_OUTPUT_ACTION_FORMAT}; the speed/course action "
                "fields would be zeros. Re-run with --allow-placeholder-actions if that is intended."
            )
    if subtask_field == reasoning_field:
        raise SystemExit("[cast_hl_to_rlds] subtask and reasoning fields must differ.")

    roots = [Path(r) for r in args.hl_root]
    if args.split == "supervision":
        jobs = [
            (f"{args.dataset_name}_corrective", "corrective"),
            (f"{args.dataset_name}_reinforce", "reinforcing"),
        ]
    else:
        jobs = [(args.dataset_name, args.keep)]

    if not args.dry_run and not args.out_dir:
        raise SystemExit("[cast_hl_to_rlds] --out-dir is required unless --dry-run.")

    for name, keep in jobs:
        print(f"\n[cast_hl_to_rlds] === {name} (keep={keep}) ===")
        samples, stats = load_corpus(
            roots, keep=keep, min_subtask_chars=args.min_subtask_chars, limit=args.limit
        )
        episodes = group_into_episodes(samples, episode_key=args.episode_key)
        _print_stats(stats, episodes)
        if not samples:
            print(f"[cast_hl_to_rlds] no samples for {name}; nothing written.")
            continue
        if args.dry_run:
            continue

        description = (
            "CAST-relabel offline collection: SteerVLA rollouts on CARLA Bench2Drive with per-chunk "
            "VLM credit assignment and corrected subtasks, in the SimLingo RLDS layout. "
            f"CoT targets: subtask={subtask_field!r}, reasoning={reasoning_field!r}. "
            "future_10_speed_course_delta_t / future_10_course_delta_space are ZERO placeholders "
            f"(rollouts are {SUPPORTED_OUTPUT_ACTION_FORMAT} only)."
        )
        built = write_tfds_dataset(
            episodes,
            out_dir=Path(args.out_dir),
            dataset_name=name,
            image_size=args.image_size,
            ego_history_len=args.ego_history_len,
            subtask_field=subtask_field,
            reasoning_field=reasoning_field,
            version=args.version,
            description=description,
        )
        stats_path = Path(built) / "cast_conversion_stats.json"
        stats_path.write_text(
            json.dumps(
                {
                    "dataset_name": name,
                    "keep": keep,
                    "hl_roots": [str(r) for r in roots],
                    "subtask_field": subtask_field,
                    "reasoning_field": reasoning_field,
                    "image_size": args.image_size,
                    "ego_history_len": args.ego_history_len,
                    "episode_key": args.episode_key,
                    "num_episodes": len(episodes),
                    "placeholder_action_fields": [
                        "future_10_speed_course_delta_t",
                        "future_10_course_delta_space",
                    ],
                    **stats.to_json(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[cast_hl_to_rlds] wrote {built} ({stats.num_samples} frames, {len(episodes)} episodes)")
        print(f"[cast_hl_to_rlds] stats -> {stats_path}")
        print(
            f"[cast_hl_to_rlds] register it in the OpenPI train config, e.g. "
            f'hl_dataset_name_weight_mappings={{"{name}": 1.0}} with rlds_data_dir="{args.out_dir}"'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
