"""Build an offline SimLingo HL replay pool for ``vlas.simlingo_steervla`` (``steervla.hl_replay_pools``).

Samples frames from the SimLingo database and writes one ``hl_samples.json`` in the HL-pool layout
``HLPoolSamplingMixin._scan_pool`` reads. Each sample carries the HL training target exactly as
``simlingo_training/dataloader/dataset_driving.py`` builds it for ``model_type: hl`` with
``use_refined_labels`` + ``use_reasoning_traces`` + ``use_ego_state_history``:

    prompt  = "Speed history: ... Heading history: ...\\nCurrent speed: X m/s\\nCommand: <routing>"
    answer  = "<vehicle_movement_description> <traffic_light_status>\\n\\nDriving Behavior: <meta_action>"

``reasoning`` is the part before the marker, ``subtask`` the meta action. With ``--simplified``
(the bellman HL's ``use_simplified_reasoning``) coordinate clauses are removed with simlingo's own
``filter_coordinate_clauses``.

Images: by default ``sample_file`` is the absolute path of the database JPEG, which
``SimLingoSteerVLAActor._read_hl_record`` decodes on the fly. With ``--embed-images`` the frame is
decoded once and stored beside the manifest as ``sample_%06d.npz`` (key ``image``, RGB uint8) and
``sample_file`` becomes that relative name -- the same self-contained layout as the SteerVLA pools in
``steervla_hl_pools/``. Use it when the pool has to move to a machine that does not mount the SimLingo
database (~1.6 MB per 512x1024 frame, so ~6.3 GB for 4000).

    .venv/bin/python impls/vlas/extract_simlingo_hl_replay.py \\
      --data-path /raid/datasets/simlingo/database/simlingo \\
      --simlingo-source-root /home/cglossop/simlingo-steervla \\
      --out-root /raid/users/cglossop/simlingo_hl_pools --name simlingo_hl_simplified --n 4000

    # self-contained copy (images in the pool):
    .venv/bin/python impls/vlas/extract_simlingo_hl_replay.py ... --embed-images \\
      --name simlingo_hl_simplified_img
"""

from __future__ import annotations

import argparse
import glob
import gzip
import importlib.util
import json
import random
from pathlib import Path

# ``dataset_base.get_navigational_conditioning`` (route_as="command", non-lmdrive branch).
MAP_COMMAND = {
    1: "go left at the next intersection",
    2: "go right at the next intersection",
    3: "go straight at the next intersection",
    4: "follow the road",
    5: "do a lane change to the left",
    6: "do a lane change to the right",
}


def command_string(measurement: dict) -> str:
    import numpy as np

    dist = int(np.linalg.norm(np.asarray(measurement["target_point"], dtype=np.float64)))
    command = MAP_COMMAND[int(measurement["command"])]
    next_command = MAP_COMMAND[int(measurement["next_command"])]
    suffix = f" then {next_command}" if command != next_command else ""
    if int(measurement["command"]) == 4:
        return f"Command: {command}{suffix}."
    return f"Command: {command} in {dist} meter{suffix}."


def ego_history_line(trace: dict, count: int) -> str:
    speeds = list(trace.get("speed_history") or [])[:count]
    headings = list(trace.get("heading_history") or [])[:count]
    if not speeds or not headings:
        return ""
    speed_str = " ".join(f"{round(s, 1)} m/s" for s in speeds)
    heading_str = " ".join(f"{round(h, 1)} degrees" for h in headings)
    return f"Speed history: {speed_str} Heading history: {heading_str}\n"


def _load_gz(path: Path) -> dict | None:
    try:
        with gzip.open(path, "rt") as f:
            return json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _load_filter(source_root: Path):
    path = source_root / "scripts" / "preprocess_reasoning_traces.py"
    spec = importlib.util.spec_from_file_location("simlingo_preprocess_reasoning_traces", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.filter_coordinate_clauses


def build_sample(db: Path, route_rel: str, frame: int, *, simplify, ego_history_count: int) -> dict | None:
    name = f"{frame:04d}"
    image = db / "data" / "simlingo" / route_rel / "rgb" / f"{name}.jpg"
    measurement = _load_gz(db / "data" / "simlingo" / route_rel / "measurements" / f"{name}.json.gz")
    trace = _load_gz(db / "reasoning_traces" / "simlingo" / route_rel / "reasoning" / f"{name}.json.gz")
    meta = _load_gz(db / "meta_action" / "simlingo" / route_rel / "meta_action" / f"{name}.json.gz")
    if not image.is_file() or measurement is None or trace is None or meta is None:
        return None
    subtask = " ".join(str(meta.get("commentary") or "").replace("..", ".").split())
    if not subtask:
        return None
    vehicle = str(trace.get("vehicle_movement_description") or "")
    traffic = str(trace.get("traffic_light_status") or "")
    if simplify is not None:
        vehicle, traffic = simplify(vehicle), simplify(traffic)
    reasoning = " ".join(part for part in (vehicle, traffic) if part)
    speed = round(float(measurement["speed"]), 1)
    prompt = f"{ego_history_line(trace, ego_history_count)}Current speed: {speed} m/s\n{command_string(measurement)}"
    return {
        "sample_file": str(image),
        "prompt": prompt.replace("..", "."),
        "subtask": subtask,
        "reasoning": reasoning,
        "label": None,
        "route": route_rel,
        "frame": frame,
        "current_speed": float(measurement["speed"]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", required=True, help="SimLingo database root (contains data/, meta_action/, reasoning_traces/).")
    p.add_argument("--simlingo-source-root", required=True, help="simlingo-steervla checkout (for filter_coordinate_clauses).")
    p.add_argument("--out-root", required=True)
    p.add_argument("--name", default="simlingo_hl_simplified")
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-first-n-frames", type=int, default=10, help="dataset skip_first_n_frames")
    p.add_argument("--ego-history-count", type=int, default=3)
    p.add_argument("--simplified", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    p.add_argument(
        "--embed-images",
        action="store_true",
        help="Store each frame in the pool as sample_%%06d.npz (RGB uint8) instead of pointing at the "
        "database JPEG, so the pool is self-contained. ~1.6 MB per frame.",
    )
    args = p.parse_args()

    db = Path(args.data_path)
    simplify = _load_filter(Path(args.simlingo_source_root)) if args.simplified else None
    route_dirs = sorted(
        d for d in glob.glob(str(db / "data" / "simlingo" / "*" / "*" / "*" / "Town*")) if "validation" not in d
    )
    if not route_dirs:
        raise SystemExit(f"no route dirs under {db}/data/simlingo")
    rng = random.Random(args.seed)
    samples, seen, attempts = [], set(), 0
    while len(samples) < args.n and attempts < args.n * 50:
        attempts += 1
        route = Path(rng.choice(route_dirs))
        frames = sorted(int(f.name[:4]) for f in (route / "measurements").glob("*.json.gz"))
        frames = [f for f in frames if f >= args.skip_first_n_frames]
        if not frames:
            continue
        frame = rng.choice(frames)
        route_rel = str(route.relative_to(db / "data" / "simlingo"))
        if (route_rel, frame) in seen:
            continue
        seen.add((route_rel, frame))
        sample = build_sample(db, route_rel, frame, simplify=simplify, ego_history_count=args.ego_history_count)
        if sample is not None:
            samples.append(sample)
            if len(samples) % 500 == 0:
                print(f"{len(samples)}/{args.n} samples ({attempts} frames tried)", flush=True)

    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.embed_images:
        # Decode once into the pool. RGB, because _read_hl_record's JPEG path converts BGR->RGB and the
        # two layouts must feed the model identical pixels. Uncompressed, like the SteerVLA pools: the
        # trainer reads these at random every HL update and decode time is what matters, not disk.
        import cv2  # local: only this mode needs OpenCV
        import numpy as np

        kept = []
        for i, s in enumerate(samples):
            src = s["sample_file"]
            bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"  unreadable, dropping: {src}", flush=True)
                continue
            npz_name = f"sample_{len(kept):06d}.npz"
            np.savez(
                out_dir / npz_name,
                image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8),
                current_speed=np.float32(s["current_speed"]),
            )
            s = dict(s, sample_file=npz_name, source_image=str(src))
            kept.append(s)
            if len(kept) % 500 == 0:
                print(f"  embedded {len(kept)}/{len(samples)} images", flush=True)
        samples = kept
    manifest = {
        "dataset_format": "simlingo_hl_dataset_format",
        "schema_version": 1,
        "action_supervision": False,
        "supervise_fast": False,
        "num_samples": len(samples),
        "source": {k: v for k, v in vars(args).items()},
        "samples": samples,
    }
    (out_dir / "hl_samples.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(samples)} samples ({attempts} frames tried) -> {out_dir / 'hl_samples.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
