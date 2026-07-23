"""Offline extractor: SteerVLA RLDS pretraining datasets -> npz HL replay pools.

The online CAST-relabel HL update (``SteerVLAActor.update_hl``) can mix in a small amount of the
original pretraining data to stabilize the VLM backbone (prevent catastrophic forgetting as it is
fine-tuned on relabeled subtasks). Reading the pretraining data live needs the TF/RLDS stack
(``tensorflow`` + ``tensorflow_datasets`` + ``dlimp``), which the CARLA training venv does not carry.

This script runs the openpi RLDS loader **once, offline** (in an env that HAS the TF stack) and dumps
a fixed subset of each pretraining dataset into an npz "pool" that the runtime loads with plain numpy
— no TF at training time. Each pool mirrors the steervla-pi mixing convention: it is one weighted
source, tagged with the supervision it should receive:

- ``action_supervision`` — regular datasets (``dataset_name_weight_mappings``, e.g. the SimLingo
  datasets) train the action flow head too (True); HL datasets
  (``hl_dataset_name_weight_mappings``, e.g. ``simplified_reasoning_dataset``) mask it (False).
- ``supervise_fast`` — whether the FAST action-token CE is supervised. For the ``simplified_reasoning``
  HL pool this is set False (train reasoning + subtask only), per the run's spec.

The restructure params (dataset_format / cot keys / ego-history / proprio-norm / action format) are
pulled from an OpenPI ``TrainConfig`` (``--actor-config``) so the pool matches how the checkpoint was
pretrained. The pool ``state`` is stored as the already-normalized proprio the restructure emits, and
tagged ``state_format=proprio`` so the runtime uses it directly (the online pool instead stores raw
CARLA ego vectors, ``state_format=carla_raw``).

Pool layout (one dir per dataset under ``--out-root``)::

    <out-root>/<dataset>/sample_000000.npz   # image (224x224x3), state (proprio), current_speed, actions
    <out-root>/<dataset>/hl_samples.json     # pool-level supervision flags + per-sample text targets

Example (run in a TF-equipped env)::

    uv run --extra <rlds-extra> python impls/vlas/extract_hl_replay.py \
        --data-dir /scratch/current/cglossop/steervla_datasets \
        --out-root /scratch/current/cglossop/steervla_hl_pools \
        --actor-config pi05_steervla_cot_simplified_reasoning \
        --dataset simlingo_dataset_all_img512_1116 --regular --n 2000 \
        --dataset simplified_reasoning_dataset --hl --supervise-fast=false --n 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Model image resolution (matches ResizeImages(224, 224) in the SteerVLA model transforms). Stored at
# this size so replay images stack directly with the (also-224) online HL images at update time.
HL_IMAGE_HW = (224, 224)


def _decode_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.ndarray):
        return _decode_str(x.reshape(-1)[0]) if x.size else ""
    return str(x)


def _resize_image(img: np.ndarray) -> np.ndarray:
    """Plain distorting resize (stretch) to ``HL_IMAGE_HW`` — no aspect-preserving pad.

    The SteerVLA pretraining dataset was preprocessed by stretching frames to square, so this keeps
    replay images identical to that distribution (and to the runtime HL loader's ``resize_stretch_np``
    / ``_resize_hl_image``, which now stretch too). For the typical already-square RLDS frame this is a
    plain downscale; a pad would introduce black bars the backbone was never trained on.
    """
    import cv2  # type: ignore

    img = np.asarray(img, dtype=np.uint8)
    height, width = HL_IMAGE_HW
    cur_h, cur_w = img.shape[:2]
    if (cur_h, cur_w) == (height, width):
        return np.ascontiguousarray(img)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized, dtype=np.uint8)


def _resolve_restructure_params(actor_config: str) -> dict:
    """Pull the SIMLINGO restructure params from an OpenPI TrainConfig so pools match pretraining."""
    from openpi.training import config as openpi_train_config

    cfg = openpi_train_config.get_config(actor_config)
    df = cfg.data
    model = cfg.model
    return {
        "dataset_format": getattr(df, "dataset_format"),
        "cot_reasoning_key": getattr(df, "cot_reasoning_key", "commentary"),
        "cot_subtask_key": getattr(df, "cot_subtask_key", "gemini_refined_label"),
        "hl_cot_reasoning_key": getattr(df, "hl_cot_reasoning_key", "gemini_refined_label"),
        "hl_cot_subtask_key": getattr(df, "hl_cot_subtask_key", "prompt"),
        "include_ego_history": bool(getattr(df, "include_ego_history", False)),
        "include_xy_action": bool(getattr(df, "include_xy_action", False)),
        "speed_in_prompt": bool(getattr(df, "speed_in_prompt", True)),
        "proprio_norm": bool(getattr(df, "proprio_norm", True)),
        "output_action_format": getattr(df, "output_action_format"),
        "lang_label_type": getattr(df, "lang_label_type"),
        "routing_command_in_prompt": bool(getattr(df, "routing_command_in_prompt", False)),
        "add_suffix_to_prompt": bool(getattr(df, "add_suffix_to_prompt", False)),
        "action_chunk_size": int(getattr(model, "action_horizon", 10)),
        "image_size": int(getattr(df, "image_size", 512)) if hasattr(df, "image_size") else 512,
    }


def extract_pool(
    *,
    data_dir: str,
    out_root: Path,
    dataset: str,
    is_hl: bool,
    n_samples: int,
    action_supervision: bool,
    supervise_fast: bool,
    params: dict,
    rlds_batch_size: int = 16,
) -> None:
    from openpi.training import steervla_rlds_dataset as srds

    spec = [srds.SteerVLARLDSDataset(name=dataset, weight=1.0)]
    kwargs = dict(
        data_dir=data_dir,
        batch_size=rlds_batch_size,
        dataset_format=params["dataset_format"],
        enable_cot=True,
        action_chunk_size=params["action_chunk_size"],
        include_ego_history=params["include_ego_history"],
        include_xy_action=params["include_xy_action"],
        speed_in_prompt=params["speed_in_prompt"],
        proprio_norm=params["proprio_norm"],
        output_action_format=params["output_action_format"],
        lang_label_type=params["lang_label_type"],
        routing_command_in_prompt=params["routing_command_in_prompt"],
        add_suffix_to_prompt=params["add_suffix_to_prompt"],
        image_size=params["image_size"],
        shuffle=True,
        shuffle_buffer_size=min(10_000, max(1000, n_samples)),
    )
    if is_hl:
        # HL source: restructure with the HL cot keys and action_supervision=False semantics.
        ds = srds.SteerVLARldsDataset(
            datasets=(),
            hl_datasets=spec,
            hl_dataset_format=params["dataset_format"],
            hl_cot_reasoning_key=params["hl_cot_reasoning_key"],
            hl_cot_subtask_key=params["hl_cot_subtask_key"],
            **kwargs,
        )
    else:
        ds = srds.SteerVLARldsDataset(
            datasets=spec,
            cot_reasoning_key=params["cot_reasoning_key"],
            cot_subtask_key=params["cot_subtask_key"],
            **kwargs,
        )

    pool_dir = out_root / dataset
    pool_dir.mkdir(parents=True, exist_ok=True)

    manifest_samples: list[dict] = []
    written = 0
    for batch in ds:
        bs = int(np.asarray(batch["actions"]).shape[0])
        images = np.asarray(batch["observation"]["image"])
        states = np.asarray(batch["observation"]["state"], dtype=np.float32)
        speeds = np.asarray(batch["observation"]["current_speed"], dtype=np.float32).reshape(-1)
        actions = np.asarray(batch["actions"], dtype=np.float32)
        prompts = batch["prompt"]
        subtasks = batch.get("subtask")
        reasonings = batch.get("reasoning")
        for i in range(bs):
            if written >= n_samples:
                break
            subtask = _decode_str(subtasks[i]) if subtasks is not None else ""
            reasoning = _decode_str(reasonings[i]) if reasonings is not None else ""
            if not subtask.strip():
                continue
            sample_file = f"sample_{written:06d}.npz"
            np.savez_compressed(
                pool_dir / sample_file,
                image=_resize_image(images[i]),
                state=np.asarray(states[i], dtype=np.float32).reshape(-1),
                current_speed=np.float32(speeds[i]),
                actions=np.asarray(actions[i], dtype=np.float32),
            )
            manifest_samples.append(
                {
                    "sample_file": sample_file,
                    "prompt": _decode_str(prompts[i]),
                    "subtask": subtask,
                    "reasoning": reasoning,
                    # Real, consistent pretraining action <-> subtask pair.
                    "action_matches_subtask": True,
                }
            )
            written += 1
        if written >= n_samples:
            break

    (pool_dir / "hl_samples.json").write_text(
        json.dumps(
            {
                "dataset_format": "steervla_hl_replay_pool",
                "source_dataset": dataset,
                "is_hl_dataset": bool(is_hl),
                # Pool-level supervision flags applied to every sample at load time.
                "action_supervision": bool(action_supervision),
                "supervise_fast": bool(supervise_fast),
                "state_format": "proprio",
                "num_samples": len(manifest_samples),
                "samples": manifest_samples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[extract_hl_replay] wrote {len(manifest_samples)} samples -> {pool_dir} "
        f"(is_hl={is_hl} action_supervision={action_supervision} supervise_fast={supervise_fast})",
        flush=True,
    )


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="Root holding the RLDS/TFDS dataset dirs.")
    ap.add_argument("--out-root", required=True, help="Where to write the npz replay pools.")
    ap.add_argument(
        "--actor-config",
        default="pi05_steervla_cot_simplified_reasoning",
        help="OpenPI TrainConfig to pull restructure params from (must match the checkpoint).",
    )
    ap.add_argument("--rlds-batch-size", type=int, default=16)
    # Override the restructure proprio params to match the RUNTIME actor (CARLA steervla config),
    # which may differ from the installed openpi TrainConfig. The stored ``state`` must be normalized
    # the same way the online actor normalizes ego state, or the replay proprio will not match.
    ap.add_argument("--proprio-norm", default=None, help="true/false; overrides the config's proprio_norm.")
    ap.add_argument(
        "--include-ego-history", default=None, help="true/false; overrides the config's include_ego_history."
    )
    # One --dataset group per pool; flags after it apply to that dataset.
    ap.add_argument("--dataset", action="append", default=[], help="Dataset name (repeatable).")
    ap.add_argument("--n", action="append", default=[], type=int, help="Samples for the matching --dataset.")
    ap.add_argument("--kind", action="append", default=[], help="'regular' or 'hl' for the matching --dataset.")
    ap.add_argument(
        "--supervise-fast",
        action="append",
        default=[],
        help="true/false FAST supervision for the matching --dataset (default: regular=true, hl=true).",
    )
    args = ap.parse_args()

    if not args.dataset:
        raise SystemExit("Pass at least one --dataset.")
    if len(args.n) != len(args.dataset):
        raise SystemExit("Provide one --n per --dataset.")
    kinds = args.kind or ["regular"] * len(args.dataset)
    if len(kinds) != len(args.dataset):
        raise SystemExit("Provide one --kind per --dataset (or none for all-regular).")

    params = _resolve_restructure_params(args.actor_config)
    if args.proprio_norm is not None:
        params["proprio_norm"] = _str2bool(args.proprio_norm)
    if args.include_ego_history is not None:
        params["include_ego_history"] = _str2bool(args.include_ego_history)
    print(
        f"[extract_hl_replay] restructure params: proprio_norm={params['proprio_norm']} "
        f"include_ego_history={params['include_ego_history']} "
        f"output_action_format={params['output_action_format']} action_chunk_size={params['action_chunk_size']}",
        flush=True,
    )
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for idx, name in enumerate(args.dataset):
        is_hl = str(kinds[idx]).strip().lower() == "hl"
        action_supervision = not is_hl  # regular -> flow supervised; hl -> masked.
        if idx < len(args.supervise_fast):
            supervise_fast = _str2bool(args.supervise_fast[idx])
        else:
            supervise_fast = True  # default: FAST supervised (matches pretraining) unless overridden.
        extract_pool(
            data_dir=args.data_dir,
            out_root=out_root,
            dataset=name,
            is_hl=is_hl,
            n_samples=int(args.n[idx]),
            action_supervision=action_supervision,
            supervise_fast=supervise_fast,
            params=params,
            rlds_batch_size=int(args.rlds_batch_size),
        )


if __name__ == "__main__":
    main()
