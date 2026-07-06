"""Verify the residual-RL action decode path equals the reference (main_carla.py) path.

Background
----------
``main_carla_residual.py`` (this branch) returns the **raw** normalized SteerVLA
chunk from the actor and lets the CARLA env denormalize it with
``action_input_space="normalized"`` (``steervla_simlingo_control._denormalize_action_chunk``
-> ``openpi ... denormalize_actions``).

``main_carla.py`` (routing-commands / master) returns the **postprocessed** chunk
(``SteerVLAActor._postprocess_action_trajectory`` = OpenPI ``model_transforms.outputs``
+ disabled Unnormalize + ``_SliceActionDim`` + fixed ``denormalize_actions`` scaling) and
lets the env pass it through with ``action_input_space="policy_output"`` (identity slice).

These two paths are only equivalent if ``model_transforms.outputs`` is a pure slice
(true for a Pi05 model, where the factory adds *no* output transforms) and OpenPI
``Normalize`` / ``Unnormalize`` stay disabled. This script asserts both structural
facts and then checks numerical equivalence on random chunks, so the equivalence
claim regresses loudly if the checkpoint/config ever changes (e.g. a PI0_FAST config
whose ``model_transforms.outputs`` is ``ExtractFASTActions``).

Run (on a box where ``openpi`` is installed, e.g. the training GPU host)::

    uv run python impls/tools/verify_residual_decode.py \
        --actor_config pi05_steervla_cot_simplified_reasoning

Exits 0 on success or clean skip (openpi unavailable), 1 on a real mismatch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_IMPLS_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPLS_ROOT))


def _skip(msg: str) -> int:
    print(f"[verify_residual_decode] SKIP: {msg}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actor_config", default="pi05_steervla_cot_simplified_reasoning")
    ap.add_argument("--output_action_format", default="DELTA_XY_T_DELTA_XY_SPACE")
    ap.add_argument("--action_horizon", type=int, default=10)
    ap.add_argument("--action_dim", type=int, default=4, help="Env action dim (vla_action_dim).")
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args()

    try:
        from openpi.training import config as openpi_train_config
        from openpi.visualizing.steervla_visualization import denormalize_actions
    except Exception as exc:  # openpi not installed on this host (e.g. mac dev box).
        return _skip(f"openpi import failed ({exc!r}); run this on the training host.")

    try:
        from vlas.steervla import (
            build_openpi_policy_transforms,
            steervla_physical_denormalize_actions,
        )
    except Exception as exc:
        return _skip(f"impls.vlas.steervla import failed ({exc!r}).")

    train_cfg = openpi_train_config.get_config(args.actor_config)
    model_cfg = train_cfg.model
    model_ah = int(getattr(model_cfg, "action_horizon", args.action_horizon))
    model_ad = int(getattr(model_cfg, "action_dim", 32))
    env_ah = int(args.action_horizon)
    env_ad = int(args.action_dim)
    fmt = str(args.output_action_format)

    # ---- structural guards: the two decode paths are equivalent only under these. --
    data_config, input_transform, output_transform = build_openpi_policy_transforms(
        train_cfg, Path("/tmp/__verify_residual_decode_dummy__")
    )
    model_outputs = list(getattr(data_config.model_transforms, "outputs", []))
    if model_outputs:
        names = [type(t).__name__ for t in model_outputs]
        print(
            "[verify_residual_decode] FAIL: model_transforms.outputs is non-empty "
            f"({names}). The env 'normalized' path skips these transforms, so it is "
            "NOT equivalent to the reference postprocess path. This checkpoint/config "
            "needs the reference (postprocess + policy_output) decode instead."
        )
        return 1

    # Norm must be disabled (Normalize/Unnormalize commented out / gated off). If a
    # Normalize/Unnormalize slipped into the transforms, values would differ.
    def _has_norm(compose_fn) -> bool:
        inner = getattr(compose_fn, "transforms", []) or []
        return any("ormalize" in type(t).__name__ for t in inner)

    if _has_norm(input_transform) or _has_norm(output_transform):
        print(
            "[verify_residual_decode] FAIL: OpenPI Normalize/Unnormalize is enabled; the "
            "raw-chunk 'normalized' env path does not apply it, breaking equivalence."
        )
        return 1

    # ---- numerical check on random model-space chunks. ----------------------------
    rng = np.random.default_rng(args.seed)
    max_abs = 0.0
    for _ in range(int(args.trials)):
        # Raw model output space: (B=1, model_ah, model_ad), roughly normalized units.
        traj = rng.standard_normal((1, model_ah, model_ad)).astype(np.float32)

        # Path A — main_carla_residual.py: actor returns raw traj[:, :env_ah, :env_ad];
        # env decodes with action_input_space="normalized" == denormalize_actions(...).
        raw_chunk = traj[:, :env_ah, :env_ad]
        path_a = np.asarray(denormalize_actions(raw_chunk, env_ad, fmt), dtype=np.float64)

        # Path B — main_carla.py: actor returns _postprocess_action_trajectory(traj)
        # (model_transforms.outputs + SliceActionDim + fixed denormalize scaling); env
        # passes it through with action_input_space="policy_output" (identity slice).
        state = np.zeros((1, env_ad), dtype=np.float32)
        post = output_transform({"actions": traj[0], "state": state[0]})["actions"]
        post = steervla_physical_denormalize_actions(
            np.asarray(post)[None], action_dim=env_ad, output_action_format=fmt
        )
        path_b = np.asarray(post[:, :env_ah, :env_ad], dtype=np.float64)

        max_abs = max(max_abs, float(np.max(np.abs(path_a - path_b))))

    if max_abs > float(args.atol):
        print(
            f"[verify_residual_decode] FAIL: decode paths differ (max|A-B|={max_abs:.3e} "
            f"> atol={args.atol:.1e})."
        )
        return 1

    print(
        "[verify_residual_decode] OK: 'normalized' (raw-chunk) decode == reference "
        f"postprocess+'policy_output' decode over {args.trials} trials "
        f"(max|A-B|={max_abs:.3e}; model={model_ah}x{model_ad}, env={env_ah}x{env_ad}, fmt={fmt})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
