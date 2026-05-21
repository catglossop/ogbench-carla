"""FastAPI HTTP server exposing OpenPI SteerVLA via :meth:`openpi.policies.policy.Policy.infer_with_cot`.

Matches the endpoints used by :class:`~impls.vlas.utils.RemoteActor`:

- ``GET  /get_info``
- ``POST /gen_action`` — body: CARLA-style observation dict (``image``, ``state``; optional ``routing_command``).
  Runs full CoT + flow (same as ``infer_with_cot``); returns the first predicted action step, flattened.
- ``POST /gen_cot`` — same body; runs ``infer_with_cot`` and returns token buffers + ``policy_timing`` (also
  computes actions on the server; omit large ``actions`` from the JSON response).
- ``POST /update`` — no-op ack.

Observation layout is translated into OpenPI SteerVLA policy inputs (``observation/image``, ``prompt``, etc.);
see :class:`openpi.policies.steervla_policy.SteerVLAInputs`.

Run (from repo root, with CARLA/OpenPI deps installed):

.. code-block:: bash

   PYTHONPATH=. python -m impls.vlas.steervla_server \\
       --actor-config pi05_steervla_cot_ki \\
       --checkpoint /path/to/checkpoint \\
       --cot-jit-decode false \\
       --cot-jit-transformer-forward false \\
       --host 0.0.0.0 --port 8000

Alternatively set ``STEERVLA_ACTOR_CONFIG`` and ``STEERVLA_CHECKPOINT`` (and optional vars below)
and invoke ``python -m impls.vlas.steervla_server`` with no positional args required.

Environment (optional): ``STEERVLA_ROUTING_COMMAND``, ``STEERVLA_COT_TEMPERATURE``,
``STEERVLA_SAMPLE_ACTIONS_NUM_STEPS``, ``STEERVLA_SAMPLE_ACTIONS_LOW_MEMORY``,
``STEERVLA_SAMPLE_ACTIONS_JIT_DENOISE_STEPS``, ``STEERVLA_TRAINING_GPU_RANK``,
``STEERVLA_COT_JIT_DECODE``, ``STEERVLA_COT_JIT_TRANSFORMER_FORWARD``,
``STEERVLA_COT_REPLAY_REASONING`` (passed into ``Pi0CoTConfig`` / ``sample_cot``).

CLI can override CoT options without env vars: ``--cot-jit-decode false``,
``--cot-jit-transformer-forward false``, ``--cot-replay-reasoning false``, or suffix
``true`` explicitly.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from openpi.policies import policy as openpi_policy
from openpi.shared import download
from openpi.training import checkpoints as openpi_checkpoints
from openpi.training import config as openpi_train_config
import openpi.transforms as openpi_transforms
from openpi.models.pi0_config import Pi0CoTConfig

if __package__ in (None, ""):
    from steervla import CARLA_STEERVLA_IMAGE_KEYS, restore_openpi_params_on_single_gpu
else:
    from .steervla import CARLA_STEERVLA_IMAGE_KEYS, restore_openpi_params_on_single_gpu

# Tuple (not list): ``policy.Policy`` passes ``image_keys`` as a ``jax.jit`` static argument (must be hashable).
_POLICY_IMAGE_KEYS = CARLA_STEERVLA_IMAGE_KEYS

_INFO_ATTRS = (
    "actor_config",
    "checkpoint_path",
    "routing_command",
    "cot_temperature",
    "sample_actions_num_steps",
    "sample_actions_low_memory",
    "sample_actions_jit_denoise_steps",
    "cot_jit_decode",
    "cot_jit_transformer_forward",
    "cot_replay_reasoning",
)

_STATE_REQUIRED_KEYS = ("image", "state")


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None else v


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return default if v is None else float(v)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v is None else int(v)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _normalize_obs_payload(body: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in _STATE_REQUIRED_KEYS if k not in body]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing observation keys: {missing}")
    out = dict(body)
    out["image"] = np.asarray(out["image"], dtype=np.uint8)
    out["state"] = np.asarray(out["state"], dtype=np.float32)
    return out


def _jsonify_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonify_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify_numpy(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj




def _parse_optional_bool_cli(cli_value: str | None, env_name: str, env_if_unset_default: bool) -> bool:
    """If ``cli_value`` is ``'true'`` or ``'false'``, use it; otherwise read ``env_name``."""
    if cli_value is not None:
        return cli_value == "true"
    return _env_bool(env_name, env_if_unset_default)


def _steervla_server_kwargs(
    *,
    actor_config: str | None,
    checkpoint_path: str | None,
    cot_jit_decode_cli: str | None = None,
    cot_jit_transformer_forward_cli: str | None = None,
    cot_replay_reasoning_cli: str | None = None,
) -> dict[str, Any]:
    ac = actor_config or os.environ.get("STEERVLA_ACTOR_CONFIG")
    ck = checkpoint_path or os.environ.get("STEERVLA_CHECKPOINT")
    if not ac or not ck:
        raise ValueError(
            "Provide OpenPI TrainConfig name and checkpoint via STEERVLA_ACTOR_CONFIG "
            "and STEERVLA_CHECKPOINT (or CLI --actor-config / --checkpoint)."
        )
    return {
        "actor_config": ac,
        "checkpoint_path": ck,
        "routing_command": _env_str("STEERVLA_ROUTING_COMMAND", "Follow the route."),
        "cot_temperature": _env_float("STEERVLA_COT_TEMPERATURE", 0.0),
        "sample_actions_num_steps": _env_int("STEERVLA_SAMPLE_ACTIONS_NUM_STEPS", 10),
        "sample_actions_low_memory": _env_bool("STEERVLA_SAMPLE_ACTIONS_LOW_MEMORY", True),
        "sample_actions_jit_denoise_steps": _env_bool("STEERVLA_SAMPLE_ACTIONS_JIT_DENOISE_STEPS", False),
        "training_gpu_rank": _env_int("STEERVLA_TRAINING_GPU_RANK", -1),
        "cot_jit_decode": _parse_optional_bool_cli(
            cot_jit_decode_cli, "STEERVLA_COT_JIT_DECODE", True
        ),
        "cot_jit_transformer_forward": _parse_optional_bool_cli(
            cot_jit_transformer_forward_cli, "STEERVLA_COT_JIT_TRANSFORMER_FORWARD", True
        ),
        "cot_replay_reasoning": _parse_optional_bool_cli(
            cot_replay_reasoning_cli, "STEERVLA_COT_REPLAY_REASONING", True
        ),
    }


def _pop_request_options(body: dict[str, Any]) -> dict[str, Any]:
    opts = body.pop("__steervla_options__", None)
    return opts if isinstance(opts, dict) else {}


def _request_cot_sample_kwargs(cfg: dict[str, Any], request_options: dict[str, Any]) -> dict[str, Any]:
    replay_reasoning = request_options.get("replay_reasoning", cfg.get("cot_replay_reasoning", True))
    return {"replay_reasoning": bool(replay_reasoning)}


def _infer_with_request_options(
    policy: openpi_policy.Policy,
    policy_obs: dict[str, Any],
    *,
    cot_sample_kwargs: dict[str, Any],
) -> dict[str, Any]:
    original = dict(policy._cot_sample_kwargs)
    try:
        policy._cot_sample_kwargs = {**original, **cot_sample_kwargs}
        return policy.infer_with_cot(policy_obs)
    finally:
        policy._cot_sample_kwargs = original


def _carla_gym_to_policy_obs(raw: dict[str, Any], *, default_routing: str) -> dict[str, Any]:
    """Map Bench2Drive / CARLA gym dict to keys expected by :class:`openpi.policies.steervla_policy.SteerVLAInputs`."""
    img = raw["image"]
    state_vec = np.asarray(raw["state"], dtype=np.float32).reshape(-1)
    spd = float(state_vec[15])
    yaw_deg = float(state_vec[5])
    obs_state = np.array([spd, yaw_deg], dtype=np.float32)
    rc = raw.get("routing_command")
    prompt = (rc if isinstance(rc, str) and rc.strip() else default_routing).strip()
    subtask = raw.get("subtask", "Dummy subtask")
    reasoning = raw.get("reasoning", "Dummy reasoning")
    return {
        "observation/image": img,
        "observation/state": obs_state,
        "observation/current_speed": spd,
        "prompt": prompt,
        "subtask": subtask,
        "reasoning": reasoning,
    }


def _build_steervla_openpi_policy(cfg: dict[str, Any]) -> openpi_policy.Policy:
    """Same transform stack as :func:`openpi.policies.policy_config.create_trained_policy`, single-GPU restore."""
    train_cfg = openpi_train_config.get_config(cfg["actor_config"])
    ckpt_root = Path(download.maybe_download(str(cfg["checkpoint_path"]))).resolve()
    params_dir = ckpt_root / "params"
    if not params_dir.is_dir():
        raise FileNotFoundError(f"Expected OpenPI checkpoint params at {params_dir}")

    restored, _ = restore_openpi_params_on_single_gpu(
        params_dir, training_gpu_rank=int(cfg["training_gpu_rank"])
    )

    model_cfg = train_cfg.model
    if isinstance(model_cfg, Pi0CoTConfig):
        model_cfg = dataclasses.replace(
            model_cfg,
            cot_jit_decode=bool(cfg["cot_jit_decode"]),
            cot_jit_transformer_forward=bool(cfg["cot_jit_transformer_forward"]),
            cot_replay_reasoning=bool(cfg["cot_replay_reasoning"]),
        )
    model = model_cfg.load(restored)
    model.eval()

    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_config.asset_id is None:
        raise ValueError("TrainConfig data requires asset_id to load norm stats for the policy.")
    norm_stats = openpi_checkpoints.load_norm_stats(ckpt_root / "assets", data_config.asset_id)

    repack = openpi_transforms.Group()
    sample_kwargs = {
        "num_steps": int(cfg["sample_actions_num_steps"]),
        "low_memory_denoise": bool(cfg["sample_actions_low_memory"]),
        "jit_denoise_steps": bool(cfg["sample_actions_jit_denoise_steps"]),
        "image_keys": _POLICY_IMAGE_KEYS,
    }
    cot_sample_kwargs = {
        "temperature": float(cfg["cot_temperature"]),
        "image_keys": _POLICY_IMAGE_KEYS,
        "replay_reasoning": bool(cfg["cot_replay_reasoning"]),
    }

    return openpi_policy.Policy(
        model,
        transforms=[
            *repack.inputs,
            openpi_transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            openpi_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            openpi_transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack.outputs,
        ],
        sample_kwargs=sample_kwargs,
        cot_sample_kwargs=cot_sample_kwargs,
        metadata=train_cfg.policy_metadata,
        is_pytorch=False,
        pytorch_device=None,
    )


def create_app(
    *,
    actor_config: str | None = None,
    checkpoint_path: str | None = None,
    cot_jit_decode_cli: str | None = None,
    cot_jit_transformer_forward_cli: str | None = None,
    cot_replay_reasoning_cli: str | None = None,
) -> FastAPI:
    """Build ASGI app; loads OpenPI :class:`~openpi.policies.policy.Policy` on startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            kw = _steervla_server_kwargs(
                actor_config=actor_config,
                checkpoint_path=checkpoint_path,
                cot_jit_decode_cli=cot_jit_decode_cli,
                cot_jit_transformer_forward_cli=cot_jit_transformer_forward_cli,
                cot_replay_reasoning_cli=cot_replay_reasoning_cli,
            )
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        print(
            f"[steervla_server] lifespan: loading OpenPI Policy (infer_with_cot) "
            f"(actor_config={kw['actor_config']!r}, checkpoint={kw['checkpoint_path']!r}, "
            f"cot_jit_decode={kw['cot_jit_decode']}, "
            f"cot_jit_transformer_forward={kw['cot_jit_transformer_forward']}, "
            f"cot_replay_reasoning={kw['cot_replay_reasoning']})",
            flush=True,
        )
        app.state.policy = _build_steervla_openpi_policy(kw)
        app.state.policy_lock = Lock()
        app.state.server_cfg = {k: kw[k] for k in _INFO_ATTRS}
        print("[steervla_server] lifespan: policy ready.", flush=True)
        yield

    app = FastAPI(title="SteerVLA policy (infer_with_cot)", lifespan=lifespan)

    @app.get("/get_info")
    def get_info() -> dict[str, Any]:
        policy: openpi_policy.Policy = app.state.policy
        info: dict[str, Any] = {
            "name": "steervla_pi0_cot",
            "service": "steervla_server",
            "inference": "openpi.policies.policy.Policy.infer_with_cot",
        }
        cfg = getattr(app.state, "server_cfg", {})
        for k in _INFO_ATTRS:
            if k in cfg:
                info[k] = cfg[k]
        md = getattr(policy, "metadata", None)
        if isinstance(md, dict) and md:
            info["policy_metadata"] = dict(md)
        return info

    @app.post("/gen_action")
    def gen_action(body: dict[str, Any]) -> list[float]:
        cfg = getattr(app.state, "server_cfg", {})
        request_options = _pop_request_options(body)
        obs = _normalize_obs_payload(body)
        policy_obs = _carla_gym_to_policy_obs(obs, default_routing=str(cfg.get("routing_command", "Follow the route.")))
        policy: openpi_policy.Policy = app.state.policy
        with app.state.policy_lock:
            out = _infer_with_request_options(
                policy,
                policy_obs,
                cot_sample_kwargs=_request_cot_sample_kwargs(cfg, request_options),
            )
        act = np.asarray(out["actions"], dtype=np.float32)
        return act.reshape(-1).tolist()

    @app.post("/gen_cot")
    def gen_cot(body: dict[str, Any]) -> dict[str, Any]:
        cfg = getattr(app.state, "server_cfg", {})
        request_options = _pop_request_options(body)
        obs = _normalize_obs_payload(body)
        policy_obs = _carla_gym_to_policy_obs(obs, default_routing=str(cfg.get("routing_command", "Follow the route.")))
        policy: openpi_policy.Policy = app.state.policy
        try:
            with app.state.policy_lock:
                out = _infer_with_request_options(
                    policy,
                    policy_obs,
                    cot_sample_kwargs=_request_cot_sample_kwargs(cfg, request_options),
                )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        payload = {
            "tokenized_reasoning": out["tokenized_reasoning"],
            "tokenized_reasoning_mask": out["tokenized_reasoning_mask"],
            "tokenized_subtask": out["tokenized_subtask"],
            "tokenized_subtask_mask": out["tokenized_subtask_mask"],
            "policy_timing": out.get("policy_timing", {}),
        }
        if "tokenized_fast" in out:
            payload["tokenized_fast"] = out["tokenized_fast"]
            payload["tokenized_fast_mask"] = out["tokenized_fast_mask"]
        return _jsonify_numpy(payload)

    @app.post("/update")
    def update() -> dict[str, Any]:
        return {"ok": True, "message": "No weight update applied (Pi0-CoT inference-only server)."}

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="SteerVLA FastAPI server for RemoteActor clients.")
    p.add_argument("--actor-config", type=str, default=None, help="OpenPI TrainConfig key (env: STEERVLA_ACTOR_CONFIG).")
    p.add_argument("--checkpoint", type=str, default=None, help="Checkpoint URI or path (env: STEERVLA_CHECKPOINT).")
    p.add_argument("--host", type=str, default=os.environ.get("STEERVLA_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("STEERVLA_PORT", "8000")))
    p.add_argument(
        "--cot-jit-decode",
        nargs="?",
        const="true",
        choices=("true", "false"),
        default=None,
        metavar="{true|false}",
        help="Pi0CoT decode (logits/argmax) nnx.jit. Omit to use STEERVLA_COT_JIT_DECODE / default true.",
    )
    p.add_argument(
        "--cot-jit-transformer-forward",
        nargs="?",
        const="true",
        choices=("true", "false"),
        default=None,
        metavar="{true|false}",
        help="Pi0CoT single-token transformer nnx.jit; omit for env STEERVLA_COT_JIT_TRANSFORMER_FORWARD / default true.",
    )
    p.add_argument(
        "--cot-replay-reasoning",
        nargs="?",
        const="true",
        choices=("true", "false"),
        default=None,
        metavar="{true|false}",
        help="Replay generated reasoning before subtask generation; omit for STEERVLA_COT_REPLAY_REASONING / default true.",
    )
    args = p.parse_args()
    if args.actor_config:
        os.environ["STEERVLA_ACTOR_CONFIG"] = args.actor_config
    if args.checkpoint:
        os.environ["STEERVLA_CHECKPOINT"] = args.checkpoint
    if args.cot_jit_decode is not None:
        os.environ["STEERVLA_COT_JIT_DECODE"] = args.cot_jit_decode
    if args.cot_jit_transformer_forward is not None:
        os.environ["STEERVLA_COT_JIT_TRANSFORMER_FORWARD"] = args.cot_jit_transformer_forward
    if args.cot_replay_reasoning is not None:
        os.environ["STEERVLA_COT_REPLAY_REASONING"] = args.cot_replay_reasoning

    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit(
            "uvicorn is required to run the server. Install optional deps, e.g. "
            "`pip install 'ogbench[carla]'` (includes fastapi/uvicorn) or "
            "`pip install uvicorn fastapi`."
        ) from e

    uvicorn.run(
        create_app(
            actor_config=args.actor_config,
            checkpoint_path=args.checkpoint,
            cot_jit_decode_cli=args.cot_jit_decode,
            cot_jit_transformer_forward_cli=args.cot_jit_transformer_forward,
            cot_replay_reasoning_cli=args.cot_replay_reasoning,
        ),
        host=args.host,
        port=args.port,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()
else:
    app = create_app()
