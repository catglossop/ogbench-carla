"""SteerVLA / OpenPI: smoke-test loader, remote actor, and CARLA Pi0-CoT inference.

**Checkpoint smoke tests** — :class:`SteerVLAActor` (local checkpoint mode) restores OpenPI weights **once**, builds
:class:`openpi.training.utils.TrainingState` via :func:`init_train_state`, and shares those params with the Pi0-CoT
module used at inference.

**Pi0-CoT + DSRL** — :class:`SteerVLAActor` and :func:`create_steervla_pi0_cot_sample_fn`
implement ``vla_sample_fn`` for :class:`jax_agents.dsrl.DSRLAgent.sample_actions_with_vla`.

Uses ``openpi.models.pi0_cot.Pi0CoT.sample_cot`` then ``sample_actions`` (see
``openpi/visualizing/steervla_visualization.py``). Prompt layout follows
:class:`openpi.models.tokenizer.CoTPaligemmaTokenizer` (``Prompt:...;State:...``
through ``<start_of_reasoning>``). When ``Pi0CoTConfig.use_fast_tokens`` is enabled,
``sample_cot`` autoregressively generates FAST action tokens; ``sample_actions`` attends
to subtask + FAST (not reasoning).

Routing text mirrors ``SteerVLAInputs`` / ``simlingo/team_code/agent_steervla.py``:
``The current speed is X m/s. <routing_command>``. Continuous ``Observation.state``
uses :func:`openpi.policies.steervla_policy.normalize_ego_state` from CARLA
``obs["state"]`` (speed index 15, yaw index 5).

**Training:** Action-expert-only updates belong in OpenPI (``TrainConfig.freeze_filter``).
DSRL keeps OpenPI inference frozen in the online loop unless you add an NNx bridge.
:class:`SteerVLAActor` uses per-step helpers inside ``Pi0CoT.sample_cot``.

:class:`SteerVLAActor` ``update`` runs a VLA train step when DSRL is wired via :meth:`SteerVLAActor.attach_dsrl`.

``raw_obs_holder["obs"]`` must be the latest gym dict with ``"state"`` and ``"image"``
before each VLA sample when using :meth:`SteerVLAActor.__call__` (``main_carla`` maintains this).
For :meth:`SteerVLAActor.get_action` / :meth:`SteerVLAActor.get_cot`, pass that dict as ``state``.
"""

from __future__ import annotations

import dataclasses
import functools
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, MutableMapping, Optional

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os

from openpi.models import model as _openpi_model
from openpi.models import pi0 as _openpi_pi0
from openpi.models.pi0_config import Pi0CoTConfig
from openpi.models.tokenizer import CoTPaligemmaTokenizer
from openpi.policies import steervla_policy as sv_policy
from openpi.shared import array_typing as at
from openpi.shared import download
from openpi.shared import nnx_utils as nnx_utils
from openpi.training import checkpoints as openpi_checkpoints
from openpi.training import config as openpi_train_config
from openpi.training import optimizer as _optimizer
from openpi.training import utils as training_utils
from openpi.training import weight_loaders as _weight_loaders
import openpi.transforms as openpi_transforms
from openpi.transforms import pad_to_dim
import openpi.training.sharding as sharding
from jax.sharding import SingleDeviceSharding


from impls.vlas.utils import RemoteActor

# Only the front camera for OpenPI preprocess + SigLIP (skip zero-padded wrist streams).
CARLA_STEERVLA_IMAGE_KEYS: tuple[str, ...] = ("base_0_rgb",)

# Where OpenPI caches downloaded checkpoints. ``openpi.shared.download.maybe_download``
# reads ``OPENPI_DATA_HOME`` (default ``~/.cache/openpi``) and lays out files as
# ``<OPENPI_DATA_HOME>/<netloc>/<path>``. Point the cache at NFS so large GCS
# checkpoints are shared across hosts/users instead of filling each box's home dir;
# the on-disk layout under it is unchanged.
STEERVLA_CACHE_DIR = "/home/carla/.cache/openpi"


def _ensure_openpi_cache_dir() -> None:
    """Redirect OpenPI's download cache to NFS unless the caller overrode it.

    Uses ``setdefault`` so an explicit ``OPENPI_DATA_HOME`` in the environment still
    wins. Must run before any ``download.maybe_download`` call.
    """
    os.environ.setdefault("OPENPI_DATA_HOME", STEERVLA_CACHE_DIR)


def restore_openpi_params_on_single_gpu(
    params_dir: Path | str,
    *,
    training_gpu_rank: int = -1,
):
    print(f"Restoring OpenPI params on single GPU {training_gpu_rank}", flush=True)
    """Load OpenPI checkpoint onto **one** accelerator.

    ``openpi.models.model.restore_params`` defaults to a mesh over ``jax.devices()``,
    which **replicates** weights on every GPU and doubles VRAM. Use an explicit
    :class:`jax.sharding.SingleDeviceSharding` instead.

    Args:
        params_dir: Directory containing Orbax ``params`` (same as ``restore_params``).
        training_gpu_rank: Which JAX-visible GPU index to use (same semantics as agent
            ``training_gpu_rank``). If ``< 0``, uses GPU ``0`` when GPUs exist.

    Returns:
        ``(params_tree, device)`` for optional ``jax.device_put`` of inputs.
    """

    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    if gpus:
        idx = training_gpu_rank if training_gpu_rank >= 0 else 0
        idx = min(max(idx, 0), len(gpus) - 1)
        device = gpus[idx]
    else:
        device = jax.devices()[0]
    sharding = SingleDeviceSharding(device)
    params = _openpi_model.restore_params(params_dir, sharding=sharding)
    return params, device


@dataclasses.dataclass(frozen=True)
class _SliceActionDim(openpi_transforms.DataTransformFn):
    """Slice env action dims on the last axis (safe for ``(H, D)`` and ``(B, H, D)``)."""

    action_dim: int

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        return {
            **data,
            "actions": np.asarray(data["actions"], dtype=np.float32)[..., : int(self.action_dim)],
        }


def build_openpi_policy_transforms(
    train_cfg: openpi_train_config.TrainConfig,
    checkpoint_dir: Path,
):
    """Input/output transforms matching :func:`impls.vlas.steervla_server._build_steervla_openpi_policy`."""
    data_factory = train_cfg.data
    data_config = data_factory.create(train_cfg.assets_dirs, train_cfg.model)
    if data_config.asset_id is None:
        raise ValueError("TrainConfig data requires asset_id to load norm stats.")
    env_action_dim = int(getattr(data_factory, "action_dim", 4))
    # norm_stats = openpi_checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    input_transform = openpi_transforms.compose(
        [
            # openpi_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        ]
    )
    output_transform = openpi_transforms.compose(
        [
            *data_config.model_transforms.outputs,
            # openpi_transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            _SliceActionDim(action_dim=env_action_dim),
        ]
    )
    return data_config, input_transform, output_transform


def steervla_physical_denormalize_actions(
    actions: np.ndarray,
    *,
    action_dim: int,
    output_action_format: str | None,
) -> np.ndarray:
    """Apply fixed RLDS scaling (``* 7``, ``* 180``, etc.) after OpenPI ``Unnormalize``."""
    from openpi.visualizing.steervla_visualization import denormalize_actions

    arr = np.asarray(actions, dtype=np.float32)
    ad = min(int(action_dim), int(arr.shape[-1]))
    return np.asarray(
        denormalize_actions(arr, ad, output_action_format),
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Pi0-CoT CARLA inference helpers
# ---------------------------------------------------------------------------

def routing_instruction_prompt(*, routing_command: str, current_speed_mps: float) -> str:
    """High-level instruction line (speed prefix matches ``SteerVLAInputs`` when ``speed_in_prompt``)."""
    rc = routing_command.strip()
    speed_prefix = round(float(current_speed_mps), 1)
    return f"The current speed is {speed_prefix} m/s. {rc}"


def steervla_prompt_state_dim(*, include_ego_history: bool) -> int:
    """Proprio dimensions embedded in the CoT prompt (matches ``TokenizeCoTPrompt.prompt_state_dim``)."""
    return 8 if include_ego_history else 2


def format_steervla_cot_prompt(
    prompt: str,
    state: np.ndarray,
    *,
    state_dim: int,
) -> str:
    """Human-readable prefix string before BOS (matches ``CoTPaligemmaTokenizer.tokenize_prompt``)."""
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    state_arr = np.asarray(state, dtype=np.float32).reshape(-1)[:state_dim]
    discretized = np.digitize(state_arr, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized))
    return f"Prompt:{cleaned};State:{state_str};"


def carla_state_vec_to_steervla_state(
    carla_vec: np.ndarray,
    *,
    include_ego_history: bool,
    proprio_norm: bool,
) -> np.ndarray:
    """Map CARLA ego+command vector (``carla_utils.STATE_DIM``-dim) to padded proprio; uses indices 5 and 15."""
    flat_sv = np.asarray(carla_vec, dtype=np.float32).reshape(-1)
    speed = float(flat_sv[15])
    yaw_deg = float(flat_sv[5])
    raw_pair = np.array([speed, yaw_deg], dtype=np.float32)
    normalized = sv_policy.normalize_ego_state(
        raw_pair,
        include_ego_history=include_ego_history,
        proprio_norm=proprio_norm,
    )
    flat = np.asarray(normalized, dtype=np.float32).reshape(-1)
    return flat

def _observation_has_cot_tokens(observation: _openpi_model.Observation) -> bool:
    """Whether an OpenPI observation already carries non-empty CoT tokens."""
    try:
        r_mask = jnp.asarray(observation.tokenized_reasoning_mask)
        s_mask = jnp.asarray(observation.tokenized_subtask_mask)
        return bool(jnp.any(r_mask) or jnp.any(s_mask))
    except Exception:
        return False


def _merge_cot_output_into_observation(
    openpi_observation: _openpi_model.Observation,
    cot_out: dict[str, Any],
) -> _openpi_model.Observation:
    """Attach sampled reasoning/subtask/FAST tokens for ``sample_actions``."""
    obs_full = dataclasses.replace(
        openpi_observation,
        tokenized_reasoning=cot_out["tokenized_reasoning"],
        tokenized_reasoning_mask=cot_out["tokenized_reasoning_mask"],
        tokenized_subtask=cot_out["tokenized_subtask"],
        tokenized_subtask_mask=cot_out["tokenized_subtask_mask"],
    )
    if "tokenized_fast" in cot_out and "tokenized_fast_mask" in cot_out:
        obs_full = dataclasses.replace(
            obs_full,
            tokenized_fast=cot_out["tokenized_fast"],
            tokenized_fast_mask=cot_out["tokenized_fast_mask"],
        )
    return obs_full


def _model_uses_fast_tokens(model_cfg: Pi0CoTConfig | None) -> bool:
    return bool(model_cfg is not None and getattr(model_cfg, "use_fast_tokens", False))


def empty_openpi_replay_fast_fields(max_fast_len: int) -> dict[str, np.ndarray]:
    """Zero FAST token buffers for replay schema / transitions without FAST yet."""
    fast = np.zeros((int(max_fast_len),), dtype=np.int32)
    fast_mask = np.zeros((int(max_fast_len),), dtype=bool)
    return {
        "openpi_tokenized_fast": fast,
        "openpi_tokenized_fast_mask": fast_mask,
        "fast": fast.copy(),
        "fast_mask": fast_mask.copy(),
    }


def with_openpi_replay_fast_fields(
    fields: dict[str, np.ndarray],
    *,
    max_fast_len: int | None,
    use_fast_tokens: bool,
) -> dict[str, np.ndarray]:
    """Ensure replay dicts always carry FAST keys when the model uses FAST tokens."""
    if not use_fast_tokens or max_fast_len is None:
        return fields
    out = dict(fields)
    if "openpi_tokenized_fast" not in out:
        out.update(empty_openpi_replay_fast_fields(int(max_fast_len)))
        return out
    if "fast" not in out:
        out["fast"] = out["openpi_tokenized_fast"]
    if "fast_mask" not in out:
        out["fast_mask"] = out["openpi_tokenized_fast_mask"]
    return out


def _pad_action_chunk_for_fast(actions: np.ndarray, *, model_action_dim: int) -> np.ndarray:
    """Pad a single normalized action chunk ``(H, D)`` to Pi0 ``action_dim``."""
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk.reshape(-1, model_action_dim)
    pad_dim = int(model_action_dim) - int(chunk.shape[-1])
    if pad_dim > 0:
        chunk = np.pad(chunk, [(0, 0), (0, pad_dim)], mode="constant")
    return chunk


def _tokenize_fast_actions_batch(
    tokenizer: CoTPaligemmaTokenizer,
    actions: np.ndarray,
    *,
    model_action_dim: int,
    action_horizon: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize normalized action chunks into FAST PaliGemma token buffers."""
    if not tokenizer.use_fast_tokens:
        raise RuntimeError("FAST tokenizer is disabled on CoTPaligemmaTokenizer.")
    actions = np.asarray(actions, dtype=np.float32)
    ah = int(action_horizon)
    ad = int(action_dim)
    flat_dim = ah * ad

    if actions.ndim == 1:
        if int(actions.size) != flat_dim:
            raise ValueError(
                f"Expected flat action length {flat_dim} (= {ah}×{ad}), got {actions.size}."
            )
        actions = actions.reshape(1, ah, ad)
    elif actions.ndim == 2:
        if int(actions.shape[-1]) == flat_dim:
            # Replay batches store flattened chunks as (B, H*D).
            actions = actions.reshape(int(actions.shape[0]), ah, ad)
        elif actions.shape == (ah, ad):
            actions = actions[None, ...]
        else:
            raise ValueError(
                f"Expected actions shape (B, {flat_dim}), ({ah}, {ad}), or (B, {ah}, {ad}); "
                f"got {actions.shape}."
            )
    elif actions.ndim == 3:
        if int(actions.shape[-2]) != ah or int(actions.shape[-1]) != ad:
            raise ValueError(
                f"Expected actions shape (B, {ah}, {ad}); got {actions.shape}."
            )
    else:
        raise ValueError(f"Expected 1D–3D actions array; got ndim={actions.ndim}.")

    batch_size = int(actions.shape[0])
    fast_len = int(tokenizer.max_fast_len)
    fast = np.zeros((batch_size, fast_len), dtype=np.int32)
    fast_mask = np.zeros((batch_size, fast_len), dtype=bool)
    for i in range(batch_size):
        row = _pad_action_chunk_for_fast(actions[i], model_action_dim=model_action_dim)
        tok, mask = tokenizer.tokenize_fast_actions(row)
        fast[i] = tok
        fast_mask[i] = mask
    return fast, fast_mask


def _fast_arrays_from_raw(
    raw: dict[str, Any] | None,
    *,
    batch_size: int,
    fast_len: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read FAST token buffers from a CARLA/raw obs dict when present."""
    if raw is None or not isinstance(raw, dict):
        return None, None
    fast_src = raw.get("openpi_tokenized_fast")
    if fast_src is None:
        fast_src = raw.get("fast")
    if fast_src is None:
        return None, None
    fast_arr = np.asarray(fast_src, dtype=np.int32).reshape(-1)
    mask_src = raw.get("openpi_tokenized_fast_mask")
    if mask_src is None:
        mask_src = raw.get("fast_mask")
    if mask_src is not None:
        mask_arr = np.asarray(mask_src, dtype=bool).reshape(-1)
    else:
        mask_arr = fast_arr != 0
    n = min(fast_len, fast_arr.size)
    fast = np.zeros((batch_size, fast_len), dtype=np.int32)
    fast_mask = np.zeros((batch_size, fast_len), dtype=bool)
    fast[:, :n] = fast_arr[:n]
    fast_mask[:, :n] = mask_arr[:n]
    return fast, fast_mask


def with_replay_cot_tokens(
    openpi_observation: _openpi_model.Observation,
    replay_batch: dict[str, Any],
    *,
    prefix: str = "",
) -> _openpi_model.Observation:
    """Overlay replay-stored CoT / FAST tokens onto an OpenPI observation when present."""
    rk = f"{prefix}reasoning"
    rmk = f"{prefix}reasoning_mask"
    sk = f"{prefix}subtask"
    smk = f"{prefix}subtask_mask"
    fk = f"{prefix}openpi_tokenized_fast"
    fmk = f"{prefix}openpi_tokenized_fast_mask"
    has_cot = rk in replay_batch or sk in replay_batch
    has_fast = fk in replay_batch or f"{prefix}fast" in replay_batch
    if not has_cot and not has_fast:
        return openpi_observation

    reasoning = jnp.asarray(replay_batch.get(rk, openpi_observation.tokenized_reasoning), dtype=jnp.int32)
    subtask = jnp.asarray(replay_batch.get(sk, openpi_observation.tokenized_subtask), dtype=jnp.int32)

    if rmk in replay_batch:
        reasoning_mask = jnp.asarray(replay_batch[rmk], dtype=bool)
    else:
        reasoning_mask = reasoning != 0
    if smk in replay_batch:
        subtask_mask = jnp.asarray(replay_batch[smk], dtype=bool)
    else:
        subtask_mask = subtask != 0

    out = dataclasses.replace(
        openpi_observation,
        tokenized_reasoning=reasoning,
        tokenized_reasoning_mask=reasoning_mask,
        tokenized_subtask=subtask,
        tokenized_subtask_mask=subtask_mask,
    )
    fast_src = replay_batch.get(fk, replay_batch.get(f"{prefix}fast"))
    if fast_src is None:
        return out
    fast = jnp.asarray(fast_src, dtype=jnp.int32)
    fast_mask_src = replay_batch.get(fmk, replay_batch.get(f"{prefix}fast_mask"))
    if fast_mask_src is not None:
        fast_mask = jnp.asarray(fast_mask_src, dtype=bool)
    else:
        fast_mask = fast != 0
    return dataclasses.replace(
        out,
        tokenized_fast=fast,
        tokenized_fast_mask=fast_mask,
    )


def openpi_replay_fields_from_observation(
    obs_struct: _openpi_model.Observation,
    *,
    include_legacy_cot_keys: bool = True,
) -> dict[str, np.ndarray]:
    """Serialize OpenPI observation token/image/state fields for replay storage."""
    out: dict[str, np.ndarray] = {
        "openpi_image_base_0_rgb": np.asarray(obs_struct.images["base_0_rgb"][0], dtype=np.uint8),
        "openpi_image_mask_base_0_rgb": np.asarray(obs_struct.image_masks["base_0_rgb"][0], dtype=bool),
        "openpi_state": np.asarray(obs_struct.state[0], dtype=np.float32),
        "openpi_tokenized_prompt": np.asarray(obs_struct.tokenized_prompt[0], dtype=np.int32),
        "openpi_tokenized_prompt_mask": np.asarray(obs_struct.tokenized_prompt_mask[0], dtype=bool),
        "openpi_tokenized_reasoning": np.asarray(obs_struct.tokenized_reasoning[0], dtype=np.int32),
        "openpi_tokenized_reasoning_mask": np.asarray(obs_struct.tokenized_reasoning_mask[0], dtype=bool),
        "openpi_tokenized_subtask": np.asarray(obs_struct.tokenized_subtask[0], dtype=np.int32),
        "openpi_tokenized_subtask_mask": np.asarray(obs_struct.tokenized_subtask_mask[0], dtype=bool),
    }
    if obs_struct.tokenized_fast is not None and obs_struct.tokenized_fast_mask is not None:
        out["openpi_tokenized_fast"] = np.asarray(obs_struct.tokenized_fast[0], dtype=np.int32)
        out["openpi_tokenized_fast_mask"] = np.asarray(obs_struct.tokenized_fast_mask[0], dtype=bool)
        out["fast"] = out["openpi_tokenized_fast"]
        out["fast_mask"] = out["openpi_tokenized_fast_mask"]
    if include_legacy_cot_keys:
        out["reasoning"] = out["openpi_tokenized_reasoning"]
        out["reasoning_mask"] = out["openpi_tokenized_reasoning_mask"]
        out["subtask"] = out["openpi_tokenized_subtask"]
        out["subtask_mask"] = out["openpi_tokenized_subtask_mask"]
    return out


def openpi_replay_fields_with_fast_placeholders(
    fields: dict[str, np.ndarray],
    *,
    model_cfg: Pi0CoTConfig | None,
) -> dict[str, np.ndarray]:
    """Add empty FAST replay keys when needed so buffer schema stays fixed."""
    max_fast_len = int(getattr(model_cfg, "max_fast_len", 0)) if model_cfg is not None else 0
    return with_openpi_replay_fast_fields(
        fields,
        max_fast_len=max_fast_len if max_fast_len > 0 else None,
        use_fast_tokens=_model_uses_fast_tokens(model_cfg),
    )


def openpi_cot_replay_fields_from_raw(raw: dict[str, Any] | None) -> dict[str, np.ndarray]:
    """CoT/FAST token fields from a raw CARLA obs dict after VLA ``_stash_cot_in_raw``.

    Returns an empty dict when CoT has not been generated for this observation yet.
    """
    if raw is None or not isinstance(raw, dict) or "reasoning" not in raw:
        return {}

    reasoning = np.asarray(raw["reasoning"], dtype=np.int32)
    reasoning_mask = np.asarray(raw.get("reasoning_mask", reasoning != 0), dtype=bool)
    subtask = np.asarray(raw["subtask"], dtype=np.int32)
    subtask_mask = np.asarray(raw.get("subtask_mask", subtask != 0), dtype=bool)
    out: dict[str, np.ndarray] = {
        "openpi_tokenized_reasoning": reasoning,
        "openpi_tokenized_reasoning_mask": reasoning_mask,
        "openpi_tokenized_subtask": subtask,
        "openpi_tokenized_subtask_mask": subtask_mask,
        "reasoning": reasoning,
        "reasoning_mask": reasoning_mask,
        "subtask": subtask,
        "subtask_mask": subtask_mask,
    }
    fast = raw.get("openpi_tokenized_fast", raw.get("fast"))
    fast_mask = raw.get("openpi_tokenized_fast_mask", raw.get("fast_mask"))
    if fast is not None and fast_mask is not None:
        out["openpi_tokenized_fast"] = np.asarray(fast, dtype=np.int32)
        out["openpi_tokenized_fast_mask"] = np.asarray(fast_mask, dtype=bool)
        out["fast"] = out["openpi_tokenized_fast"]
        out["fast_mask"] = out["openpi_tokenized_fast_mask"]
    return out


def _batch_has_openpi_replay_fields(replay_batch: dict[str, Any], *, prefix: str = "") -> bool:
    return f"{prefix}openpi_state" in replay_batch and f"{prefix}openpi_tokenized_prompt" in replay_batch


def openpi_observation_from_replay_batch(
    replay_batch: dict[str, Any],
    *,
    prefix: str = "",
) -> _openpi_model.Observation:
    """Build an OpenPI observation directly from replay-aligned batch tensors.

    Expected keys use ``openpi_*`` prefixes for prompt/image/state. For CoT tokens this accepts both
    ``openpi_tokenized_reasoning``/``openpi_tokenized_subtask`` and legacy ``reasoning``/``subtask`` names.
    """
    if not _batch_has_openpi_replay_fields(replay_batch, prefix=prefix):
        raise KeyError(f"Missing replay OpenPI fields for prefix={prefix!r}.")

    def _get(*names: str):
        for n in names:
            key = f"{prefix}{n}"
            if key in replay_batch:
                return replay_batch[key]
        return None

    state = jnp.asarray(_get("openpi_state"), dtype=jnp.float32)
    if int(state.shape[-1]) < 32:
        state = jnp.asarray(pad_to_dim(np.asarray(state), 32), dtype=jnp.float32)
    prompt = jnp.asarray(_get("openpi_tokenized_prompt"), dtype=jnp.int32)
    prompt_mask = jnp.asarray(_get("openpi_tokenized_prompt_mask"), dtype=bool)
    image = jnp.asarray(_get("openpi_image_base_0_rgb"), dtype=jnp.uint8)
    image_mask_src = _get("openpi_image_mask_base_0_rgb")
    image_mask = jnp.asarray(image_mask_src if image_mask_src is not None else jnp.ones((state.shape[0],), dtype=bool), dtype=bool)

    reasoning = jnp.asarray(
        _get("openpi_tokenized_reasoning", "reasoning"),
        dtype=jnp.int32,
    )
    reasoning_mask_src = _get("openpi_tokenized_reasoning_mask", "reasoning_mask")
    reasoning_mask = (
        jnp.asarray(reasoning_mask_src, dtype=bool)
        if reasoning_mask_src is not None
        else (reasoning != 0)
    )

    subtask = jnp.asarray(
        _get("openpi_tokenized_subtask", "subtask"),
        dtype=jnp.int32,
    )
    subtask_mask_src = _get("openpi_tokenized_subtask_mask", "subtask_mask")
    subtask_mask = (
        jnp.asarray(subtask_mask_src, dtype=bool)
        if subtask_mask_src is not None
        else (subtask != 0)
    )

    fast = None
    fast_mask = None
    fast_src = _get("openpi_tokenized_fast", "fast")
    if fast_src is not None:
        fast = jnp.asarray(fast_src, dtype=jnp.int32)
        fast_mask_src = _get("openpi_tokenized_fast_mask", "fast_mask")
        fast_mask = (
            jnp.asarray(fast_mask_src, dtype=bool)
            if fast_mask_src is not None
            else (fast != 0)
        )

    data = {
        "image": {
            "base_0_rgb": image,
        },
        "image_mask": {
            "base_0_rgb": image_mask,
        },
        "state": state,
        "tokenized_prompt": prompt,
        "tokenized_prompt_mask": prompt_mask,
        "tokenized_reasoning": reasoning,
        "tokenized_reasoning_mask": reasoning_mask,
        "tokenized_subtask": subtask,
        "tokenized_subtask_mask": subtask_mask,
    }
    if fast is not None and fast_mask is not None:
        data["tokenized_fast"] = fast
        data["tokenized_fast_mask"] = fast_mask
    return _openpi_model.Observation.from_dict(data)

def _maybe_set_jax_default_gpu(training_gpu_rank: int) -> None:
    if training_gpu_rank < 0:
        return
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    if not gpus:
        return
    idx = min(max(training_gpu_rank, 0), len(gpus) - 1)
    jax.config.update("jax_default_device", gpus[idx])


class SteerVLAActor:
    """SteerVLA as a remote HTTP client or as local OpenPI Pi0-CoT inference.

    Remote mode (``actor_url`` set) delegates ``get_action``, ``get_cot``, and ``update`` to
    :class:`RemoteActor` and never downloads or restores checkpoints locally.

    Local mode loads a Pi0-CoT checkpoint (requires ``actor_config`` and ``checkpoint_path``). Use
    :meth:`__call__` for DSRL flow sampling (reads ``raw_obs_holder["obs"]``). Use
    :meth:`get_action` / :meth:`get_cot` with a CARLA gym observation dict passed as ``state``.

    OpenPI weights are loaded **once** from Orbax; :attr:`model` and :attr:`train_state` share the same parameters.
    After optional :meth:`attach_dsrl`, :meth:`update` runs :func:`train_step` (DSRL-frozen actor loss through the VLA).
    """

    def __init__(
        self,
        actor_url: Optional[str] = None,
        *,
        actor_config: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        raw_obs_holder: Optional[MutableMapping[str, Any]] = None,
        routing_command: str = "Follow the route.",
        cot_temperature: float = 0.0,
        include_ego_history: bool = False,
        proprio_norm: bool = True,
        output_action_format: Optional[str] = "DELTA_XY_T_DELTA_XY_SPACE",
        action_horizon: int = 10,
        action_dim: int = 4,
        actions_per_model_query: int = 1,
        actions_per_cot: int = 1,
        sample_actions_num_steps: int = 10,
        action_decode_batch_size: int = 2,
        training_gpu_rank: int = -1,
        return_normalized_action_chunk: bool = False,
        fixed_subtask_text: Optional[str] = None,
        fixed_reasoning_text: Optional[str] = None,
        debug_noise: bool = False,
        debug_noise_samples: int = 15,
        use_best_noise: bool = True,
        debug_noise_log_every_n_steps: int = 5,
        debug_noise_run_name: str | None = None,
        debug_noise_save_root: str | Path | None = None,
        debug_noise_route_name: str = "?",
        debug_noise_episode: int = 0,
        debug_noise_episode_step: int = 0,
        noise_scale: float = 1.0,
    ) -> None:
        self.actor_url = actor_url
        self._remote: Optional[RemoteActor] = None

        self.actor_config = actor_config
        self.checkpoint_path = checkpoint_path
        self.raw_obs_holder = raw_obs_holder
        self.routing_command = routing_command
        self.cot_temperature = float(cot_temperature)
        self.include_ego_history = include_ego_history
        self.proprio_norm = proprio_norm
        self.output_action_format = output_action_format
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.actions_per_model_query = max(1, int(actions_per_model_query))
        self.actions_per_cot = max(1, int(actions_per_cot))
        self.sample_actions_num_steps = int(sample_actions_num_steps)
        self.action_decode_batch_size = max(1, int(action_decode_batch_size))
        self.return_normalized_action_chunk = bool(return_normalized_action_chunk)
        self.fixed_subtask_text = (
            str(fixed_subtask_text).strip() if fixed_subtask_text else None
        )
        self.fixed_reasoning_text = (
            str(fixed_reasoning_text).strip() if fixed_reasoning_text else None
        )
        self.debug_noise = bool(debug_noise)
        self.debug_noise_samples = max(1, int(debug_noise_samples))
        self.use_best_noise = bool(use_best_noise)
        self.debug_noise_log_every_n_steps = max(1, int(debug_noise_log_every_n_steps))
        self.debug_noise_run_name = debug_noise_run_name
        self.debug_noise_save_root = (
            Path(debug_noise_save_root) if debug_noise_save_root is not None else None
        )
        self.debug_noise_route_name = str(debug_noise_route_name)
        self.debug_noise_episode = int(debug_noise_episode)
        self.debug_noise_episode_step = int(debug_noise_episode_step)
        self.noise_scale = float(noise_scale)
        self.prompt_state_dim = steervla_prompt_state_dim(include_ego_history=include_ego_history)
        self._call_counter = 0
        self._cached_action_chunk: np.ndarray | None = None
        self._cached_action_step = 0
        self._cached_cot: dict[str, Any] | None = None
        self._cached_cot_actions_used = 0

        self.train_cfg = None
        self.model = None
        self.tokenizer = None
        self.model_cfg = None
        self._jax_device = None
        self._local_ready = False
        self.checkpoint_dir: Path | None = None
        self._mesh: jax.sharding.Mesh | None = None
        self._train_rng: jax.Array | None = None
        self._train_state: training_utils.TrainState | None = None
        self._data_config: Any = None
        self._input_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self._output_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None

        if actor_url is not None:
            self._remote = RemoteActor(
                actor_url,
            )
        else:
            if actor_config is None or checkpoint_path is None:
                raise ValueError("Local SteerVLAActor requires actor_config and checkpoint_path.")

        self.setup(training_gpu_rank=training_gpu_rank)

    def setup(self, *, training_gpu_rank: int = -1) -> None:
        """Remote: ping ``/get_info``. Local: restore checkpoint and build Pi0-CoT module."""
        if self._remote is not None:
            self._remote.get_info()
            return
        if self._local_ready:
            return

        _maybe_set_jax_default_gpu(training_gpu_rank)

        self.train_cfg = openpi_train_config.get_config(self.actor_config)
        model_cfg = self.train_cfg.model
        if not isinstance(model_cfg, Pi0CoTConfig):
            raise TypeError(
                f"SteerVLA Pi0-CoT expects Pi0CoTConfig; got {type(model_cfg).__name__} "
                f"for actor_config={self.actor_config!r}. Use e.g. pi05_steervla_cot_ki."
            )
        model_cfg = dataclasses.replace(
            model_cfg,
        )
        self.train_cfg = dataclasses.replace(self.train_cfg, model=model_cfg)

        _ensure_openpi_cache_dir()
        ckpt_root = Path(download.maybe_download(self.checkpoint_path)).resolve()
        self.checkpoint_dir = ckpt_root
        params_dir = ckpt_root / "params"
        print("Params directory: ", params_dir, flush=True)
        if not params_dir.is_dir():
            raise FileNotFoundError(f"Expected OpenPI checkpoint params at {params_dir}")

        self.train_cfg = dataclasses.replace(
            self.train_cfg,
            weight_loader=_weight_loaders.CheckpointWeightLoader(str(params_dir)),
        )

        params, device = restore_openpi_params_on_single_gpu(params_dir=params_dir, training_gpu_rank=training_gpu_rank)
        
        self.model = self.train_cfg.model.load(params)
        self._jax_device = device

        self.tokenizer = CoTPaligemmaTokenizer(
            max_prompt_len=model_cfg.max_token_len,
            max_subtask_len=model_cfg.max_subtask_len,
            max_reasoning_len=model_cfg.max_reasoning_len,
            max_fast_len=model_cfg.max_fast_len,
            use_fast_tokens=bool(getattr(model_cfg, "use_fast_tokens", False)),
        )
        self.model_cfg = model_cfg
        self._data_config, self._input_transform, self._output_transform = build_openpi_policy_transforms(
            self.train_cfg,
            ckpt_root,
        )
        self._sample_actions = nnx_utils.module_jit(
            self.model.sample_actions,
            static_argnames=(
                "num_steps",
                "image_keys",
            ),
        )
        self._sample_cot = nnx_utils.module_jit(
            self.model.sample_cot,
            static_argnames=(
                "temperature",
                "max_subtask_len",
                "max_reasoning_len",
                "image_keys",
            ),
        )
        
        self._local_ready = True

    def _state_for_transform(self, state: np.ndarray | jax.Array) -> np.ndarray:
        """Proprio slice passed to OpenPI ``Normalize`` / ``Unnormalize`` (2 or 8 dims, not padded)."""
        state_np = np.asarray(jax.device_get(state), dtype=np.float32)
        if state_np.ndim == 1:
            state_np = state_np[None, ...]
        return state_np[..., : int(self.prompt_state_dim)]

    def _normalize_state_batch(self, state: np.ndarray) -> np.ndarray:
        """Apply OpenPI ``Normalize(norm_stats)`` to unpadded proprio fed to the model."""
        state_np = self._state_for_transform(state)
        if self._input_transform is None:
            return state_np
        return np.asarray(self._input_transform({"state": state_np})["state"], dtype=np.float32)

    def _reshape_model_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Normalize raw model outputs to ``(B, model_ah, model_ad)``."""
        traj_np = np.asarray(trajectory, dtype=np.float32)
        model_ah = int(self.model.action_horizon)
        model_ad = int(self.model.action_dim)
        env_ah = int(self.action_horizon)
        env_ad = int(self.action_dim)
        model_flat = model_ah * model_ad
        env_flat = env_ah * env_ad

        if traj_np.ndim == 1:
            if traj_np.shape[0] == model_flat:
                traj_np = traj_np.reshape(1, model_ah, model_ad)
            elif traj_np.shape[0] == env_flat:
                traj_np = traj_np.reshape(1, env_ah, env_ad)
            elif traj_np.shape[0] == env_ad:
                traj_np = traj_np.reshape(1, 1, env_ad)
            else:
                raise ValueError(
                    f"Cannot map 1D trajectory length {traj_np.shape[0]} to model ({model_ah}, {model_ad})."
                )
        elif traj_np.ndim == 2:
            batch, last = traj_np.shape
            if last == model_flat:
                traj_np = traj_np.reshape(batch, model_ah, model_ad)
            elif last == env_flat:
                traj_np = traj_np.reshape(batch, env_ah, env_ad)
            elif last == model_ad and batch == model_ah:
                traj_np = traj_np[None, ...]
            elif last == model_ad:
                traj_np = traj_np[:, None, :]
            elif last == env_ad:
                traj_np = traj_np[:, None, :]
            else:
                raise ValueError(
                    f"Cannot map trajectory shape {(batch, last)} to model ({model_ah}, {model_ad})."
                )
        elif traj_np.ndim != 3:
            raise ValueError(f"Expected trajectory ndim 1/2/3, got {traj_np.ndim}.")

        if traj_np.shape[-1] < model_ad:
            pad = np.zeros((*traj_np.shape[:-1], model_ad - traj_np.shape[-1]), dtype=np.float32)
            traj_np = np.concatenate([traj_np, pad], axis=-1)
        if traj_np.shape[1] < model_ah:
            pad = np.zeros((traj_np.shape[0], model_ah - traj_np.shape[1], traj_np.shape[2]), dtype=np.float32)
            traj_np = np.concatenate([traj_np, pad], axis=1)
        if traj_np.shape[1] != model_ah or traj_np.shape[2] != model_ad:
            full = np.zeros((traj_np.shape[0], model_ah, model_ad), dtype=np.float32)
            ah = min(traj_np.shape[1], model_ah)
            ad = min(traj_np.shape[2], model_ad)
            full[:, :ah, :ad] = traj_np[:, :ah, :ad]
            traj_np = full
        return traj_np

    def _postprocess_action_trajectory(
        self,
        trajectory: np.ndarray | jax.Array,
        *,
        observation_state: np.ndarray | jax.Array,
    ) -> np.ndarray:
        """Undo OpenPI norm stats, then apply fixed SteerVLA action scaling (meters / degrees)."""
        traj_np = self._reshape_model_trajectory(np.asarray(jax.device_get(trajectory), dtype=np.float32))
        state_np = self._state_for_transform(observation_state)
        if state_np.ndim == 1:
            state_np = state_np[None, ...]
        if state_np.shape[0] == 1 and traj_np.shape[0] > 1:
            state_np = np.tile(state_np, (traj_np.shape[0], 1))

        if self._output_transform is not None:
            out = self._output_transform({"actions": traj_np, "state": state_np})
            traj_np = np.asarray(out["actions"], dtype=np.float32)

        traj_np = steervla_physical_denormalize_actions(
            traj_np,
            action_dim=int(self.action_dim),
            output_action_format=self.output_action_format,
        )
        return traj_np[:, : int(self.action_horizon), : int(self.action_dim)]

    def postprocess_sampled_trajectory(
        self,
        trajectory: np.ndarray | jax.Array,
        *,
        observation_state: np.ndarray | jax.Array,
    ) -> jnp.ndarray:
        """Public wrapper for DSRL / training-time VLA forwards."""
        out = self._postprocess_action_trajectory(trajectory, observation_state=observation_state)
        return jnp.asarray(out, dtype=jnp.float32)
    
    # def flow_sample(self, rng, openpi_observation, input_noise):
    #     batch_size = int(openpi_observation.state.shape[0])
    #     cot_out = self._sample_or_reuse_cot(rng, openpi_observation, batch_size)
    #     obs_full = _merge_cot_output_into_observation(openpi_observation, cot_out)
        
    #     # Construct the noise
    #     model_ah = int(self.model.action_horizon)
    #     model_ad = int(self.model.action_dim)
    #     cfg_ah = min(int(self.action_horizon), model_ah)
    #     cfg_ad = min(int(self.action_dim), model_ad)
    #     noise_full = jnp.zeros((batch_size, model_ah, model_ad), dtype=jnp.float32)
    #     if input_noise.ndim == 3:
    #         noise_chunk = input_noise[:, :cfg_ah, :cfg_ad]
    #     elif int(input_noise.shape[-1]) == int(self.action_horizon) * int(self.action_dim):
    #         noise_chunk = input_noise.reshape(batch_size, int(self.action_horizon), int(self.action_dim))[:, :cfg_ah, :cfg_ad]
    #     else:
    #         noise_chunk = input_noise[:, None, :cfg_ad]
    #         write_ah = 1
    #     noise_full = noise_full.at[:, :write_ah, :cfg_ad].set(noise_chunk)
        
    #     traj = self._sample_actions(
    #         rng,
    #         obs_full,
    #         noise=noise_full,
    #         image_keys=CARLA_STEERVLA_IMAGE_KEYS,
    #         num_steps=int(self.sample_actions_num_steps),
    #     )
    #     traj_np = self._postprocess_action_trajectory(
    #         traj,
    #         observation_state=openpi_observation.state,
    #     )
    #     target_dim = int(self.action_dim)
    #     first_step = traj_np[:, 0, :target_dim]
    #     out = jnp.asarray(first_step, dtype=jnp.float32)
    #     return out

    def _routing_for_raw(self, raw: Dict[str, Any]) -> str:
        rc = raw.get("routing_command")
        if rc is not None and isinstance(rc, str) and rc.strip():
            return rc.strip()
        return self.routing_command

    def build_observation_batch_numpy(
        self,
        batch_size: int,
        *,
        raw: Optional[Dict[str, Any]] = None,
    ) -> _openpi_model.Observation:
        if raw is None:
            if self.raw_obs_holder is None:
                raise RuntimeError(
                    "Missing CARLA observation: pass ``raw=`` to build_observation_batch_numpy "
                    "or set ``raw_obs_holder['obs']`` before calling SteerVLAActor.__call__."
                )
            raw_holder = self.raw_obs_holder.get("obs")
            if raw_holder is None or not isinstance(raw_holder, dict):
                raise RuntimeError('raw_obs_holder["obs"] must be set to the latest CARLA gym dict.')
            raw = raw_holder

        img = np.asarray(raw["image"], dtype=np.uint8)
        if img.ndim == 3:
            img = np.broadcast_to(img[None], (batch_size, *img.shape))
        elif img.shape[0] != batch_size:
            img = np.repeat(img[:1], batch_size, axis=0)

        state_vec = np.asarray(raw["state"], dtype=np.float32).reshape(-1)
        spd = float(state_vec[15])
        prompt_text = routing_instruction_prompt(
            routing_command=self._routing_for_raw(raw),
            current_speed_mps=spd,
        )

        assert self.model_cfg is not None
        state_pad = carla_state_vec_to_steervla_state(
            state_vec,
            include_ego_history=self.include_ego_history,
            proprio_norm=self.proprio_norm,
        )
        model_action_dim = int(self.model_cfg.action_dim)
        state_norm = self._normalize_state_batch(state_pad)[0]
        state_for_model = pad_to_dim(state_norm, model_action_dim)
        state_batch = np.tile(state_for_model[None], (batch_size, 1))
        formatted_prompt = format_steervla_cot_prompt(
            prompt_text,
            state_pad,
            state_dim=self.prompt_state_dim,
        )
        if isinstance(raw, dict):
            raw["openpi_prompt_text"] = formatted_prompt

        assert self.tokenizer is not None
        tok_ids, tok_mask = self.tokenizer.tokenize_prompt(
            prompt_text,
            state_pad,
            state_dim=self.prompt_state_dim,
        )

        valid = tok_ids[tok_mask.astype(bool)]
        prompt_detokenized = self.tokenizer._tokenizer.decode(valid.tolist())
        print(f"[DEBUG - steervla] Prompt text: {prompt_detokenized}")
        
        reasoning_len = int(self.model_cfg.max_reasoning_len)
        subtask_len = int(self.model_cfg.max_subtask_len)
        # Do not preload stale CoT/FAST from raw; sample_cot generates fresh tokens each query.
        reasoning = np.zeros((batch_size, reasoning_len), dtype=np.int32)
        reasoning_mask = np.zeros((batch_size, reasoning_len), dtype=bool)
        subtask = np.zeros((batch_size, subtask_len), dtype=np.int32)
        subtask_mask = np.zeros((batch_size, subtask_len), dtype=bool)

        data = {
            "image": {
                "base_0_rgb": img,
            },
            "image_mask": {
                "base_0_rgb": np.ones(batch_size, dtype=bool),
            },
            "state": state_batch,
            "tokenized_prompt": np.tile(tok_ids[None], (batch_size, 1)),
            "tokenized_prompt_mask": np.tile(tok_mask[None], (batch_size, 1)),
            "tokenized_reasoning": reasoning,
            "tokenized_reasoning_mask": reasoning_mask,
            "tokenized_subtask": subtask,
            "tokenized_subtask_mask": subtask_mask,
        }
        return _openpi_model.Observation.from_dict(data)

    def _stash_cot_in_raw(self, raw: Optional[Dict[str, Any]], cot_out: dict[str, Any]) -> None:
        """Persist latest CoT tokens/masks in raw obs dict for downstream training."""
        if raw is None:
            return
        try:
            reason_tokens = np.asarray(jax.device_get(cot_out["tokenized_reasoning"][0]), dtype=np.int32)
            reason_mask = np.asarray(jax.device_get(cot_out["tokenized_reasoning_mask"][0]), dtype=bool)
            subtask_tokens = np.asarray(jax.device_get(cot_out["tokenized_subtask"][0]), dtype=np.int32)
            subtask_mask = np.asarray(jax.device_get(cot_out["tokenized_subtask_mask"][0]), dtype=bool)
            raw["reasoning"] = reason_tokens
            raw["reasoning_mask"] = reason_mask
            raw["subtask"] = subtask_tokens
            raw["subtask_mask"] = subtask_mask
            raw["reasoning_text"] = self.tokenizer._tokenizer.decode(reason_tokens[reason_mask].tolist())
            raw["subtask_text"] = self.tokenizer._tokenizer.decode(subtask_tokens[subtask_mask].tolist())
            if "tokenized_fast" in cot_out and "tokenized_fast_mask" in cot_out:
                fast_tokens = np.asarray(jax.device_get(cot_out["tokenized_fast"][0]), dtype=np.int32)
                fast_mask = np.asarray(jax.device_get(cot_out["tokenized_fast_mask"][0]), dtype=bool)
                if np.any(fast_mask):
                    raw["openpi_tokenized_fast"] = fast_tokens
                    raw["openpi_tokenized_fast_mask"] = fast_mask
                    raw["fast"] = fast_tokens
                    raw["fast_mask"] = fast_mask
        except Exception:
            # Keep rollout robust if CoT payload changes shape unexpectedly.
            return

    def attach_replay_tokens(
        self,
        openpi_observation: _openpi_model.Observation,
        replay_batch: dict[str, Any],
        *,
        prefix: str = "",
    ) -> _openpi_model.Observation:
        """Overlay replay CoT/FAST tokens; tokenize FAST from actions when needed."""
        obs = with_replay_cot_tokens(openpi_observation, replay_batch, prefix=prefix)
        if not _model_uses_fast_tokens(self.model_cfg) or self.tokenizer is None:
            return obs
        if obs.tokenized_fast is not None:
            try:
                if bool(jnp.any(jnp.asarray(obs.tokenized_fast_mask))):
                    return obs
            except Exception:
                pass
        actions_key = f"{prefix}openpi_actions"
        if actions_key not in replay_batch:
            actions_key = f"{prefix}actions" if f"{prefix}actions" in replay_batch else "actions"
        if actions_key not in replay_batch:
            return obs
        try:
            fast, fast_mask = _tokenize_fast_actions_batch(
                self.tokenizer,
                np.asarray(replay_batch[actions_key]),
                model_action_dim=int(self.model_cfg.action_dim),
                action_horizon=int(self.action_horizon),
                action_dim=int(self.action_dim),
            )
        except Exception:
            return obs
        return dataclasses.replace(
            obs,
            tokenized_fast=jnp.asarray(fast, dtype=jnp.int32),
            tokenized_fast_mask=jnp.asarray(fast_mask, dtype=bool),
        )

    def _shift_cached_action_chunk(self, action: np.ndarray, step: int) -> np.ndarray:
        flat = np.asarray(action, dtype=np.float32)
        expected = int(self.action_horizon) * int(self.action_dim)
        if flat.ndim != 2 or flat.shape[-1] != expected or step <= 0:
            return flat
        chunks = flat.reshape(flat.shape[0], int(self.action_horizon), int(self.action_dim))
        shifted = np.zeros_like(chunks)
        keep = max(0, int(self.action_horizon) - int(step))
        if keep > 0:
            shifted[:, :keep, :] = chunks[:, int(step): int(step) + keep, :]
            shifted[:, keep:, :] = shifted[:, keep - 1: keep, :]
        return shifted.reshape(flat.shape[0], expected)

    def _next_cached_action(self, batch_size: int) -> jnp.ndarray | None:
        if self.actions_per_model_query <= 1 or batch_size != 1 or self._cached_action_chunk is None:
            return None
        max_cached_steps = min(self.actions_per_model_query, int(self.action_horizon))
        if self._cached_action_step >= max_cached_steps:
            self._cached_action_chunk = None
            self._cached_action_step = 0
            return None
        out = self._shift_cached_action_chunk(self._cached_action_chunk, self._cached_action_step)
        self._cached_action_step += 1
        return jnp.asarray(out)

    def _remember_action_chunk(self, action: Any, batch_size: int) -> None:
        if self.actions_per_model_query <= 1 or batch_size != 1:
            return
        self._cached_action_chunk = np.asarray(jax.device_get(action), dtype=np.float32)
        # Step 0 was just returned from the fresh model query.
        self._cached_action_step = 1

    def _cot_cache_enabled(self, batch_size: int) -> bool:
        return self.actions_per_cot > 1 and batch_size == 1

    def _uses_fixed_cot(self) -> bool:
        return self.fixed_subtask_text is not None

    def _build_fixed_cot_out(
        self,
        batch_size: int,
        *,
        ref_array: jnp.ndarray,
    ) -> dict[str, Any]:
        """Teacher-forced CoT matching ``inspect_outputs.ipynb`` (no ``sample_cot``)."""
        if self.tokenizer is None or self.model_cfg is None:
            raise RuntimeError("Fixed CoT requires a loaded local SteerVLAActor tokenizer/model.")
        reasoning_text = self.fixed_reasoning_text or "Follow the route."
        rea_np, rea_mask_np = self.tokenizer.tokenize_reasoning(reasoning_text)
        sub_np, sub_mask_np = self.tokenizer.tokenize_subtask(self.fixed_subtask_text)
        device = self._jax_device if self._jax_device is not None else ref_array.devices().pop()

        def _tile(tok: np.ndarray, mask: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            tokens = jnp.asarray(np.tile(tok[None, :], (batch_size, 1)), dtype=ref_array.dtype)
            masks = jnp.asarray(np.tile(mask[None, :], (batch_size, 1)), dtype=bool)
            return jax.device_put(tokens, device), jax.device_put(masks, device)

        reasoning, reasoning_mask = _tile(rea_np, rea_mask_np)
        subtask, subtask_mask = _tile(sub_np, sub_mask_np)
        out: dict[str, Any] = {
            "tokenized_reasoning": reasoning,
            "tokenized_reasoning_mask": reasoning_mask,
            "tokenized_subtask": subtask,
            "tokenized_subtask_mask": subtask_mask,
        }
        if self.tokenizer.use_fast_tokens:
            mf = int(self.model_cfg.max_fast_len)
            out["tokenized_fast"] = jax.device_put(
                jnp.zeros((batch_size, mf), dtype=ref_array.dtype),
                device,
            )
            out["tokenized_fast_mask"] = jax.device_put(
                jnp.zeros((batch_size, mf), dtype=bool),
                device,
            )
        return out

    def _sample_or_reuse_cot(
        self,
        rng: jax.Array,
        obs_jax: _openpi_model.Observation,
        batch_size: int,
    ) -> dict[str, Any]:
        if self._uses_fixed_cot():
            return self._build_fixed_cot_out(
                batch_size,
                ref_array=obs_jax.tokenized_prompt,
            )
        if (
            self._cot_cache_enabled(batch_size)
            and self._cached_cot is not None
            and self._cached_cot_actions_used < self.actions_per_cot
        ):
            return self._cached_cot
        cot_out = self._sample_cot(
            rng,
            obs_jax,
            temperature=float(self.cot_temperature), 
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
        )
        if self._cot_cache_enabled(batch_size):
            self._cached_cot = dict(cot_out)
            self._cached_cot_actions_used = 0
        return cot_out

    def _mark_action_served(self, batch_size: int) -> None:
        if self._cot_cache_enabled(batch_size) and self._cached_cot is not None:
            self._cached_cot_actions_used += 1

    def reset_action_cache(self) -> None:
        self._cached_action_chunk = None
        self._cached_action_step = 0
        self._cached_cot = None
        self._cached_cot_actions_used = 0

    def _forward_pi0(
        self,
        batch_size: int,
        noise_jax: jax.Array,
        *,
        raw: Optional[Dict[str, Any]] = None,
        rng: jax.Array | None = None,
        force_accel_steer: bool = False,
    ) -> jax.Array:
        assert self.model is not None and self._jax_device is not None
        if rng is None:
            self._call_counter += 1
            rng = jax.random.PRNGKey(self._call_counter)

        # Build the observation from the batch
        obs_np_struct = self.build_observation_batch_numpy(batch_size, raw=raw)
        obs_jax = jax.tree.map(
            lambda x: jax.device_put(jnp.asarray(x), self._jax_device),
            obs_np_struct,
        )
        noise_jax = jax.device_put(noise_jax, self._jax_device)
        rng_cot, rng_act = jax.random.split(rng)
        
        # Either sample or reuse the CoT
        cot_out = self._sample_or_reuse_cot(rng_cot, obs_jax, batch_size)
        
        reason_tokens = cot_out["tokenized_reasoning"]
        reason_mask = cot_out["tokenized_reasoning_mask"]
        reason_valid = reason_tokens[reason_mask.astype(bool)]
        reason_text = self.tokenizer._tokenizer.decode(reason_valid.tolist())
        print(f"[DEBUG - steervla] Reason text: {reason_text}")
        
        subtask_tokens = cot_out["tokenized_subtask"]
        subtask_mask = cot_out["tokenized_subtask_mask"]
        subtask_valid = subtask_tokens[subtask_mask.astype(bool)]
        subtask_text = self.tokenizer._tokenizer.decode(subtask_valid.tolist())
        print(f"[DEBUG - steervla] Subtask text: {subtask_text}")
        if "tokenized_fast" in cot_out and self.tokenizer.use_fast_tokens:
            fast_tokens = cot_out["tokenized_fast"]
            fast_mask = cot_out["tokenized_fast_mask"]
            fast_valid = fast_tokens[fast_mask.astype(bool)]
            fast_text = self.tokenizer._tokenizer.decode(fast_valid.tolist())
            print(f"[DEBUG - steervla] FAST segment ({int(jnp.sum(fast_mask))} tok): {fast_text[:120]}")
        
        # Keep latest generated CoT in raw holder (batch-1 online CARLA path).
        if batch_size == 1:
            if raw is not None:
                self._stash_cot_in_raw(raw, cot_out)
            elif self.raw_obs_holder is not None and isinstance(self.raw_obs_holder.get("obs"), dict):
                self._stash_cot_in_raw(self.raw_obs_holder["obs"], cot_out)

        obs_full = _merge_cot_output_into_observation(obs_jax, cot_out)

        # Prepare noise for inference
        batch_size = obs_jax.state.shape[0]
        model_ah = int(self.model.action_horizon)
        model_ad = int(self.model.action_dim)
        cfg_ah = min(int(self.action_horizon), model_ah)
        cfg_ad = min(int(self.action_dim), model_ad)
        noise_full = jnp.zeros((batch_size, model_ah, model_ad), dtype=jnp.float32)
        if noise_jax.ndim == 3:
            noise_chunk = noise_jax[:, :cfg_ah, :cfg_ad]
        elif int(noise_jax.shape[-1]) == int(self.action_horizon) * int(self.action_dim):
            noise_chunk = noise_jax.reshape(batch_size, int(self.action_horizon), int(self.action_dim))[
                :, :cfg_ah, :cfg_ad
            ]
        else:
            noise_chunk = noise_jax[:, None, :cfg_ad]
            cfg_ah = 1
        noise_full = noise_full.at[:, :cfg_ah, :cfg_ad].set(noise_chunk)
        
        # Sample the actions
        sample_actions_time = time.time()
        traj = self._sample_actions(
            rng_act,
            obs_full,
            noise=noise_full,
            num_steps=int(self.sample_actions_num_steps),
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
        )
        jax.block_until_ready(traj)
        sample_actions_time = time.time() - sample_actions_time

        print(f"[DEBUG - steervla] Sample actions time: {sample_actions_time} seconds")
        traj_np = self._postprocess_action_trajectory(traj, observation_state=obs_jax.state)

        if self.return_normalized_action_chunk and not force_accel_steer:
            flat = traj_np.reshape(batch_size, -1)
            expected = int(self.action_horizon) * int(self.action_dim)
            if flat.shape[-1] != expected:
                raise ValueError(
                    f"SteerVLA action chunk has length {flat.shape[-1]}, expected "
                    f"{expected} (= {self.action_horizon} x {self.action_dim}). "
                    f"Postprocessed trajectory shape: {tuple(traj_np.shape)}."
                )
            return jax.device_put(jnp.asarray(flat, dtype=jnp.float32), self._jax_device)

        first_step = traj_np[:, 0, : int(self.action_dim)]
        return jax.device_put(jnp.asarray(first_step, dtype=jnp.float32), self._jax_device)

    @staticmethod
    def _sanitize_debug_path_component(name: str) -> str:
        s = str(name).strip() or "unknown"
        s = re.sub(r"[^\w\-.]+", "_", s)
        return s[:120]

    def set_debug_noise_context(
        self,
        *,
        run_name: str | None = None,
        save_root: str | Path | None = None,
        route_name: str | None = None,
        episode: int | None = None,
        episode_step: int | None = None,
    ) -> None:
        """Update where debug-noise plots and npz logs are written (per run / route / episode)."""
        if run_name is not None:
            self.debug_noise_run_name = str(run_name)
        if save_root is not None:
            self.debug_noise_save_root = Path(save_root)
        if route_name is not None:
            self.debug_noise_route_name = str(route_name)
        if episode is not None:
            self.debug_noise_episode = int(episode)
        if episode_step is not None:
            self.debug_noise_episode_step = int(episode_step)

    def _should_log_debug_noise_plot(self) -> bool:
        step = int(self.debug_noise_episode_step)
        every = int(self.debug_noise_log_every_n_steps)
        return step > 0 and step % every == 0

    def _debug_noise_artifact_dir(self) -> Path:
        run_name = self._sanitize_debug_path_component(
            self.debug_noise_run_name or "run"
        )
        route = self._sanitize_debug_path_component(self.debug_noise_route_name)
        ep = int(self.debug_noise_episode)
        root = self.debug_noise_save_root or Path("debug_noises")
        out_dir = root / "debug_noises" / f"{run_name}_{route}" / f"ep{ep:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _debug_speed_score_from_flat(self, flat_action: Any) -> float:
        """First-step speed delta magnitude (``delta_xy`` cols 0:2); lower means slower."""
        ah = int(self.action_horizon)
        ad = int(self.action_dim)
        flat = np.asarray(jax.device_get(flat_action), dtype=np.float32).reshape(-1)
        expected = ah * ad
        if flat.size != expected:
            return float("inf")
        speed_xy = flat.reshape(ah, ad)[:, :2]
        speed_xy_final = np.cumsum(speed_xy, axis=0)[-1, :2]
        return float(np.linalg.norm(speed_xy_final))

    def _debug_xy_cumsum_from_flat(self, flat_action: Any) -> np.ndarray:
        """Cumulative ``delta_xy`` in meters, shape ``(action_horizon, 2)``."""
        ah = int(self.action_horizon)
        ad = int(self.action_dim)
        flat = np.asarray(jax.device_get(flat_action), dtype=np.float32).reshape(-1)
        expected = ah * ad
        if flat.size != expected:
            return np.full((ah, 2), np.nan, dtype=np.float64)
        deltas = flat.reshape(ah, ad)[:, :2].astype(np.float64)
        return np.cumsum(deltas, axis=0)

    def _sample_best_of_random_noises(self, batch_size: int, noise_jax: jax.Array) -> jnp.ndarray:
        """Debug rollout: sample random noises, log distributions, optionally pick the slowest chunk."""
        if not self.use_best_noise and not self._should_log_debug_noise_plot():
            return self._forward_pi0(batch_size, noise_jax, raw=None)

        import matplotlib.pyplot as plt

        n = int(self.debug_noise_samples)
        self._call_counter += 1
        rng = jax.random.PRNGKey(self._call_counter)
        rng, samp_rng = jax.random.split(rng)

        ref_noise = np.asarray(jax.device_get(noise_jax), dtype=np.float32)
        if ref_noise.ndim == 1:
            ref_noise = ref_noise.reshape(1, -1)
        noise_dim = int(ref_noise.shape[-1])
        candidate_noises = np.asarray(
            jax.device_get(
                jax.random.normal(samp_rng, (n, batch_size, noise_dim), dtype=jnp.float32)
            ),
            dtype=np.float32,
        )

        best_out: jnp.ndarray | None = None
        best_score = float("inf")
        best_idx = -1
        scores: list[float] = []
        xy_avgs: list[tuple[float, float]] = []
        xy_cumsum_steps: list[np.ndarray] = []
        for i in range(n):
            noise_i = jnp.asarray(candidate_noises[i], dtype=jnp.float32)
            out = self._forward_pi0(batch_size, noise_i * self.noise_scale, raw=None)
            score = self._debug_speed_score_from_flat(out)
            scores.append(score)
            cumsum_xy = self._debug_xy_cumsum_from_flat(out)
            xy_cumsum_steps.append(cumsum_xy)
            final_xy = cumsum_xy[-1]
            xy_avgs.append((float(final_xy[0]), float(final_xy[1])))
            if self.use_best_noise and score < best_score:
                best_score = score
                best_out = out
                best_idx = i

        scores_arr = np.asarray(scores, dtype=np.float64)
        xy_arr = np.asarray(xy_avgs, dtype=np.float64)
        xy_cumsum_arr = np.stack(xy_cumsum_steps, axis=0)
        score_mean = float(np.mean(scores_arr))
        score_std = float(np.std(scores_arr))
        score_max = float(np.max(scores_arr))
        score_min = float(np.min(scores_arr))

        should_log_plot = self._should_log_debug_noise_plot()
        plot_path: Path | None = None
        npz_path: Path | None = None
        if should_log_plot:
            fig, (ax_score, ax_xy) = plt.subplots(1, 2, figsize=(13, 5))

            ax_score.scatter(range(n), scores_arr, alpha=0.75, label="candidates")
            if self.use_best_noise and best_idx >= 0:
                ax_score.scatter(
                    [best_idx],
                    [scores_arr[best_idx]],
                    color="red",
                    s=80,
                    zorder=3,
                    label=f"best ({best_idx})",
                )
            ax_score.axhline(score_mean, color="C1", linestyle="-", linewidth=1.5, label="mean")
            ax_score.axhline(score_max, color="C2", linestyle="--", linewidth=1.2, label="max")
            ax_score.axhline(score_min, color="C3", linestyle="--", linewidth=1.2, label="min")
            ax_score.axhspan(
                score_mean - score_std,
                score_mean + score_std,
                color="C1",
                alpha=0.15,
                label="±1 std",
            )
            stats_text = (
                f"mean = {score_mean:.4f}\n"
                f"std  = {score_std:.4f}\n"
                f"max  = {score_max:.4f}\n"
                f"min  = {score_min:.4f}"
            )
            ax_score.text(
                0.02,
                0.98,
                stats_text,
                transform=ax_score.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.85},
            )
            ax_score.set_title("Speed score (lower = slower)")
            ax_score.set_xlabel("Candidate")
            ax_score.set_ylabel("||cumsum delta_xy||")
            ax_score.legend(loc="upper right", fontsize=8)

            cmap = plt.get_cmap("tab20", max(n, 1))
            valid_cumsum_mask = np.isfinite(xy_cumsum_arr).all(axis=(1, 2))
            final_points: list[np.ndarray] = []
            for i in range(n):
                if not valid_cumsum_mask[i]:
                    continue
                cumsum_xy = xy_cumsum_arr[i]
                color = cmap(i % cmap.N)
                ax_xy.plot(
                    cumsum_xy[:, 0],
                    cumsum_xy[:, 1],
                    color=color,
                    alpha=0.55,
                    linewidth=1.2,
                    zorder=2,
                )
                ax_xy.scatter(
                    cumsum_xy[:, 0],
                    cumsum_xy[:, 1],
                    color=[color],
                    alpha=0.35,
                    s=14,
                    edgecolors="none",
                    zorder=2,
                )
                final_xy = cumsum_xy[-1]
                final_points.append(final_xy)
                marker_edge = "red" if i == best_idx else "black"
                marker_lw = 1.0 if i == best_idx else 0.35
                ax_xy.scatter(
                    final_xy[0],
                    final_xy[1],
                    color=color,
                    marker="D",
                    s=52,
                    edgecolors=marker_edge,
                    linewidths=marker_lw,
                    zorder=4,
                )
                ax_xy.annotate(
                    f"c{i}",
                    (final_xy[0], final_xy[1]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    fontweight="bold" if i == best_idx else "normal",
                )

            if final_points:
                finals = np.stack(final_points, axis=0)
                overall_mean = finals.mean(axis=0)
                ax_xy.scatter(
                    overall_mean[0],
                    overall_mean[1],
                    color="black",
                    marker="X",
                    s=90,
                    linewidths=1.5,
                    zorder=5,
                )
                ax_xy.annotate(
                    "overall",
                    (overall_mean[0], overall_mean[1]),
                    xytext=(5, -10),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )

            ax_xy.axhline(0.0, color="0.7", linewidth=0.8)
            ax_xy.axvline(0.0, color="0.7", linewidth=0.8)
            ax_xy.set_title("Cumulative delta_xy per candidate")
            ax_xy.set_xlabel("x (m)")
            ax_xy.set_ylabel("y (m)")
            ax_xy.set_aspect("equal", adjustable="datalim")

            mode_label = "best_of_n" if self.use_best_noise else "log_only"
            fig.suptitle(
                f"debug_noise call={self._call_counter} step={self.debug_noise_episode_step} "
                f"mode={mode_label} route={self.debug_noise_route_name} "
                f"ep={self.debug_noise_episode}",
                fontsize=10,
            )
            fig.tight_layout()

            artifact_dir = self._debug_noise_artifact_dir()
            stem = f"debug_noise_step{self.debug_noise_episode_step:04d}_{self._call_counter:04d}"
            plot_path = artifact_dir / f"{stem}.png"
            npz_path = artifact_dir / f"{stem}.npz"
            fig.savefig(plot_path)
            plt.close(fig)

            np.savez(
                npz_path,
                candidate_noises=candidate_noises,
                scores=scores_arr,
                xy_final=xy_arr,
                xy_cumsum=xy_cumsum_arr,
                ref_noise=ref_noise,
                best_idx=np.int32(best_idx),
                use_best_noise=np.bool_(self.use_best_noise),
                call_counter=np.int32(self._call_counter),
                episode=np.int32(self.debug_noise_episode),
                episode_step=np.int32(self.debug_noise_episode_step),
                route_name=np.array(self.debug_noise_route_name),
            )

        best_speed_str = f"{best_score:.4f}" if best_idx >= 0 else "n/a"
        plot_msg = f" plot={plot_path} npz={npz_path}" if should_log_plot else " plot=skipped"
        print(
            f"[debug_noise] step={self.debug_noise_episode_step} candidates={n} "
            f"use_best_noise={self.use_best_noise} "
            f"best_idx={best_idx} best_speed_delta_xy={best_speed_str} "
            f"mean={score_mean:.4f} std={score_std:.4f} "
            f"min={score_min:.4f} max={score_max:.4f}{plot_msg}",
            flush=True,
        )

        if self.use_best_noise:
            if best_out is None:
                raise RuntimeError("debug_noise search failed to produce any action.")
            return jnp.asarray(best_out)
        return self._forward_pi0(batch_size, noise_jax, raw=None)

    def __call__(self, observations_jax: jax.Array, noise_jax: jax.Array) -> jax.Array:
        """DSRL ``vla_sample_fn``: map encoder observations + noise to CARLA actions.

        Image and proprio come from ``raw_obs_holder`` (not ``observations_jax``).
        Remote mode calls the HTTP actor once per batch row; ``noise_jax`` is ignored (server samples).
        """
        batch_size = int(noise_jax.shape[0])
        cached = self._next_cached_action(batch_size)
        if cached is not None:
            # Cache-hit path still corresponds to a real env step with a fresh `raw_obs_holder["obs"]`.
            # Re-stash the currently reused CoT so replay capture for this step remains aligned.
            if (
                batch_size == 1 
                and self._cached_cot is not None
                and self.raw_obs_holder is not None
                and isinstance(self.raw_obs_holder.get("obs"), dict)
            ):
                print(f"[DEBUG - steervla] Stashing cached CoT in raw obs holder")
                self._stash_cot_in_raw(self.raw_obs_holder["obs"], self._cached_cot)
            self._mark_action_served(batch_size)
            return cached

        if self._remote is not None:
            if self.raw_obs_holder is None:
                raise RuntimeError(
                    "Remote SteerVLAActor.__call__ requires ``raw_obs_holder`` with the latest CARLA gym dict."
                )
            raw = self.raw_obs_holder.get("obs")
            if not isinstance(raw, dict):
                raise RuntimeError('raw_obs_holder["obs"] must be a dict for remote VLA sampling.')
            rows: list[np.ndarray] = []
            for _ in range(batch_size):
                a = np.asarray(self._remote.get_action(raw), dtype=np.float32).reshape(-1)
                rows.append(a)
            out = jnp.asarray(np.stack(rows, axis=0))
            self._remember_action_chunk(out, batch_size)
            self._mark_action_served(batch_size)
            return out

        out = (
            self._sample_best_of_random_noises(batch_size, noise_jax)
            if self.debug_noise
            else self._forward_pi0(batch_size, noise_jax, raw=None)
        )
        self._remember_action_chunk(out, batch_size)
        self._mark_action_served(batch_size)
        return out

    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """Local: Pi0-CoT sample from CARLA gym dict ``state`` (``image``, ``state``, optional ``routing_command``)."""
        if self._remote is not None:
            out = self._remote.get_action(state)
            return np.asarray(out, dtype=np.float32)

        assert self.model is not None and self._jax_device is not None
        self._call_counter += 1
        rng = jax.random.PRNGKey(self._call_counter)
        rng_noise, rng_act = jax.random.split(rng)
        noise_jax = jax.random.normal(rng_noise, (1, self.model.action_dim), dtype=jnp.float32)
        noise_jax = noise_jax * jnp.asarray(self.noise_scale, dtype=jnp.float32)
        actions = self._forward_pi0(1, noise_jax, raw=state, rng=rng_act, force_accel_steer=True)
        self._mark_action_served(1)
        return np.asarray(jax.device_get(actions[0]), dtype=np.float32)

    def get_cot(self, state: Dict[str, Any]) -> dict:
        """Local: run ``sample_cot`` only; returns arrays converted to NumPy."""
        if self._remote is not None:
            return self._remote.get_cot(state)

        assert self.model is not None and self.tokenizer is not None and self._jax_device is not None
        self._call_counter += 1
        rng = jax.random.PRNGKey(self._call_counter)

        obs_np_struct = self.build_observation_batch_numpy(1, raw=state)
        obs_jax = jax.tree.map(
            lambda x: jax.device_put(jnp.asarray(x), self._jax_device),
            obs_np_struct,
        )
        cot_out = self._sample_or_reuse_cot(rng, obs_jax, 1)

        def _to_numpy(x: Any) -> Any:
            return np.asarray(jax.device_get(x))

        return jax.tree.map(_to_numpy, dict(cot_out))

    def sample_candidates(
        self,
        n: int,
        *,
        temperature: float,
        noise: jax.Array | None = None,
        raw: Optional[Dict[str, Any]] = None,
        rng: jax.Array | None = None,
    ) -> dict[str, Any]:
        """Best-of-N support: sample ``n`` CoTs at ``temperature`` and decode each subtask.

        One batched forward samples ``n`` diverse chains-of-thought (``temperature`` drives
        per-row diversity), then ``sample_actions`` produces one normalized action chunk per
        candidate. Each candidate's reasoning + subtask is printed for debugging.

        Returns a dict with:
          - ``actions``: ``(n, action_horizon * action_dim)`` float32 normalized chunks.
          - ``subtask_texts`` / ``reasoning_texts``: decoded strings, one per candidate.
          - ``cot_out``: the batched CoT dict; slice row ``i`` (via :meth:`stash_candidate_cot`)
            to persist the executed candidate's tokens for replay/training.
        """
        assert (
            self.model is not None
            and self._jax_device is not None
            and self.tokenizer is not None
        ), "sample_candidates requires a local SteerVLAActor (checkpoint loaded)."
        n = max(1, int(n))
        if rng is None:
            self._call_counter += 1
            rng = jax.random.PRNGKey(self._call_counter)
        else:
            rng = jax.random.fold_in(jnp.asarray(rng), self._call_counter)
        rng_cot, rng_act, rng_noise = jax.random.split(rng, 3)

        obs_np_struct = self.build_observation_batch_numpy(n, raw=raw)
        obs_jax = jax.tree.map(
            lambda x: jax.device_put(jnp.asarray(x), self._jax_device),
            obs_np_struct,
        )

        # n diverse CoTs in one batched call (temperature gives independent per-row samples).
        cot_out = self._sample_cot(
            rng_cot,
            obs_jax,
            temperature=float(temperature),
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
        )

        reason_tokens = np.asarray(jax.device_get(cot_out["tokenized_reasoning"]), dtype=np.int32)
        reason_mask = np.asarray(jax.device_get(cot_out["tokenized_reasoning_mask"]), dtype=bool)
        subtask_tokens = np.asarray(jax.device_get(cot_out["tokenized_subtask"]), dtype=np.int32)
        subtask_mask = np.asarray(jax.device_get(cot_out["tokenized_subtask_mask"]), dtype=bool)
        reasoning_texts: list[str] = []
        subtask_texts: list[str] = []
        for i in range(n):
            r_txt = self.tokenizer._tokenizer.decode(reason_tokens[i][reason_mask[i]].tolist())
            s_txt = self.tokenizer._tokenizer.decode(subtask_tokens[i][subtask_mask[i]].tolist())
            reasoning_texts.append(r_txt)
            subtask_texts.append(s_txt)
            print(
                f"[best_of_n][cand {i}] temp={float(temperature):.2f} "
                f"subtask={s_txt!r} reasoning={r_txt!r}",
                flush=True,
            )

        obs_full = _merge_cot_output_into_observation(obs_jax, cot_out)

        # Build per-candidate noise in model space.
        model_ah = int(self.model.action_horizon)
        model_ad = int(self.model.action_dim)
        cfg_ah = min(int(self.action_horizon), model_ah)
        cfg_ad = min(int(self.action_dim), model_ad)
        noise_scale = jnp.asarray(self.noise_scale, dtype=jnp.float32)
        if noise is None:
            noise_chunk = (
                jax.random.normal(rng_noise, (n, cfg_ah, cfg_ad), dtype=jnp.float32) * noise_scale
            )
        else:
            noise_arr = jnp.asarray(noise, dtype=jnp.float32)
            if noise_arr.ndim == 2:
                noise_arr = noise_arr.reshape(
                    noise_arr.shape[0], int(self.action_horizon), int(self.action_dim)
                )
            noise_chunk = noise_arr[:, :cfg_ah, :cfg_ad]
            if noise_chunk.shape[0] == 1 and n > 1:
                noise_chunk = jnp.broadcast_to(noise_chunk, (n, cfg_ah, cfg_ad))
        noise_full = jnp.zeros((n, model_ah, model_ad), dtype=jnp.float32)
        noise_full = noise_full.at[:, :cfg_ah, :cfg_ad].set(
            jax.device_put(noise_chunk, self._jax_device)
        )

        decode_bs = min(n, int(self.action_decode_batch_size))
        traj_parts: list[np.ndarray] = []
        for start in range(0, n, decode_bs):
            end = min(start + decode_bs, n)
            chunk_rng = jax.random.fold_in(rng_act, start)
            chunk_obs = jax.tree.map(lambda x: x[start:end], obs_full)
            chunk_noise = noise_full[start:end]
            traj = self._sample_actions(
                chunk_rng,
                chunk_obs,
                noise=chunk_noise,
                num_steps=int(self.sample_actions_num_steps),
                image_keys=CARLA_STEERVLA_IMAGE_KEYS,
            )
            jax.block_until_ready(traj)
            traj_np = self._postprocess_action_trajectory(
                traj, observation_state=jax.tree.map(lambda x: x[start:end], obs_jax.state)
            )
            traj_parts.append(np.asarray(traj_np, dtype=np.float32))
        actions_flat = np.concatenate(traj_parts, axis=0).reshape(n, -1)

        return {
            "actions": actions_flat,
            "subtask_texts": subtask_texts,
            "reasoning_texts": reasoning_texts,
            "cot_out": cot_out,
        }

    def stash_candidate_cot(
        self,
        cot_out: dict[str, Any],
        index: int,
        raw: Optional[Dict[str, Any]],
    ) -> None:
        """Persist candidate ``index`` from a batched :meth:`sample_candidates` CoT into ``raw``."""
        if raw is None:
            return
        i = int(index)
        sliced = {k: v[i : i + 1] for k, v in cot_out.items()}
        self._stash_cot_in_raw(raw, sliced)


def create_steervla_pi0_cot_sample_fn(
    steervla_cfg: MutableMapping[str, Any],
    raw_obs_holder: MutableMapping[str, Any],
    *,
    training_gpu_rank: int = -1,
    noise_scale: float = 1.0,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Build ``vla_sample_fn`` for :class:`jax_agents.dsrl.DSRLAgent`."""
    srank = steervla_cfg.get("training_gpu_rank", None)
    if srank is None:
        srank = training_gpu_rank
    url = steervla_cfg.get("actor_url")
    url_clean = str(url).strip() if url else ""

    ctor_kw = dict(
        raw_obs_holder=raw_obs_holder,
        routing_command=str(steervla_cfg.get("routing_command", "Follow the route.")),
        cot_temperature=float(steervla_cfg.get("cot_temperature", 0.0)),
        include_ego_history=bool(steervla_cfg.get("include_ego_history", False)),
        proprio_norm=bool(steervla_cfg.get("proprio_norm", True)),
        output_action_format=steervla_cfg.get("output_action_format") or "DELTA_XY_T_DELTA_XY_SPACE",
        action_horizon=int(steervla_cfg.get("action_horizon", 10)),
        action_dim=int(steervla_cfg.get("action_dim", 4)),
        actions_per_model_query=int(steervla_cfg.get("actions_per_model_query", 1)),
        actions_per_cot=int(steervla_cfg.get("actions_per_cot", 1)),
        sample_actions_num_steps=int(steervla_cfg.get("sample_actions_num_steps", 10)),
        action_decode_batch_size=int(steervla_cfg.get("action_decode_batch_size", 2)),
        training_gpu_rank=int(srank),
        return_normalized_action_chunk=bool(steervla_cfg.get("use_pi_action_chunk_for_env", True)),
        fixed_subtask_text=steervla_cfg.get("fixed_subtask_text"),
        fixed_reasoning_text=steervla_cfg.get("fixed_reasoning_text"),
        debug_noise=bool(steervla_cfg.get("debug_noise", False)),
        debug_noise_samples=int(steervla_cfg.get("debug_noise_samples", 15)),
        use_best_noise=bool(steervla_cfg.get("use_best_noise", True)),
        debug_noise_log_every_n_steps=int(
            steervla_cfg.get("debug_noise_log_every_n_steps", 5)
        ),
        noise_scale=float(steervla_cfg.get("noise_scale", noise_scale)),
    )

    if url_clean:
        actor = SteerVLAActor(actor_url=url_clean, actor_config=None, checkpoint_path=None, **ctor_kw)
    else:
        actor = SteerVLAActor(
            actor_config=str(steervla_cfg["actor_config"]),
            checkpoint_path=str(steervla_cfg["checkpoint"]),
            **ctor_kw,
        )

    def sample_fn(observations: jax.Array, noise: jax.Array) -> jax.Array:
        return actor(observations, noise)

    sample_fn.reset_action_cache = actor.reset_action_cache  # type: ignore[attr-defined]
    return sample_fn, actor
