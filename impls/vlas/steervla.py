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

import einops

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _openpi_model
from openpi.models import pi0 as _openpi_pi0
from openpi.models.pi0_config import Pi0CoTConfig
from openpi.models.tokenizer import CoTPaligemmaTokenizer
from openpi.policies import steervla_policy as sv_policy
from openpi.shared import array_typing as at
from openpi.shared import download
from openpi.training import config as openpi_train_config
from openpi.training import utils as training_utils
from openpi.training import weight_loaders as _weight_loaders
import openpi.training.sharding as sharding
from jax.sharding import SingleDeviceSharding


from impls.vlas.utils import RemoteActor

# Only the front camera for OpenPI preprocess + SigLIP (skip zero-padded wrist streams).
CARLA_STEERVLA_IMAGE_KEYS: tuple[str, ...] = ("base_0_rgb",)


def _denormalize_action_chunk_numpy(
    actions: np.ndarray,
    *,
    action_horizon: int,
    action_dim: int,
    output_action_format: str,
) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected batched flat action chunks, got shape {arr.shape}.")
    chunks = arr.reshape(arr.shape[0], action_horizon, action_dim)
    from openpi.visualizing.steervla_visualization import denormalize_actions

    denorm = denormalize_actions(chunks, action_dim, output_action_format)
    return np.asarray(denorm, dtype=np.float32).reshape(arr.shape[0], action_horizon * action_dim)


def _pad_action_chunk_to_model_numpy(
    actions: np.ndarray,
    *,
    src_action_horizon: int,
    src_action_dim: int,
    dst_action_horizon: int,
    dst_action_dim: int,
) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected batched flat action chunks, got shape {arr.shape}.")
    src = arr.reshape(arr.shape[0], src_action_horizon, src_action_dim)
    dst = np.zeros((arr.shape[0], dst_action_horizon, dst_action_dim), dtype=np.float32)
    copy_h = min(int(src_action_horizon), int(dst_action_horizon))
    copy_d = min(int(src_action_dim), int(dst_action_dim))
    dst[:, :copy_h, :copy_d] = src[:, :copy_h, :copy_d]
    return dst


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


# ---------------------------------------------------------------------------
# Pi0-CoT CARLA inference helpers
# ---------------------------------------------------------------------------

def routing_instruction_prompt(*, routing_command: str, current_speed_mps: float) -> str:
    """High-level instruction line (speed prefix matches ``SteerVLAInputs`` when ``speed_in_prompt``)."""
    rc = routing_command.strip()
    speed_prefix = round(float(current_speed_mps), 1)
    return f"The current speed is {speed_prefix} m/s. {rc}"


def carla_state_vec_to_steervla_state(
    carla_vec: np.ndarray,
    *,
    include_ego_history: bool,
    proprio_norm: bool,
) -> np.ndarray:
    """Map CARLA ego vector (``ogbench.carla.carla_utils._ego_state_vector`` layout) to padded proprio."""
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
    prompt = jnp.asarray(_get("openpi_tokenized_prompt"), dtype=jnp.int32)
    prompt_mask = jnp.asarray(_get("openpi_tokenized_prompt_mask"), dtype=bool)
    base_image = jnp.asarray(_get("openpi_image_base_0_rgb"), dtype=jnp.uint8)
    batch_size = int(state.shape[0])

    images: dict[str, jax.Array] = {}
    image_masks: dict[str, jax.Array] = {}
    for image_key in _openpi_model.IMAGE_KEYS:
        image_src = _get(f"openpi_image_{image_key}")
        if image_src is None:
            image = jnp.zeros_like(base_image)
            image_mask = jnp.zeros((batch_size,), dtype=bool)
        else:
            image = jnp.asarray(image_src, dtype=jnp.uint8)
            image_mask_src = _get(f"openpi_image_mask_{image_key}")
            image_mask = jnp.asarray(
                image_mask_src if image_mask_src is not None else jnp.ones((batch_size,), dtype=bool),
                dtype=bool,
            )
        images[image_key] = image
        image_masks[image_key] = image_mask

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
        "image": images,
        "image_mask": image_masks,
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
        training_gpu_rank: int = -1,
        return_normalized_action_chunk: bool = False,
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
        self.return_normalized_action_chunk = bool(return_normalized_action_chunk)
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
        # Persistent state-parameterized JIT functions (created once, never recreated).
        # _infer_state is updated in-place on each refresh so inference always uses
        # current weights without triggering XLA recompilation.
        self._infer_state = None
        self._persistent_jit_sample_actions = None
        self._persistent_jit_sample_cot = None

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
        self._refresh_compiled_inference_fns()
        self._local_ready = True

    def _refresh_compiled_inference_fns(self) -> None:
        """Initialize or refresh inference JIT functions.

        On the first call, creates persistent state-parameterised JIT functions so that
        later weight updates only swap ``self._infer_state`` without creating new
        ``jax.jit`` objects, avoiding XLA recompilation.
        """
        assert self.model is not None
        graphdef, state = nnx.split(self.model)
        self._infer_state = state

        if self._persistent_jit_sample_actions is None:
            # Created once. ``state`` is a dynamic arg, so different weight values never
            # trigger recompilation.
            def _sa_fn(state, rng, obs, *, noise=None, image_keys=CARLA_STEERVLA_IMAGE_KEYS, num_steps=10):
                model = nnx.merge(graphdef, state)
                return model.sample_actions(rng, obs, noise=noise, image_keys=image_keys, num_steps=num_steps)

            def _sc_fn(state, rng, obs, *, temperature=0.0, max_subtask_len=None, max_reasoning_len=None, image_keys=CARLA_STEERVLA_IMAGE_KEYS):
                model = nnx.merge(graphdef, state)
                return model.sample_cot(rng, obs, temperature=temperature, max_subtask_len=max_subtask_len, max_reasoning_len=max_reasoning_len, image_keys=image_keys)

            self._persistent_jit_sample_actions = jax.jit(
                _sa_fn, static_argnames=("num_steps", "image_keys")
            )
            self._persistent_jit_sample_cot = jax.jit(
                _sc_fn, static_argnames=("temperature", "max_subtask_len", "max_reasoning_len", "image_keys")
            )

        # Wrappers read self._infer_state at call time, so refreshing the state
        # propagates updated weights without touching the JIT objects.
        self._sample_actions = lambda *a, **kw: self._persistent_jit_sample_actions(self._infer_state, *a, **kw)
        self._sample_cot = lambda *a, **kw: self._persistent_jit_sample_cot(self._infer_state, *a, **kw)

    def _build_frozen_prefix_cache(
        self,
        model,
        observation: _openpi_model.Observation,
    ) -> tuple[_openpi_model.Observation, Any, jax.Array, jax.Array]:
        """Precompute frozen prefix KV/cache for suffix-only action-head training."""
        observation = _openpi_model.preprocess_observation(
            None,
            observation,
            train=False,
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
        )

        img_tokens, img_masks, img_ar = model._embed_images(observation)
        n_img = sum(t.shape[1] for t in img_tokens)

        prompt_emb = model._embed_text_tokens(observation.tokenized_prompt)
        prompt_mask = observation.tokenized_prompt_mask
        n_prompt = prompt_emb.shape[1]

        reasoning_emb = model._embed_text_tokens(observation.tokenized_reasoning)
        reasoning_mask = observation.tokenized_reasoning_mask
        n_reasoning = reasoning_emb.shape[1]

        subtask_emb = model._embed_text_tokens(observation.tokenized_subtask)
        subtask_mask = observation.tokenized_subtask_mask
        n_subtask = subtask_emb.shape[1]

        prefix_tokens = jnp.concatenate(img_tokens + [prompt_emb, reasoning_emb, subtask_emb], axis=1)
        prefix_mask = jnp.concatenate(img_masks + [prompt_mask, reasoning_mask, subtask_mask], axis=1)
        prefix_ar = jnp.array(img_ar + [False] * n_prompt + [True] * n_reasoning + [True] * n_subtask)

        prefix_attn_mask = _openpi_pi0.make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        reasoning_start = n_img + n_prompt
        reasoning_end = reasoning_start + n_reasoning
        prefix_len = prefix_mask.shape[1]
        col_is_reasoning = (jnp.arange(prefix_len) >= reasoning_start) & (jnp.arange(prefix_len) < reasoning_end)
        prefix_mask_no_reasoning = prefix_mask & ~col_is_reasoning[None, :]

        return (
            observation,
            jax.tree.map(jax.lax.stop_gradient, kv_cache),
            jax.lax.stop_gradient(prefix_mask),
            jax.lax.stop_gradient(prefix_mask_no_reasoning),
        )

    def encode_image_features(
        self,
        observation: _openpi_model.Observation,
    ) -> jax.Array:
        """Return a pooled Pi image feature vector for each batch row.

        Uses the same OpenPI preprocessing and image embedding path as the
        direct-DAgger trainer, but stops after the vision backbone and mean-pools
        the valid image tokens into a single feature vector per row.
        """
        if self._remote is not None:
            raise RuntimeError("Pi image features are not available in remote SteerVLAActor mode.")
        if self.model is None or self._jax_device is None:
            raise RuntimeError("Local SteerVLA model is not initialized.")

        obs_jax = jax.tree.map(
            lambda x: jax.device_put(jnp.asarray(x), self._jax_device),
            observation,
        )
        obs_proc = _openpi_model.preprocess_observation(
            None,
            obs_jax,
            train=False,
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
        )
        img_tokens, img_masks, _img_ar = self.model._embed_images(obs_proc)
        tokens = jnp.concatenate(img_tokens, axis=1)
        masks = jnp.concatenate(img_masks, axis=1).astype(tokens.dtype)
        denom = jnp.maximum(masks.sum(axis=1, keepdims=True), 1.0)
        pooled = jnp.sum(tokens * masks[..., None], axis=1) / denom
        return jax.lax.stop_gradient(pooled)

    def encode_prefix_tokens(
        self,
        observation: _openpi_model.Observation,
    ) -> tuple[jax.Array, jax.Array]:
        """Return the un-pooled frozen Pi prefix hidden states and their mask.

        Runs the full frozen prefix path (image tokens plus prompt/reasoning/
        subtask through the Pi transformer) and returns the per-token final-layer
        embeddings ``prefix_out (B, P, H)`` plus a boolean validity ``mask
        (B, P)``. These feed an external RL-token encoder; :meth:`encode_prefix_features`
        is the mean-pooled view of the same tensor.
        """
        if self._remote is not None:
            raise RuntimeError("Pi prefix tokens are not available in remote SteerVLAActor mode.")
        if self.model is None or self._jax_device is None:
            raise RuntimeError("Local SteerVLA model is not initialized.")

        obs_jax = jax.tree.map(
            lambda x: jax.device_put(jnp.asarray(x), self._jax_device),
            observation,
        )
        obs_proc = _openpi_model.preprocess_observation(
            None,
            obs_jax,
            train=False,
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
        )

        img_tokens, img_masks, img_ar = self.model._embed_images(obs_proc)
        prompt_emb = self.model._embed_text_tokens(obs_proc.tokenized_prompt)
        prompt_mask = obs_proc.tokenized_prompt_mask
        n_prompt = prompt_emb.shape[1]

        reasoning_emb = self.model._embed_text_tokens(obs_proc.tokenized_reasoning)
        reasoning_mask = obs_proc.tokenized_reasoning_mask
        n_reasoning = reasoning_emb.shape[1]

        subtask_emb = self.model._embed_text_tokens(obs_proc.tokenized_subtask)
        subtask_mask = obs_proc.tokenized_subtask_mask
        n_subtask = subtask_emb.shape[1]

        prefix_tokens = jnp.concatenate(
            img_tokens + [prompt_emb, reasoning_emb, subtask_emb], axis=1,
        )
        prefix_mask = jnp.concatenate(
            img_masks + [prompt_mask, reasoning_mask, subtask_mask], axis=1,
        )
        prefix_ar = jnp.array(
            img_ar + [False] * n_prompt + [True] * n_reasoning + [True] * n_subtask
        )

        prefix_attn_mask = _openpi_pi0.make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _kv_cache = self.model.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )
        return jax.lax.stop_gradient(prefix_out), prefix_mask

    def encode_prefix_features(
        self,
        observation: _openpi_model.Observation,
    ) -> jax.Array:
        """Mean-pooled view of :meth:`encode_prefix_tokens` (one vector per row)."""
        prefix_out, prefix_mask = self.encode_prefix_tokens(observation)
        prefix_mask_f = prefix_mask.astype(prefix_out.dtype)
        denom = jnp.maximum(prefix_mask_f.sum(axis=1, keepdims=True), 1.0)
        pooled = jnp.sum(prefix_out * prefix_mask_f[..., None], axis=1) / denom
        return jax.lax.stop_gradient(pooled)

    def encode_suffix_features(
        self,
        observation: _openpi_model.Observation,
        base_action: jax.Array | np.ndarray,
    ) -> jax.Array:
        """Return a frozen Pi suffix/action-head feature for each batch row.

        This reuses the same frozen prefix context as direct DAgger, then runs
        the Pi action suffix stack conditioned on the current base action chunk.
        The returned feature is the pooled hidden state over the final action
        suffix tokens immediately before ``action_out_proj``.
        """
        if self._remote is not None:
            raise RuntimeError("Pi suffix features are not available in remote SteerVLAActor mode.")
        if self.model is None or self._jax_device is None:
            raise RuntimeError("Local SteerVLA model is not initialized.")

        obs_jax = jax.tree.map(
            lambda x: jax.device_put(jnp.asarray(x), self._jax_device),
            observation,
        )
        obs_proc, kv_cache, prefix_mask, prefix_mask_no_reasoning = self._build_frozen_prefix_cache(
            self.model, obs_jax,
        )

        base_action_np = np.asarray(base_action, dtype=np.float32)
        padded_action = _pad_action_chunk_to_model_numpy(
            base_action_np,
            src_action_horizon=int(self.action_horizon),
            src_action_dim=int(self.action_dim),
            dst_action_horizon=int(self.model.action_horizon),
            dst_action_dim=int(self.model.action_dim),
        )
        x_t = jax.device_put(jnp.asarray(padded_action), self._jax_device)
        # Use a tiny fixed diffusion time so the suffix stack sees a nearly-final
        # action chunk while remaining in-distribution for the time embedding.
        time = jnp.full((x_t.shape[0],), 1e-3, dtype=jnp.float32)

        suffix_tokens, suffix_mask, suffix_ar_list, adarms_cond = self.model._embed_action_suffix(
            obs_proc, x_t, time,
        )
        suffix_ar = jnp.array(suffix_ar_list)
        suffix_attn_mask = _openpi_pi0.make_attn_mask(suffix_mask, suffix_ar)
        action_to_prefix = einops.repeat(
            prefix_mask_no_reasoning, "b p -> b s p", s=suffix_tokens.shape[1],
        )
        full_attn_mask = jnp.concatenate([action_to_prefix, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (_, suffix_out), _ = self.model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        action_hidden = suffix_out[:, -int(self.model.action_horizon) :]
        pooled = jnp.mean(action_hidden, axis=1)
        return jax.lax.stop_gradient(pooled)

    def flow_sample(self, rng, openpi_observation, input_noise):
        # Always run sample_cot so reasoning/subtask/FAST tokens match the current prompt.
        cot_out = self._sample_cot(
            rng,
            openpi_observation,
            temperature=float(self.cot_temperature),
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
        )
        obs_full = _merge_cot_output_into_observation(openpi_observation, cot_out)
        
        # Construct the noise
        batch_size = int(openpi_observation.state.shape[0])
        model_ah = int(self.model.action_horizon)
        model_ad = int(self.model.action_dim)
        cfg_ah = min(int(self.action_horizon), model_ah)
        cfg_ad = min(int(self.action_dim), model_ad)
        noise_full = jnp.zeros((batch_size, model_ah, model_ad), dtype=jnp.float32)
        full_chunk = None
        if input_noise.ndim == 3:
            noise_chunk = input_noise[:, :cfg_ah, :cfg_ad]
        elif int(input_noise.shape[-1]) == model_ah * model_ad:
            # Full Pi0 chunk from DSRL noise actor (actor_action_dim * action_horizon):
            # reshape directly so every dim/timestep gets noise (training uses N(0,I) over (B,H,D)).
            full_chunk = input_noise.reshape(batch_size, model_ah, model_ad).astype(jnp.float32)
        elif int(input_noise.shape[-1]) == int(self.action_horizon) * int(self.action_dim):
            noise_chunk = input_noise.reshape(batch_size, int(self.action_horizon), int(self.action_dim))[:, :cfg_ah, :cfg_ad]
        elif int(input_noise.shape[-1]) == model_ad:
            noise_chunk = jnp.broadcast_to(input_noise[:, None, :], (batch_size, cfg_ah, model_ad))
            cfg_ad = model_ad
        else:
            noise_chunk = input_noise[:, None, :cfg_ad]
            cfg_ah = 1
        if full_chunk is not None:
            noise_full = full_chunk
        else:
            noise_full = noise_full.at[:, :cfg_ah, :cfg_ad].set(noise_chunk)
        
        # Sample the actions
        traj = self._sample_actions(
            rng,
            obs_full,
            noise=noise_full,
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
            num_steps=int(self.sample_actions_num_steps),
        )
        target_dim = int(self.action_dim)
        first_step = traj[:, 0, :].astype(jnp.float32)
        out = jnp.zeros((batch_size, target_dim), dtype=jnp.float32)
        copy_dim = min(target_dim, int(first_step.shape[-1]))
        out = out.at[:, :copy_dim].set(first_step[:, :copy_dim])
        return out

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

        assert self.tokenizer is not None
        tok_ids, tok_mask = self.tokenizer.tokenize_prompt(prompt_text, state_pad)

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
            "state": np.tile(state_pad[None], (batch_size, 1)),
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

    def _sample_or_reuse_cot(
        self,
        rng: jax.Array,
        obs_jax: _openpi_model.Observation,
        batch_size: int,
    ) -> dict[str, Any]:
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

    def sample_action_distribution(
        self,
        raw: Dict[str, Any],
        *,
        num_samples: int = 32,
        seed: int | None = None,
    ) -> dict[str, np.ndarray] | None:
        """Sample the frozen Pi action expert on one observation for support diagnostics.

        Returns both normalized OpenPI chunks and ``policy_output`` chunks so they can
        be compared directly to CARLA ``expert_action`` labels.
        """
        if self._remote is not None:
            return None
        assert self.model is not None and self._jax_device is not None

        saved_action_chunk = self._cached_action_chunk
        saved_action_step = self._cached_action_step
        saved_cot = self._cached_cot
        saved_cot_actions_used = self._cached_cot_actions_used
        saved_counter = self._call_counter
        try:
            if seed is None:
                self._call_counter += 1
                seed = self._call_counter
            rng = jax.random.PRNGKey(int(seed))
            rng_noise, rng_zero = jax.random.split(rng)
            noise = jax.random.normal(
                rng_noise,
                (int(num_samples), int(self.action_horizon), int(self.action_dim)),
                dtype=jnp.float32,
            )
            zero_noise = jnp.zeros((1, int(self.action_horizon), int(self.action_dim)), dtype=jnp.float32)

            sampled_norm = self._forward_pi0(int(num_samples), noise, raw=raw, rng=rng_noise, force_accel_steer=False)
            zero_norm = self._forward_pi0(1, zero_noise, raw=raw, rng=rng_zero, force_accel_steer=False)

            sampled_norm_np = np.asarray(jax.device_get(sampled_norm), dtype=np.float32)
            zero_norm_np = np.asarray(jax.device_get(zero_norm), dtype=np.float32)
            sampled_policy = _denormalize_action_chunk_numpy(
                sampled_norm_np,
                action_horizon=int(self.action_horizon),
                action_dim=int(self.action_dim),
                output_action_format=str(self.output_action_format or "DELTA_XY_T_DELTA_XY_SPACE"),
            )
            zero_policy = _denormalize_action_chunk_numpy(
                zero_norm_np,
                action_horizon=int(self.action_horizon),
                action_dim=int(self.action_dim),
                output_action_format=str(self.output_action_format or "DELTA_XY_T_DELTA_XY_SPACE"),
            )
            return {
                "normalized": sampled_norm_np,
                "policy_output": sampled_policy,
                "zero_normalized": zero_norm_np,
                "zero_policy_output": zero_policy,
            }
        finally:
            self._cached_action_chunk = saved_action_chunk
            self._cached_action_step = saved_action_step
            self._cached_cot = saved_cot
            self._cached_cot_actions_used = saved_cot_actions_used
            self._call_counter = saved_counter

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
        full_chunk = None
        if noise_jax.ndim == 3:
            noise_chunk = noise_jax[:, :cfg_ah, :cfg_ad]
        elif int(noise_jax.shape[-1]) == model_ah * model_ad:
            # Full Pi0 chunk from DSRL noise actor (actor_action_dim * action_horizon):
            # reshape directly so every dim/timestep gets noise (training uses N(0,I) over (B,H,D)).
            full_chunk = noise_jax.reshape(batch_size, model_ah, model_ad).astype(jnp.float32)
        elif int(noise_jax.shape[-1]) == int(self.action_horizon) * int(self.action_dim):
            noise_chunk = noise_jax.reshape(batch_size, int(self.action_horizon), int(self.action_dim))[
                :, :cfg_ah, :cfg_ad
            ]
        elif int(noise_jax.shape[-1]) == model_ad:
            # DSRL noise is in full model action_dim space (e.g. actor_action_dim=32 == model_ad=32).
            # Broadcast it to all action steps so every step gets guidance, not just step 0.
            noise_chunk = jnp.broadcast_to(noise_jax[:, None, :], (batch_size, cfg_ah, model_ad))
            cfg_ad = model_ad
        else:
            noise_chunk = noise_jax[:, None, :cfg_ad]
            cfg_ah = 1
        if full_chunk is not None:
            noise_full = full_chunk
        else:
            noise_full = noise_full.at[:, :cfg_ah, :cfg_ad].set(noise_chunk)
        
        # Sample the actions
        traj = self._sample_actions(
            rng_act,
            obs_full,
            noise=noise_full,
            num_steps=int(self.sample_actions_num_steps),
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
        )
        if self.return_normalized_action_chunk and not force_accel_steer:
            traj_clip = traj[:, : int(self.action_horizon), : int(self.action_dim)]
            flat = traj_clip.reshape(batch_size, -1)
            return jax.device_put(flat.astype(jnp.float32), self._jax_device)

        first_step = traj[:, 0, :].astype(jnp.float32)
        target_dim = int(self.action_dim)
        out = jnp.zeros((batch_size, target_dim), dtype=jnp.float32)
        copy_dim = min(target_dim, int(first_step.shape[-1]))
        out = out.at[:, :copy_dim].set(first_step[:, :copy_dim])
        return out

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

        out = self._forward_pi0(batch_size, noise_jax, raw=None)
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
        cot_out = self._sample_cot(
            rng,
            obs_jax,
            temperature=float(self.cot_temperature),
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
            # replay_reasoning=self.cot_replay_reasoning,
        )

        def _to_numpy(x: Any) -> Any:
            return np.asarray(jax.device_get(x))

        return jax.tree.map(_to_numpy, dict(cot_out))


def create_steervla_pi0_cot_sample_fn(
    steervla_cfg: MutableMapping[str, Any],
    raw_obs_holder: MutableMapping[str, Any],
    *,
    training_gpu_rank: int = -1,
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
        training_gpu_rank=int(srank),
        return_normalized_action_chunk=bool(steervla_cfg.get("use_pi_action_chunk_for_env", True)),
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
