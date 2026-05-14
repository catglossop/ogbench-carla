"""SteerVLA / OpenPI: smoke-test loader, remote actor, and CARLA Pi0-CoT inference.

**Checkpoint smoke tests** — :class:`SteerVLAActor` (local checkpoint mode) restores OpenPI weights **once**, builds
:class:`openpi.training.utils.TrainingState` via :func:`init_train_state`, and shares those params with the Pi0-CoT
module used at inference.

**Pi0-CoT + DSRL** — :class:`SteerVLAActor` and :func:`create_steervla_pi0_cot_sample_fn`
implement ``vla_sample_fn`` for :class:`jax_agents.dsrl.DSRLAgent.sample_actions_with_vla`.

Uses ``openpi.models.pi0_cot.Pi0CoT.sample_cot`` then ``sample_actions`` (see
``openpi/visualizing/steervla_visualization.py``). Prompt layout follows
:class:`openpi.models.tokenizer.CoTPaligemmaTokenizer` (``Prompt:...;State:...``
through ``<start_of_reasoning>``).

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

from numpy.ma import innerproduct

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import model as _openpi_model
from openpi.models.pi0_config import Pi0CoTConfig
from openpi.models.tokenizer import CoTPaligemmaTokenizer
from openpi.policies import steervla_policy as sv_policy
from openpi.shared import array_typing as at
from openpi.shared import download
from openpi.shared import nnx_utils as nnx_utils
from openpi.training import config as openpi_train_config
from openpi.training import optimizer as _optimizer
from openpi.training import utils as training_utils
from openpi.training import weight_loaders as _weight_loaders
import openpi.training.sharding as sharding
from jax.sharding import SingleDeviceSharding


from impls.vlas.utils import RemoteActor

# Only the front camera for OpenPI preprocess + SigLIP (skip zero-padded wrist streams).
CARLA_STEERVLA_IMAGE_KEYS: tuple[str, ...] = ("base_0_rgb",)


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
    pad_to_dim: int,
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
    out = np.zeros((pad_to_dim,), dtype=np.float32)
    out[: min(flat.size, pad_to_dim)] = flat[:pad_to_dim]
    return out

def _observation_has_cot_tokens(observation: _openpi_model.Observation) -> bool:
    """Whether an OpenPI observation already carries non-empty CoT tokens."""
    try:
        r_mask = jnp.asarray(observation.tokenized_reasoning_mask)
        s_mask = jnp.asarray(observation.tokenized_subtask_mask)
        return bool(jnp.any(r_mask) or jnp.any(s_mask))
    except Exception:
        return False


def with_replay_cot_tokens(
    openpi_observation: _openpi_model.Observation,
    replay_batch: dict[str, Any],
    *,
    prefix: str = "",
) -> _openpi_model.Observation:
    """Overlay replay-stored CoT tokens onto an OpenPI observation when present."""
    rk = f"{prefix}reasoning"
    rmk = f"{prefix}reasoning_mask"
    sk = f"{prefix}subtask"
    smk = f"{prefix}subtask_mask"
    if rk not in replay_batch and sk not in replay_batch:
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

    return dataclasses.replace(
        openpi_observation,
        tokenized_reasoning=reasoning,
        tokenized_reasoning_mask=reasoning_mask,
        tokenized_subtask=subtask,
        tokenized_subtask_mask=subtask_mask,
    )


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
        )
        self.model_cfg = model_cfg
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
    
    def flow_sample(self, rng, openpi_observation, input_noise):
        
        if _observation_has_cot_tokens(openpi_observation):
            print(f"[DEBUG - steervla] Using stored CoT")
            obs_full = openpi_observation
        else:
            cot_out = self._sample_cot(
                rng,
                openpi_observation,
                temperature=float(self.cot_temperature),
                image_keys=CARLA_STEERVLA_IMAGE_KEYS,
            )
            obs_full = dataclasses.replace(
                openpi_observation, 
                tokenized_reasoning=cot_out["tokenized_reasoning"], tokenized_reasoning_mask=cot_out["tokenized_reasoning_mask"], tokenized_subtask=cot_out["tokenized_subtask"], tokenized_subtask_mask=cot_out["tokenized_subtask_mask"])
        
        # Construct the noise
        batch_size = int(openpi_observation.state.shape[0])
        model_ah = int(self.model.action_horizon)
        model_ad = int(self.model.action_dim)
        cfg_ah = min(int(self.action_horizon), model_ah)
        cfg_ad = min(int(self.action_dim), model_ad)
        noise_full = jnp.zeros((batch_size, model_ah, model_ad), dtype=jnp.float32)
        if input_noise.ndim == 3:
            noise_chunk = input_noise[:, :cfg_ah, :cfg_ad]
        elif int(input_noise.shape[-1]) == int(self.action_horizon) * int(self.action_dim):
            noise_chunk = input_noise.reshape(batch_size, int(self.action_horizon), int(self.action_dim))[:, :cfg_ah, :cfg_ad]
        else:
            noise_chunk = input_noise[:, None, :cfg_ad]
            write_ah = 1
        noise_full = noise_full.at[:, :write_ah, :cfg_ad].set(noise_chunk)
        
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
            pad_to_dim=self.model_cfg.action_dim,
        )

        assert self.tokenizer is not None
        tok_ids, tok_mask = self.tokenizer.tokenize_prompt(prompt_text, state_pad)

        # Optional CoT fields carried in raw obs holder from previous VLA inference.
        reasoning_len = int(self.model_cfg.max_reasoning_len)
        subtask_len = int(self.model_cfg.max_subtask_len)
        reasoning = np.zeros((batch_size, reasoning_len), dtype=np.int32)
        reasoning_mask = np.zeros((batch_size, reasoning_len), dtype=bool)
        subtask = np.zeros((batch_size, subtask_len), dtype=np.int32)
        subtask_mask = np.zeros((batch_size, subtask_len), dtype=bool)

        if isinstance(raw, dict):
            rr = raw.get("reasoning")
            rrm = raw.get("reasoning_mask")
            ss = raw.get("subtask")
            ssm = raw.get("subtask_mask")
            if rr is not None:
                rr_arr = np.asarray(rr, dtype=np.int32).reshape(-1)
                n = min(reasoning_len, rr_arr.size)
                reasoning[:, :n] = rr_arr[:n]
            if rrm is not None:
                rrm_arr = np.asarray(rrm, dtype=bool).reshape(-1)
                n = min(reasoning_len, rrm_arr.size)
                reasoning_mask[:, :n] = rrm_arr[:n]
            else:
                reasoning_mask = reasoning != 0
            if ss is not None:
                ss_arr = np.asarray(ss, dtype=np.int32).reshape(-1)
                n = min(subtask_len, ss_arr.size)
                subtask[:, :n] = ss_arr[:n]
            if ssm is not None:
                ssm_arr = np.asarray(ssm, dtype=bool).reshape(-1)
                n = min(subtask_len, ssm_arr.size)
                subtask_mask[:, :n] = ssm_arr[:n]
            else:
                subtask_mask = subtask != 0

        # Single base camera only (see CARLA_STEERVLA_IMAGE_KEYS + Pi0CoT ``image_keys``).
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
            raw["reasoning"] = np.asarray(jax.device_get(cot_out["tokenized_reasoning"][0]), dtype=np.int32)
            raw["reasoning_mask"] = np.asarray(jax.device_get(cot_out["tokenized_reasoning_mask"][0]), dtype=bool)
            raw["subtask"] = np.asarray(jax.device_get(cot_out["tokenized_subtask"][0]), dtype=np.int32)
            raw["subtask_mask"] = np.asarray(jax.device_get(cot_out["tokenized_subtask_mask"][0]), dtype=bool)
        except Exception:
            # Keep rollout robust if CoT payload changes shape unexpectedly.
            return

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

        # Replace the reasoning, reasoning mask, subtask, and subtask mask in the observation
        obs_full = dataclasses.replace(
            obs_jax,
            tokenized_reasoning=cot_out["tokenized_reasoning"],
            tokenized_reasoning_mask=cot_out["tokenized_reasoning_mask"],
            tokenized_subtask=cot_out["tokenized_subtask"],
            tokenized_subtask_mask=cot_out["tokenized_subtask_mask"],
        )
        
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
        if self.return_normalized_action_chunk and not force_accel_steer:
            traj = self._sample_actions(
                rng_act,
                obs_full,
                noise=noise_full,
                num_steps=int(self.sample_actions_num_steps),
                image_keys=CARLA_STEERVLA_IMAGE_KEYS,
            )
            traj_clip = traj[:, : int(self.action_horizon), : int(self.action_dim)]
            flat = traj_clip.reshape(batch_size, -1)
            out = jax.device_put(flat.astype(jnp.float32), self._jax_device)
            return out

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
