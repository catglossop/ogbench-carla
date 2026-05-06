"""SteerVLA / OpenPI: smoke-test loader, remote actor, and CARLA Pi0-CoT inference.

**Checkpoint smoke tests** — :class:`SteerVLALocalActor` loads params via ``TrainConfig.model.load``.

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
:class:`SteerVLAActor` uses per-step helpers inside ``Pi0CoT.sample_cot``; whether those use
``module_jit`` is controlled by ``Pi0CoTConfig.cot_jit_decode`` and ``cot_jit_transformer_forward``
(override from ``steervla`` config or env on the inference server).
``cot_replay_reasoning`` controls whether CoT generation replays generated reasoning before subtask generation.

By default it also runs ``Pi0CoT.sample_actions`` **fully eagerly** via
``low_memory_denoise=True`` (Python loop over flow steps, no outer ``nnx.jit`` on the action
path): **lower peak VRAM**, slower than fused JIT. Set ``sample_actions_low_memory: false`` and
``sample_actions_jit_denoise_steps: false`` to use the outer ``nnx.jit`` wrapper
(``_jit_pi0_sample_actions``, highest speed / highest VRAM). Set ``sample_actions_jit_denoise_steps:
true`` (with ``sample_actions_low_memory: false``) for per-step jitted ``_denoise_flow_step`` only.
If both ``sample_actions_low_memory`` and ``sample_actions_jit_denoise_steps`` are true, **eager**
wins. Lower ``sample_actions_num_steps`` to save time and memory. Tokenizer and NumPy packing stay
outside JIT. CARLA uses only ``base_0_rgb`` (see :data:`CARLA_STEERVLA_IMAGE_KEYS`).

:class:`SteerVLAActor` ``update`` on a local actor is a no-op for OpenPI weights.

``raw_obs_holder["obs"]`` must be the latest gym dict with ``"state"`` and ``"image"``
before each VLA sample when using :meth:`SteerVLAActor.__call__` (``main_carla`` maintains this).
For :meth:`SteerVLAActor.get_action` / :meth:`SteerVLAActor.get_cot`, pass that dict as ``state``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable, Dict, MutableMapping, Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _openpi_model
from openpi.models.pi0_config import Pi0CoTConfig
from openpi.models.tokenizer import CoTPaligemmaTokenizer
from openpi.policies import steervla_policy as sv_policy
from openpi.shared import download
from openpi.training import config as openpi_train_config

from impls.vlas.utils import LocalActor, RemoteActor

# Only the front camera for OpenPI preprocess + SigLIP (skip zero-padded wrist streams).
CARLA_STEERVLA_IMAGE_KEYS: tuple[str, ...] = ("base_0_rgb",)


def restore_openpi_params_on_single_gpu(
    params_dir: Path | str,
    *,
    training_gpu_rank: int = -1,
):
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
    from jax.sharding import SingleDeviceSharding

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
# Smoke-test local actor (generic OpenPI load)
# ---------------------------------------------------------------------------


class SteerVLALocalActor(LocalActor):
    """Loads JAX params via ``TrainConfig.model.load`` after ``maybe_download``."""

    def __init__(self, actor_config: str, checkpoint_path: str) -> None:
        super().__init__(actor_config, checkpoint_path)
        self.checkpoint_dir: Optional[Path] = None
        self.policy = None

    def setup(self, *, training_gpu_rank: int = -1) -> None:
        cfg = openpi_train_config.get_config(self.actor_config)
        path = download.maybe_download(self.checkpoint_path)
        self.checkpoint_dir = Path(path).resolve()
        params_dir = self.checkpoint_dir / "params"
        if not params_dir.exists():
            raise FileNotFoundError(f"Expected OpenPI checkpoint params at {params_dir}")
        params, dev = restore_openpi_params_on_single_gpu(params_dir, training_gpu_rank=training_gpu_rank)
        print(f"[SteerVLA] Loaded checkpoint params on {dev}", flush=True)
        self.policy = cfg.model.load(params)

    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError("Wire observation dict + policy inference for offline eval.")

    def get_cot(self, state: Dict[str, Any]) -> dict:
        raise NotImplementedError

    def update(self) -> None:
        raise NotImplementedError("Update is not supported for SteerVLA actor at this time.")


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


def _jax_denormalize_actions_chunk(
    actions: jnp.ndarray,
    *,
    action_dim: int,
    output_action_format: str | None,
) -> jnp.ndarray:
    """JAX equivalent of ``openpi.visualizing.steervla_visualization.denormalize_actions`` (subset of formats)."""
    x = actions[..., :action_dim]
    fmt = output_action_format

    if fmt in (
        "delta_speed_t_delta_course_t_delta_course_space",
        "DELTA_SPEED_T_DELTA_COURSE_T_DELTA_COURSE_SPACE",
    ):
        out = jnp.empty_like(x)
        out = out.at[..., 0].set(x[..., 0] * 10.0)
        out = out.at[..., 1].set(x[..., 1] * 180.0)
        out = out.at[..., 2].set(x[..., 2] * 180.0)
        return out

    if fmt in (
        "delta_xy_t_delta_xy_space",
        "DELTA_XY_T_DELTA_XY_SPACE",
    ):
        out = jnp.empty_like(x)
        out = out.at[..., :2].set(x[..., :2] * 7.0)
        out = out.at[..., 2:].set(x[..., 2:])
        return out

    if fmt in (
        "delta_xy_t_delta_course_space",
        "DELTA_XY_T_DELTA_COURSE_SPACE",
    ):
        out = jnp.empty_like(x)
        out = out.at[..., :2].set(x[..., :2] * 7.0)
        out = out.at[..., 2].set(x[..., 2] * 180.0)
        return out

    # Default nuScenes-style scaling from ``denormalize_actions``.
    out = jnp.empty_like(x)
    out = out.at[..., 0].set(x[..., 0] * 10.0)
    out = out.at[..., 1].set(x[..., 1] * 180.0)
    if action_dim > 2:
        out = out.at[..., 2:].set(x[..., 2:] * 15.0)
    return out


def _jax_chunk_to_accel_steer(
    actions: jnp.ndarray,
    *,
    output_action_format: str | None,
    action_dim_clip: int,
) -> jnp.ndarray:
    """First trajectory step → CARLA ``[accel, steer]`` in ``[-1, 1]`` (same mapping as the former NumPy path)."""
    chunk = actions[:, :1, :action_dim_clip]
    denorm = _jax_denormalize_actions_chunk(
        chunk, action_dim=chunk.shape[-1], output_action_format=output_action_format
    )
    dx = denorm[:, 0, 0]
    dy = denorm[:, 0, 1]
    accel = jnp.clip(dx / 7.0, -1.0, 1.0)
    steer = jnp.clip(
        jnp.arctan2(dy, jnp.maximum(jnp.abs(dx), jnp.float32(1e-3))) / (jnp.pi / 4),
        -1.0,
        1.0,
    )
    return jnp.stack([accel, steer], axis=-1)


def _pi0_sample_cot(
    model: Any,
    rng: jax.Array,
    observation: Any,
    *,
    temperature: float,
    image_keys_tuple: tuple[str, ...],
    replay_reasoning: bool,
) -> dict[str, jax.Array]:
    """Eager ``sample_cot`` — see module docstring (avoid jitting the full CoT HLO)."""
    return model.sample_cot(
        rng,
        observation,
        temperature=temperature,
        image_keys=image_keys_tuple,
        replay_reasoning=replay_reasoning,
    )


def _finalize_pi0_actions_for_carla(
    model: Any,
    traj: jax.Array,
    *,
    output_action_format: str | None,
) -> jax.Array:
    ad_clip = min(4, int(model.action_dim))
    return _jax_chunk_to_accel_steer(traj, output_action_format=output_action_format, action_dim_clip=ad_clip)


@nnx.jit(static_argnames=("num_steps", "output_action_format", "image_keys_tuple"))
def _jit_pi0_sample_actions(
    model: Any,
    rng: jax.Array,
    observation: Any,
    noise_full: jax.Array,
    *,
    num_steps: int,
    output_action_format: str | None,
    image_keys_tuple: tuple[str, ...],
) -> jax.Array:
    traj = model.sample_actions(
        rng,
        observation,
        num_steps=num_steps,
        noise=noise_full,
        image_keys=image_keys_tuple,
        low_memory_denoise=False,
        jit_denoise_steps=False,
    )
    return _finalize_pi0_actions_for_carla(model, traj, output_action_format=output_action_format)


def _eager_pi0_sample_actions(
    model: Any,
    rng: jax.Array,
    observation: Any,
    noise_full: jax.Array,
    *,
    num_steps: int,
    output_action_format: str | None,
    image_keys_tuple: tuple[str, ...],
) -> jax.Array:
    """No ``nnx.jit``: ``low_memory_denoise`` runs flow steps as separate forwards (less VRAM)."""
    traj = model.sample_actions(
        rng,
        observation,
        num_steps=num_steps,
        noise=noise_full,
        image_keys=image_keys_tuple,
        low_memory_denoise=True,
        jit_denoise_steps=False,
    )
    return _finalize_pi0_actions_for_carla(model, traj, output_action_format=output_action_format)


def _jit_denoise_step_pi0_sample_actions(
    model: Any,
    rng: jax.Array,
    observation: Any,
    noise_full: jax.Array,
    *,
    num_steps: int,
    output_action_format: str | None,
    image_keys_tuple: tuple[str, ...],
) -> jax.Array:
    """No outer ``nnx.jit``; each flow step uses :meth:`Pi0CoT._denoise_flow_step` (``nnx.jit``)."""
    traj = model.sample_actions(
        rng,
        observation,
        num_steps=num_steps,
        noise=noise_full,
        image_keys=image_keys_tuple,
        low_memory_denoise=False,
        jit_denoise_steps=True,
    )
    return _finalize_pi0_actions_for_carla(model, traj, output_action_format=output_action_format)


def _pi0_sample_cot_only(
    model: Any,
    rng: jax.Array,
    observation: Any,
    *,
    temperature: float,
    image_keys_tuple: tuple[str, ...],
    replay_reasoning: bool,
) -> dict[str, jax.Array]:
    """Eager ``sample_cot`` for :meth:`SteerVLAActor.get_cot`."""
    return model.sample_cot(
        rng,
        observation,
        temperature=temperature,
        image_keys=image_keys_tuple,
        replay_reasoning=replay_reasoning,
    )


# ---------------------------------------------------------------------------
# Remote HTTP actor or local Pi0-CoT CARLA actor
# ---------------------------------------------------------------------------


class SteerVLAActor:
    """SteerVLA as a remote HTTP client or as local OpenPI Pi0-CoT inference.

    Remote mode (``actor_url`` set) delegates ``get_action``, ``get_cot``, and ``update`` to
    :class:`RemoteActor` and never downloads or restores checkpoints locally.

    Local mode loads a Pi0-CoT checkpoint (requires ``actor_config`` and ``checkpoint_path``). Use
    :meth:`__call__` for DSRL flow sampling (reads ``raw_obs_holder["obs"]``). Use
    :meth:`get_action` / :meth:`get_cot` with a CARLA gym observation dict passed as ``state``.

    Local ``update`` does not mutate OpenPI weights (train via OpenPI / ``freeze_filter``).
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
        sample_actions_low_memory: bool = True,
        sample_actions_jit_denoise_steps: bool = False,
        training_gpu_rank: int = -1,
        cot_jit_decode: bool = True,
        cot_jit_transformer_forward: bool = True,
        cot_replay_reasoning: bool = True,
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
        self.sample_actions_low_memory = bool(sample_actions_low_memory)
        self.sample_actions_jit_denoise_steps = bool(sample_actions_jit_denoise_steps)
        self.cot_jit_decode = bool(cot_jit_decode)
        self.cot_jit_transformer_forward = bool(cot_jit_transformer_forward)
        self.cot_replay_reasoning = bool(cot_replay_reasoning)
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

        if actor_url is not None:
            self._remote = RemoteActor(
                actor_url,
                request_options={"replay_reasoning": self.cot_replay_reasoning},
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

        assert self.actor_config is not None and self.checkpoint_path is not None
        self.train_cfg = openpi_train_config.get_config(self.actor_config)
        model_cfg = self.train_cfg.model
        if not isinstance(model_cfg, Pi0CoTConfig):
            raise TypeError(
                f"SteerVLA Pi0-CoT expects Pi0CoTConfig; got {type(model_cfg).__name__} "
                f"for actor_config={self.actor_config!r}. Use e.g. pi05_steervla_cot_ki."
            )
        model_cfg = dataclasses.replace(
            model_cfg,
            cot_jit_decode=self.cot_jit_decode,
            cot_jit_transformer_forward=self.cot_jit_transformer_forward,
            cot_replay_reasoning=self.cot_replay_reasoning,
        )
        ckpt_root = Path(download.maybe_download(self.checkpoint_path)).resolve()
        params_dir = ckpt_root / "params"
        if not params_dir.is_dir():
            raise FileNotFoundError(f"Expected OpenPI checkpoint params at {params_dir}")
        restored, self._jax_device = restore_openpi_params_on_single_gpu(
            params_dir, training_gpu_rank=training_gpu_rank
        )
        print(f"[SteerVLA] Pi0-CoT checkpoint loaded on {self._jax_device}", flush=True)
        self.model = model_cfg.load(restored)
        self.model.eval()

        self.tokenizer = CoTPaligemmaTokenizer(
            max_prompt_len=model_cfg.max_token_len,
            max_subtask_len=model_cfg.max_subtask_len,
            max_reasoning_len=model_cfg.max_reasoning_len,
        )
        self.model_cfg = model_cfg
        self._local_ready = True

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
        }
        return _openpi_model.Observation.from_dict(data)

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
        cot_out = _pi0_sample_cot(
            self.model,
            rng,
            obs_jax,
            temperature=float(self.cot_temperature),
            image_keys_tuple=CARLA_STEERVLA_IMAGE_KEYS,
            replay_reasoning=self.cot_replay_reasoning,
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

        obs_np_struct = self.build_observation_batch_numpy(batch_size, raw=raw)
        obs_jax = jax.tree.map(
            lambda x: jax.device_put(jnp.asarray(x), self._jax_device),
            obs_np_struct,
        )
        noise_jax = jax.device_put(noise_jax, self._jax_device)
        rng_cot, rng_act = jax.random.split(rng)
        cot_out = self._sample_or_reuse_cot(rng_cot, obs_jax, batch_size)
        obs_full = dataclasses.replace(
            obs_jax,
            tokenized_reasoning=cot_out["tokenized_reasoning"],
            tokenized_reasoning_mask=cot_out["tokenized_reasoning_mask"],
            tokenized_subtask=cot_out["tokenized_subtask"],
            tokenized_subtask_mask=cot_out["tokenized_subtask_mask"],
        )
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
        if self.return_normalized_action_chunk and not force_accel_steer:
            traj = self.model.sample_actions(
                rng_act,
                obs_full,
                num_steps=int(self.sample_actions_num_steps),
                noise=noise_full,
                image_keys=CARLA_STEERVLA_IMAGE_KEYS,
                low_memory_denoise=self.sample_actions_low_memory,
                jit_denoise_steps=self.sample_actions_jit_denoise_steps,
            )
            traj_clip = traj[:, : int(self.action_horizon), : int(self.action_dim)]
            flat = traj_clip.reshape(batch_size, -1)
            return jax.device_put(flat.astype(jnp.float32), self._jax_device)

        sa_kw = dict(
            model=self.model,
            rng=rng_act,
            observation=obs_full,
            noise_full=noise_full,
            num_steps=int(self.sample_actions_num_steps),
            output_action_format=self.output_action_format,
            image_keys_tuple=CARLA_STEERVLA_IMAGE_KEYS,
        )
        if self.sample_actions_low_memory:
            carla_act = _eager_pi0_sample_actions(**sa_kw)
        elif self.sample_actions_jit_denoise_steps:
            carla_act = _jit_denoise_step_pi0_sample_actions(**sa_kw)
        else:
            carla_act = _jit_pi0_sample_actions(**sa_kw)
        return jax.device_put(carla_act.astype(jnp.float32), self._jax_device)

    def __call__(self, observations_jax: jax.Array, noise_jax: jax.Array) -> jax.Array:
        """DSRL ``vla_sample_fn``: map encoder observations + noise to CARLA actions.

        Image and proprio come from ``raw_obs_holder`` (not ``observations_jax``).
        Remote mode calls the HTTP actor once per batch row; ``noise_jax`` is ignored (server samples).
        """
        del observations_jax
        batch_size = int(noise_jax.shape[0])
        cached = self._next_cached_action(batch_size)
        if cached is not None:
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
        cot_out = _pi0_sample_cot_only(
            self.model,
            rng,
            obs_jax,
            temperature=float(self.cot_temperature),
            image_keys_tuple=CARLA_STEERVLA_IMAGE_KEYS,
            replay_reasoning=self.cot_replay_reasoning,
        )

        def _to_numpy(x: Any) -> Any:
            return np.asarray(jax.device_get(x))

        return jax.tree.map(_to_numpy, dict(cot_out))

    def update(self) -> Optional[Any]:
        """Remote: POST ``/update``. Local: no-op (train OpenPI separately)."""
        if self._remote is not None:
            return self._remote.update()
        return None


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
        sample_actions_low_memory=bool(steervla_cfg.get("sample_actions_low_memory", True)),
        sample_actions_jit_denoise_steps=bool(steervla_cfg.get("sample_actions_jit_denoise_steps", False)),
        training_gpu_rank=int(srank),
        cot_jit_decode=bool(steervla_cfg.get("cot_jit_decode", True)),
        cot_jit_transformer_forward=bool(steervla_cfg.get("cot_jit_transformer_forward", True)),
        cot_replay_reasoning=bool(steervla_cfg.get("cot_replay_reasoning", True)),
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
    return sample_fn


def openpi_action_expert_trainable_hint(train_cfg_name: str) -> str:
    """Note on freezing OpenPI backbone vs training the Gemma action expert only."""
    cfg = openpi_train_config.get_config(train_cfg_name)
    return (
        "[SteerVLA RL] DSRL keeps OpenPI inference frozen here; to train only the action expert, "
        "use OpenPI's TrainConfig.freeze_filter / trainable_filter (see Pi0CoTConfig.get_freeze_filter "
        "and nnx.Param routing for the expert stack). "
        f"Loaded TrainConfig={train_cfg_name!r}, model={type(cfg.model).__name__}."
    )
