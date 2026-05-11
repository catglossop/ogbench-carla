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

:class:`SteerVLAActor` ``update`` runs a VLA train step when DSRL is wired via :meth:`SteerVLAActor.attach_dsrl`.

``raw_obs_holder["obs"]`` must be the latest gym dict with ``"state"`` and ``"image"``
before each VLA sample when using :meth:`SteerVLAActor.__call__` (``main_carla`` maintains this).
For :meth:`SteerVLAActor.get_action` / :meth:`SteerVLAActor.get_cot`, pass that dict as ``state``.
"""

from __future__ import annotations

import dataclasses
import functools
import re
from pathlib import Path
from typing import Any, Callable, Dict, MutableMapping, Optional

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


from impls.vlas.utils import RemoteActor
from utils.flax_utils import TrainState
from jax_agents.dsrl import (
    DSRLAgent,
    dsrl_critic_min_q,
    dsrl_encode_obs_sample_noise_logprob,
)

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


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Load and validate weights for the provided parameter shape (can be a trainable subset)."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _merge_openpi_checkpoint_trees(loaded_params: at.Params, reference_params: at.Params, *, missing_regex: str) -> at.Params:
    """Same semantics as ``openpi.training.weight_loaders._merge_params`` (released checkpoints + LoRA holes)."""
    flat_ref = traverse_util.flatten_dict(reference_params, sep="/")
    flat_loaded = traverse_util.flatten_dict(loaded_params, sep="/")
    result: dict[str, Any] = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref_v = flat_ref[k]
            if hasattr(ref_v, "dtype") and hasattr(v, "dtype") and v.dtype != ref_v.dtype:
                result[k] = jnp.asarray(v, dtype=ref_v.dtype)
            else:
                result[k] = v

    flat_loaded.clear()
    pattern = re.compile(missing_regex)
    for k in {kk for kk in flat_ref if pattern.fullmatch(str(kk))}:
        if k not in result:
            result[k] = flat_ref[k]

    return traverse_util.unflatten_dict(result, sep="/")


def _partial_params_from_preloaded_checkpoint(
    preloaded_tree: at.Params,
    params_shape: at.Params,
) -> at.Params:
    """Merge Orbax/restored weights into the train-state param spec (LoRA holes), validate, drop struct leaves."""
    merged = _merge_openpi_checkpoint_trees(
        preloaded_tree,
        params_shape,
        missing_regex=".*lora.*",
    )
    at.check_pytree_equality(expected=params_shape, got=merged, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(merged).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: openpi_train_config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
    preloaded_partial_tree: at.Params | None = None,
) -> tuple[training_utils.TrainState, Any]:
    """Build OpenPI training state.

    If ``preloaded_partial_tree`` is set (e.g. from :func:`restore_openpi_params_on_single_gpu`), those weights are
    merged into the model shape and **no** ``config.weight_loader`` / disk read is used.
    Only parameters selected by ``config.trainable_filter`` are restored; frozen params stay at init values.
    """
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    trainable_params_shape = train_state_shape.params.filter(config.trainable_filter).to_pure_dict()
    if preloaded_partial_tree is not None:
        partial_params = _partial_params_from_preloaded_checkpoint(preloaded_partial_tree, trainable_params_shape)
    else:
        partial_params = _load_weights_and_validate(config.weight_loader, trainable_params_shape)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def pi0_cot_action_flow_matching_loss_per_step(
    model: Any,
    rng: at.KeyArrayLike,
    observation: _openpi_model.Observation,
    actions: _openpi_model.Actions,
    *,
    train: bool = False,
) -> jnp.ndarray:
    """Action flow-matching term from ``Pi0CoT.compute_loss`` (shape ``(batch, action_horizon)`` only, no CoT CE)."""
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _openpi_model.preprocess_observation(preprocess_rng, observation, train=train, image_keys=CARLA_STEERVLA_IMAGE_KEYS)

    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

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
    prefix_ar = jnp.array(
        img_ar
        + [False] * n_prompt
        + [True] * n_reasoning
        + [True] * n_subtask
    )

    suffix_tokens, suffix_mask, suffix_ar_list, adarms_cond = model._embed_action_suffix(observation, x_t, time)
    suffix_ar = jnp.array(suffix_ar_list)
    n_action = suffix_tokens.shape[1]

    attn_mask = model._build_attention_mask(
        prefix_mask,
        prefix_ar,
        suffix_mask,
        suffix_ar,
        n_img,
        n_prompt,
        n_subtask,
        n_reasoning,
        n_action,
    )

    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    positions = jnp.cumsum(input_mask, axis=1) - 1

    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
    )

    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)


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


def _vla_accel_steer_from_dsrl_noise(
    model: _openpi_model.BaseModel,
    rng: at.KeyArrayLike,
    openpi_observation: _openpi_model.Observation,
    noise_dsrl: jnp.ndarray,
    *,
    cot_temperature: float,
    cot_replay_reasoning: bool,
    sample_actions_num_steps: int,
    sample_actions_low_memory: bool,
    sample_actions_jit_denoise_steps: bool,
    action_horizon: int,
    action_dim: int,
    output_action_format: str | None,
) -> jnp.ndarray:
    """Map DSRL noise tensor to VLA action vector for RL losses (no device_put)."""
    rng_cot, rng_act = jax.random.split(rng)
    cot_out = model.sample_cot(
        rng_cot,
        openpi_observation,
        temperature=cot_temperature,
        image_keys=CARLA_STEERVLA_IMAGE_KEYS,
        replay_reasoning=cot_replay_reasoning,
    )
    obs_full = dataclasses.replace(
        openpi_observation,
        tokenized_reasoning=cot_out["tokenized_reasoning"],
        tokenized_reasoning_mask=cot_out["tokenized_reasoning_mask"],
        tokenized_subtask=cot_out["tokenized_subtask"],
        tokenized_subtask_mask=cot_out["tokenized_subtask_mask"],
    )
    batch_size = int(openpi_observation.state.shape[0])
    model_ah = int(model.action_horizon)
    model_ad = int(model.action_dim)
    cfg_ah = min(int(action_horizon), model_ah)
    cfg_ad = min(int(action_dim), model_ad)
    noise_full = jnp.zeros((batch_size, model_ah, model_ad), dtype=jnp.float32)
    if noise_dsrl.ndim == 3:
        noise_chunk = noise_dsrl[:, :cfg_ah, :cfg_ad]
    elif int(noise_dsrl.shape[-1]) == int(action_horizon) * int(action_dim):
        noise_chunk = noise_dsrl.reshape(batch_size, int(action_horizon), int(action_dim))[:, :cfg_ah, :cfg_ad]
    else:
        noise_chunk = noise_dsrl[:, None, :cfg_ad]
        cfg_ah = 1
    noise_full = noise_full.at[:, :cfg_ah, :cfg_ad].set(noise_chunk)
    # For RL losses, keep representation aligned with VLA action space (e.g. 4-D DELTA_XY_T_DELTA_XY_SPACE).
    traj = model.sample_actions(
        rng_act,
        obs_full,
        num_steps=int(sample_actions_num_steps),
        noise=noise_full,
        image_keys=CARLA_STEERVLA_IMAGE_KEYS,
        low_memory_denoise=bool(sample_actions_low_memory),
        jit_denoise_steps=bool(sample_actions_jit_denoise_steps),
    )
    out_dim = min(int(action_dim), int(model.action_dim))
    return traj[:, 0, :out_dim].astype(jnp.float32)


def flow_sample_with_vla(
    model: Any,
    rng: at.KeyArrayLike,
    openpi_observation: _openpi_model.Observation,
    noise_rl: jnp.ndarray,
    *,
    cot_temperature: float,
    cot_replay_reasoning: bool,
    sample_actions_num_steps: int,
    sample_actions_low_memory: bool,
    sample_actions_jit_denoise_steps: bool,
    action_horizon: int,
    action_dim: int,
    output_action_format: str | None,
) -> jnp.ndarray:
    """Map DSRL noise to env actions through Pi0-CoT flow sampling (denoise / ``sample_actions``)."""
    return _vla_accel_steer_from_dsrl_noise(
        model,
        rng,
        openpi_observation,
        noise_rl,
        cot_temperature=cot_temperature,
        cot_replay_reasoning=cot_replay_reasoning,
        sample_actions_num_steps=sample_actions_num_steps,
        sample_actions_low_memory=sample_actions_low_memory,
        sample_actions_jit_denoise_steps=sample_actions_jit_denoise_steps,
        action_horizon=action_horizon,
        action_dim=action_dim,
        output_action_format=output_action_format,
    )


@at.typecheck
def train_step(
    config: openpi_train_config.TrainConfig,
    alpha: float,
    noise_scale: float,
    cot_temperature: float,
    cot_replay_reasoning: bool,
    sample_actions_num_steps: int,
    sample_actions_low_memory: bool,
    sample_actions_jit_denoise_steps: bool,
    action_horizon: int,
    action_dim: int,
    output_action_format: str | None,
    dsrl_network: TrainState,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_openpi_model.Observation, jnp.ndarray],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """OpenPI SGD step with the same objective as :meth:`DSRLAgent.actor_loss`, but actions from the VLA.

    Samples noise from the (frozen) DSRL noise policy, maps it through the trainable Pi0-CoT model to env actions,
    and minimizes ``mean(alpha * log_prob(noise|s) - min_k Q_k(s, a))``. Only OpenPI trainable params get gradients.
    """
    openpi_observation, observations_rl = batch
    model = nnx.merge(state.model_def, state.params)
    model.train()

    train_rng = jax.random.fold_in(rng, state.step)
    rng_dsrl, rng_vla = jax.random.split(train_rng)

    @at.typecheck
    def loss_fn(
        model: _openpi_model.BaseModel,
        rng_vla_inner: at.KeyArrayLike,
        openpi_obs: _openpi_model.Observation,
        obs_rl: jnp.ndarray,
        rng_noise: at.KeyArrayLike,
    ):
        obs_e, noise, log_prob = dsrl_encode_obs_sample_noise_logprob(
            dsrl_network, noise_scale, obs_rl, rng_noise
        )
        actions_vla = flow_sample_with_vla(
            model,
            rng_vla_inner,
            openpi_obs,
            noise,
            cot_temperature=cot_temperature,
            cot_replay_reasoning=cot_replay_reasoning,
            sample_actions_num_steps=sample_actions_num_steps,
            sample_actions_low_memory=sample_actions_low_memory,
            sample_actions_jit_denoise_steps=sample_actions_jit_denoise_steps,
            action_horizon=action_horizon,
            action_dim=action_dim,
            output_action_format=output_action_format,
        )
        q = dsrl_critic_min_q(dsrl_network, obs_e, actions_vla)
        return jnp.mean(jnp.asarray(alpha, dtype=log_prob.dtype) * log_prob - q)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, rng_vla, openpi_observation, observations_rl, rng_dsrl
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def vla_flow_only_train_step(
    config: openpi_train_config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_openpi_model.Observation, _openpi_model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Single OpenPI optimizer step using only the Pi0-CoT **action flow-matching** term."""
    observation, actions = batch
    model = nnx.merge(state.model_def, state.params)
    model.train()

    train_rng = jax.random.fold_in(rng, state.step)

    @at.typecheck
    def loss_fn(
        model: _openpi_model.BaseModel,
        rng_inner: at.KeyArrayLike,
        obs: _openpi_model.Observation,
        act: _openpi_model.Actions,
    ):
        # expand actions to the dimensions of the base pi05 model 
        per = pi0_cot_action_flow_matching_loss_per_step(model, rng_inner, obs, act, train=True)
        return jnp.mean(per)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "vla_flow_loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


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


def _single_device_openpi_mesh(training_gpu_rank: int) -> tuple[jax.sharding.Mesh, jax.Device]:
    """Create a 1x1 mesh on one GPU so params are not replicated."""
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
    mesh = jax.sharding.Mesh(
        np.asarray([[device]], dtype=object),
        (sharding.BATCH_AXIS, sharding.FSDP_AXIS),
    )
    return mesh, device


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
        self.checkpoint_dir: Path | None = None
        self._mesh: jax.sharding.Mesh | None = None
        self._train_rng: jax.Array | None = None
        self._train_state: training_utils.TrainState | None = None
        self._ptrain_step: Callable[..., Any] | None = None
        self._dsrl_network: TrainState | None = None
        self._dsrl_alpha: float = 0.1
        self._dsrl_noise_scale: float = 1.0

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
            cot_jit_decode=self.cot_jit_decode,
            cot_jit_transformer_forward=self.cot_jit_transformer_forward,
            cot_replay_reasoning=self.cot_replay_reasoning,
        )
        self.train_cfg = dataclasses.replace(self.train_cfg, model=model_cfg)

        ckpt_root = Path(download.maybe_download(self.checkpoint_path)).resolve()
        self.checkpoint_dir = ckpt_root
        params_dir = ckpt_root / "params"
        print("Params directory: ", params_dir, flush=True)
        if not params_dir.is_dir():
            raise FileNotFoundError(f"Expected OpenPI checkpoint params at {params_dir}")

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
        self._jax_device = device
        
        print(f"[SteerVLA] Pi0-CoT checkpoint tensors loaded on {device}", flush=True)

        self.train_cfg = dataclasses.replace(
            self.train_cfg,
            weight_loader=_weight_loaders.CheckpointWeightLoader(str(params_dir)),
        )
        mesh, mesh_device = _single_device_openpi_mesh(training_gpu_rank)
        if int(getattr(self.train_cfg, "fsdp_devices", 1)) != 1:
            print(
                "[SteerVLA] Overriding TrainConfig.fsdp_devices for local actor: using single-device mesh.",
                flush=True,
            )
        self._mesh = mesh
        rng = jax.random.key(self.train_cfg.seed)
        self._train_rng, init_rng = jax.random.split(rng)
        train_state, _train_state_sharding = init_train_state(
            self.train_cfg,
            init_rng,
            mesh,
            resume=False,
        )
        jax.block_until_ready(train_state)
        self._train_state = train_state

        assert self.actor_config is not None and self.checkpoint_path is not None

        self.model = nnx.merge(train_state.model_def, train_state.params)
        self.model.eval()

        self.tokenizer = CoTPaligemmaTokenizer(
            max_prompt_len=model_cfg.max_token_len,
            max_subtask_len=model_cfg.max_subtask_len,
            max_reasoning_len=model_cfg.max_reasoning_len,
        )
        self.model_cfg = model_cfg
        self._refresh_ptrain_step()
        self._local_ready = True
        print(f"[SteerVLA] Train state + Pi0-CoT module on {mesh_device}", flush=True)

    def _refresh_ptrain_step(self) -> None:
        cfg = self.train_cfg
        if cfg is None:
            return
        self._ptrain_step = jax.jit(
            functools.partial(
                train_step,
                cfg,
                float(self._dsrl_alpha),
                float(self._dsrl_noise_scale),
                float(self.cot_temperature),
                bool(self.cot_replay_reasoning),
                int(self.sample_actions_num_steps),
                bool(self.sample_actions_low_memory),
                bool(self.sample_actions_jit_denoise_steps),
                int(self.action_horizon),
                int(self.action_dim),
                self.output_action_format,
            )
        )

    def attach_dsrl(self, agent: DSRLAgent) -> None:
        """Attach current DSRL state so actor-side updates use fresh RL parameters."""
        self.set_dsrl_network(agent.network)
        self._dsrl_alpha = float(agent.config["alpha"])
        self._dsrl_noise_scale = float(agent.config["noise_scale"])
        self._refresh_ptrain_step()

    @property
    def train_state(self) -> training_utils.TrainState | None:
        return self._train_state

    def set_dsrl_network(self, network: TrainState) -> None:
        """Update cached DSRL network snapshot used by :meth:`update`."""
        self._dsrl_network = network

    def apply_train_state(self, train_state: training_utils.TrainState) -> None:
        """Apply externally-updated OpenPI state and refresh local Pi0-CoT params."""
        self._train_state = train_state
        self.model = nnx.merge(train_state.model_def, train_state.params)
        self.model.eval()

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
        # Keep latest generated CoT in raw holder (batch-1 online CARLA path).
        if batch_size == 1:
            if raw is not None:
                self._stash_cot_in_raw(raw, cot_out)
            elif self.raw_obs_holder is not None and isinstance(self.raw_obs_holder.get("obs"), dict):
                self._stash_cot_in_raw(self.raw_obs_holder["obs"], cot_out)
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

    def update(
        self,
        batch: tuple[_openpi_model.Observation, jnp.ndarray] | None = None,
    ) -> dict[str, Any] | None:
        """One VLA step with DSRL actor_loss. ``batch = (openpi_observation, rl_observations)``."""
        if self._remote is not None:
            return self._remote.update(batch)
        if batch is None:
            return None
        if self._dsrl_network is None:
            raise RuntimeError("Call SteerVLAActor.attach_dsrl(DSRLAgent) before update(batch).")
        if self._ptrain_step is None or self._train_state is None or self._mesh is None:
            return None
        assert self._train_rng is not None
        with sharding.set_mesh(self._mesh):
            new_state, info = self._ptrain_step(
                self._dsrl_network,
                self._train_rng,
                self._train_state,
                batch,
            )
        self._train_state = new_state
        self.model = nnx.merge(new_state.model_def, new_state.params)
        self.model.eval()
        return jax.device_get(info)


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
    return sample_fn, actor


def openpi_action_expert_trainable_hint(train_cfg_name: str) -> str:
    """Note on freezing OpenPI backbone vs training the Gemma action expert only."""
    cfg = openpi_train_config.get_config(train_cfg_name)
    return (
        "[SteerVLA RL] DSRL keeps OpenPI inference frozen here; to train only the action expert, "
        "use OpenPI's TrainConfig.freeze_filter / trainable_filter (see Pi0CoTConfig.get_freeze_filter "
        "and nnx.Param routing for the expert stack). "
        f"Loaded TrainConfig={train_cfg_name!r}, model={type(cfg.model).__name__}."
    )
