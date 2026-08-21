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
import inspect
import json
import os
import re
import time
import types as _types
from pathlib import Path
from typing import Any, Callable, Dict, MutableMapping, Optional

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import model as _openpi_model
# Used by routing-commands' QGF guidance / frozen-prefix helpers (make_attn_mask).
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
STEERVLA_CACHE_DIR = "/raid/users/cglossop/openpi"


def _ensure_openpi_cache_dir() -> None:
    """Redirect OpenPI's download cache to NFS unless the caller overrode it.

    Uses ``setdefault`` so an explicit ``OPENPI_DATA_HOME`` in the environment still
    wins. Must run before any ``download.maybe_download`` call.
    """
    os.environ.setdefault("OPENPI_DATA_HOME", STEERVLA_CACHE_DIR)


def _pool_prefix_hidden(prefix_out: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Mean-pool valid prefix token hiddens to a single vector per batch row."""
    mask_f = mask.astype(prefix_out.dtype)[..., None]
    summed = jnp.sum(prefix_out * mask_f, axis=1)
    denom = jnp.maximum(jnp.sum(mask_f, axis=1), 1.0)
    return summed / denom


def _frozen_prefix_embed_forward(model, observation):
    """Pooled frozen-prefix embedding (jit target; bound to ``model`` via ``module_jit``).

    Same prefix construction as ``SteerVLAActor._build_frozen_prefix_cache`` but
    returns only the pooled embedding — the per-step policy_embed path doesn't
    need the KV cache, and running this eagerly costs ~5 s/step on an A100
    versus ~tens of ms jitted.
    """
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

    prefix_parts = img_tokens + [prompt_emb, reasoning_emb, subtask_emb]
    prefix_mask_parts = img_masks + [prompt_mask, reasoning_mask, subtask_mask]
    prefix_ar_list = img_ar + [False] * n_prompt + [True] * n_reasoning + [True] * n_subtask
    if getattr(model, "_use_fast_tokens", False) and observation.tokenized_fast is not None:
        fast_emb = model._embed_text_tokens(observation.tokenized_fast)
        prefix_parts.append(fast_emb)
        prefix_mask_parts.append(observation.tokenized_fast_mask)
        prefix_ar_list += [True] * int(fast_emb.shape[1])

    prefix_tokens = jnp.concatenate(prefix_parts, axis=1)
    prefix_mask = jnp.concatenate(prefix_mask_parts, axis=1)
    prefix_ar = jnp.array(prefix_ar_list)

    prefix_attn_mask = _openpi_pi0.make_attn_mask(prefix_mask, prefix_ar)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), _kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=positions,
    )

    reasoning_start = n_img + n_prompt
    reasoning_end = reasoning_start + n_reasoning
    prefix_len = prefix_mask.shape[1]
    col_is_reasoning = (jnp.arange(prefix_len) >= reasoning_start) & (jnp.arange(prefix_len) < reasoning_end)
    prefix_mask_no_reasoning = prefix_mask & ~col_is_reasoning[None, :]
    return jax.lax.stop_gradient(_pool_prefix_hidden(prefix_out, prefix_mask_no_reasoning))


def _compute_prefix_cache_for_qgf(model, observation):
    """Build prefix KV-cache and masks needed by QGF guided denoising.

    Like _frozen_prefix_embed_forward but returns kv_cache + masks instead of
    discarding them. Returns (preprocessed_obs, kv_cache, prefix_mask,
    prefix_mask_no_reasoning, pooled_embed).
    - preprocessed_obs: preprocess_observation output (needed by _denoise_flow_step)
    - kv_cache: prefix KV cache to reuse across all N denoising steps
    - prefix_mask / prefix_mask_no_reasoning: attention masks for suffix pass
    - pooled_embed: pooled prefix hidden (same as _frozen_prefix_embed_forward)
    """
    observation = _openpi_model.preprocess_observation(
        None, observation, train=False, image_keys=CARLA_STEERVLA_IMAGE_KEYS
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

    prefix_parts = img_tokens + [prompt_emb, reasoning_emb, subtask_emb]
    prefix_mask_parts = img_masks + [prompt_mask, reasoning_mask, subtask_mask]
    prefix_ar_list = img_ar + [False] * n_prompt + [True] * n_reasoning + [True] * n_subtask
    if getattr(model, "_use_fast_tokens", False) and observation.tokenized_fast is not None:
        fast_emb = model._embed_text_tokens(observation.tokenized_fast)
        prefix_parts.append(fast_emb)
        prefix_mask_parts.append(observation.tokenized_fast_mask)
        prefix_ar_list += [True] * int(fast_emb.shape[1])

    prefix_tokens = jnp.concatenate(prefix_parts, axis=1)
    prefix_mask = jnp.concatenate(prefix_mask_parts, axis=1)
    prefix_ar = jnp.array(prefix_ar_list)

    prefix_attn_mask = _openpi_pi0.make_attn_mask(prefix_mask, prefix_ar)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )

    prefix_len = prefix_mask.shape[1]
    reasoning_start = n_img + n_prompt
    reasoning_end = reasoning_start + n_reasoning
    col_is_reasoning = (jnp.arange(prefix_len) >= reasoning_start) & (jnp.arange(prefix_len) < reasoning_end)
    prefix_mask_no_reasoning = prefix_mask & ~col_is_reasoning[None, :]

    pooled = jax.lax.stop_gradient(_pool_prefix_hidden(prefix_out, prefix_mask_no_reasoning))
    return observation, kv_cache, prefix_mask, prefix_mask_no_reasoning, pooled


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
    single_sharding = SingleDeviceSharding(device)
    params = _openpi_model.restore_params(params_dir, sharding=single_sharding)
    return params, device


def _pick_single_gpu_device(training_gpu_rank: int = -1) -> jax.Device:
    """Resolve the single accelerator to place SteerVLA on (same policy as the restore helper)."""
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    if gpus:
        idx = training_gpu_rank if training_gpu_rank >= 0 else 0
        idx = min(max(idx, 0), len(gpus) - 1)
        return gpus[idx]
    return jax.devices()[0]


def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader,
    params_shape: at.Params,
) -> at.Params:
    """Load + validate a checkpoint subset against the target param shapes.

    Verbatim port of ``scripts/train.py :: _load_weights_and_validate`` so the trainable-state path
    here matches OpenPI training exactly.
    """
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    # Drop jax.ShapeDtypeStruct leaves so only the actually-loaded params are returned.
    return traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )


def init_openpi_train_state_single_gpu(
    train_cfg: openpi_train_config.TrainConfig,
    *,
    training_gpu_rank: int = -1,
):
    """Build a **full trainable** OpenPI ``TrainState`` pinned to one accelerator.

    Mirrors ``scripts/train.py :: init_train_state`` (fresh model → merge checkpoint params →
    cast frozen params to bf16 → optimizer + ``opt_state`` over the trainable filter), but places
    the whole state on a single-device mesh instead of the training FSDP mesh so the online CARLA
    actor does not replicate weights across every visible GPU.

    Returns ``(train_state, mesh, device)``. ``train_state.tx`` / ``opt_state`` are ready for
    gradient steps; ``nnx.merge(train_state.model_def, train_state.params)`` reconstructs the model.
    """
    device = _pick_single_gpu_device(training_gpu_rank)
    mesh = jax.sharding.Mesh(
        np.asarray([device]).reshape(1, 1),
        (sharding.BATCH_AXIS, sharding.FSDP_AXIS),
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    tx = _optimizer.create_optimizer(train_cfg.optimizer, train_cfg.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = train_cfg.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # Errors if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        # Convert frozen params to bfloat16 (trainable params stay full precision).
        params = nnx_utils.state_map(
            params, train_cfg.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
        )
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(train_cfg.trainable_filter)),
            ema_decay=train_cfg.ema_decay,
            ema_params=None if train_cfg.ema_decay is None else params,
        )

    init_rng = jax.random.key(0)
    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)

    partial_params = _load_weights_and_validate(train_cfg.weight_loader, train_state_shape.params.to_pure_dict())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, mesh, device


def _openpi_hl_train_step(
    config: openpi_train_config.TrainConfig,
    rng: jax.Array,
    state: training_utils.TrainState,
    batch: tuple[_openpi_model.Observation, jnp.ndarray],
):
    """One OpenPI gradient step, jit-friendly (bind ``config`` via ``functools.partial``).

    Verbatim port of ``scripts/train.py :: train_step`` (grads filtered to
    ``config.trainable_filter``, ``tx.update`` + ``optax.apply_updates``, EMA). For the CAST-relabel
    high-level (VLM-backbone) update the batch is built with ``action_loss_mask`` all-``False`` so the
    action-flow loss is zero — the action-expert params receive no gradient and only the CoT/VLM
    backbone is updated, exactly like OpenPI's ``steervla_hl_datasets`` (``action_supervision=False``).
    """
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, rng, observation, actions):
        if hasattr(model, "compute_loss_with_aux"):
            chunked_loss, aux_metrics = model.compute_loss_with_aux(rng, observation, actions, train=True)
        else:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            aux_metrics = {}
        loss = jnp.mean(chunked_loss)
        reduced_aux = {k: jnp.mean(v) for k, v in aux_metrics.items()}
        return loss, reduced_aux

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
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
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    info = {"loss": loss, "grad_norm": optax.global_norm(grads)}
    info.update(aux_metrics)
    return new_state, info


def _cot_ce_per_example(model, rng, observation, actions) -> jnp.ndarray:
    """Per-example CoT cross-entropy ``(B,)`` = ``-logπ(cot | state)`` of the teacher-forced tokens.

    With the batch's ``action_loss_mask`` all-``False`` the flow loss is zero, so this is purely the
    reasoning+subtask CE — the same quantity the HL BC step trains, just kept per-example.
    """
    if hasattr(model, "compute_loss_with_aux"):
        chunked_loss, _ = model.compute_loss_with_aux(rng, observation, actions, train=True)
    else:
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
    ce = jnp.asarray(chunked_loss)
    return ce.reshape(ce.shape[0], -1).mean(axis=-1) if ce.ndim > 1 else ce


def _openpi_hl_grpo_step(
    config: openpi_train_config.TrainConfig,
    rng: jax.Array,
    state: training_utils.TrainState,
    ref_params,
    batch: tuple[_openpi_model.Observation, jnp.ndarray],
    advantages: jnp.ndarray,
    beta: jnp.ndarray,
):
    """One GRPO gradient step on the HL (CoT/subtask) policy; the action expert is never modified.

    Trajectory-level GRPO: ``advantages`` is one group-relative scalar per example (every CoT sampled
    in a rollout shares its episode advantage). Since CoT CE is ``-logπ(cot)``, the policy-gradient
    surrogate is ``mean(A · ce_theta)`` (minimizing it raises ``logπ`` for positive-advantage samples),
    and the KL(πθ‖π_ref) penalty uses Schulman's k3 estimator from the per-example log-ratio against the
    frozen ``ref_params``. Grads are filtered to ``config.trainable_filter`` (same freeze story as the BC
    step); the action-expert params never enter the CoT CE, so they receive no gradient regardless.
    """
    model = nnx.merge(state.model_def, state.params)
    model.train()
    ref_model = nnx.merge(state.model_def, ref_params)
    ref_model.eval()

    observation, actions = batch
    train_rng = jax.random.fold_in(rng, state.step)
    # Reference log-prob is a constant (frozen params, forward-only): compute outside the grad path so
    # XLA keeps no backward activations for it and no gradient can flow into the reference.
    ce_ref = jax.lax.stop_gradient(_cot_ce_per_example(ref_model, train_rng, observation, actions))

    def loss_fn(model, rng, observation, actions):
        ce_theta = _cot_ce_per_example(model, rng, observation, actions)
        pg_loss = jnp.mean(advantages * ce_theta)
        log_ratio = ce_ref - ce_theta  # logπθ - logπ_ref
        kl = jnp.mean(jnp.exp(-log_ratio) + log_ratio - 1.0)  # k3 KL(πθ‖π_ref) >= 0
        loss = pg_loss + beta * kl
        return loss, {"pg_loss": pg_loss, "kl": kl, "ce_theta": jnp.mean(ce_theta), "ce_ref": jnp.mean(ce_ref)}

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
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
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    info = {"loss": loss, "grad_norm": optax.global_norm(grads), "mean_adv": jnp.mean(advantages)}
    info.update(aux)
    return new_state, info


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
    # Checkpoint norm-stats Normalize/Unnormalize must stay DISABLED for this checkpoint
    # (matches origin/master 8d10a05, where they are commented out): the model predicts
    # raw RLDS-scaled actions, so the fixed *7 physical scaling alone is the correct
    # decode (offline A/B 2026-06-11, /tmp/cmp + logs/norm_ab_f2d: with norm OFF the
    # speed-waypoint deltas track ego speed — 1.24 m at 5 m/s = maintain; with norm ON,
    # Unnormalize biases every prediction to the dataset-mean ~2.5 m → desired_speed
    # ~10 m/s regardless of state, so the car cannot express a stop and runs red lights).
    # Set STEERVLA_ENABLE_OPENPI_NORM=1 to re-enable for A/B testing.
    disable_norm = os.environ.get("STEERVLA_ENABLE_OPENPI_NORM", "0") != "1"
    if not disable_norm:
        print("[steervla] STEERVLA_ENABLE_OPENPI_NORM=1: applying Normalize/Unnormalize", flush=True)
    norm_stats = None if disable_norm else openpi_checkpoints.load_norm_stats(
        checkpoint_dir / "assets", data_config.asset_id
    )
    input_transform = openpi_transforms.compose(
        []
        if disable_norm
        else [
            openpi_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        ]
    )
    output_transform = openpi_transforms.compose(
        [
            *data_config.model_transforms.outputs,
            *(
                []
                if disable_norm
                else [openpi_transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm)]
            ),
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


_COT_PROMPT_WRAPPER_RE = re.compile(r"^Prompt:(.*?);State:[-\d\s]*;$", re.DOTALL)


def unformat_steervla_cot_prompt(text: str) -> str:
    """Strip a ``Prompt:<instruction>;State:<digits>;`` wrapper, returning the bare instruction.

    Defensive inverse of :func:`format_steervla_cot_prompt`. Anything that re-tokenizes a *stored*
    prompt must pass ``tokenize_prompt`` the bare instruction, because that method applies the
    wrapper itself — handing it an already-wrapped string yields
    ``Prompt:Prompt:<instr>;State:<s>;;State:<s>;``, a prefix format the model never sees at
    inference. Producers should store the raw instruction (``openpi_prompt_raw_text``); this exists
    so datasets already written with the wrapped form still train on the correct prefix.

    Returns ``text`` unchanged when it is not wrapped. Applied to a fixed point, so a prompt that
    was already double-wrapped on disk unwraps all the way back to the bare instruction.
    """
    stripped = str(text or "").strip()
    for _ in range(4):  # Bounded: real prompts are wrapped at most once or twice.
        match = _COT_PROMPT_WRAPPER_RE.match(stripped)
        if match is None:
            break
        stripped = match.group(1).strip()
    return stripped


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


# PaliGemma location sentinels (``<loc0000>``..``<loc1023>``). The CoT decode
# (``tokenizer._tokenizer.decode``) emits these verbatim, so the subtask/reasoning text captured
# at rollout — and therefore any HL sample that *reinforces* the model's original CoT — carries
# them: 1019-1022 wrap each segment, lower ids appear mid-text. They are not natural language and
# must not be trained on as CoT targets.
_LOC_SENTINEL_RE = re.compile(r"<loc\d+>")


def strip_cot_sentinels(text: Any) -> str:
    """Strip ``<locNNNN>`` sentinels from decoded CoT text and tidy the segment delimiter.

    ``'<loc1022>The vehicle accelerates normally.;<loc1021>'`` -> ``'The vehicle accelerates normally.'``
    Internal punctuation is preserved; only the trailing ``;`` left behind by the delimiter is
    removed.
    """
    s = _LOC_SENTINEL_RE.sub(" ", str(text or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(" ;").strip()


# Model image resolution. The SteerVLA pretraining dataset was preprocessed by **stretching** each
# camera frame to a square (a plain distorting resize to 224x224), NOT by aspect-preserving pad — the
# stretch happened upstream during dataset creation, so it is invisible in the runtime transforms
# (``ResizeImages`` -> ``resize_with_pad`` is then a near no-op on an already-square frame). The raw
# CARLA obs image is rectangular (144x256), so to feed the backbone the same distribution it was
# trained on we must stretch it to square here too — padding it would introduce black bars the model
# never saw. This resolution is applied both to the HL training frames and (via
# :meth:`build_observation_batch_numpy`) to the live rollout frame, so the two stay identical.
_HL_IMAGE_HW: tuple[int, int] = (224, 224)

# How many samples of an HL batch the wandb/disk visualization panel renders. Batches are typically
# 32; rendering all of them makes an unreadably large figure and costs real wall-clock per update.
_HL_BATCH_FIG_MAX_SAMPLES: int = 6


def resize_stretch_np(image: Any, height: int, width: int) -> np.ndarray:
    """Plain distorting resize (stretch) to ``(height, width)`` — no aspect preservation, no pad.

    Matches how the SteerVLA pretraining dataset was preprocessed (frames stretched to square), so a
    rectangular CARLA frame fills the whole square instead of being letterboxed. Uses bilinear to match
    the model's own ``jax.image.resize`` LINEAR method.
    """
    import cv2  # type: ignore

    img = np.asarray(image, dtype=np.uint8)
    cur_h, cur_w = img.shape[:2]
    if (cur_h, cur_w) == (height, width):
        return np.ascontiguousarray(img)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized, dtype=np.uint8)


def _resize_hl_image(image: Any, hw: tuple[int, int] = _HL_IMAGE_HW) -> np.ndarray:
    """Resize an ``(H, W, 3)`` uint8 image to the model resolution (no-op if already there).

    Stretches to square (see :func:`resize_stretch_np`) to match the pretraining preprocessing rather
    than pad — a padded CARLA frame would train/query the backbone on black bars it never saw.
    """
    return resize_stretch_np(image, hw[0], hw[1])


def cot_special_token_map(tokenizer: CoTPaligemmaTokenizer) -> dict[int, str]:
    """Map the reserved CoT / sentencepiece control ids to readable tags, for token decoding.

    The CoT delimiters live at the top of the PaliGemma vocab (``vocab_size - 1 - skip - slot``) and
    sentencepiece ``decode`` renders them as empty strings, so a plain decode silently swallows exactly
    the tokens that say where the reasoning and subtask segments begin and end. Tagging them keeps a
    decoded segment structurally legible (e.g. ``<sor>...<eor>``).
    """
    sp = tokenizer._tokenizer  # openpi exposes no public accessor for the sentencepiece processor.
    out: dict[int, str] = {}
    for name, fn in (
        ("<start_of_subtask>", tokenizer._start_of_subtask),
        ("<end_of_subtask>", tokenizer._end_of_subtask),
        ("<start_of_reasoning>", tokenizer._start_of_reasoning),
        ("<end_of_reasoning>", tokenizer._end_of_reasoning),
    ):
        try:
            out[int(fn())] = name
        except Exception:  # noqa: BLE001 - a missing delimiter must not break decoding.
            continue
    for name, fn in (("<bos>", sp.bos_id), ("<eos>", sp.eos_id), ("<pad>", sp.pad_id)):
        try:
            tid = int(fn())
        except Exception:  # noqa: BLE001
            continue
        if tid >= 0:
            out.setdefault(tid, name)
    return out


def decode_cot_token_ids(
    tokenizer: CoTPaligemmaTokenizer,
    ids: Any,
    mask: Any = None,
    *,
    keep_padding: bool = False,
) -> str:
    """Decode one tokenized segment (prompt / reasoning / subtask / fast) back to text.

    This is the inverse of ``CoTPaligemmaTokenizer.tokenize_*`` and is meant for *inspection of the
    exact tensors a training step consumes*, not for generation. Reserved delimiter ids are rendered
    as tags (see :func:`cot_special_token_map`) and the remaining runs are decoded with sentencepiece.

    ``mask`` (the segment's ``tokenized_*_mask``) selects the real tokens; the padding tail is dropped
    unless ``keep_padding``. Never raises — decode failures degrade to raw piece concatenation.
    """
    arr = np.asarray(jax.device_get(ids)).reshape(-1).astype(np.int64)
    if mask is not None and not keep_padding:
        m = np.asarray(jax.device_get(mask)).reshape(-1).astype(bool)
        n = min(arr.shape[0], m.shape[0])
        arr = arr[:n][m[:n]]
    specials = cot_special_token_map(tokenizer)
    sp = tokenizer._tokenizer
    pieces: list[str] = []
    buf: list[int] = []

    def _flush() -> None:
        if not buf:
            return
        try:
            pieces.append(sp.decode(list(buf)))
        except Exception:  # noqa: BLE001 - out-of-range ids (e.g. FAST codes) can trip decode.
            pieces.append(
                "".join(_piece_or_id(sp, t) for t in buf).replace("▁", " ")
            )
        buf.clear()

    for tid in arr.tolist():
        if tid in specials:
            _flush()
            pieces.append(specials[tid])
        else:
            buf.append(int(tid))
    _flush()
    return "".join(pieces)


def _piece_or_id(sp: Any, token_id: int) -> str:
    """``id_to_piece`` with an ``<id:N>`` fallback for ids outside the sentencepiece vocab."""
    try:
        return str(sp.id_to_piece(int(token_id)))
    except Exception:  # noqa: BLE001
        return f"<id:{int(token_id)}>"


def _fit_chunk_to_horizon(chunk: np.ndarray, horizon: int, dim: int) -> np.ndarray:
    """Coerce a ``(H, D)`` chunk to exactly ``(horizon, dim)`` (pad-last-row / truncate as needed)."""
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, dim)
    out = np.zeros((int(horizon), int(dim)), dtype=np.float32)
    h = min(int(horizon), int(arr.shape[0]))
    d = min(int(dim), int(arr.shape[1]))
    out[:h, :d] = arr[:h, :d]
    if arr.shape[0] < horizon and arr.shape[0] > 0:
        out[arr.shape[0]:, :d] = arr[-1, :d]  # repeat last row (matches RLDS chunk padding).
    return out


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
        cot_resample_on_overflow: bool = True,
        cot_overflow_max_resamples: int = 2,
        include_ego_history: bool = False,
        proprio_norm: bool = True,
        output_action_format: Optional[str] = "DELTA_XY_T_DELTA_XY_SPACE",
        action_horizon: int = 10,
        action_dim: int = 4,
        actions_per_model_query: int = 1,
        actions_per_cot: int = 1,
        env_steps_per_chunk_row: int = 5,
        reanchor_cached_chunk: bool = True,
        sample_actions_num_steps: int = 10,
        action_decode_batch_size: int = 2,
        training_gpu_rank: int = -1,
        hl_training_gpu_rank: int = -1,
        load_trainable_params: bool = False,
        hl_dataset_dir: str | Path | None = None,
        hl_update_every: int = 1,
        hl_update_batch_size: int = 2,
        hl_update_num_steps: int = 1,
        hl_lr: float | None = None,
        hl_freeze_regexes: list[str] | None = None,
        hl_replay_root: str | Path | None = None,
        hl_replay_pools: list[dict] | None = None,
        hl_online_weight: float = 1.0,
        hl_online_bad_fraction: float = -1.0,
        hl_online_precursor_fraction: float = -1.0,
        hl_min_online_samples: int = 1,
        hl_keep_last_rounds: int = 0,
        hl_log_batch_tokens: bool = True,
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
        t_context: float | None = None,
        sample_t_context: bool = False,
        t_context_min: float = 0.0,
        t_context_max: float = 1.0,
    ) -> None:
        self.actor_url = actor_url
        self._remote: Optional[RemoteActor] = None

        self.actor_config = actor_config
        self.checkpoint_path = checkpoint_path
        self.load_trainable_params = bool(load_trainable_params)
        # High-level (VLM-backbone) online update from a stored steervla_hl_dataset_format dataset
        # (written by coaches.cast_relabel). Consumed by :meth:`update_hl`, which the DSRL agent
        # calls from ``update_with_vla``. ``hl_dataset_dir`` is usually set by ``main_carla`` after
        # construction (it links the actor to the CAST-relabel session's dataset dir).
        self.hl_dataset_dir: Path | None = Path(hl_dataset_dir) if hl_dataset_dir is not None else None
        self.hl_update_every = max(1, int(hl_update_every))
        self.hl_update_batch_size = max(1, int(hl_update_batch_size))
        self.hl_update_num_steps = max(1, int(hl_update_num_steps))
        # Flat LR override for the HL optimizer. The pretraining actor_config ships a warmup->1e-4
        # cosine schedule (tuned for from-scratch BC); RL fine-tuning wants a small constant rate, so
        # when set this replaces the schedule with a flat ``hl_lr`` (no warmup ramp). None = keep config.
        self.hl_lr: float | None = float(hl_lr) if hl_lr else None
        # Optional regexes of param paths to FREEZE in the trainable state (e.g. the SigLIP vision
        # tower and the tied token embedder), so their grad + Adam buffers are dropped and the HL
        # update fits. Applied in :meth:`setup` before the train state is built. None = full fine-tune.
        self.hl_freeze_regexes: list[str] | None = (
            [str(r) for r in hl_freeze_regexes if str(r).strip()] if hl_freeze_regexes else None
        )
        # HL replay pools: a small amount of the original pretraining data (pre-extracted to npz by
        # ``impls/vlas/extract_hl_replay.py``) mixed into every HL update to stabilize the VLM
        # backbone. Weighted like steervla-pi's dataset mixture: the online cast_relabel pool gets
        # ``hl_online_weight`` and each replay pool its own weight; per HL update the batch is split
        # across pools by normalized weight. Each replay pool carries its own supervision flags
        # (``action_supervision`` = train the flow head; ``supervise_fast`` = train the FAST CE).
        self.hl_online_weight = float(hl_online_weight)
        # Within the online cast_relabel pool, bias the per-batch draw toward corrective chunks:
        # ``hl_online_bad_fraction`` of each online-share slot is filled from the BAD / BAD(precursor)
        # bucket (``label == "BAD"``, either ``credit_source``) and the remainder from the GOOD / null
        # bucket (``label`` GOOD or absent). Whichever bucket is short is topped up from the other so a
        # full batch is still produced. ``< 0`` disables the split (uniform draw over the online pool).
        # Only the online pool is bucketed; replay pools keep their weighted share untouched.
        self.hl_online_bad_fraction = float(hl_online_bad_fraction)
        # Within the corrective (BAD) bucket, further balance BAD(precursor) chunks
        # (``credit_source == "precursor"``) against direct BAD chunks: ``hl_online_precursor_fraction``
        # of the corrective slots are filled from the precursor sub-bucket and the remainder from the
        # direct sub-bucket, again topping up from whichever sub-bucket has spares. ``< 0`` disables the
        # sub-split (corrective slots drawn uniformly over BAD/precursor). Only meaningful when
        # ``hl_online_bad_fraction >= 0`` (the corrective bucket must exist to be sub-split).
        self.hl_online_precursor_fraction = float(hl_online_precursor_fraction)
        # How many online cast_relabel samples must exist before the FIRST HL update runs. The batch
        # does NOT wait for the online pool to fill its whole weighted share: as soon as this many
        # online samples are on disk, the update takes whatever the online pool has and fills the rest
        # of the batch from the offline replay pools (and, if those can't cover it either, by resampling
        # what's available). Set to 0 to allow replay-only updates before any online sample lands.
        self.hl_min_online_samples = max(0, int(hl_min_online_samples))
        # Pooled training: keep only samples from the last N policy versions (0 = keep everything).
        self.hl_keep_last_rounds = max(0, int(hl_keep_last_rounds))
        # Decode every HL batch's tokens back to text right before the gradient step consumes them
        # (:meth:`_dump_hl_batch_tokens`). Cheap (one host transfer + sentencepiece decode per update)
        # and the only place tokenization damage — truncated reasoning, empty segments, masked
        # padding — is observable, so it defaults on; set steervla.hl_log_batch_tokens=False to skip.
        self.hl_log_batch_tokens = bool(hl_log_batch_tokens)
        self._hl_replay_root: Path | None = Path(hl_replay_root) if hl_replay_root is not None else None
        self._hl_replay_pool_specs: list[dict[str, Any]] = self._resolve_replay_pool_specs(hl_replay_pools)
        self._hl_replay_logged_once = False
        self._hl_replay_missing_warned = False
        self._hl_train_step = None
        self._hl_grpo_step = None
        self._hl_ref_params = None
        self._hl_update_calls = 0
        # Diagnostics for :meth:`update_hl`, whose skip paths are otherwise silent (it returns
        # ``{}``), which makes an absent ``vla_hl/batch_text`` table impossible to explain.
        self._hl_pool_size = 0
        self._last_hl_skip_reason: str | None = None
        self._last_hl_note: str | None = None
        self._hl_logged_batch_once = False
        self._hl_logged_json_once = False
        # Rows accumulated for the wandb step currently being logged. ``updates_per_step`` sends
        # several update_with_vla calls per env step, all sharing one ``global_step``; wandb keeps
        # only the LAST value logged per (step, key), so same-step batches are merged into a single
        # table instead of overwriting each other.
        self._hl_table_step: int | None = None
        self._hl_table_rows: list[list[Any]] = []
        # Same accumulate-per-env-step bookkeeping for the decoded-token table.
        self._hl_token_table_step: int | None = None
        self._hl_token_table_rows: list[list[Any]] = []
        self._hl_logged_tokens_once = False
        # (segment, pool) pairs already warned about for tokenizer truncation — warn once each.
        self._hl_truncation_warned: set[tuple[str, str]] = set()
        self._weights_dirty = False
        self.raw_obs_holder = raw_obs_holder
        self.routing_command = routing_command
        self.cot_temperature = float(cot_temperature)
        self.cot_resample_on_overflow = bool(cot_resample_on_overflow)
        self.cot_overflow_max_resamples = max(0, int(cot_overflow_max_resamples))
        self._cot_overflow_count = 0
        self._cot_sample_count = 0
        self.include_ego_history = include_ego_history
        self.proprio_norm = proprio_norm
        self.output_action_format = output_action_format
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.actions_per_model_query = max(1, int(actions_per_model_query))
        self.actions_per_cot = max(1, int(actions_per_cot))
        # Env steps per chunk row: CARLA ticks at ``carla_fps`` (20) and one env step is one tick,
        # but a chunk row is a waypoint at the model's ``policy_fps`` (4). See _next_cached_action.
        self.env_steps_per_chunk_row = max(1, int(env_steps_per_chunk_row))
        # Transform a replayed chunk's route waypoints into the ego's *current* body frame before
        # serving it. Only meaningful when ``actions_per_model_query > 1``. See
        # :meth:`_reanchor_route_to_current_pose`; set False to reproduce the old open-loop replay.
        self.reanchor_cached_chunk = bool(reanchor_cached_chunk)
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
        # Context-Smoothed Pre-training (CSP): the context noise level fed to the action expert.
        # ``None`` (and sample_t_context=False) is the clean regime -- identical to a model trained
        # without CSP, and the only setting that works with a non-CSP checkpoint. ``sample_t_context``
        # draws a fresh t_context ~ U[t_context_min, t_context_max] per row on every model query;
        # ``t_context`` alone pins a fixed level (what a TMRL high-level policy would select).
        self.t_context = None if t_context is None else float(t_context)
        self.sample_t_context = bool(sample_t_context)
        self.t_context_min = float(t_context_min)
        self.t_context_max = float(t_context_max)
        if self.t_context is not None and not 0.0 <= self.t_context <= 1.0:
            raise ValueError(f"t_context must be in [0, 1], got {self.t_context}")
        if not 0.0 <= self.t_context_min <= self.t_context_max <= 1.0:
            raise ValueError(
                "t_context_min/t_context_max must satisfy 0 <= min <= max <= 1, got "
                f"({self.t_context_min}, {self.t_context_max})"
            )
        # Most recent per-row t_context actually used (numpy, shape (b,)); None in the clean regime.
        self._last_t_context: np.ndarray | None = None
        self.prompt_state_dim = steervla_prompt_state_dim(include_ego_history=include_ego_history)
        self._call_counter = 0
        self._cached_action_chunk: np.ndarray | None = None
        self._cached_action_step = 0
        # Ego pose (world x, y, yaw_rad) the cached chunk was sampled from, and the diagnostics
        # from the last re-anchor. ``last_action_was_cached`` lets the caller tell a replayed
        # action from a fresh model query (the noise argument is ignored on replayed steps).
        self._cached_action_pose: np.ndarray | None = None
        self._reanchor_disabled_reason: str | None = None
        self.last_reanchor: dict[str, float] = {}
        self.last_action_was_cached: bool = False
        self._cached_cot: dict[str, Any] | None = None
        self._cached_cot_actions_used = 0
        # Pooled prefix embedding cache (DSRL / residual state encoders). Both are read
        # before their first assignment in _forward_pi0 / ensure_policy_embedding, so they
        # must exist from construction.
        self._cached_policy_embed: np.ndarray | None = None
        self._cached_policy_embed_obs_id: int | None = None
        # Latest CoT output (reasoning/subtask/FAST tokens) the base policy sampled,
        # needed by the RLT state encoder to reproduce the prefix the policy acted on.
        self._last_cot_out: dict[str, Any] | None = None

        # Frozen prefix cached by _sample_actions_cached so pi_prefix / rl_token reuse it instead of
        # a second PaliGemma forward. out: f32[B, M, D], mask: bool[B, M] (B = N for EXPO else 1);
        # _prefix_cache_row picks the row, _last_prefix_n_fast = trailing FAST cols pi_prefix drops.
        self._prefix_reuse: bool = False
        self._last_prefix_out: jax.Array | None = None
        self._last_prefix_mask: jax.Array | None = None
        self._last_prefix_n_fast: int = 0
        self._prefix_cache_row: int = 0
        # Black-image sanity check: when set, the prefix encoders ignore the cache and run a fresh
        # prefix forward from the (blacked) obs they are handed, so the feature reflects that image.
        self._recompute_prefix_from_obs: bool = False

        self.train_cfg = None
        self.model = None
        self.tokenizer = None
        self.model_cfg = None
        self._jax_device = None
        # Optional dedicated device for the high-level (VLM-backbone) update. When
        # ``hl_training_gpu_rank`` selects a GPU different from ``training_gpu_rank``, the trainable
        # ``_train_state`` (params + optimizer + backward activations — the memory hog) lives here,
        # isolated from the inference model + DSRL RL updates on ``_jax_device``. This frees the whole
        # HL GPU for a larger ``hl_update_batch_size``. When they match (or the split is disabled),
        # ``_hl_jax_device`` is just ``_jax_device`` and behavior is unchanged. Set in :meth:`setup`.
        self._hl_jax_device = None
        # Requested HL device rank (``< 0`` or equal-to-inference => no split).
        self.hl_training_gpu_rank = int(hl_training_gpu_rank)
        self._local_ready = False
        self._qgf_config: dict | None = None  # set by setup_qgf()
        self._last_batch_subtask: tuple[np.ndarray, np.ndarray] | None = None
        self._last_batch_reasoning: tuple[np.ndarray, np.ndarray] | None = None
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
        # Remembered so :meth:`reload_params_from_checkpoint` can restore a later checkpoint onto the
        # same single device this actor was placed on.
        self._reload_gpu_rank = int(training_gpu_rank)
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

        # Optional extra freeze filter, ORed onto the config's own freeze_filter. The HL update
        # supervises only the CoT/VLM backbone (action loss masked out), so the big pretrained
        # subtrees the caller lists here (e.g. ``.*img.*`` SigLIP vision tower, ``.*embedder.*``
        # tied token embedder) rarely need to move. Freezing them drops their gradient + Adam
        # (mu/nu) buffers and casts them to bf16 inside ``init_openpi_train_state_single_gpu``,
        # which is what makes the update fit. Must be applied BEFORE the train state is built so
        # ``opt_state`` is allocated over the trainable subset only.
        if self.load_trainable_params and self.hl_freeze_regexes:
            extra = [nnx_utils.PathRegex(r) for r in self.hl_freeze_regexes]
            self.train_cfg = dataclasses.replace(
                self.train_cfg,
                freeze_filter=nnx.Any(self.train_cfg.freeze_filter, *extra),
            )
            print(f"[steervla] extra freeze regexes for trainable state: {self.hl_freeze_regexes}", flush=True)

        if self.load_trainable_params and self.hl_lr:
            # Flat constant LR (warmup_steps=0, peak == decay) instead of the pretraining warmup-cosine.
            self.train_cfg = dataclasses.replace(
                self.train_cfg,
                lr_schedule=_optimizer.CosineDecaySchedule(
                    warmup_steps=0, peak_lr=self.hl_lr, decay_steps=10**9, decay_lr=self.hl_lr
                ),
            )
            print(f"[steervla] HL optimizer LR overridden to flat {self.hl_lr:g}", flush=True)

        if self.load_trainable_params:
            # Full trainable state (optimizer + opt_state + freeze/trainable filters), pinned to one
            # GPU. Matches ``scripts/train.py :: init_train_state`` so the actor's weights can be
            # gradient-updated. ``self.model`` is reconstructed from the live (non-EMA) params.
            print("Loading SteerVLA as a trainable model (full train state).", flush=True)
            # Inference (+ DSRL RL) device. The HL train state can optionally live on a *different*
            # GPU (``hl_training_gpu_rank``) so its optimizer/backward memory doesn't compete with the
            # inference model, leaving that GPU free for a bigger HL batch.
            infer_device = _pick_single_gpu_device(training_gpu_rank)
            hl_rank = self.hl_training_gpu_rank
            hl_device = _pick_single_gpu_device(hl_rank) if hl_rank >= 0 else infer_device
            split_hl = hl_device != infer_device
            train_state, mesh, device = init_openpi_train_state_single_gpu(
                self.train_cfg,
                training_gpu_rank=(hl_rank if split_hl else training_gpu_rank),
            )
            self._train_state = train_state
            self._mesh = mesh
            self._jax_device = infer_device
            self._hl_jax_device = device  # where the train state / HL gradient step live.
            if split_hl:
                # Keep the inference model on ``infer_device``; the train state (and its params) stay
                # on ``hl_device``. Copy the params across once for the initial inference model; each
                # HL update later re-syncs via :meth:`_refresh_inference_weights`.
                print(
                    f"[steervla] HL update isolated on {hl_device} (rank {hl_rank}); "
                    f"inference + RL on {infer_device} (rank {training_gpu_rank}).",
                    flush=True,
                )
                infer_params = self._params_for_inference(train_state.params)
                self.model = nnx.merge(train_state.model_def, infer_params)
            else:
                self.model = nnx.merge(train_state.model_def, self._params_for_inference(train_state.params))

            # Confirm the freeze actually matched: a regex that matches nothing would silently keep
            # the whole model trainable and OOM again. Report trainable vs total param counts and the
            # rough VRAM saved (frozen params drop grad + Adam mu/nu and go bf16 ~= 14 B/param).
            try:
                total = sum(int(x.size) for x in jax.tree.leaves(train_state.params.filter(nnx.Param)))
                trainable = sum(
                    int(x.size)
                    for x in jax.tree.leaves(train_state.params.filter(self.train_cfg.trainable_filter))
                )
                frozen = max(0, total - trainable)
                print(
                    f"[steervla] trainable params {trainable / 1e6:.1f}M / {total / 1e6:.1f}M "
                    f"({100.0 * trainable / max(1, total):.1f}%); frozen {frozen / 1e6:.1f}M "
                    f"(~{14 * frozen / 1e9:.1f} GB saved vs full fine-tune).",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic only, never fatal.
                print(f"[steervla] trainable-param diagnostic unavailable ({exc}).", flush=True)
        else:
            # Inference-only: restore params once onto a single GPU and share them with the model.
            params, device = restore_openpi_params_on_single_gpu(
                params_dir=params_dir, training_gpu_rank=training_gpu_rank
            )
            self.model = self.train_cfg.model.load(params)
            self._jax_device = device
            self._hl_jax_device = device  # no HL update in inference-only mode; keep them aligned.

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
        self._build_sample_wrappers()

        if self.load_trainable_params and self._train_state is not None:
            # Jitted high-level train step; ``config`` is bound via partial so it stays static.
            # Donate the input TrainState (arg index 1 after ``config`` is bound by partial) so XLA
            # aliases the new state onto the old buffers instead of holding both (~40 GB each for a
            # full-FT 3.3B model). Without this the step's input+output args double-count the state
            # (seen as a ~107 GB I/O footprint / 71.5 GiB working set) and OOM. Matches the ``init``
            # jit above and OpenPI ``scripts/train.py``. Safe: ``update_hl`` reassigns
            # ``self._train_state`` to the output and never reuses the donated input. Only the state
            # is donated, not the batch — the batch is reused across ``hl_update_num_steps`` steps.
            self._hl_train_step = jax.jit(
                functools.partial(_openpi_hl_train_step, self.train_cfg),
                donate_argnums=(1,),
            )
            # GRPO (HL policy) step: same donate/static convention as above. Snapshot the loaded params
            # as the frozen KL reference. Must be a real buffer copy (jnp.copy), not an identity map:
            # the step donates the train state, so a shared buffer would be freed after the first update
            # and corrupt the reference. Both are used only by ``update_hl_grpo``.
            self._hl_grpo_step = jax.jit(
                functools.partial(_openpi_hl_grpo_step, self.train_cfg),
                donate_argnums=(1,),
            )
            self._hl_ref_params = jax.tree.map(
                lambda x: jnp.copy(x) if isinstance(x, jax.Array) else x, self._train_state.params
            )
            if self._train_rng is None:
                self._train_rng = jax.random.key(0)

        self._local_ready = True

    def _build_sample_wrappers(self) -> None:
        """(Re)build the jitted inference kernels bound to the current ``self.model``.

        ``nnx_utils.module_jit`` freezes the module state at creation, so these must be rebuilt
        after any in-place weight update (see :meth:`_refresh_inference_weights`).
        """
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
        # Jit sample_actions_with_prefix when the installed openpi has it, so base-chunk sampling
        # caches the frozen prefix for the pi_prefix / rl_token encoders (surya/rl-token). Built
        # here rather than in ``init`` so it is rebuilt alongside the other kernels after an
        # in-place weight update — see :meth:`_refresh_inference_weights`.
        self._prefix_reuse = hasattr(self.model, "sample_actions_with_prefix")
        if self._prefix_reuse:
            self._sample_actions_with_prefix = nnx_utils.module_jit(
                self.model.sample_actions_with_prefix,
                static_argnames=(
                    "num_steps",
                "image_keys",
            ),
        )
        
        # Jitted pooled-prefix embedding for the per-step policy_embed path; the
        # eager _build_frozen_prefix_cache costs ~5 s/step (see _frozen_prefix_embed_forward).
        self._prefix_embed_fn = nnx_utils.module_jit(
            _types.MethodType(_frozen_prefix_embed_forward, self.model)
        )

        # QGF: jitted prefix-cache builder and single denoising step (used when setup_qgf() is called).
        self._prefix_cache_fn = nnx_utils.module_jit(
            _types.MethodType(_compute_prefix_cache_for_qgf, self.model)
        )
        self._denoise_step_fn = nnx_utils.module_jit(self.model._denoise_flow_step)

    # ------------------------------------------------------------------
    # Context-Smoothed Pre-training (CSP): context noise level ``t_context``
    # ------------------------------------------------------------------
    def set_t_context(
        self,
        t_context: float | None,
        *,
        sample: bool | None = None,
        t_context_min: float | None = None,
        t_context_max: float | None = None,
    ) -> None:
        """Set the context noise level used by subsequent action samples.

        Call this per action chunk to drive the policy from a high-level (TMRL) controller:
        ``set_t_context(0.0)`` for precise imitation, values toward 1.0 for the broad marginal.
        ``sample=True`` switches to a fresh uniform draw on every model query instead.
        """
        self.t_context = None if t_context is None else float(t_context)
        if sample is not None:
            self.sample_t_context = bool(sample)
        if t_context_min is not None:
            self.t_context_min = float(t_context_min)
        if t_context_max is not None:
            self.t_context_max = float(t_context_max)

    def _context_smoothing_supported(self) -> bool:
        """True iff the checkpoint was trained with CSP *and* this openpi exposes ``t_context``."""
        if self.model is None or getattr(self.model, "context_smoothing", None) is None:
            return False
        try:
            return "t_context" in inspect.signature(type(self.model).sample_actions).parameters
        except (TypeError, ValueError):
            return False

    def _resolve_t_context(self, rng: jax.Array, batch_size: int) -> jax.Array | None:
        """Per-row context noise level in [0, 1], or ``None`` for the clean (no-CSP) regime.

        Always returns an array of shape ``(b,)`` rather than a Python float so that a new value
        does not force ``module_jit`` to retrace ``sample_actions``.
        """
        if self.t_context is None and not self.sample_t_context:
            return None
        if not self._context_smoothing_supported():
            raise ValueError(
                "t_context / sample_t_context require a checkpoint trained with "
                "context_smoothing enabled (Pi0CoTConfig.context_smoothing) and an openpi build "
                "whose Pi0CoT.sample_actions accepts t_context. Loaded config "
                f"{self.actor_config!r} does not."
            )
        if self.sample_t_context:
            t = jax.random.uniform(
                rng,
                (batch_size,),
                dtype=jnp.float32,
                minval=self.t_context_min,
                maxval=self.t_context_max,
            )
        else:
            t = jnp.full((batch_size,), float(self.t_context), dtype=jnp.float32)
        t = jax.device_put(t, self._jax_device)
        self._last_t_context = np.asarray(jax.device_get(t), dtype=np.float32)
        return t

    @property
    def last_t_context(self) -> np.ndarray | None:
        """Per-row ``t_context`` used by the most recent action sample (``None`` if clean)."""
        return self._last_t_context

    def _params_for_inference(self, params):
        """Return train-state params as buffers the inference model can own outright.

        ``_hl_train_step`` is jitted with ``donate_argnums=(1,)``, so every HL update deletes the
        buffers of the ``TrainState`` handed to it. Two cases:

        * **Split HL GPU** (``hl_training_gpu_rank`` != ``training_gpu_rank``): the freshly-updated
          params live on ``_hl_jax_device``; ``device_put`` to ``_jax_device`` is a real cross-device
          copy, so inference already owns independent buffers.
        * **Shared GPU** (``hl_training_gpu_rank`` unset, -1, or equal): ``device_put`` onto the
          device the array is already on is a no-op that returns the *same* buffer. The inference
          model would then alias the donated train state and the next forward dies with
          ``INVALID_ARGUMENT: Buffer has been deleted or donated``. Copy explicitly instead.
        """
        if (
            self._hl_jax_device is not None
            and self._jax_device is not None
            and self._hl_jax_device != self._jax_device
        ):
            return jax.device_put(params, self._jax_device)
        return jax.tree.map(lambda x: x.copy() if hasattr(x, "copy") else x, params)

    def _refresh_inference_weights(self) -> None:
        """Rebuild the model + inference kernels from the train state after a HL update.

        Called lazily at the start of each inference entrypoint so multiple HL updates between
        rollouts trigger at most one rebuild. ``module_jit`` snapshots weights, hence the rebuild.
        """
        if not self._weights_dirty or self._train_state is None:
            return
        params = self._params_for_inference(self._train_state.params)
        self.model = nnx.merge(self._train_state.model_def, params)
        self._build_sample_wrappers()
        self._weights_dirty = False

    def reload_params_from_checkpoint(self, checkpoint_dir: str | Path) -> bool:
        """Swap in a params-only checkpoint **in place**, without rebuilding the actor.

        This is the worker half of a pooled CAST run (:mod:`cast_pool`): the trainer publishes a new
        params-only export and every rollout worker hot-reloads it mid-episode, so all routes keep
        driving the newest policy without a CARLA restart. ``checkpoint_dir`` is the version dir --
        the same thing ``steervla.checkpoint`` accepts -- and must contain ``params/``.

        Only valid for a **local, inference-only** actor. A trainable actor owns an optimizer state
        that a params-only restore would silently desynchronize from its params, so reloading one is
        refused; that process is the trainer, and it updates its own weights through ``update_hl``.

        Rebuilding the sample wrappers is required, not optional: ``module_jit`` snapshots the
        weights, so without :meth:`_build_sample_wrappers` the new params would never reach
        inference. The action/CoT caches are dropped for the same reason -- a chunk sampled by the
        previous policy must not go on being executed by the new one.

        Returns True on success. On failure the previous weights are left untouched (the model is
        only rebound after the restore returns), so a torn read costs one round's freshness rather
        than the run.
        """
        if self._remote is not None:
            print("[steervla.reload_params] skipped: actor is remote.", flush=True)
            return False
        if self.load_trainable_params:
            print(
                "[steervla.reload_params] skipped: actor is trainable (its opt_state would desync); "
                "only inference-only rollout workers hot-reload.",
                flush=True,
            )
            return False
        params_dir = Path(checkpoint_dir) / "params"
        if not params_dir.is_dir():
            print(f"[steervla.reload_params] no params/ under {checkpoint_dir}; keeping current weights.", flush=True)
            return False
        t0 = time.time()
        try:
            params, device = restore_openpi_params_on_single_gpu(
                params_dir=params_dir, training_gpu_rank=self._reload_gpu_rank
            )
            model = self.train_cfg.model.load(params)
        except Exception as exc:  # noqa: BLE001 - a failed reload must never kill the rollout.
            print(
                f"[steervla.reload_params] FAILED to load {params_dir} ({exc}); keeping current weights.",
                flush=True,
            )
            return False
        self.model = model
        self._jax_device = device
        self._hl_jax_device = device
        self._build_sample_wrappers()
        self.reset_action_cache()
        print(
            f"[steervla.reload_params] loaded {checkpoint_dir} in {time.time() - t0:.1f}s "
            "(sample wrappers rebuilt, action/CoT cache cleared).",
            flush=True,
        )
        return True

    # ------------------------------------------------------------------
    # High-level (VLM-backbone) online update from the cast_relabel HL dataset
    # ------------------------------------------------------------------

    def _resolve_replay_pool_specs(self, hl_replay_pools) -> list[dict[str, Any]]:
        """Normalize the configured replay-pool list into ``{dir, name, weight, kind}`` specs.

        Each entry is a ``{name, weight}`` dict (``name`` is a dir under ``hl_replay_root`` or an
        absolute path). Supervision flags (``action_supervision`` / ``supervise_fast`` / state format)
        are NOT set here — they live in each pool's ``hl_samples.json`` (written by
        ``extract_hl_replay.py``) and are read at scan time.
        """
        specs: list[dict[str, Any]] = []
        for p in (hl_replay_pools or []):
            try:
                p = dict(p)
            except Exception:
                continue
            name = str(p.get("name") or p.get("dir") or "").strip()
            weight = float(p.get("weight", 0.0) or 0.0)
            if not name or weight <= 0.0:
                continue
            d = Path(name)
            if not d.is_absolute() and self._hl_replay_root is not None:
                d = self._hl_replay_root / name
            specs.append({"dir": d, "name": name, "weight": weight, "kind": "replay"})
        return specs

    def _hl_pools(self) -> list[dict[str, Any]]:
        """Active HL sources: the online cast_relabel pool plus any configured replay pools."""
        pools: list[dict[str, Any]] = []
        if self.hl_dataset_dir is not None and float(self.hl_online_weight) > 0.0:
            pools.append(
                {"dir": Path(self.hl_dataset_dir), "name": "online", "weight": float(self.hl_online_weight), "kind": "online"}
            )
        pools.extend(self._hl_replay_pool_specs)
        return [p for p in pools if float(p.get("weight", 0.0)) > 0.0]

    def _scan_pool(self, pool: dict[str, Any]) -> list[dict[str, Any]]:
        """List all samples in one pool, tagging each with the pool's supervision flags.

        The online cast_relabel dir keeps a ``hl_samples.json`` per window subdir; an extracted replay
        pool keeps a single ``hl_samples.json`` at its root — both globs are checked. A **pooled** run
        (``impls/cast_pool.py``) adds a third depth, ``<pool_root>/<worker>/<window>/``, so several
        rollout workers can write one shared corpus that the trainer reads whole. ``action_supervision``
        / ``supervise_fast`` / ``state_format`` come from the manifest (defaults match the online
        ``steervla_hl_dataset_format``: no flow, per-sample FAST, raw CARLA state).

        Window dirs still being written carry the ``cast_pool.TMP_PREFIX`` and are skipped, so a
        half-written manifest is never scanned.
        """
        entries: list[dict[str, Any]] = []
        root = pool.get("dir")
        if root is None or not Path(root).is_dir():
            return entries
        root = Path(root)
        manifests = (
            sorted(root.glob("hl_samples.json"))
            + sorted(root.glob("*/hl_samples.json"))
            + sorted(root.glob("*/*/hl_samples.json"))
        )
        manifests = [
            m for m in manifests if not any(p.name.startswith(".tmp-") for p in m.relative_to(root).parents)
        ]
        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception:
                continue
            work_dir = manifest_path.parent
            pool_action_sup = bool(manifest.get("action_supervision", False))
            pool_fast = manifest.get("supervise_fast", None)  # None -> resolve per-sample.
            state_format = str(manifest.get("state_format", "carla_raw"))
            for s in manifest.get("samples", []):
                entries.append(
                    {
                        "dir": work_dir,
                        "file": s.get("sample_file"),
                        "prompt": s.get("prompt", ""),
                        "subtask": s.get("subtask", ""),
                        "reasoning": s.get("reasoning", ""),
                        # Only reinforced (original) online chunks have an action matching their subtask.
                        "action_matches_subtask": bool(s.get("action_matches_subtask", False)),
                        # cast_relabel per-chunk verdict: "GOOD"/"BAD"/None and (for BAD) whether the
                        # blame is "direct" or "precursor". Used to bucket the online pool 80/20
                        # (BAD or BAD-precursor vs GOOD or null) in :meth:`_load_hl_batch`. Replay
                        # pools have no such labels; they fall into the "not BAD" bucket harmlessly
                        # (the label split is only applied to the online pool).
                        "label": s.get("label"),
                        "credit_source": s.get("credit_source", ""),
                        "action_supervision": pool_action_sup,
                        "supervise_fast": pool_fast,
                        "state_format": state_format,
                        "pool": pool.get("name", "online"),
                        # Which policy version produced this sample (pooled runs; see
                        # impls/cast_pool.py). -1 for pools that predate the field or for the
                        # offline replay pools, which are version-less and never age out.
                        # NOTE: version 0 (the pre-round base policy) is a real version — read it
                        # with an explicit None check, not ``or -1``, which would make 0 unversioned
                        # and thus permanently exempt from the staleness filter.
                        "policy_version": (
                            -1 if s.get("policy_version") is None else int(s["policy_version"])
                        ),
                    }
                )
        return entries

    def _filter_stale_entries(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop online samples produced more than ``hl_keep_last_rounds`` policy versions ago.

        Pooled training swaps the policy every round, so an old sample supervises a backbone that no
        longer produces the behavior it was correcting. Keeping a sliding window of versions bounds
        both that staleness and how many times any one sample can be re-drawn. Version ``-1``
        (unversioned pools, including every offline replay pool) is always kept -- those are
        pretraining data, not on-policy corrections.
        """
        keep = int(self.hl_keep_last_rounds)
        if keep <= 0:
            return entries
        versions = [int(e.get("policy_version", -1)) for e in entries]
        newest = max((v for v in versions if v >= 0), default=-1)
        if newest < 0:
            return entries  # nothing versioned; nothing to age out.
        floor = newest - keep + 1
        return [e for e in entries if int(e.get("policy_version", -1)) < 0 or int(e["policy_version"]) >= floor]

    @staticmethod
    def _largest_remainder(total: int, fracs: list[float]) -> list[int]:
        """Split ``total`` into integer per-source counts closest to ``fracs`` (sum == total)."""
        raw = [f * total for f in fracs]
        floors = [int(np.floor(x)) for x in raw]
        rem = int(total - sum(floors))
        if rem > 0:
            order = np.argsort([-(raw[i] - floors[i]) for i in range(len(raw))])
            for k in range(rem):
                floors[int(order[k % len(order)])] += 1
        return floors

    @staticmethod
    def _is_bad_entry(e: dict[str, Any]) -> bool:
        """True for a cast_relabel BAD / BAD(precursor) chunk (either ``credit_source``)."""
        return str(e.get("label") or "").strip().upper() == "BAD"

    @staticmethod
    def _is_precursor_entry(e: dict[str, Any]) -> bool:
        """True for a cast_relabel BAD(precursor) chunk (``credit_source == "precursor"``)."""
        return str(e.get("credit_source") or "").strip().lower() == "precursor"

    @staticmethod
    def _balance_buckets(
        primary: list[dict[str, Any]],
        secondary: list[dict[str, Any]],
        cnt: int,
        primary_fraction: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Pick up to ``cnt`` entries — ~``primary_fraction`` from ``primary``, the rest from
        ``secondary`` — topping up from whichever bucket has spares when the other underfills.
        Both inputs are assumed pre-shuffled. Returns ``(selected, leftovers)``.
        """
        n_primary = min(len(primary), int(round(cnt * float(primary_fraction))))
        n_secondary = cnt - n_primary
        if n_secondary > len(secondary):  # secondary bucket short -> pull more primary to reach cnt.
            n_secondary = len(secondary)
            n_primary = min(len(primary), cnt - n_secondary)
        selected = primary[:n_primary] + secondary[:n_secondary]
        leftovers = primary[n_primary:] + secondary[n_secondary:]
        return selected, leftovers

    def _order_online_entries(
        self,
        entries: list[dict[str, Any]],
        cnt: int,
        bad_fraction: float,
        precursor_fraction: float = -1.0,
    ) -> list[dict[str, Any]]:
        """Order online-pool entries so the first ``cnt`` are ~``bad_fraction`` BAD, the rest GOOD/null.

        BAD and BAD(precursor) chunks form the corrective bucket; GOOD and unlabeled/null chunks the
        reinforce bucket. Selects ``round(cnt * bad_fraction)`` corrective + the remainder reinforce,
        topping up from whichever bucket has spares when the other underfills so ``cnt`` is met when
        possible. When ``precursor_fraction >= 0``, the corrective slots are themselves balanced
        between BAD(precursor) and direct BAD chunks (``round(n_bad * precursor_fraction)`` precursor,
        the rest direct), with the same underfill top-up. Remaining entries are appended (shuffled) as
        leftover top-up candidates for the caller. The returned list is a full permutation of
        ``entries``.
        """
        bad = [e for e in entries if self._is_bad_entry(e)]
        good = [e for e in entries if not self._is_bad_entry(e)]
        np.random.shuffle(bad)
        np.random.shuffle(good)
        n_bad = min(len(bad), int(round(cnt * float(bad_fraction))))
        n_good = cnt - n_bad
        if n_good > len(good):  # reinforce bucket short -> pull more corrective to reach cnt.
            n_good = len(good)
            n_bad = min(len(bad), cnt - n_good)
        if precursor_fraction >= 0.0:
            # Sub-split the corrective bucket by BAD(precursor) vs direct BAD. ``bad`` is already
            # shuffled, so the comprehensions inherit that shuffle.
            precursor = [e for e in bad if self._is_precursor_entry(e)]
            direct = [e for e in bad if not self._is_precursor_entry(e)]
            bad_selected, bad_leftover = self._balance_buckets(precursor, direct, n_bad, precursor_fraction)
        else:
            bad_selected, bad_leftover = bad[:n_bad], bad[n_bad:]
        selected = bad_selected + good[:n_good]
        leftovers = bad_leftover + good[n_good:]
        np.random.shuffle(selected)
        np.random.shuffle(leftovers)
        return selected + leftovers

    def _read_hl_record(self, e: dict[str, Any]) -> dict[str, Any] | None:
        """Load one sample's arrays + resolve its per-sample supervision into a record dict."""
        npz_path = Path(e["dir"]) / str(e["file"])
        if not npz_path.is_file():
            return None
        try:
            with np.load(npz_path) as z:
                files = set(getattr(z, "files", []))
                image = np.asarray(z["image"], dtype=np.uint8)
                state = np.asarray(z["state"], dtype=np.float32).reshape(-1)
                action = np.asarray(z["actions"], dtype=np.float32) if "actions" in files else None
        except Exception:
            # Half-written .npz (the coach writes these concurrently with training); skip it.
            return None
        # Strip PaliGemma <loc> sentinels from CoT targets (online reinforced samples carry them from
        # the rollout decode; training the backbone on them teaches it to emit sentinels as prose).
        subtask = strip_cot_sentinels(e.get("subtask", ""))
        if not subtask:
            return None
        action_supervision = bool(e.get("action_supervision", False))
        pool_fast = e.get("supervise_fast", None)
        pool_name = str(e.get("pool", "online"))
        # FAST supervision: explicit pool flag wins. The online cast_relabel pool NEVER supervises FAST
        # — its sampled chunks (reinforced or relabeled) must not train the shared LLM's FAST head, or
        # the action-token predictions drift. FAST is supervised only from the pretraining replay pools
        # whose manifests set ``supervise_fast`` explicitly.
        if pool_fast is not None:
            supervise_fast = bool(pool_fast)
        elif pool_name == "online":
            supervise_fast = False
        else:
            supervise_fast = bool(e.get("action_matches_subtask", False))
        # Drop the action if neither loss will use it, and never supervise a zero/absent chunk.
        if not (supervise_fast or action_supervision):
            action = None
        if action is not None and not np.any(action):
            action = None
            supervise_fast = False
            action_supervision = False
        return {
            "image": image,
            "state": state,
            "prompt": str(e.get("prompt", "")),
            "subtask": subtask,
            "reasoning": strip_cot_sentinels(e.get("reasoning", "")),
            "action_chunk": action,
            "action_supervision": action_supervision,
            "supervise_fast": supervise_fast,
            "state_format": str(e.get("state_format", "carla_raw")),
            "pool": e.get("pool", "online"),
            "label": e.get("label"),
            "credit_source": e.get("credit_source", ""),
        }

    def _load_hl_batch(self, batch_size: int):
        """Draw a weighted mixed batch of HL records across the online + replay pools.

        Counts are split across active pools by their (normalized) weights, mirroring steervla-pi's
        dataset mixture. The online cast_relabel pool does NOT have to fill its whole weighted share:
        as soon as it holds ``hl_min_online_samples`` samples the update runs, taking every online
        sample available (up to its share) and topping the batch up from the offline replay pools.
        Returns ``None`` only when the online pool is still below ``hl_min_online_samples`` (or no
        pool has any readable sample at all).
        """
        bs = int(batch_size)
        pools = self._hl_pools()
        if not pools:
            self._hl_pool_size = 0
            return None
        scanned = [{"pool": p, "entries": self._filter_stale_entries(self._scan_pool(p))} for p in pools]
        # Warn once if replay pools were configured but none resolved on disk (extraction not run).
        if self._hl_replay_pool_specs and not getattr(self, "_hl_replay_missing_warned", False):
            replay_have = any(
                s["entries"] for s in scanned if s["pool"].get("kind") == "replay"
            )
            if not replay_have:
                self._hl_replay_missing_warned = True
                print(
                    "[steervla.update_hl] WARNING: hl_replay_pools configured but no replay samples "
                    f"found under {self._hl_replay_root} — training online-only. Run "
                    "impls/vlas/extract_hl_replay.py to populate the pools.",
                    flush=True,
                )
        online = next((s for s in scanned if s["pool"].get("kind") == "online"), None)
        # ``_hl_pool_size`` reports the *online* pool fill (what the skip messages talk about).
        self._hl_pool_size = (
            len(online["entries"]) if online is not None else sum(len(s["entries"]) for s in scanned)
        )
        # Gate only on *starting* the online stream, not on it filling its share: below
        # ``hl_min_online_samples`` we'd be training on replay alone, which is not the point of the
        # online HL update. At or above it, whatever is on disk goes in and replay covers the rest.
        if online is not None and len(online["entries"]) < self.hl_min_online_samples:
            return None

        active = [s for s in scanned if s["entries"]]
        if not active:
            return None
        wsum = sum(float(s["pool"]["weight"]) for s in active)
        counts = self._largest_remainder(bs, [float(s["pool"]["weight"]) / wsum for s in active])

        records: list[dict[str, Any]] = []
        leftovers: list[dict[str, Any]] = []
        for s, cnt in zip(active, counts):
            # The online pool is bucketed 80/20 (BAD/precursor vs GOOD/null) when enabled; replay
            # pools (and disabled split) draw uniformly at random.
            if s["pool"].get("kind") == "online" and self.hl_online_bad_fraction >= 0.0:
                ordered = self._order_online_entries(
                    s["entries"], cnt, self.hl_online_bad_fraction, self.hl_online_precursor_fraction
                )
            else:
                ordered = [s["entries"][int(j)] for j in np.random.permutation(len(s["entries"]))]
            taken = 0
            for e in ordered:
                if taken >= cnt:
                    leftovers.append(e)  # spare candidate for top-up if another pool underfills.
                    continue
                rec = self._read_hl_record(e)
                if rec is None:
                    continue
                records.append(rec)
                taken += 1
        if len(records) < bs and leftovers:
            np.random.shuffle(leftovers)
            for e in leftovers:
                if len(records) >= bs:
                    break
                rec = self._read_hl_record(e)
                if rec is not None:
                    records.append(rec)
        if not records:
            return self._hl_short_batch(0, bs)
        records = records[:bs]
        # Records are assembled pool-by-pool (online block, then each replay pool), so without this
        # shuffle the batch — and the logged HL panel, which shows the leading rows — would be all
        # online. Interleave the pools so the batch (and its inspection panel) is a random mix.
        np.random.shuffle(records)
        if len(records) < bs:
            records = self._pad_hl_batch(records, bs)
        else:
            self._last_hl_note = None  # Cleared: a later partial batch should re-announce itself.
        if not self._hl_replay_logged_once and (
            self._hl_replay_pool_specs or self.hl_online_bad_fraction >= 0.0
        ):
            self._hl_replay_logged_once = True
            comp = {}
            for r in records:
                comp[r["pool"]] = comp.get(r["pool"], 0) + 1
            msg = f"[steervla.update_hl] HL batch mix (pool -> count): {comp}"
            if self.hl_online_bad_fraction >= 0.0:
                n_bad = sum(1 for r in records if r.get("pool") == "online" and self._is_bad_entry(r))
                n_online = sum(1 for r in records if r.get("pool") == "online")
                msg += f"; online label split (BAD/precursor -> {n_bad}, GOOD/null -> {n_online - n_bad})"
                if self.hl_online_precursor_fraction >= 0.0:
                    n_precursor = sum(
                        1
                        for r in records
                        if r.get("pool") == "online" and self._is_bad_entry(r) and self._is_precursor_entry(r)
                    )
                    msg += f"; corrective split (precursor -> {n_precursor}, direct BAD -> {n_bad - n_precursor})"
            print(msg, flush=True)
        return records

    def _hl_short_batch(self, got: int, bs: int):
        """Report a batch with nothing readable in it and skip the update."""
        self._hl_skip(
            f"HL pool has {self._hl_pool_size} samples but only {got}/{bs} were readable "
            f"(missing or half-written .npz under {self.hl_dataset_dir}); skipping this update"
        )
        return None

    def _pad_hl_batch(self, records: list[dict[str, Any]], bs: int) -> list[dict[str, Any]]:
        """Repeat-sample ``records`` up to ``bs`` rows so the batch shape stays fixed.

        Only reached when the online pool has started but neither it nor the replay pools can cover a
        full batch yet (typically the first few updates of a run, or replay pools not extracted). The
        alternative — training on a genuinely smaller batch — retraces/recompiles ``_hl_train_step``
        for every new size and re-allocates its backward buffers, so instead the available rows are
        cycled (in shuffled passes) to fill ``bs``. Gradient-wise this is the mean over the distinct
        rows; it just costs the extra compute of the duplicated rows.
        """
        n = len(records)
        if n >= bs or n == 0:
            return records[:bs]
        padded = list(records)
        while len(padded) < bs:
            extra = list(records)
            np.random.shuffle(extra)
            padded.extend(extra[: bs - len(padded)])
        self._hl_note(
            f"partial HL batch: {n}/{bs} distinct samples available "
            f"(online pool {self._hl_pool_size}); repeating them to fill the batch"
        )
        return padded

    def _hl_note(self, msg: str) -> None:
        """Print an HL-update note once per distinct message (same de-dup idea as :meth:`_hl_skip`)."""
        if msg != getattr(self, "_last_hl_note", None):
            print(f"[steervla.update_hl] {msg}", flush=True)
            self._last_hl_note = msg

    def _build_hl_observation_batch(self, records: list[dict[str, Any]]):
        """Build ``(Observation, actions)`` for :meth:`update_hl` from mixed HL records.

        Each record carries per-sample supervision resolved by :meth:`_read_hl_record`:

        - ``action_supervision`` — when True the (real, consistent) action chunk is the flow-matching
          target and ``action_loss_mask`` is True for that row (regular pretraining pools, e.g. the
          SimLingo replay pool). When False the flow loss is masked (online cast_relabel samples and HL
          pools like ``simplified_reasoning``).
        - ``supervise_fast`` — when True the chunk is FAST-tokenized into ``tokenized_fast`` so the FAST
          CE (``cot_fast_ce``) is supervised, mirroring OpenPI's ``TokenizeCoTPrompt``. When False the
          FAST tokens are zero + masked (online relabeled chunks whose action no longer matches the
          corrected subtask; the ``simplified_reasoning`` HL pool, which trains reasoning + subtask only).
        - ``state_format`` — ``carla_raw`` runs the raw CARLA ego vector through
          :func:`carla_state_vec_to_steervla_state`; ``proprio`` (extracted replay pools) uses the stored
          normalized proprio directly.

        Images are resized to the model resolution so online (CARLA) and replay (512px) frames stack.
        """
        assert self.model is not None and self.model_cfg is not None and self.tokenizer is not None
        model_action_dim = int(self.model_cfg.action_dim)
        model_ah = int(self.model.action_horizon)
        model_ad = int(self.model.action_dim)
        use_fast = bool(getattr(self.tokenizer, "use_fast_tokens", False))
        max_fast_len = int(self.tokenizer.max_fast_len)

        img_batch, state_batch = [], []
        prompt_ids, prompt_masks = [], []
        rea_ids, rea_masks, sub_ids, sub_masks = [], [], [], []
        fast_ids, fast_masks = [], []
        act_rows, loss_rows = [], []
        for rec in records:
            state_vec = np.asarray(rec["state"], dtype=np.float32).reshape(-1)
            if str(rec.get("state_format", "carla_raw")) == "proprio":
                # Already the normalized proprio the model consumes (extracted replay pools).
                state_pad = state_vec
            else:
                state_pad = carla_state_vec_to_steervla_state(
                    state_vec,
                    include_ego_history=self.include_ego_history,
                    proprio_norm=self.proprio_norm,
                )
            # ``tokenize_prompt`` applies the ``Prompt:...;State:...;`` wrapper itself, so unwrap
            # any stored prompt that already carries it (datasets written before the producer was
            # fixed) rather than double-wrapping into a prefix format the model never sees.
            prompt_text = unformat_steervla_cot_prompt(rec.get("prompt", "")) or self.routing_command
            tok_ids, tok_mask = self.tokenizer.tokenize_prompt(
                prompt_text, state_pad, state_dim=self.prompt_state_dim
            )
            state_norm = self._normalize_state_batch(state_pad)[0]
            state_for_model = pad_to_dim(state_norm, model_action_dim)

            # GRPO records carry the exact sampled token ids. Text records
            # (cast_relabel / replay pools) tokenize their reasoning/subtask strings as before.
            if "reasoning_ids" in rec:
                rea_tok, rea_mask = rec["reasoning_ids"], rec["reasoning_mask"]
            else:
                rea_tok, rea_mask = self.tokenizer.tokenize_reasoning(
                    str(rec.get("reasoning", "")).strip() or "Follow the route."
                )
            if "subtask_ids" in rec:
                sub_tok, sub_mask = rec["subtask_ids"], rec["subtask_mask"]
            else:
                sub_tok, sub_mask = self.tokenizer.tokenize_subtask(
                    str(rec.get("subtask", "")).strip() or "Follow the route."
                )

            img_batch.append(_resize_hl_image(rec["image"]))
            state_batch.append(np.asarray(state_for_model, dtype=np.float32))
            prompt_ids.append(np.asarray(tok_ids, dtype=np.int32))
            prompt_masks.append(np.asarray(tok_mask, dtype=bool))
            rea_ids.append(np.asarray(rea_tok, dtype=np.int32))
            rea_masks.append(np.asarray(rea_mask, dtype=bool))
            sub_ids.append(np.asarray(sub_tok, dtype=np.int32))
            sub_masks.append(np.asarray(sub_mask, dtype=bool))

            chunk = rec.get("action_chunk")
            supervise_fast = bool(rec.get("supervise_fast")) and chunk is not None
            action_supervised = bool(rec.get("action_supervision")) and chunk is not None
            # Normalized chunk padded to the model action dim ((H, model_ad)); reused for both the flow
            # target and FAST. Zero placeholder (unsupervised flow) when no usable chunk.
            if chunk is not None:
                chunk_padded = _pad_action_chunk_for_fast(
                    np.asarray(chunk, dtype=np.float32), model_action_dim=model_ad
                )
                chunk_padded = _fit_chunk_to_horizon(chunk_padded, model_ah, model_ad)
            else:
                chunk_padded = np.zeros((model_ah, model_ad), dtype=np.float32)

            if action_supervised:
                act_rows.append(chunk_padded)
                loss_rows.append(np.ones((model_ah,), dtype=bool))
            else:
                act_rows.append(np.zeros((model_ah, model_ad), dtype=np.float32))
                loss_rows.append(np.zeros((model_ah,), dtype=bool))

            if use_fast:
                if supervise_fast:
                    f_tok, f_mask = self.tokenizer.tokenize_fast_actions(chunk_padded)
                else:
                    f_tok = np.zeros((max_fast_len,), dtype=np.int32)
                    f_mask = np.zeros((max_fast_len,), dtype=bool)
                fast_ids.append(np.asarray(f_tok, dtype=np.int32))
                fast_masks.append(np.asarray(f_mask, dtype=bool))

        batch_size = len(img_batch)
        # SteerVLA/Pi0 preprocess (inside compute_loss) requires all of ``_model.IMAGE_KEYS``.
        # Match the training format (openpi.policies.steervla_policy): the front camera is real and
        # unmasked; the wrist streams are zeros with ``image_mask=False`` so only ``base_0_rgb``
        # contributes ("base_0_rgb only" in effect). Inference restricts to base via CARLA_STEERVLA_IMAGE_KEYS.
        base_imgs = np.stack(img_batch, axis=0)
        zero_imgs = np.zeros_like(base_imgs)
        image_dict: dict[str, np.ndarray] = {}
        image_mask_dict: dict[str, np.ndarray] = {}
        for i, key in enumerate(_openpi_model.IMAGE_KEYS):
            image_dict[key] = base_imgs if i == 0 else zero_imgs
            image_mask_dict[key] = (
                np.ones(batch_size, dtype=bool) if i == 0 else np.zeros(batch_size, dtype=bool)
            )
        data = {
            "image": image_dict,
            "image_mask": image_mask_dict,
            "state": np.stack(state_batch, axis=0),
            "tokenized_prompt": np.stack(prompt_ids, axis=0),
            "tokenized_prompt_mask": np.stack(prompt_masks, axis=0),
            "tokenized_reasoning": np.stack(rea_ids, axis=0),
            "tokenized_reasoning_mask": np.stack(rea_masks, axis=0),
            "tokenized_subtask": np.stack(sub_ids, axis=0),
            "tokenized_subtask_mask": np.stack(sub_masks, axis=0),
            # Per-sample: True for action_supervision pools (flow trained), False otherwise.
            "action_loss_mask": np.stack(loss_rows, axis=0),
        }
        if use_fast and fast_ids:
            # Supervise the FAST action-token CE (cot_fast_ce) only for rows with supervise_fast=True.
            data["tokenized_fast"] = np.stack(fast_ids, axis=0)
            data["tokenized_fast_mask"] = np.stack(fast_masks, axis=0)
        observation = _openpi_model.Observation.from_dict(data)
        actions = np.stack(act_rows, axis=0).astype(np.float32)

        # The HL gradient step runs on ``_hl_jax_device`` (== ``_jax_device`` unless the HL update is
        # isolated on its own GPU), so colocate the batch with the train state.
        device = self._hl_jax_device or self._jax_device
        observation = jax.tree.map(lambda x: jax.device_put(jnp.asarray(x), device), observation)
        actions = jax.device_put(jnp.asarray(actions), device)
        return observation, actions

    def _hl_skip(self, reason: str) -> dict[str, float]:
        """No-op the HL update, printing *why* — but only when the reason changes.

        ``update_hl`` runs on every ``update_with_vla`` call, so an unconditional print would
        spam the log. Printing on transitions keeps one line per distinct cause (and re-prints
        if the run regresses into a previously-cleared state).
        """
        if reason != self._last_hl_skip_reason:
            print(f"[steervla.update_hl] no HL update: {reason}", flush=True)
            self._last_hl_skip_reason = reason
        return {}

    def _render_hl_batch_figure(self, records: list[dict[str, Any]], *, global_step: int | None):
        """Render the HL batch as a matplotlib grid: each sample's observation image with its
        pool / supervision flags / prompt / subtask / reasoning printed underneath.

        Same idea as the best-of-N candidate panel logged next to the rollout video — the point is
        to *see* what the backbone is being trained on (is the image the frame you expect? does the
        subtask match it?), which a text-only table can't answer. Returns an ``(H, W, 3)`` uint8
        array, or ``None`` if the batch has no usable images.
        """
        import textwrap
        from io import BytesIO

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        show = records[: _HL_BATCH_FIG_MAX_SAMPLES]
        if not show:
            return None

        def _wrap(label: str, text: str, width: int = 62, max_lines: int = 6) -> str:
            lines = textwrap.wrap(str(text), width=width) or ["(none)"]
            if len(lines) > max_lines:
                lines = lines[:max_lines] + ["..."]
            return f"{label}: " + f"\n{' ' * (len(label) + 2)}".join(lines)

        ncols = min(3, len(show))
        nrows = int(np.ceil(len(show) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.6 * nrows), squeeze=False)
        for idx in range(nrows * ncols):
            ax = axes[idx // ncols][idx % ncols]
            ax.axis("off")
            if idx >= len(show):
                continue
            rec = show[idx]
            img = np.asarray(rec.get("image"))
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            if img.ndim == 3 and img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            if img.ndim != 3 or img.shape[-1] != 3:
                continue
            # Show the model-resolution frame (same resize+pad _build_hl_observation_batch applies),
            # so the panel reflects what the backbone actually sees, padding artifacts included.
            ax.imshow(_resize_hl_image(img))
            ax.set_title(
                f"[{idx}] pool={rec.get('pool', '')}  "
                f"fast={bool(rec.get('supervise_fast', False))}  "
                f"flow={bool(rec.get('action_supervision', False))}",
                fontsize=9,
            )
            caption = "\n".join(
                [
                    _wrap("prompt", rec.get("prompt", ""), max_lines=3),
                    _wrap("subtask", rec.get("subtask", "")),
                    _wrap("reasoning", rec.get("reasoning", "")),
                ]
            )
            ax.text(
                0.0,
                -0.04,
                caption,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=6.5,
                family="monospace",
            )
        fig.suptitle(
            f"HL update batch — call {int(self._hl_update_calls)}"
            + ("" if global_step is None else f" @ step {int(global_step)}")
            + f" ({len(show)}/{len(records)} samples shown)"
        )
        # Leave vertical room for the out-of-axes captions between rows.
        fig.subplots_adjust(top=0.92, hspace=0.75)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        arr = plt.imread(buf)  # float32 RGBA in [0, 1]
        return (np.asarray(arr)[..., :3] * 255.0).astype(np.uint8)

    def _log_hl_batch_to_wandb(
        self,
        records: list[dict[str, Any]],
        *,
        figure: np.ndarray | None = None,
        global_step: int | None,
    ) -> None:
        """Log the HL update batch to wandb as **readable text** (prompt / subtask / reasoning).

        These are the decoded supervision targets straight from the ``hl_samples.json`` manifest —
        the actual strings the CoT/VLM backbone is trained on this step, NOT the tokenized ids.
        Called on **every** successful :meth:`update_hl`, so each logged table is the complete batch
        that update actually trained on.

        wandb history holds one value per ``(step, key)``, and ``updates_per_step`` sends several
        update_with_vla calls per env step under a single ``global_step``. Logging each batch
        straight to ``vla_hl/batch_text`` would therefore leave only the last one visible. So
        batches sharing a ``global_step`` are accumulated and re-logged as one merged table, with
        ``hl_update_call`` identifying which update each row came from.

        Three channels are logged, because a wandb ``Table`` alone is easy to miss (it renders only
        in the run's Tables/media section and is silently dropped by some workspace configs):

        - ``vla_hl/batch_text`` — the merged table.
        - ``vla_hl/batch_images`` — the matplotlib panel from :meth:`_render_hl_batch_figure`.
        - ``vla_hl/n_samples`` + ``vla_hl/pool/<name>`` — plain scalars, so the batch always shows
          up in Charts even if the media panels don't.
        """
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        if wandb.run is None:
            print(
                "[steervla.update_hl] wandb.run is None (WANDB_MODE=disabled?); "
                "vla_hl/batch_text cannot be logged.",
                flush=True,
            )
            return
        try:
            prompts = [str(r.get("prompt", "")) for r in records]
            subtasks = [str(r.get("subtask", "")) for r in records]
            reasonings = [str(r.get("reasoning", "")) for r in records]
            pools = [str(r.get("pool", "online")) for r in records]
            rows = [
                [
                    int(self._hl_update_calls),
                    i,
                    pools[i],
                    prompts[i],
                    subtasks[i],
                    reasonings[i],
                ]
                for i in range(len(records))
            ]
            if global_step is not None and global_step == self._hl_table_step:
                self._hl_table_rows.extend(rows)  # Same env step: merge, don't overwrite.
            else:
                self._hl_table_step = global_step
                self._hl_table_rows = rows

            table = wandb.Table(columns=["hl_update_call", "sample", "pool", "prompt", "subtask", "reasoning"])
            for row in self._hl_table_rows:
                table.add_data(*row)
            payload: dict[str, Any] = {
                "vla_hl/batch_text": table,
                "vla_hl/n_samples": float(len(records)),
                "vla_hl/hl_update_call": float(self._hl_update_calls),
            }
            for p in set(pools):
                payload[f"vla_hl/pool/{p}"] = float(pools.count(p))
            if figure is not None:
                payload["vla_hl/batch_images"] = wandb.Image(
                    figure,
                    caption=f"HL update call {int(self._hl_update_calls)} @ step {global_step}",
                )
            # wandb silently drops a log whose ``step`` is behind the run's current step. The HL
            # update runs after the env-step logging, so clamp forward rather than lose the batch.
            step = None if global_step is None else max(int(global_step), int(getattr(wandb.run, "step", 0) or 0))
            wandb.log(payload, step=step)
            if not self._hl_logged_batch_once:
                self._hl_logged_batch_once = True
                print(
                    f"[steervla.update_hl] logged first HL batch ({len(rows)} rows) at wandb step "
                    f"{step} (global_step={global_step}): vla_hl/batch_text (Table), "
                    f"vla_hl/batch_images ({'panel' if figure is not None else 'MISSING'}), "
                    f"vla_hl/n_samples (scalar). Tables/Images live under the run's media panels, "
                    f"not in Charts — vla_hl/n_samples is the one visible in Charts.",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - logging must never break the update.
            print(f"[steervla.update_hl] wandb batch logging failed (non-fatal): {exc}", flush=True)

    def _dump_hl_batch_json(
        self,
        records: list[dict[str, Any]],
        *,
        figure: np.ndarray | None = None,
        global_step: int | None,
    ) -> None:
        """Write the exact HL update batch (readable text + image panel) to local files.

        Reliable, wandb-independent record of what each :meth:`update_hl` trained on: one JSON per
        update call under ``<hl_dataset_dir>/hl_update_batches/`` (plus the matching ``.png`` panel),
        and an appended one-line summary in ``hl_update_batches.jsonl``. Written unconditionally on
        every successful update, so the batch is always inspectable on disk even when wandb is
        disabled or drops the media table.
        """
        try:
            prompts = [str(r.get("prompt", "")) for r in records]
            subtasks = [str(r.get("subtask", "")) for r in records]
            reasonings = [str(r.get("reasoning", "")) for r in records]
            pools = [str(r.get("pool", "online")) for r in records]
            root = self.hl_dataset_dir if self.hl_dataset_dir is not None else Path(".")
            out_dir = Path(root) / "hl_update_batches"
            out_dir.mkdir(parents=True, exist_ok=True)
            pool_mix: dict[str, int] = {}
            for p in pools:
                pool_mix[p] = pool_mix.get(p, 0) + 1
            record = {
                "hl_update_call": int(self._hl_update_calls),
                "global_step": None if global_step is None else int(global_step),
                "num_samples": len(subtasks),
                "pool_mix": pool_mix,
                "samples": [
                    {
                        "sample": i,
                        "pool": pools[i],
                        "prompt": prompts[i],
                        "subtask": subtasks[i],
                        "reasoning": reasonings[i],
                        "supervise_fast": bool(records[i].get("supervise_fast", False)),
                        "action_supervision": bool(records[i].get("action_supervision", False)),
                    }
                    for i in range(len(records))
                ],
            }
            fname = f"hl_batch_call{int(self._hl_update_calls):06d}"
            if global_step is not None:
                fname += f"_step{int(global_step):09d}"
            if figure is not None:
                # Save the panel alongside the JSON via PIL (already a dep of the image pipeline);
                # matplotlib is not re-imported here so the failure path stays cheap.
                from PIL import Image as _PILImage

                _PILImage.fromarray(np.asarray(figure, dtype=np.uint8)).save(out_dir / f"{fname}.png")
                record["figure"] = f"{fname}.png"
            (out_dir / f"{fname}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            with (out_dir / "hl_update_batches.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            if not getattr(self, "_hl_logged_json_once", False):
                self._hl_logged_json_once = True
                print(
                    f"[steervla.update_hl] writing HL update batches to {out_dir} "
                    f"(one JSON + one PNG panel per update + hl_update_batches.jsonl).",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - local logging must never break the update.
            print(f"[steervla.update_hl] local HL batch JSON dump failed (non-fatal): {exc}", flush=True)

    def _decode_hl_batch_tokens(
        self,
        observation: Any,
        actions: Any,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Decode the *tokenized* HL batch back to text, sample by sample.

        Everything else logged around the HL update (``_log_hl_batch_to_wandb``,
        ``_dump_hl_batch_json``, the figure) reads the manifest strings — i.e. the supervision as it
        was *written by cast_relabel*. This reads the tensors that :meth:`update_hl` is about to hand
        to ``_hl_train_step``, so anything the tokenizer did in between (truncation of a long
        reasoning trace, an empty segment falling back to ``"Follow the route."``, a mask that
        supervises padding, a subtask/reasoning swap) shows up here and nowhere else.

        Per sample it reports the decoded prompt / reasoning / subtask (and FAST segment when
        supervised), the token counts against each segment's max length, a ``truncated`` flag, and
        whether the decode round-trips to the source string.
        """
        assert self.tokenizer is not None
        max_prompt = int(getattr(self.tokenizer, "_max_prompt_len", 0))
        max_reason = int(getattr(self.tokenizer, "_max_reasoning_len", 0))
        max_subtask = int(getattr(self.tokenizer, "_max_subtask_len", 0))
        segments = (
            ("prompt", observation.tokenized_prompt, observation.tokenized_prompt_mask, max_prompt),
            ("reasoning", observation.tokenized_reasoning, observation.tokenized_reasoning_mask, max_reason),
            ("subtask", observation.tokenized_subtask, observation.tokenized_subtask_mask, max_subtask),
            ("fast", observation.tokenized_fast, observation.tokenized_fast_mask, int(self.tokenizer.max_fast_len)),
        )
        # One host transfer per array rather than one per (sample, segment).
        seg_np: dict[str, tuple[np.ndarray | None, np.ndarray | None, int]] = {}
        for name, ids, mask, max_len in segments:
            seg_np[name] = (
                None if ids is None else np.asarray(jax.device_get(ids)),
                None if mask is None else np.asarray(jax.device_get(mask)),
                max_len,
            )
        loss_mask = (
            None
            if observation.action_loss_mask is None
            else np.asarray(jax.device_get(observation.action_loss_mask))
        )
        act_np = np.asarray(jax.device_get(actions), dtype=np.float32)

        out: list[dict[str, Any]] = []
        for i, rec in enumerate(records):
            row: dict[str, Any] = {
                "sample": i,
                "pool": str(rec.get("pool", "online")),
                "label": rec.get("label"),
                "credit_source": rec.get("credit_source"),
                "supervise_fast": bool(rec.get("supervise_fast", False)),
                "action_supervision": bool(rec.get("action_supervision", False)),
            }
            for name, (ids, mask, max_len) in seg_np.items():
                if ids is None or i >= ids.shape[0]:
                    row[f"decoded_{name}"] = None
                    continue
                mrow = None if mask is None else mask[i].astype(bool)
                n_tok = int(mrow.sum()) if mrow is not None else int(ids.shape[-1])
                row[f"decoded_{name}"] = decode_cot_token_ids(self.tokenizer, ids[i], mrow)
                row[f"n_{name}_tokens"] = n_tok
                row[f"max_{name}_tokens"] = int(max_len)
                # The tokenizer truncates silently (it only logs a warning): a segment whose mask runs
                # to the very last slot almost certainly lost its tail.
                row[f"{name}_truncated"] = bool(max_len and n_tok >= int(max_len))
                row[f"{name}_token_ids"] = [int(t) for t in np.asarray(ids[i]).reshape(-1).tolist()][:max_len or None]
            # Round-trip check against the manifest string the tokenizer was handed.
            for name in ("prompt", "reasoning", "subtask"):
                src = str(rec.get(name, "") or "").strip()
                dec = str(row.get(f"decoded_{name}") or "")
                row[f"source_{name}"] = src
                row[f"{name}_roundtrip_ok"] = bool(src) and src.replace("_", " ").replace("\n", " ") in dec
            if loss_mask is not None and i < loss_mask.shape[0]:
                row["action_loss_mask_true"] = int(loss_mask[i].astype(bool).sum())
                row["action_loss_mask_len"] = int(loss_mask[i].shape[0])
            if i < act_np.shape[0]:
                row["action_chunk_absmax"] = float(np.abs(act_np[i]).max())
            out.append(row)
        return out

    def _dump_hl_batch_tokens(
        self,
        observation: Any,
        actions: Any,
        records: list[dict[str, Any]],
        *,
        global_step: int | None,
    ) -> None:
        """Write / log the decoded token batch produced by :meth:`_decode_hl_batch_tokens`.

        Local JSON is the primary channel (``<hl_dataset_dir>/hl_update_batches/hl_tokens_*.json``
        plus an appended ``hl_update_batch_tokens.jsonl`` without the raw ids); the wandb table
        ``vla_hl/batch_tokens`` mirrors it when a run is active. Never raises — diagnostics must not
        be able to break the update.
        """
        if not self.hl_log_batch_tokens:
            return
        try:
            rows = self._decode_hl_batch_tokens(observation, actions, records)
        except Exception as exc:  # noqa: BLE001 - decoding must never break the update.
            print(f"[steervla.update_hl] HL token decode failed (non-fatal): {exc}", flush=True)
            return

        n_trunc = sum(1 for r in rows if r.get("reasoning_truncated"))
        n_bad_rt = sum(1 for r in rows if not r.get("reasoning_roundtrip_ok", True))
        # The tokenizer truncates silently (``_pad_or_truncate`` only emits a logging.warning that
        # is easily lost in the CARLA log), and a reasoning trace that loses its tail — and its
        # <end_of_reasoning> delimiter — is exactly the kind of damage this dump exists to catch.
        # Announce it per (segment, pool) once so it can't pass unnoticed.
        for seg in ("prompt", "reasoning", "subtask", "fast"):
            hits = [r for r in rows if r.get(f"{seg}_truncated")]
            if not hits:
                continue
            for pool in sorted({str(r["pool"]) for r in hits}):
                key = (seg, pool)
                if key in self._hl_truncation_warned:
                    continue
                self._hl_truncation_warned.add(key)
                n = sum(1 for r in hits if str(r["pool"]) == pool)
                max_len = next(r.get(f"max_{seg}_tokens") for r in hits)
                print(
                    f"[steervla.update_hl] WARNING: {n} sample(s) from pool '{pool}' have a "
                    f"TRUNCATED {seg} segment ({max_len}/{max_len} tokens used). The tail — "
                    f"including the closing CoT delimiter — was dropped before training. Raise the "
                    f"tokenizer's max_{seg}_len or shorten the source text.",
                    flush=True,
                )
        record = {
            "hl_update_call": int(self._hl_update_calls),
            "global_step": None if global_step is None else int(global_step),
            "num_samples": len(rows),
            "num_reasoning_truncated": n_trunc,
            "num_reasoning_roundtrip_failed": n_bad_rt,
            "samples": rows,
        }
        try:
            root = self.hl_dataset_dir if self.hl_dataset_dir is not None else Path(".")
            out_dir = Path(root) / "hl_update_batches"
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"hl_tokens_call{int(self._hl_update_calls):06d}"
            if global_step is not None:
                fname += f"_step{int(global_step):09d}"
            (out_dir / f"{fname}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            slim = dict(record)
            slim["samples"] = [
                {k: v for k, v in r.items() if not k.endswith("_token_ids")} for r in rows
            ]
            with (out_dir / "hl_update_batch_tokens.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(slim) + "\n")
            if not getattr(self, "_hl_logged_tokens_once", False):
                self._hl_logged_tokens_once = True
                print(
                    f"[steervla.update_hl] writing DECODED HL batch tokens to {out_dir} "
                    f"(hl_tokens_*.json + hl_update_batch_tokens.jsonl): the prompt/reasoning/subtask "
                    f"decoded from the exact tensors each update consumes.",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[steervla.update_hl] HL token JSON dump failed (non-fatal): {exc}", flush=True)

        try:
            import wandb  # type: ignore

            if wandb.run is None:
                return
            columns = [
                "hl_update_call",
                "sample",
                "pool",
                "decoded_prompt",
                "decoded_reasoning",
                "decoded_subtask",
                "decoded_fast",
                "source_reasoning",
                "source_subtask",
                "n_reasoning_tokens",
                "n_subtask_tokens",
                "reasoning_truncated",
                "reasoning_roundtrip_ok",
                "subtask_roundtrip_ok",
            ]
            new_rows = [
                [int(self._hl_update_calls)] + [r.get(c) for c in columns[1:]] for r in rows
            ]
            # Same merge-by-global_step rule as ``_log_hl_batch_to_wandb``: several update_with_vla
            # calls share one env step and wandb keeps only the last value per (step, key).
            if global_step is not None and global_step == self._hl_token_table_step:
                self._hl_token_table_rows.extend(new_rows)
            else:
                self._hl_token_table_step = global_step
                self._hl_token_table_rows = new_rows
            table = wandb.Table(columns=columns)
            for row in self._hl_token_table_rows:
                table.add_data(*row)
            step = None if global_step is None else max(int(global_step), int(getattr(wandb.run, "step", 0) or 0))
            payload: dict[str, Any] = {
                "vla_hl/batch_tokens": table,
                "vla_hl/tokens_reasoning_roundtrip_failed": float(n_bad_rt),
            }
            for seg in ("prompt", "reasoning", "subtask", "fast"):
                lens = [r[f"n_{seg}_tokens"] for r in rows if f"n_{seg}_tokens" in r]
                if not lens:
                    continue
                payload[f"vla_hl/tokens_{seg}_truncated"] = float(
                    sum(1 for r in rows if r.get(f"{seg}_truncated"))
                )
                payload[f"vla_hl/tokens_mean_{seg}_len"] = float(np.mean(lens))
                payload[f"vla_hl/tokens_max_{seg}_len"] = float(np.max(lens))
            wandb.log(payload, step=step)
        except ImportError:
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[steervla.update_hl] HL token wandb logging failed (non-fatal): {exc}", flush=True)

    def update_hl(
        self,
        *,
        batch_size: int | None = None,
        num_steps: int | None = None,
        rng=None,
        global_step: int | None = None,
    ) -> dict[str, float]:
        """Run high-level (VLM-backbone) gradient steps on the stored cast_relabel HL dataset.

        No-op (returns ``{}``) unless the actor was loaded trainable (``load_trainable_params``) and the
        online cast_relabel pool holds at least ``hl_min_online_samples`` samples. It does *not* wait
        for ``batch_size`` online samples: the batch takes whatever the online pool has (up to its
        weighted share) and fills the rest from the offline replay pools, repeating rows only if even
        those can't cover ``batch_size``. Throttled by ``hl_update_every`` calls. Updates
        ``self._train_state`` in place and marks the inference weights dirty so the next rollout uses
        the updated backbone.
        """
        if self._remote is not None:
            return self._hl_skip(
                "actor is remote (steervla.actor_url set); the HL update only runs for a local "
                "trainable actor, so no vla_hl/batch_text will be logged"
            )
        if self._train_state is None or self._hl_train_step is None:
            return self._hl_skip(
                "actor was not loaded trainable (needs steervla.load_trainable_params=True)"
            )
        if self.hl_dataset_dir is None:
            return self._hl_skip(
                "hl_dataset_dir is unset — main_carla only wires it when cast_relabel.enabled "
                "and load_trainable_params are both true"
            )
        self._hl_update_calls += 1
        if self.hl_update_every > 1 and (self._hl_update_calls % self.hl_update_every != 0):
            return {}  # Normal throttle, not a fault — stay silent.

        bs = int(batch_size or self.hl_update_batch_size)
        ns = int(num_steps or self.hl_update_num_steps)
        loaded = self._load_hl_batch(bs)
        if loaded is None:
            return self._hl_skip(
                f"waiting for the online HL pool to start: {self._hl_pool_size}/"
                f"{self.hl_min_online_samples} samples under {self.hl_dataset_dir} "
                f"(lower steervla.hl_min_online_samples, or let more cast_relabel windows finish). "
                f"Once it starts, the batch fills from replay rather than waiting for {bs} online samples"
            )
        records = loaded
        self._last_hl_skip_reason = None  # Cleared: a later skip should re-announce itself.
        # Log the batch (observation image + prompt / subtask / reasoning / source pool) to wandb AND
        # to local files. The local dump is the reliable channel — it is written unconditionally on
        # every HL update, independent of wandb run state / step ordering, so the exact batch each
        # update trained on is always inspectable on disk.
        try:
            figure = self._render_hl_batch_figure(records, global_step=global_step)
        except Exception as exc:  # noqa: BLE001 - visualization must never break the update.
            print(f"[steervla.update_hl] HL batch figure render failed (non-fatal): {exc}", flush=True)
            figure = None
        self._log_hl_batch_to_wandb(records, figure=figure, global_step=global_step)
        self._dump_hl_batch_json(records, figure=figure, global_step=global_step)
        observation, actions = self._build_hl_observation_batch(records)
        # Decode the *tokenized* batch (prompt / reasoning / subtask / FAST) immediately before the
        # gradient step consumes it. This is the only view of the supervision after tokenization —
        # truncated or empty reasoning, mis-tokenized subtasks, and masks over padding are invisible
        # in the manifest-text logs above.
        self._dump_hl_batch_tokens(observation, actions, records, global_step=global_step)

        info: dict[str, Any] = {}
        for _ in range(max(1, ns)):
            if rng is None:
                self._train_rng, step_rng = jax.random.split(self._train_rng)
            else:
                step_rng = rng
            self._train_state, info = self._hl_train_step(step_rng, self._train_state, (observation, actions))
        self._weights_dirty = True

        out: dict[str, float] = {}
        for k, v in info.items():
            try:
                out[str(k)] = float(jax.device_get(v))
            except Exception:
                continue
        out["n_samples"] = float(len(records))

        # True device memory around the HL gradient step. ``nvidia-smi`` only shows the
        # preallocated XLA pool (XLA_PYTHON_CLIENT_MEM_FRACTION), not live usage, so read it
        # from JAX directly. Block first so the step has actually materialized before we
        # sample ``peak_bytes_in_use`` (async dispatch would otherwise read a stale peak).
        try:
            jax.block_until_ready(self._train_state.params)
            dev = self._hl_jax_device or self._jax_device or jax.local_devices()[0]
            stats = dev.memory_stats() or {}
            live = stats.get("bytes_in_use")
            peak = stats.get("peak_bytes_in_use")
            limit = stats.get("bytes_limit")
            if live is not None:
                out["hl_mem_live_gb"] = float(live) / 1e9
            if peak is not None:
                out["hl_mem_peak_gb"] = float(peak) / 1e9
            if limit is not None:
                out["hl_mem_limit_gb"] = float(limit) / 1e9
            print(
                f"[steervla.update_hl] device={dev} live={out.get('hl_mem_live_gb', float('nan')):.2f}GB "
                f"peak={out.get('hl_mem_peak_gb', float('nan')):.2f}GB "
                f"limit={out.get('hl_mem_limit_gb', float('nan')):.2f}GB "
                f"(bs={len(records)}, steps={ns})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - memory telemetry must never break the update.
            print(f"[steervla.update_hl] memory_stats unavailable ({exc}).", flush=True)

        return out

    def update_hl_grpo(
        self,
        records: list[dict[str, Any]],
        advantages: np.ndarray,
        *,
        beta: float,
        num_epochs: int = 1,
        global_step: int | None = None,
    ) -> dict[str, float]:
        """GRPO gradient steps on the HL policy over a pooled group of CoT samples.

        ``records`` is the pooled candidate CoTs across scored states (schema from
        :meth:`grpo_records_from_candidates`); ``advantages`` is the aligned per-record group-relative
        advantage (the K candidates of one scored state share that state's normalized VLM scores). The
        pool can exceed one forward, so it is shuffled and processed in ``hl_update_batch_size`` minibatches
        (remainder dropped, or padded by repetition when the pool is smaller than one minibatch) for
        ``num_epochs`` passes.
        Updates ``self._train_state`` in place and marks the inference weights dirty. No-op unless the
        actor was loaded trainable and has a frozen reference.
        """
        if self._remote is not None or self._train_state is None or self._hl_grpo_step is None:
            return self._hl_skip("GRPO update needs a local trainable actor (load_trainable_params=True)")
        records = list(records)
        advantages = np.asarray(advantages, dtype=np.float32).reshape(-1)
        if not records or advantages.shape[0] != len(records):
            return self._hl_skip("GRPO update got no records or mismatched advantages")

        mb = int(self.hl_update_batch_size)
        n = len(records)
        rng_np = np.random.default_rng(int(global_step or 0))
        beta_arr = jnp.asarray(float(beta), dtype=jnp.float32)
        infos: list[dict[str, Any]] = []
        for _ in range(max(1, int(num_epochs))):
            order = rng_np.permutation(n)
            if n >= mb:
                batches = [order[i : i + mb] for i in range(0, n - mb + 1, mb)]  # drop remainder
            else:
                batches = [np.resize(order, mb)]  # pad the whole (small) group up to one minibatch
            for idx in batches:
                sub_records = [records[i] for i in idx]
                sub_adv = jnp.asarray(advantages[idx], dtype=jnp.float32)
                observation, actions = self._build_hl_observation_batch(sub_records)
                self._train_rng, step_rng = jax.random.split(self._train_rng)
                self._train_state, info = self._hl_grpo_step(
                    step_rng, self._train_state, self._hl_ref_params, (observation, actions), sub_adv, beta_arr
                )
                infos.append(info)
        self._weights_dirty = True

        out: dict[str, float] = {}
        if infos:
            for k in infos[-1]:
                try:
                    out[str(k)] = float(np.mean([float(jax.device_get(i[k])) for i in infos]))
                except Exception:
                    continue
        out["n_samples"] = float(n)
        out["n_minibatches"] = float(len(infos))
        return out

    def save_checkpoint(self, out_root: str | Path, step: int, *, keep_last: int = 0) -> Path | None:
        """Export the (HL-fine-tuned) backbone as a redeployable, params-only OpenPI checkpoint.

        Writes ``<out_root>/<step>/params`` in the layout :meth:`setup` loads, so it can be redeployed
        frozen via ``steervla.checkpoint=<out_root>/<step>`` (same ``actor_config``,
        ``load_trainable_params=False``). Saves the current inference params as ``{"params": params}``
        like :func:`openpi.training.checkpoints.save_state` (no optimizer state). Norm stats are not
        copied — this stack runs norm-off by default; to redeploy with ``STEERVLA_ENABLE_OPENPI_NORM=1``
        also copy the source checkpoint's ``assets/``. No-op for a remote/non-trainable actor.

        ``keep_last`` > 0 prunes older step directories under ``out_root`` after a successful write,
        retaining only the newest ``keep_last``. Each checkpoint is ~10 GB, so an un-pruned run at
        ``hl_checkpoint_every_steps=2000`` over 20k env steps leaves ~100 GB behind; the pruning is
        deliberately post-write so a failed save can never delete a good earlier checkpoint.
        """
        if self._remote is not None or self._train_state is None:
            print("[steervla.save_checkpoint] skipped: actor is remote or not loaded trainable.", flush=True)
            return None
        import orbax.checkpoint as ocp

        params_dir = Path(out_root) / str(int(step)) / "params"
        params = self._params_for_inference(self._train_state.params)
        jax.block_until_ready(params)
        with ocp.PyTreeCheckpointer() as ckptr:
            ckptr.save(params_dir, args=ocp.args.PyTreeSave({"params": params}), force=True)
        print(f"[steervla.save_checkpoint] wrote params-only checkpoint -> {params_dir.parent}", flush=True)
        if int(keep_last) > 0:
            self._prune_checkpoints(out_root, keep_last=int(keep_last))
        return params_dir.parent

    @staticmethod
    def _prune_checkpoints(out_root: str | Path, *, keep_last: int) -> None:
        """Delete all but the ``keep_last`` newest numeric step dirs under ``out_root``.

        Best-effort: a pruning failure must never take down a training run, and a partially-removed
        directory is no worse than the disk pressure it was trying to relieve.
        """
        import shutil

        try:
            root = Path(out_root)
            steps = sorted(
                (int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
            )
            for stale in steps[: max(0, len(steps) - int(keep_last))]:
                shutil.rmtree(root / str(stale), ignore_errors=True)
                print(f"[steervla.save_checkpoint] pruned old checkpoint {root / str(stale)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - pruning is best-effort.
            print(f"[steervla.save_checkpoint] checkpoint pruning failed (non-fatal): {exc}", flush=True)

    def setup_qgf(
        self,
        critic_def,
        critic_params: dict,
        guidance_weight: float,
        siglip_encoder,
    ) -> None:
        """Configure QGF inference-time Q-gradient guidance.

        Must be called after setup() (model must be loaded).
        critic_def / critic_params: from qgf_guidance.load_pretrained_critic()
        guidance_weight: scalar weight on the Q-gradient (tune via sweep)
        siglip_encoder: SigLIPEncoder instance (image-only, 1152-D) for the critic
        """
        from qgf_guidance import make_q_grad_fn, make_q_fn
        # model.action_dim is 32 (the full model action space) but the pretrained critic
        # was trained on 10 × 4 = 40-D actions (DELTA_XY_T_DELTA_XY_SPACE only).
        # critic_ah / critic_ad are the dims the critic actually understands.
        CRITIC_AH, CRITIC_AD = 10, 4
        self._qgf_config = {
            "q_grad_fn": make_q_grad_fn(critic_def, critic_params),
            "q_fn": make_q_fn(critic_def, critic_params),
            "guidance_weight": float(guidance_weight),
            "siglip_encoder": siglip_encoder,
            "model_ah": int(self.model.action_horizon),
            "model_ad": int(self.model.action_dim),
            "critic_ah": CRITIC_AH,
            "critic_ad": CRITIC_AD,
            "capture_step": -1,
            "capture_data": None,
        }
        print(
            f"[SteerVLA QGF] guidance_weight={guidance_weight}  "
            f"action_horizon={self.model.action_horizon}  action_dim={self.model.action_dim}",
            flush=True,
        )

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
        
        # Construct the noise
        model_ah = int(self.model.action_horizon)
        model_ad = int(self.model.action_dim)
        cfg_ah = min(int(self.action_horizon), model_ah)
        cfg_ad = min(int(self.action_dim), model_ad)
        noise_full = jnp.zeros((batch_size, model_ah, model_ad), dtype=jnp.float32)
        write_ah = cfg_ah
        if input_noise.ndim == 3:
            noise_chunk = input_noise[:, :cfg_ah, :cfg_ad]
        elif int(input_noise.shape[-1]) == int(self.action_horizon) * int(self.action_dim):
            noise_chunk = input_noise.reshape(batch_size, int(self.action_horizon), int(self.action_dim))[:, :cfg_ah, :cfg_ad]
        else:
            noise_chunk = input_noise[:, None, :cfg_ad]
            write_ah = 1
        noise_full = noise_full.at[:, :write_ah, :cfg_ad].set(noise_chunk)
        
        # Sample the actions
        traj = self._sample_actions_cached(
            rng,
            obs_full,
            noise=noise_full,
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
            num_steps=int(self.sample_actions_num_steps),
        )
        traj_np = self._postprocess_action_trajectory(
            traj,
            observation_state=openpi_observation.state,
        )
        target_dim = int(self.action_dim)
        first_step = traj_np[:, 0, :target_dim]
        out = jnp.asarray(first_step, dtype=jnp.float32)
        return out

    def sample_candidates(
        self,
        n: int,
        *,
        temperature: float,
        raw: Optional[Dict[str, Any]] = None,
        rng: jax.Array | None = None,
    ) -> dict[str, Any]:
        """Best-of-N / EXPO: sample ``n`` CoTs at ``temperature`` (one batched forward), decode a
        chunk per candidate. Diversity comes from the CoTs/subtasks, not the flow noise.

        Returns ``actions`` (n, action_horizon * action_dim), ``subtask_texts`` (per candidate),
        and the batched ``cot_out`` (row i = candidate i's tokens, overlaid via ``_last_cot_out``).
        """
        assert (
            self.model is not None and self._jax_device is not None and self.tokenizer is not None
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
            lambda x: jax.device_put(jnp.asarray(x), self._jax_device), obs_np_struct
        )

        # n diverse CoTs in one batched call (temperature gives independent per-row samples).
        cot_out = self._sample_cot(
            rng_cot,
            obs_jax,
            temperature=float(temperature),
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
        )

        subtask_tokens = np.asarray(jax.device_get(cot_out["tokenized_subtask"]), dtype=np.int32)
        subtask_mask = np.asarray(jax.device_get(cot_out["tokenized_subtask_mask"]), dtype=bool)
        subtask_texts = [
            self.tokenizer._tokenizer.decode(subtask_tokens[i][subtask_mask[i]].tolist()) for i in range(n)
        ]

        obs_full = _merge_cot_output_into_observation(obs_jax, cot_out)

        # One N(0, I) flow seed per candidate (cfg region only), as flow_sample does.
        model_ah = int(self.model.action_horizon)
        model_ad = int(self.model.action_dim)
        cfg_ah = min(int(self.action_horizon), model_ah)
        cfg_ad = min(int(self.action_dim), model_ad)
        noise_chunk = jax.random.normal(rng_noise, (n, cfg_ah, cfg_ad), dtype=jnp.float32)
        noise_full = jnp.zeros((n, model_ah, model_ad), dtype=jnp.float32)
        noise_full = noise_full.at[:, :cfg_ah, :cfg_ad].set(
            jax.device_put(noise_chunk, self._jax_device)
        )

        traj = self._sample_actions_cached(
            rng_act,
            obs_full,
            noise=noise_full,
            num_steps=int(self.sample_actions_num_steps),
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
        )
        jax.block_until_ready(traj)
        traj_np = self._postprocess_action_trajectory(traj, observation_state=obs_jax.state)
        actions_flat = np.asarray(traj_np, dtype=np.float32).reshape(n, -1)

        return {
            "actions": actions_flat,
            "subtask_texts": subtask_texts,
            "cot_out": cot_out,
        }

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

        # Stretch the rectangular CARLA frame to the square model resolution to match the pretraining
        # preprocessing (see ``resize_stretch_np``). Doing it here means the model's internal
        # ``preprocess_observation`` resize_with_pad is a no-op (frame is already 224x224), so the
        # backbone never sees the black bars an aspect-preserving pad would introduce — and the rollout
        # frame is identical to what the HL update trains on (``_resize_hl_image``).
        img = _resize_hl_image(np.asarray(raw["image"], dtype=np.uint8))
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
            # The *raw* routing instruction, i.e. what ``tokenize_prompt`` is actually given.
            # ``openpi_prompt_text`` above is the already-wrapped ``Prompt:...;State:...;`` display
            # form; anything that re-tokenizes a stored prompt (the CAST-relabel HL dataset) must
            # use this one, or the wrapper gets applied twice.
            raw["openpi_prompt_raw_text"] = prompt_text

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

    def policy_embedding_dim(self) -> int:
        """Hidden size of the frozen prefix (PaliGemma) representation."""
        if self.model_cfg is None:
            return 2048
        variant = getattr(self.model_cfg, "paligemma_variant", "gemma_2b")
        return int(_openpi_gemma.get_config(variant).width)

    def _build_frozen_prefix_cache(
        self,
        model,
        observation: _openpi_model.Observation,
    ) -> tuple[_openpi_model.Observation, Any, jax.Array, jax.Array, jax.Array]:
        """Precompute frozen prefix KV/cache and a pooled prefix embedding for DSRL."""
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

        prefix_parts = img_tokens + [prompt_emb, reasoning_emb, subtask_emb]
        prefix_mask_parts = img_masks + [prompt_mask, reasoning_mask, subtask_mask]
        prefix_ar_list = img_ar + [False] * n_prompt + [True] * n_reasoning + [True] * n_subtask
        n_fast = 0
        if getattr(model, "_use_fast_tokens", False) and observation.tokenized_fast is not None:
            fast_emb = model._embed_text_tokens(observation.tokenized_fast)
            fast_mask = observation.tokenized_fast_mask
            n_fast = int(fast_emb.shape[1])
            prefix_parts.append(fast_emb)
            prefix_mask_parts.append(fast_mask)
            prefix_ar_list += [True] * n_fast

        prefix_tokens = jnp.concatenate(prefix_parts, axis=1)
        prefix_mask = jnp.concatenate(prefix_mask_parts, axis=1)
        prefix_ar = jnp.array(prefix_ar_list)

        prefix_attn_mask = _openpi_pi0.make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )

        reasoning_start = n_img + n_prompt
        reasoning_end = reasoning_start + n_reasoning
        prefix_len = prefix_mask.shape[1]
        col_is_reasoning = (jnp.arange(prefix_len) >= reasoning_start) & (jnp.arange(prefix_len) < reasoning_end)
        prefix_mask_no_reasoning = prefix_mask & ~col_is_reasoning[None, :]
        prefix_embed = jax.lax.stop_gradient(
            _pool_prefix_hidden(prefix_out, prefix_mask_no_reasoning)
        )

        return (
            observation,
            jax.tree.map(jax.lax.stop_gradient, kv_cache),
            jax.lax.stop_gradient(prefix_mask),
            jax.lax.stop_gradient(prefix_mask_no_reasoning),
            prefix_embed,
        )

    def _stash_policy_embedding(self, embed: np.ndarray | jax.Array, *, raw: Optional[Dict[str, Any]] = None) -> np.ndarray:
        embed_np = np.asarray(jax.device_get(embed), dtype=np.float32)
        if embed_np.ndim == 1:
            embed_np = embed_np[None, ...]
        self._cached_policy_embed = embed_np
        vec = embed_np[0] if embed_np.shape[0] == 1 else embed_np
        targets: list[Dict[str, Any]] = []
        if isinstance(raw, dict):
            targets.append(raw)
        if self.raw_obs_holder is not None and isinstance(self.raw_obs_holder.get("obs"), dict):
            holder_obs = self.raw_obs_holder["obs"]
            if holder_obs is not raw:
                targets.append(holder_obs)
        for tgt in targets:
            tgt["policy_embedding"] = np.asarray(vec, dtype=np.float32)
        return embed_np

    def ensure_policy_embedding(
        self,
        batch_size: int = 1,
        *,
        raw: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> np.ndarray:
        """Run CoT (if needed) + frozen prefix forward; cache pooled embedding for DSRL."""
        if self._remote is not None:
            raise RuntimeError("Policy prefix embeddings require local SteerVLAActor inference.")
        assert self.model is not None and self._jax_device is not None

        obs_id = id(raw) if raw is not None else id(self.raw_obs_holder.get("obs") if self.raw_obs_holder else None)
        if (
            not force
            and self._cached_policy_embed is not None
            and int(self._cached_policy_embed.shape[0]) == int(batch_size)
            and self._cached_policy_embed_obs_id == obs_id
        ):
            return self._cached_policy_embed

        self._call_counter += 1
        rng = jax.random.PRNGKey(self._call_counter)
        obs_np_struct = self.build_observation_batch_numpy(batch_size, raw=raw)
        obs_jax = jax.tree.map(
            lambda x: jax.device_put(jnp.asarray(x), self._jax_device),
            obs_np_struct,
        )
        cot_out = self._sample_or_reuse_cot(rng, obs_jax, batch_size)
        if batch_size == 1:
            stash_raw = raw
            if stash_raw is None and self.raw_obs_holder is not None:
                stash_raw = self.raw_obs_holder.get("obs")
            if isinstance(stash_raw, dict):
                self._stash_cot_in_raw(stash_raw, cot_out)
        obs_full = _merge_cot_output_into_observation(obs_jax, cot_out)
        prefix_embed = self._prefix_embed_fn(obs_full)
        embed_np = self._stash_policy_embedding(prefix_embed, raw=raw)
        self._cached_policy_embed_obs_id = obs_id
        return embed_np

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

    # Indices into the CARLA gym ``obs["state"]`` vector built by
    # ``ogbench/carla/carla_utils.py :: _ego_state_vector``: world x, y and yaw (degrees).
    _EGO_STATE_IDX_X = 0
    _EGO_STATE_IDX_Y = 1
    _EGO_STATE_IDX_YAW_DEG = 5

    # Output formats whose trailing two action columns are ego-frame xy route deltas, i.e. the
    # ones a rigid SE(2) transform is valid for. The ``*_delta_course_space`` formats store a
    # heading angle in that slot instead, and rotating it as if it were a point is meaningless.
    _ROUTE_XY_FORMATS = frozenset({"delta_xy_t_delta_xy_space"})

    def _ego_pose_from_state(self, state: Any) -> np.ndarray | None:
        """``(x, y, yaw_rad)`` in CARLA world coords from an ``obs["state"]`` vector, else None."""
        if state is None:
            return None
        s = np.asarray(state, dtype=np.float64).reshape(-1)
        if s.size <= self._EGO_STATE_IDX_YAW_DEG:
            return None
        x = float(s[self._EGO_STATE_IDX_X])
        y = float(s[self._EGO_STATE_IDX_Y])
        yaw = float(np.deg2rad(s[self._EGO_STATE_IDX_YAW_DEG]))
        if not np.isfinite((x, y, yaw)).all():
            return None
        if x == 0.0 and y == 0.0:
            # ``_get_state_vector`` returns all-zeros before the ego actor exists.
            return None
        return np.array([x, y, yaw], dtype=np.float64)

    def _current_ego_pose(self) -> np.ndarray | None:
        if self.raw_obs_holder is None:
            return None
        raw = self.raw_obs_holder.get("obs")
        if not isinstance(raw, dict):
            return None
        return self._ego_pose_from_state(raw.get("state"))

    def _reanchor_disabled(self, reason: str) -> None:
        """Log the first reason re-anchoring is inactive, then stay quiet."""
        if self._reanchor_disabled_reason is None:
            self._reanchor_disabled_reason = reason
            print(f"[reanchor] inactive: {reason}", flush=True)

    def _reanchor_route_to_current_pose(
        self, out_flat: np.ndarray, base_flat: np.ndarray
    ) -> np.ndarray:
        """Re-express a replayed chunk's route waypoints in the ego's *current* body frame.

        The route columns are ego-frame xy deltas along a **spatial** path anchored at the pose the
        chunk was sampled from. Serving that chunk verbatim for ``actions_per_model_query`` ticks
        leaves the lateral controller open-loop: ``LateralPIDController`` derives its heading error
        purely from these waypoints (``ogbench/carla/steervla_simlingo_control.py`` passes it only
        the ego *speed* besides the chunk), and ``interpolate_waypoints`` re-zeroes arc length at
        the plan's first point. The decoder therefore *assumes* the ego is sitting exactly at the
        pose the chunk was sampled from, and re-issues the steering appropriate to that pose for
        the whole hold. Any cross-track or heading error the ego accumulates in between is
        invisible until the next model query -- which is precisely the error a controller exists
        to remove. Measured on a 30 m-radius arc at 10 m/s over a 5-tick hold
        (``test_reanchor_cached_chunk.py``): with 0.4 m of lateral drift the un-re-anchored steer
        stays pinned at 0.152 while the re-anchored one corrects to -0.109; with 4 deg of yaw lag
        it stays at 0.152 versus 0.320. When the ego tracks the plan exactly the two agree, as
        they should.

        Fix: rigid SE(2) transform of the cumulative waypoints into the current body frame, drop
        the points now behind the ego, and re-derive the deltas. This is the continuous analogue of
        :meth:`_shift_cached_action_chunk`'s integer row shift, driven by the ego's *measured*
        progress instead of by assuming it tracked the plan exactly.

        The speed columns are deliberately left untouched. The PID reads them as
        ``|wp[2] - wp[0]| * 2`` -- a displacement magnitude over a fixed 0.5 s window -- which is
        invariant under rotation and translation. That channel is time-indexed, so only the
        whole-row shift can advance it, and it stays closed-loop anyway via the fresh ego speed.
        """
        if not self.reanchor_cached_chunk:
            self._reanchor_disabled("reanchor_cached_chunk=False")
            return out_flat
        horizon = int(self.action_horizon)
        adim = int(self.action_dim)
        if adim < 4:
            self._reanchor_disabled(f"action_dim={adim} has no xy route columns")
            return out_flat
        fmt = str(self.output_action_format or "").strip().lower()
        if fmt not in self._ROUTE_XY_FORMATS:
            self._reanchor_disabled(f"output_action_format={self.output_action_format!r} is not an xy route format")
            return out_flat
        pose_now = self._current_ego_pose()
        pose_query = self._cached_action_pose
        if pose_now is None or pose_query is None:
            self._reanchor_disabled("no ego pose available from raw_obs_holder['obs']['state']")
            return out_flat

        # Old ego origin expressed in the current ego frame, using the same world->ego convention
        # as ``carla_utils._compute_target_point_ego`` (x forward): t = R(yaw_now)^T (o_old - o_now).
        d_world = pose_query[:2] - pose_now[:2]
        cy, sy = float(np.cos(pose_now[2])), float(np.sin(pose_now[2]))
        t = np.array(
            [d_world[0] * cy + d_world[1] * sy, -d_world[0] * sy + d_world[1] * cy],
            dtype=np.float64,
        )
        dyaw_raw = float(pose_now[2] - pose_query[2])
        dyaw = float(np.arctan2(np.sin(dyaw_raw), np.cos(dyaw_raw)))
        if float(np.hypot(t[0], t[1])) < 1e-3 and abs(dyaw) < 1e-4:
            # Ego has not moved since the query -- the cached frame *is* the current frame.
            self.last_reanchor = {"dx_m": 0.0, "dy_m": 0.0, "dyaw_deg": 0.0, "dropped": 0.0, "padded": 0.0}
            return out_flat

        base = np.asarray(base_flat, dtype=np.float64).reshape(-1, horizon, adim)[0]
        # Cumulative route points in the query-time ego frame; pts[0] is the query origin itself.
        pts = np.cumsum(
            np.concatenate([np.zeros((1, 2), dtype=np.float64), base[:, 2:4]], axis=0), axis=0
        )
        c, s = float(np.cos(-dyaw)), float(np.sin(-dyaw))
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        pts_now = pts @ rot.T + t

        ahead = np.flatnonzero(pts_now[:, 0] > 1e-3)
        if ahead.size == 0:
            # Whole plan is behind the ego (it overran the chunk); nothing sane to steer toward.
            self._reanchor_disabled("cached route fully behind the ego")
            return out_flat
        kept = pts_now[int(ahead[0]) :][:horizon]
        new_deltas = np.diff(
            np.concatenate([np.zeros((1, 2), dtype=np.float64), kept], axis=0), axis=0
        )
        padded = horizon - new_deltas.shape[0]
        if padded > 0:
            # Same tail convention as _shift_cached_action_chunk: extend along the last delta.
            new_deltas = np.concatenate(
                [new_deltas, np.repeat(new_deltas[-1:], padded, axis=0)], axis=0
            )

        out = np.array(out_flat, dtype=np.float32).reshape(-1, horizon, adim)
        out[:, :, 2:4] = new_deltas.astype(np.float32)[None]
        self.last_reanchor = {
            "dx_m": float(t[0]),
            "dy_m": float(t[1]),
            "dyaw_deg": float(np.degrees(dyaw)),
            "dropped": float(int(ahead[0])),
            "padded": float(max(padded, 0)),
        }
        print(
            f"[reanchor] ego moved dx={t[0]:+.3f} dy={t[1]:+.3f} m dyaw={np.degrees(dyaw):+.2f} deg "
            f"-> dropped {int(ahead[0])} route pts, padded {max(padded, 0)}",
            flush=True,
        )
        return out.reshape(out.shape[0], horizon * adim)

    def _next_cached_action(self, batch_size: int) -> jnp.ndarray | None:
        """Serve one env action from the cached chunk, re-anchored to the ego's *actual* progress.

        ``actions_per_model_query`` counts **env steps**, not chunk rows. The two are not the same
        rate: a chunk row is a waypoint at the policy rate the model was trained at (4 Hz for the
        SimLingo/SteerVLA data, i.e. 0.25 s apart), while a CARLA env step is a single 20 Hz tick
        (0.05 s) -- ``env_steps_per_chunk_row`` (= ``carla_fps // policy_fps`` = 5) is the ratio.
        So the plan may only be shifted forward one row every ``env_steps_per_chunk_row`` env steps;
        shifting once per env step would replay the model's trajectory 5x too fast (the longitudinal
        PID would read the speed target ~1 s into the plan after only 0.2 s of sim, braking early and
        over-throttling out of a stop). Holding the chunk for the intervening ticks and re-running the
        PID against the fresh ego speed is exactly what ``simlingo/team_code/agent_steervla.py`` does
        between VLM queries; the env re-decodes the chunk with a fresh state vector on every step.

        Holding the chunk is not enough on its own, though: the env only feeds the decoder a fresh
        ego *speed*, so the lateral loop would be open-loop for the whole hold. The route waypoints
        are therefore re-anchored to the ego's current pose on every replayed step -- see
        :meth:`_reanchor_route_to_current_pose`.
        """
        if self.actions_per_model_query <= 1 or batch_size != 1 or self._cached_action_chunk is None:
            return None
        rows_per_query = int(self.action_horizon) * self.env_steps_per_chunk_row
        max_cached_steps = min(self.actions_per_model_query, rows_per_query)
        if self._cached_action_step >= max_cached_steps:
            self._cached_action_chunk = None
            self._cached_action_step = 0
            return None
        row = self._cached_action_step // self.env_steps_per_chunk_row
        # Time-indexed shift for the speed columns, pose-based re-anchor for the route columns.
        # The re-anchor reads the *unshifted* chunk: it accounts for all progress since the query
        # continuously, so the integer row shift would double-count it on the route channel.
        out = self._shift_cached_action_chunk(self._cached_action_chunk, row)
        out = self._reanchor_route_to_current_pose(out, self._cached_action_chunk)
        self._cached_action_step += 1
        return jnp.asarray(out)

    def _remember_action_chunk(self, action: Any, batch_size: int) -> None:
        if self.actions_per_model_query <= 1 or batch_size != 1:
            return
        self._cached_action_chunk = np.asarray(jax.device_get(action), dtype=np.float32)
        # Pose this chunk's ego-frame waypoints are anchored at, for _reanchor_route_to_current_pose.
        self._cached_action_pose = self._current_ego_pose()
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

    def _preprocess_observation_on_device(
        self,
        observation: _openpi_model.Observation,
    ) -> _openpi_model.Observation:
        """Device-put an obs, overlay the last sampled CoT, and run OpenPI eval preprocessing.

        Shared prefix input for :meth:`encode_prefix_features` and :meth:`encode_prefix_tokens`, so
        both see the reasoning/subtask (and FAST) tokens the base policy actually acted on rather
        than the zeroed placeholders in a fresh observation.
        """
        obs_jax = jax.tree.map(
            lambda x: jax.device_put(jnp.asarray(x), self._jax_device),
            observation,
        )
        if self._last_cot_out is not None:
            obs_jax = _merge_cot_output_into_observation(obs_jax, self._last_cot_out)
        return _openpi_model.preprocess_observation(
            None, obs_jax, train=False, image_keys=CARLA_STEERVLA_IMAGE_KEYS,
        )

    @staticmethod
    def _mean_pool_prefix(prefix_out: jax.Array, prefix_mask: jax.Array) -> jax.Array:
        """Mask-weighted mean over the token axis -> one frozen feature per batch row."""
        m = prefix_mask.astype(prefix_out.dtype)
        denom = jnp.maximum(m.sum(axis=1, keepdims=True), 1.0)
        return jax.lax.stop_gradient(jnp.sum(prefix_out * m[..., None], axis=1) / denom)

    def _prefix_from_obs(self, raw: Dict[str, Any], *, include_fast: bool) -> tuple[jax.Array, jax.Array]:
        """Fresh CoT + prefix from ``raw`` alone: sample a new CoT from this image + prompt and run the
        prefix on it, as if the base VLM had seen only ``raw``. Bypasses the cache and the real rollout's
        sampled CoT, so no real-image info (not even via CoT text) leaks in, and does not mutate the
        rollout's CoT/prefix state. Black-image sanity check only.

        Runs the same jitted path as a normal base step (``_sample_cot`` + ``_sample_actions_with_prefix``,
        which preprocesses internally), so it is as fast as the normal pass; the sampled action is dropped.
        """
        if self._remote is not None or self.model is None or self._jax_device is None:
            raise RuntimeError("Prefix recompute requires a local SteerVLA model.")
        if not self._prefix_reuse:
            raise RuntimeError("Black-image recompute needs openpi with sample_actions_with_prefix.")
        obs_jax = jax.tree.map(
            lambda x: jax.device_put(jnp.asarray(x), self._jax_device),
            self.build_observation_batch_numpy(1, raw=raw),
        )
        cot_out = self._sample_cot(
            jax.random.PRNGKey(0),
            obs_jax,
            temperature=float(self.cot_temperature),
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
        )
        obs_full = _merge_cot_output_into_observation(obs_jax, cot_out)
        noise = jnp.zeros((1, int(self.model.action_horizon), int(self.model.action_dim)), dtype=jnp.float32)
        _traj, prefix_out, prefix_mask = self._sample_actions_with_prefix(
            jax.random.PRNGKey(0),
            obs_full,
            noise=noise,
            num_steps=int(self.sample_actions_num_steps),
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
        )
        if not include_fast and _model_uses_fast_tokens(self.model_cfg):
            keep = prefix_mask.shape[1] - int(self.model_cfg.max_fast_len)
            prefix_out, prefix_mask = prefix_out[:, :keep], prefix_mask[:, :keep]
        return prefix_out, prefix_mask

    def encode_prefix_features(
        self,
        raw: Dict[str, Any],
    ) -> jax.Array:
        """Frozen, mean-pooled prefix feature over ``[image, prompt, reasoning, subtask]``.

        Reuses the prefix cached by ``sample_actions_with_prefix`` (row ``_prefix_cache_row``,
        trailing FAST columns dropped), so call only after the base chunk for ``raw`` has been
        sampled -- an empty cache raises.
        """
        if self._recompute_prefix_from_obs:
            return self._mean_pool_prefix(*self._prefix_from_obs(raw, include_fast=False))
        self._require_prefix_cache("pi prefix features")
        row = int(self._prefix_cache_row)
        keep = self._last_prefix_mask.shape[1] - int(self._last_prefix_n_fast)
        return self._mean_pool_prefix(
            self._last_prefix_out[row : row + 1, :keep],
            self._last_prefix_mask[row : row + 1, :keep],
        )

    def _group_pool(self, prefix_out: jax.Array, prefix_mask: jax.Array) -> jax.Array:
        """Masked-mean each of ``[image, prompt, reasoning, subtask]`` separately; concat -> [B, 4D].

        ``prefix_out`` must be the non-FAST prefix (order ``[img, prompt, reasoning, subtask]``). Text
        span lengths come from model_cfg (padded); the image span is inferred from the remainder, so
        this works for any image tokenization.
        """
        cfg = self.model_cfg
        n_prompt, n_reasoning, n_subtask = int(cfg.max_token_len), int(cfg.max_reasoning_len), int(cfg.max_subtask_len)
        m = prefix_out.shape[1]
        n_img = m - (n_prompt + n_reasoning + n_subtask)
        bounds = (
            (0, n_img),
            (n_img, n_img + n_prompt),
            (n_img + n_prompt, n_img + n_prompt + n_reasoning),
            (n_img + n_prompt + n_reasoning, m),
        )
        pools = []
        for s, e in bounds:
            seg, msk = prefix_out[:, s:e], prefix_mask[:, s:e].astype(prefix_out.dtype)
            denom = jnp.maximum(msk.sum(axis=1, keepdims=True), 1.0)
            pools.append(jnp.sum(seg * msk[..., None], axis=1) / denom)
        return jax.lax.stop_gradient(jnp.concatenate(pools, axis=-1))

    def encode_prefix_group_features(self, raw: Dict[str, Any]) -> jax.Array:
        """Per-group mean-pooled prefix feature: concat of masked-mean over ``[image, prompt, reasoning,
        subtask]`` -> f32[1, 4*D]. Same CoT / cache semantics as :meth:`encode_prefix_features`, but keeps
        the four groups separate so a trained (ideally nonlinear) state head can weight them.
        """
        if self._recompute_prefix_from_obs:
            return self._group_pool(*self._prefix_from_obs(raw, include_fast=False))
        self._require_prefix_cache("pi prefix features")
        row = int(self._prefix_cache_row)
        keep = self._last_prefix_mask.shape[1] - int(self._last_prefix_n_fast)
        return self._group_pool(
            self._last_prefix_out[row : row + 1, :keep],
            self._last_prefix_mask[row : row + 1, :keep],
        )

    def encode_prefix_tokens(
        self,
        raw: Dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the *un-pooled* SteerVLA prefix tokens for the RLT state encoder.

        Mirrors the offline RL-Token embedding dump (``dump_rl_token_embeddings.py``) exactly so a
        separately trained autoencoder sees the same token layout it was trained on: image + prompt +
        reasoning + subtask + FAST tokens (order ``[base-cam vision, prompt, reasoning, subtask, fast]``;
        CARLA has a single camera so there are no wrist-cam tokens to drop), *without* the mean-pool that
        :meth:`encode_prefix_features` applies.

        Reuses the prefix cached by ``sample_actions_with_prefix`` (row ``_prefix_cache_row``), so call
        only after the base chunk for ``raw`` has been sampled -- an empty cache raises.

        Returns ``(prefix_out f32[1, M, D], prefix_mask bool[1, M])``.
        """
        if self._recompute_prefix_from_obs:
            prefix_out, prefix_mask = self._prefix_from_obs(raw, include_fast=True)
            out = np.asarray(jax.device_get(prefix_out), dtype=np.float32)
            mask = np.asarray(jax.device_get(prefix_mask), dtype=bool)
            return out, mask
        self._require_prefix_cache("pi prefix tokens")
        row = int(self._prefix_cache_row)
        out = np.asarray(jax.device_get(self._last_prefix_out[row : row + 1]), dtype=np.float32)
        mask = np.asarray(jax.device_get(self._last_prefix_mask[row : row + 1]), dtype=bool)
        return out, mask

    def _reasoning_overflowed(self, cot_out: dict[str, Any]) -> np.ndarray:
        """Per-row: did the reasoning segment run out its whole ``max_reasoning_len`` budget?

        ``Pi0CoT._sample_cot_core_impl`` decodes reasoning until it emits ``END_OF_REASONING``
        (``<loc1019>``) or exhausts ``mr = max_reasoning_len`` tokens. On exhaustion it *force-writes*
        the end marker into the tail slot, so an overflowed chain is indistinguishable from a normal
        one by text alone -- but its mask is saturated to the full budget.

        Overflow is the signature of a derailed sample: with ``cot_temperature > 0`` the decoder draws
        from the full PaliGemma vocabulary with no top-k/top-p truncation, so one junk token
        (multilingual fragments, emoji) knocks the chain off-distribution and it never emits the stop
        token. The resulting "reasoning" is the FAST action grammar (``Action: <loc....>|...``) spilled
        into the reasoning slot.
        """
        mask = np.asarray(jax.device_get(cot_out["tokenized_reasoning_mask"]))
        mask = mask.reshape(mask.shape[0], -1) if mask.ndim > 1 else mask.reshape(1, -1)
        budget = int(getattr(self.model_cfg, "max_reasoning_len", mask.shape[-1]))
        return mask.astype(bool).sum(axis=-1) >= budget

    def _sample_cot_checked(
        self,
        rng: jax.Array,
        obs_jax: _openpi_model.Observation,
    ) -> dict[str, Any]:
        """``_sample_cot`` plus a bounded resample when the reasoning overflows its budget.

        Only active when sampling stochastically (``cot_temperature > 0``); greedy decoding is
        deterministic, so a resample at the same temperature would reproduce the same chain. Each
        retry halves the temperature and the last one falls back to greedy, which always terminates
        on-distribution. See :meth:`_reasoning_overflowed` for why overflow means "garbled".
        """
        temperature = float(self.cot_temperature)
        cot_out = self._sample_cot(
            rng, obs_jax, temperature=temperature, image_keys=CARLA_STEERVLA_IMAGE_KEYS
        )
        self._cot_sample_count += 1
        if temperature <= 0.0 or not self.cot_resample_on_overflow:
            return cot_out

        for attempt in range(self.cot_overflow_max_resamples):
            if not bool(np.any(self._reasoning_overflowed(cot_out))):
                return cot_out
            self._cot_overflow_count += 1
            last = attempt == self.cot_overflow_max_resamples - 1
            temperature = 0.0 if last else temperature / 2.0
            print(
                f"[DEBUG - steervla] reasoning overflowed max_reasoning_len "
                f"(garbled CoT); resampling at temperature={temperature:.2f} "
                f"({attempt + 1}/{self.cot_overflow_max_resamples}) "
                f"[{self._cot_overflow_count}/{self._cot_sample_count} samples so far]",
                flush=True,
            )
            rng, sub_rng = jax.random.split(rng)
            cot_out = self._sample_cot(
                sub_rng, obs_jax, temperature=temperature, image_keys=CARLA_STEERVLA_IMAGE_KEYS
            )
        return cot_out

    def _sample_or_reuse_cot(
        self,
        rng: jax.Array,
        obs_jax: _openpi_model.Observation,
        batch_size: int,
    ) -> dict[str, Any]:
        if self._uses_fixed_cot():
            cot_out = self._build_fixed_cot_out(
                batch_size,
                ref_array=obs_jax.tokenized_prompt,
            )
            self._last_cot_out = cot_out
            return cot_out
        if (
            self._cot_cache_enabled(batch_size)
            and self._cached_cot is not None
            and self._cached_cot_actions_used < self.actions_per_cot
        ):
            self._last_cot_out = self._cached_cot
            return self._cached_cot
        cot_out = self._sample_cot_checked(rng, obs_jax)
        if self._cot_cache_enabled(batch_size):
            self._cached_cot = dict(cot_out)
            self._cached_cot_actions_used = 0
        self._last_cot_out = cot_out
        return cot_out

    def _grpo_scene_fields(self, raw: dict[str, Any]) -> tuple[np.ndarray, str, np.ndarray]:
        """Shared (state, prompt, image) for GRPO HL records built from a raw CARLA obs."""
        state_vec = np.asarray(raw["state"], dtype=np.float32).reshape(-1)
        speed = float(state_vec[15]) if state_vec.shape[0] > 15 else 0.0
        prompt = routing_instruction_prompt(routing_command=self.routing_command, current_speed_mps=speed)
        return state_vec, prompt, np.asarray(raw["image"])

    def grpo_records_from_candidates(
        self, cands: dict[str, Any], raw: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """K per-candidate HL records for :meth:`update_hl_grpo`, from a :meth:`sample_candidates` batch.

        Each record carries the sampled reasoning/subtask token ids for that candidate plus the shared scene.
        Schema matches _build_hl_observation_batch (action loss + FAST masked off: HL-only supervision).
        """
        cot = cands["cot_out"]
        rea = np.asarray(jax.device_get(cot["tokenized_reasoning"]), dtype=np.int32)
        rea_mask = np.asarray(jax.device_get(cot["tokenized_reasoning_mask"]), dtype=bool)
        sub = np.asarray(jax.device_get(cot["tokenized_subtask"]), dtype=np.int32)
        sub_mask = np.asarray(jax.device_get(cot["tokenized_subtask_mask"]), dtype=bool)
        state_vec, prompt, image = self._grpo_scene_fields(raw)
        return [
            {
                "state": state_vec,
                "state_format": "carla_raw",
                "prompt": prompt,
                "image": image,
                "reasoning_ids": rea[k],
                "reasoning_mask": rea_mask[k],
                "subtask_ids": sub[k],
                "subtask_mask": sub_mask[k],
                "action_supervision": False,
                "supervise_fast": False,
            }
            for k in range(rea.shape[0])
        ]

    def grpo_stop_record(self, raw: dict[str, Any], *, reasoning: str, subtask: str) -> dict[str, Any]:
        """Canned text-based HL record (debug stop-injection); the CoT is tokenized from the given
        reasoning/subtask text by _build_hl_observation_batch (no ``*_ids``)."""
        state_vec, prompt, image = self._grpo_scene_fields(raw)
        return {
            "state": state_vec,
            "state_format": "carla_raw",
            "prompt": prompt,
            "image": image,
            "reasoning": str(reasoning),
            "subtask": str(subtask),
            "action_supervision": False,
            "supervise_fast": False,
        }

    def _mark_action_served(self, batch_size: int) -> None:
        if self._cot_cache_enabled(batch_size) and self._cached_cot is not None:
            self._cached_cot_actions_used += 1

    def reset_action_cache(self) -> None:
        self._cached_action_chunk = None
        self._cached_action_step = 0
        self._cached_action_pose = None
        self.last_reanchor = {}
        self._cached_cot = None
        self._cached_cot_actions_used = 0
        self._last_cot_out = None
        self._last_prefix_out = None
        self._last_prefix_mask = None
        self._last_prefix_n_fast = 0
        self._prefix_cache_row = 0

    def _sample_actions_cached(
        self,
        rng: jax.Array,
        obs_full: _openpi_model.Observation,
        *,
        noise: jax.Array,
        num_steps: int,
        image_keys: tuple[str, ...],
    ) -> jax.Array:
        """Drop-in for ``self._sample_actions`` that caches the frozen prefix when available.

        With ``sample_actions_with_prefix`` the returned prefix is stashed (stop-gradient) for
        pi_prefix / rl_token reuse; otherwise the cache is cleared and the plain sampler runs.
        """
        if self._prefix_reuse:
            traj, prefix_out, prefix_mask = self._sample_actions_with_prefix(
                rng,
                obs_full,
                noise=noise,
                num_steps=num_steps,
                image_keys=image_keys,
            )
            self._last_prefix_out = jax.lax.stop_gradient(prefix_out)
            self._last_prefix_mask = prefix_mask
            self._last_prefix_n_fast = (
                int(self.model_cfg.max_fast_len) if _model_uses_fast_tokens(self.model_cfg) else 0
            )
            return traj
        self._last_prefix_out = None
        self._last_prefix_mask = None
        self._last_prefix_n_fast = 0
        return self._sample_actions(
            rng,
            obs_full,
            noise=noise,
            num_steps=num_steps,
            image_keys=image_keys,
        )

    def _require_prefix_cache(self, what: str) -> None:
        """Guard the prefix encoders: local model up, and the base prefix already cached this step."""
        if self._remote is not None:
            raise RuntimeError(f"{what} are not available in remote SteerVLAActor mode.")
        if self.model is None or self._jax_device is None:
            raise RuntimeError("Local SteerVLA model is not initialized.")
        if self._last_prefix_out is None:
            raise RuntimeError(
                f"{what} require the cached prefix; sample the base chunk before encoding "
                "(needs openpi with sample_actions_with_prefix)."
            )

    def _qgf_guided_denoise(
        self,
        obs_full,
        noise_full: jnp.ndarray,
        batch_size: int,
    ) -> jnp.ndarray:
        """QGF-guided denoising loop.

        Replaces _sample_actions when setup_qgf() has been called.
        Runs num_steps Euler steps, injecting Q-gradient guidance at each step.

        Action convention (pi0): t goes 1→0; t=1 is pure noise, t=0 is clean action.
        QGF guidance:
          a_approx = clip(x_t + t * v_bc, -1, 1)   one-Euler denoised estimate
          v_guided = v_bc + guidance_weight * chain_rule(qgrad_phys)
        """
        from qgf_guidance import model_to_physical_flat, qgrad_physical_to_model_flat

        cfg = self._qgf_config
        q_grad_fn = cfg["q_grad_fn"]
        guidance_weight = float(cfg["guidance_weight"])
        siglip_encoder = cfg["siglip_encoder"]
        model_ah = cfg["model_ah"]   # full model action horizon (e.g. 10)
        model_ad = cfg["model_ad"]   # full model action dim (e.g. 32)
        c_ah = cfg["critic_ah"]      # critic action horizon (10)
        c_ad = cfg["critic_ad"]      # critic action dim (4, DELTA_XY_T_DELTA_XY_SPACE)
        n_steps = int(self.sample_actions_num_steps)
        dt = jnp.asarray(-1.0 / n_steps, dtype=noise_full.dtype)

        # Compute prefix KV cache + preprocessed observation (one forward pass)
        obs_preprocessed, kv_cache, prefix_mask, prefix_mask_no_reasoning, _ = (
            self._prefix_cache_fn(obs_full)
        )
        jax.block_until_ready(kv_cache)

        # Get SigLIP image-only embedding (1152-D) for the pretrained critic
        raw_obs = self.raw_obs_holder.get("obs", {}) if self.raw_obs_holder else {}
        img = raw_obs.get("image")
        if img is None:
            raise RuntimeError("[QGF] raw_obs_holder['obs']['image'] is None; cannot compute obs_enc for critic.")
        obs_enc_1152 = jnp.asarray(
            siglip_encoder.encode(img), dtype=jnp.float32
        )[None]  # (1, 1152)

        x_t = noise_full  # (B, model_ah, model_ad)
        t = jnp.asarray(1.0, dtype=noise_full.dtype)

        # Optional data capture for figure generation (set by _qgf_config["capture_step"]).
        capture_step = cfg.get("capture_step", -1)
        capture_data = cfg.get("capture_data")  # list; appended to when capture_step > 0

        for step_i in range(n_steps):
            # One BC denoising step
            x_next_bc, t_next = self._denoise_step_fn(
                obs_preprocessed, kv_cache, prefix_mask, prefix_mask_no_reasoning,
                dt, x_t, t,
            )
            # Recover BC velocity: x_next_bc = x_t + dt * v_bc  →  v_bc = (x_next - x_t) / dt
            v_bc = (x_next_bc - x_t) / dt  # (B, model_ah, model_ad)

            # One-Euler denoised action approximation (in model space).
            # Slice only the first c_ah × c_ad dims that the critic was trained on.
            a_approx_model = jnp.clip(x_t + t * v_bc, -1.0, 1.0)  # (B, model_ah, model_ad)
            a_approx_critic = a_approx_model[:, :c_ah, :c_ad].reshape(batch_size, c_ah * c_ad)

            # Critic trained in model space; model_to_physical_flat is identity.
            a_phys_flat = model_to_physical_flat(a_approx_critic, c_ah, c_ad)

            # Q-gradient in model space (no chain-rule scaling needed).
            qgrad_phys_flat = q_grad_fn(obs_enc_1152, a_phys_flat)  # (B, c_ah*c_ad)
            qgrad_model_flat = qgrad_physical_to_model_flat(qgrad_phys_flat, c_ah, c_ad)
            # Embed gradient back into the full model action shape (zeros outside critic dims)
            qgrad_model = jnp.zeros_like(v_bc)
            qgrad_model = qgrad_model.at[:, :c_ah, :c_ad].set(
                qgrad_model_flat.reshape(batch_size, c_ah, c_ad)
            )

            # Capture denoising data at the requested env step (for figure generation)
            if capture_data is not None and capture_step >= 0:
                q_bc = float(self._qgf_config["q_fn"](obs_enc_1152, a_phys_flat))
                x_guided = x_t + dt * (v_bc + guidance_weight * qgrad_model)
                capture_data.append({
                    "denoise_step": step_i,
                    "t": float(t),
                    "x_t_bc": np.array(a_approx_critic[0]),   # (c_ah*c_ad,) BC approx in model space
                    "x_t_after": np.array(jnp.clip(x_guided, -1.0, 1.0)[:, :c_ah, :c_ad].reshape(batch_size, c_ah * c_ad)[0]),
                    "x_t_phys": np.array(a_phys_flat[0]),     # (c_ah*c_ad,) in physical space
                    "q_bc": q_bc,
                    "qgrad_norm": float(jnp.linalg.norm(qgrad_phys_flat)),
                    "obs_enc": np.array(obs_enc_1152[0]),      # (1152,) for Q-landscape
                })

            # Guided update — clip to keep x_t in valid model range
            x_t = jnp.clip(x_t + dt * (v_bc + guidance_weight * qgrad_model), -1.0, 1.0)
            t = t_next

        return x_t  # (B, model_ah, model_ad) — same shape as _sample_actions output

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
        self._refresh_inference_weights()
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
        _cot_t0 = time.time()
        cot_out = self._sample_or_reuse_cot(rng_cot, obs_jax, batch_size)
        jax.block_until_ready(cot_out["tokenized_reasoning"])
        print(f"[DEBUG - steervla] CoT time: {time.time() - _cot_t0:.3f} seconds")
        
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

        # Per-row (batch) subtask/reasoning tokens, for callers that need each
        # candidate's own text separately (e.g. best-of-N candidate overlays) --
        # the debug print above flattens the whole batch into one decode.
        self._last_batch_subtask = (
            np.asarray(jax.device_get(subtask_tokens)), np.asarray(jax.device_get(subtask_mask))
        )
        self._last_batch_reasoning = (
            np.asarray(jax.device_get(reason_tokens)), np.asarray(jax.device_get(reason_mask))
        )
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

        obs_id = id(raw) if raw is not None else id(self.raw_obs_holder.get("obs") if self.raw_obs_holder else None)
        if self._cached_policy_embed is None or self._cached_policy_embed_obs_id != obs_id:
            _prefix_t0 = time.time()
            prefix_embed = self._prefix_embed_fn(obs_full)
            jax.block_until_ready(prefix_embed)
            self._stash_policy_embedding(prefix_embed, raw=raw)
            self._cached_policy_embed_obs_id = obs_id
            print(f"[DEBUG - steervla] Prefix embed time: {time.time() - _prefix_t0:.3f} seconds")

        # Prepare noise for inference.
        #
        # Two flat noise conventions exist in this stack:
        #   MODEL layout  action_horizon * model.action_dim   (Pi0 latent width; DSRL noise actor)
        #   ENV   layout  action_horizon * self.action_dim    (real driving dims only)
        # They are checked in that order; on a collision the model layout wins.
        batch_size = obs_jax.state.shape[0]
        model_ah = int(self.model.action_horizon)
        model_ad = int(self.model.action_dim)
        env_ah = int(self.action_horizon)
        env_ad = int(self.action_dim)
        model_flat = model_ah * model_ad
        env_flat = env_ah * env_ad
        noise_full = jnp.zeros((batch_size, model_ah, model_ad), dtype=jnp.float32)
        if noise_jax.ndim == 3:
            # Already (B, H, D); follow the caller's shape instead of truncating to env dims.
            cfg_ah = min(int(noise_jax.shape[1]), model_ah)
            cfg_ad = min(int(noise_jax.shape[2]), model_ad)
            noise_chunk = noise_jax[:, :cfg_ah, :cfg_ad]
        elif int(noise_jax.shape[-1]) == model_flat:
            # MODEL layout, e.g. the DSRL noise actor (``actor_action_dim * action_horizon``).
            cfg_ah, cfg_ad = model_ah, model_ad
            noise_chunk = noise_jax.reshape(batch_size, model_ah, model_ad)
        elif int(noise_jax.shape[-1]) == env_flat:
            # ENV layout: noise on the real driving dims only.
            cfg_ah, cfg_ad = min(env_ah, model_ah), min(env_ad, model_ad)
            noise_chunk = noise_jax.reshape(batch_size, env_ah, env_ad)[:, :cfg_ah, :cfg_ad]
        else:
            raise ValueError(
                f"SteerVLAActor received noise of shape {tuple(noise_jax.shape)}. Expected 3-D "
                f"(batch, horizon, dim), or a flat width of {model_flat} (model layout "
                f"{model_ah}x{model_ad}) or {env_flat} (env layout {env_ah}x{env_ad}). Silently "
                f"zero-filling the flow initialization is not safe."
            )
        noise_full = noise_full.at[:, :cfg_ah, :cfg_ad].set(noise_chunk)
        
        # Context noise level (CSP). Derived with fold_in so the CoT/action rng streams above are
        # unchanged when the feature is off. Passed only when set, so a non-CSP openpi still works.
        t_context = self._resolve_t_context(jax.random.fold_in(rng, 7), batch_size)
        t_context_kw = {} if t_context is None else {"t_context": t_context}
        if t_context is not None:
            print(f"[DEBUG - steervla] t_context: {np.asarray(self._last_t_context).round(3).tolist()}")

        # Sample the actions (standard or QGF-guided). QGF (routing-commands) runs its own
        # denoising loop over the prefix cache, so it bypasses _sample_actions_cached.
        sample_actions_time = time.time()
        if self._qgf_config is not None:
            traj = self._qgf_guided_denoise(obs_full, noise_full, batch_size)
        else:
            traj = self._sample_actions_cached(
            rng_act,
            obs_full,
            noise=noise_full,
            num_steps=int(self.sample_actions_num_steps),
            image_keys=CARLA_STEERVLA_IMAGE_KEYS,
                **t_context_kw,
        )
        jax.block_until_ready(traj)
        sample_actions_time = time.time() - sample_actions_time

        mode_label = "QGF" if self._qgf_config is not None else "standard"
        print(f"[DEBUG - steervla] Sample actions ({mode_label}) time: {sample_actions_time} seconds")

        # Baseline capture: save final unguided action + Q-value at the capture step.
        # Stored in _baseline_capture_holder so main_carla.py can retrieve it.
        if hasattr(self, "_baseline_capture_holder") and self._baseline_capture_holder is not None:
            ah = int(self.model.action_horizon)
            ad = int(self.model.action_dim)
            a_flat = np.array(traj[0, :ah, :ad].reshape(ah * ad))
            self._baseline_capture_holder["action_flat"] = a_flat
            self._baseline_capture_holder["ready"] = True
            self._baseline_capture_holder = None  # one shot

        traj_np = self._postprocess_action_trajectory(traj, observation_state=obs_jax.state)

        if self.return_normalized_action_chunk and not force_accel_steer:
            # Residual RL operates in the model's normalized action space, so return the
            # raw sampled chunk (env applies denormalize_actions via action_input_space),
            # rather than the physically-postprocessed trajectory.
            chunk = traj[:, : int(self.action_horizon), : int(self.action_dim)]
            flat = chunk.reshape(batch_size, -1)
            expected = int(self.action_horizon) * int(self.action_dim)
            if flat.shape[-1] != expected:
                raise ValueError(
                    f"SteerVLA action chunk has length {flat.shape[-1]}, expected "
                    f"{expected} (= {self.action_horizon} x {self.action_dim}). "
                    f"Sampled trajectory shape: {tuple(traj.shape)}."
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
        # Each _forward_pi0 draws its own t_context (when sample_t_context is on), so this sweep
        # searches jointly over action noise AND context noise. Record the per-candidate level so a
        # candidate's score can be attributed to one or the other after the fact.
        candidate_t_contexts: list[np.ndarray | None] = []
        for i in range(n):
            noise_i = jnp.asarray(candidate_noises[i], dtype=jnp.float32)
            out = self._forward_pi0(batch_size, noise_i * self.noise_scale, raw=None)
            score = self._debug_speed_score_from_flat(out)
            scores.append(score)
            last_t = self._last_t_context
            candidate_t_contexts.append(None if last_t is None else np.array(last_t, copy=True))
            cumsum_xy = self._debug_xy_cumsum_from_flat(out)
            xy_cumsum_steps.append(cumsum_xy)
            final_xy = cumsum_xy[-1]
            xy_avgs.append((float(final_xy[0]), float(final_xy[1])))
            if self.use_best_noise and score < best_score:
                best_score = score
                best_out = out
                best_idx = i

        # (n, b) per-candidate context noise level; all-NaN in the clean regime.
        t_context_arr = np.asarray(
            [
                np.full((batch_size,), np.nan) if t is None else np.asarray(t, dtype=np.float64)
                for t in candidate_t_contexts
            ],
            dtype=np.float64,
        )
        # _last_t_context is left pointing at the final candidate's draw; restore the executed one
        # so downstream logging/replay records the t_context that actually produced the action.
        if self.use_best_noise and best_idx >= 0 and candidate_t_contexts[best_idx] is not None:
            self._last_t_context = candidate_t_contexts[best_idx]

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

            if np.all(np.isnan(t_context_arr)):
                ax_score.scatter(range(n), scores_arr, alpha=0.75, label="candidates")
            else:
                # Color by context noise level: makes it obvious when the score spread is driven by
                # t_context rather than by the action noise this sweep is nominally searching over.
                sc = ax_score.scatter(
                    range(n),
                    scores_arr,
                    c=np.nanmean(t_context_arr, axis=1),
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    alpha=0.85,
                    label="candidates",
                )
                fig.colorbar(sc, ax=ax_score, label="t_context")
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
                candidate_t_contexts=t_context_arr,
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
        t_context_msg = ""
        if not np.all(np.isnan(t_context_arr)):
            best_t_str = (
                f"{np.nanmean(t_context_arr[best_idx]):.3f}" if best_idx >= 0 else "n/a"
            )
            t_context_msg = (
                f" t_context=[{np.nanmin(t_context_arr):.3f}, {np.nanmax(t_context_arr):.3f}] "
                f"best_t_context={best_t_str}"
            )
        print(
            f"[debug_noise] step={self.debug_noise_episode_step} candidates={n} "
            f"use_best_noise={self.use_best_noise}{t_context_msg} "
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
        # A replayed chunk ignores ``noise_jax`` entirely: with ``actions_per_model_query=k`` the
        # noise actor only influences 1 in k executed actions. main_carla logs this as
        # ``vla/action_cached`` so an RL run's true on-policy fraction is visible.
        self.last_action_was_cached = cached is not None
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
        # Full model-layout noise, matching Pi0CoT._denoise's own default
        # ``normal(rng, (batch, action_horizon, action_dim))``.
        noise_jax = jax.random.normal(
            rng_noise,
            (1, int(self.model.action_horizon), int(self.model.action_dim)),
            dtype=jnp.float32,
        )
        noise_jax = noise_jax * jnp.asarray(self.noise_scale, dtype=jnp.float32)
        actions = self._forward_pi0(1, noise_jax, raw=state, rng=rng_act, force_accel_steer=True)
        self._mark_action_served(1)
        return np.asarray(jax.device_get(actions[0]), dtype=np.float32)

    def get_cot(self, state: Dict[str, Any]) -> dict:
        """Local: run ``sample_cot`` only; returns arrays converted to NumPy."""
        if self._remote is not None:
            return self._remote.get_cot(state)

        assert self.model is not None and self.tokenizer is not None and self._jax_device is not None
        self._refresh_inference_weights()
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

    def decode_last_batch_subtasks(self) -> list[str]:
        """Per-row subtask text from the most recent batched forward pass.

        Unlike the ``[DEBUG - steervla] Subtask text:`` print (which flattens the
        whole batch into a single decode), this decodes each row separately --
        for best-of-N candidate overlays where each row is a different candidate.
        """
        if self._last_batch_subtask is None or self.tokenizer is None:
            return []
        tokens, mask = self._last_batch_subtask
        texts = []
        for i in range(tokens.shape[0]):
            row_valid = tokens[i][mask[i].astype(bool)]
            texts.append(self.tokenizer._tokenizer.decode(row_valid.tolist()))
        return texts

    def decode_last_batch_reasoning(self) -> list[str]:
        """Per-row reasoning text from the most recent batched forward pass (see
        :meth:`decode_last_batch_subtasks`)."""
        if self._last_batch_reasoning is None or self.tokenizer is None:
            return []
        tokens, mask = self._last_batch_reasoning
        texts = []
        for i in range(tokens.shape[0]):
            row_valid = tokens[i][mask[i].astype(bool)]
            texts.append(self.tokenizer._tokenizer.decode(row_valid.tolist()))
        return texts
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
        self._refresh_inference_weights()
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
        # Candidates are drawn in one batched forward, so a derailed row cannot be resampled
        # individually the way _sample_cot_checked does for the batch-1 rollout path. Flag them
        # instead: an overflowed row is garbled (see _reasoning_overflowed) and its subtask is not
        # a usable Best-of-N candidate.
        overflowed = self._reasoning_overflowed(cot_out)
        reasoning_texts: list[str] = []
        subtask_texts: list[str] = []
        for i in range(n):
            r_txt = self.tokenizer._tokenizer.decode(reason_tokens[i][reason_mask[i]].tolist())
            s_txt = self.tokenizer._tokenizer.decode(subtask_tokens[i][subtask_mask[i]].tolist())
            reasoning_texts.append(r_txt)
            subtask_texts.append(s_txt)
            print(
                f"[best_of_n][cand {i}] temp={float(temperature):.2f} "
                f"{'[GARBLED: reasoning overflowed budget] ' if bool(overflowed[i]) else ''}"
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

        # One independent context noise level per candidate: with sample_t_context the best-of-N
        # search then ranges over t_context as well as over the sampled CoT.
        t_context = self._resolve_t_context(jax.random.fold_in(rng, 7), n)
        if t_context is not None:
            print(
                f"[best_of_n] t_context per candidate: "
                f"{np.asarray(self._last_t_context).round(3).tolist()}",
                flush=True,
            )

        decode_bs = min(n, int(self.action_decode_batch_size))
        traj_parts: list[np.ndarray] = []
        for start in range(0, n, decode_bs):
            end = min(start + decode_bs, n)
            chunk_rng = jax.random.fold_in(rng_act, start)
            chunk_obs = jax.tree.map(lambda x: x[start:end], obs_full)
            chunk_noise = noise_full[start:end]
            t_context_kw = {} if t_context is None else {"t_context": t_context[start:end]}
            traj = self._sample_actions(
                chunk_rng,
                chunk_obs,
                noise=chunk_noise,
                num_steps=int(self.sample_actions_num_steps),
                image_keys=CARLA_STEERVLA_IMAGE_KEYS,
                **t_context_kw,
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
            # (n,) context noise level per candidate, or None in the clean regime.
            "t_context": None if t_context is None else np.asarray(self._last_t_context, dtype=np.float32),
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
        cot_resample_on_overflow=bool(steervla_cfg.get("cot_resample_on_overflow", True)),
        cot_overflow_max_resamples=int(steervla_cfg.get("cot_overflow_max_resamples", 2)),
        include_ego_history=bool(steervla_cfg.get("include_ego_history", False)),
        proprio_norm=bool(steervla_cfg.get("proprio_norm", True)),
        output_action_format=steervla_cfg.get("output_action_format") or "DELTA_XY_T_DELTA_XY_SPACE",
        action_horizon=int(steervla_cfg.get("action_horizon", 10)),
        action_dim=int(steervla_cfg.get("action_dim", 4)),
        actions_per_model_query=int(steervla_cfg.get("actions_per_model_query", 1)),
        actions_per_cot=int(steervla_cfg.get("actions_per_cot", 1)),
        env_steps_per_chunk_row=int(steervla_cfg.get("env_steps_per_chunk_row", 5)),
        reanchor_cached_chunk=bool(steervla_cfg.get("reanchor_cached_chunk", True)),
        sample_actions_num_steps=int(steervla_cfg.get("sample_actions_num_steps", 10)),
        action_decode_batch_size=int(steervla_cfg.get("action_decode_batch_size", 2)),
        training_gpu_rank=int(srank),
        hl_training_gpu_rank=int(steervla_cfg.get("hl_training_gpu_rank", -1)),
        load_trainable_params=bool(steervla_cfg.get("load_trainable_params", False)),
        hl_dataset_dir=steervla_cfg.get("hl_dataset_dir"),
        hl_update_every=int(steervla_cfg.get("hl_update_every", 1)),
        hl_update_batch_size=int(steervla_cfg.get("hl_update_batch_size", 2)),
        hl_update_num_steps=int(steervla_cfg.get("hl_update_num_steps", 1)),
        hl_lr=steervla_cfg.get("hl_lr"),
        hl_freeze_regexes=steervla_cfg.get("hl_freeze_regexes"),
        # HL replay pools (pretraining-data stabilization). See extract_hl_replay.py.
        hl_replay_root=steervla_cfg.get("hl_replay_root"),
        hl_replay_pools=steervla_cfg.get("hl_replay_pools"),
        hl_online_weight=float(steervla_cfg.get("hl_online_weight", 1.0)),
        hl_online_bad_fraction=float(steervla_cfg.get("hl_online_bad_fraction", -1.0)),
        hl_online_precursor_fraction=float(steervla_cfg.get("hl_online_precursor_fraction", -1.0)),
        hl_min_online_samples=int(steervla_cfg.get("hl_min_online_samples", 1)),
        hl_keep_last_rounds=int(steervla_cfg.get("hl_keep_last_rounds", 0)),
        hl_log_batch_tokens=bool(steervla_cfg.get("hl_log_batch_tokens", True)),
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
        # CSP context noise. Requires a checkpoint trained with context_smoothing enabled.
        t_context=(
            None if steervla_cfg.get("t_context") is None else float(steervla_cfg["t_context"])
        ),
        sample_t_context=bool(steervla_cfg.get("sample_t_context", False)),
        t_context_min=float(steervla_cfg.get("t_context_min", 0.0)),
        t_context_max=float(steervla_cfg.get("t_context_max", 1.0)),
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
