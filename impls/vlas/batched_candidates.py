"""Validated batched SteerVLA candidate sampling for the Qwen BoN path."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


class BatchedCandidateValidationError(RuntimeError):
    """The batched actor result is unsafe to use for environment execution."""


def sample_batched_policy_candidates(
    *,
    actor: Any,
    raw: dict[str, Any],
    rng: jax.Array,
    num_candidates: int,
    model_noise_dim: int,
    env_action_dim: int,
    noise_scale: float,
) -> tuple[np.ndarray, list[str]]:
    """Sample one normalized action-chunk batch while matching rollout noise.

    The ordinary Qwen path feeds ``tanh(N(0, 1))`` in the full Pi0 model-action
    layout to ``vla_sample_fn``.  ``SteerVLAActor.sample_candidates`` otherwise
    defaults to an unsquashed, env-dimension-only Gaussian, so construct and pass
    the full bounded noise explicitly.  Physical-unit ``actions`` are deliberately
    not accepted: CARLA applies the fixed SteerVLA denormalization downstream.
    """
    n = int(num_candidates)
    if n < 1:
        raise ValueError(f"num_candidates must be positive, got {n}")
    if not hasattr(actor, "sample_candidates"):
        raise BatchedCandidateValidationError(
            "SteerVLA actor does not provide sample_candidates()."
        )

    rng_cot, rng_noise = jax.random.split(rng)
    noise = jnp.tanh(
        jax.random.normal(rng_noise, (n, int(model_noise_dim)), dtype=jnp.float32)
    ) * jnp.asarray(noise_scale, dtype=jnp.float32)

    reset_cache = getattr(actor, "reset_action_cache", None)
    if callable(reset_cache):
        reset_cache()
    result = actor.sample_candidates(
        n,
        temperature=float(actor.cot_temperature),
        noise=noise,
        raw=raw,
        rng=rng_cot,
    )
    if not isinstance(result, Mapping):
        raise BatchedCandidateValidationError(
            f"sample_candidates() returned {type(result).__name__}, expected a mapping."
        )

    # ``actions`` are already in physical units.  Falling back to them would
    # silently denormalize twice when the environment executes the chunk.
    if "actions_normalized" not in result:
        raise BatchedCandidateValidationError(
            "sample_candidates() omitted actions_normalized; refusing to use physical-unit actions."
        )
    actions = np.asarray(jax.device_get(result["actions_normalized"]), dtype=np.float32)
    expected_shape = (n, int(env_action_dim))
    if actions.shape != expected_shape:
        raise BatchedCandidateValidationError(
            f"normalized candidate shape is {actions.shape}, expected {expected_shape}."
        )
    if not np.isfinite(actions).all():
        raise BatchedCandidateValidationError("normalized candidates contain NaN or infinity.")

    subtasks_raw = result.get("subtask_texts")
    if not isinstance(subtasks_raw, (list, tuple)) or len(subtasks_raw) != n:
        count = len(subtasks_raw) if isinstance(subtasks_raw, (list, tuple)) else 0
        raise BatchedCandidateValidationError(
            f"sample_candidates() returned {count} subtasks, expected {n}."
        )
    subtasks = [str(text) for text in subtasks_raw]
    if any(not text.strip() for text in subtasks):
        raise BatchedCandidateValidationError("sample_candidates() returned an empty subtask.")

    overflowed = result.get("reasoning_overflowed")
    if overflowed is None:
        raise BatchedCandidateValidationError(
            "sample_candidates() omitted reasoning_overflowed; cannot validate sampled CoTs."
        )
    overflowed = np.asarray(jax.device_get(overflowed), dtype=bool).reshape(-1)
    if overflowed.shape != (n,):
        raise BatchedCandidateValidationError(
            f"reasoning_overflowed shape is {overflowed.shape}, expected {(n,)}."
        )
    if overflowed.any():
        rows = np.flatnonzero(overflowed).tolist()
        raise BatchedCandidateValidationError(
            f"reasoning overflowed for candidate rows {rows}; resample them via the checked path."
        )
    return actions, subtasks
