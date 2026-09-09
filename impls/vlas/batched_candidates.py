"""Validated batched SteerVLA candidate sampling for the Qwen BoN path."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


class BatchedCandidateValidationError(RuntimeError):
    """The batched actor result is unsafe to use for environment execution."""


def candidate_labels(texts: Any, count: int, source: str) -> list[str]:
    """Select decoded actor text without rewriting its natural-language content."""
    if source not in {"commentary", "subtask"}:
        raise ValueError(f"Unknown candidate label source: {source}")
    if not isinstance(texts, (list, tuple)) or len(texts) != count:
        raise BatchedCandidateValidationError(
            f"Expected {count} {source} labels from the actor."
        )
    labels = []
    for text in texts:
        if not isinstance(text, str):
            raise BatchedCandidateValidationError(f"Invalid {source} label from the actor.")
        # Both training label formats omit the actor's segment sentinels.
        text = re.sub(r"<loc\d+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip().rstrip(" ;").strip()
        if not text.strip():
            raise BatchedCandidateValidationError(f"Empty {source} label from the actor.")
        labels.append(text)
    return labels


def sample_batched_policy_candidates(
    *,
    actor: Any,
    raw: dict[str, Any],
    rng: jax.Array,
    num_candidates: int,
    model_noise_dim: int,
    env_action_dim: int,
    noise_scale: float,
    label_source: str = "subtask",
    episode_index: int | None = None,
    episode_step: int | None = None,
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

    # Keep token reuse separate from the actor's single-candidate action cache.
    # Age is measured in executed environment steps, not candidate queries.
    cache_key = (episode_index, n, label_source, raw.get("routing_command"))
    cached = getattr(actor, "_bon_cot_cache", None)
    reuse = (
        episode_step is not None and episode_index is not None
        and cached is not None and cached[0] == cache_key
        and 0 < episode_step - cached[1] < int(getattr(actor, "actions_per_cot", 1))
    )
    cot_kwargs = {"cot_out": cached[2]} if reuse else {}
    actor._bon_cot_cache = None  # Invalid results must never seed later reuse.
    reset_cache = getattr(actor, "reset_action_cache", None)
    if callable(reset_cache):
        reset_cache()
    result = actor.sample_candidates(
        n,
        temperature=float(actor.cot_temperature),
        noise=noise,
        raw=raw,
        rng=rng_cot,
        **cot_kwargs,
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

    # The commentary actor emits raw commentary on its reasoning head; its
    # subtask head emits the separate refined action summary.
    text_key = "reasoning_texts" if label_source == "commentary" else "subtask_texts"
    subtasks = candidate_labels(result.get(text_key), n, label_source)

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
    if episode_step is not None and episode_index is not None and "cot_out" in result:
        actor._bon_cot_cache = (cache_key, cached[1] if reuse else episode_step, result["cot_out"])
    return actions, subtasks
