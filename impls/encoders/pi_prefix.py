"""Frozen SteerVLA prefix state encoders: one global mean pool, and a per-group variant."""

from __future__ import annotations

import time

import jax
import numpy as np

from .base import StateEncoder


class PiPrefixPoolEncoder(StateEncoder):
    """Mean-pooled frozen prefix over ``[image, prompt, reasoning, subtask]`` (FAST excluded).

    Folds in the CoT the base policy sampled, so it is CoT-dependent (one feature per EXPO
    candidate). The actor reuses the prefix cached by ``sample_actions_with_prefix`` when
    available, so this is usually a slice + pool rather than a second PaliGemma forward.
    """

    name = "pi_prefix"
    cot_dependent = True

    def __init__(self, steervla_actor):
        self._actor = steervla_actor

    def encode(self, obs: dict) -> np.ndarray:
        return self.encode_timed(obs)[0]

    def encode_timed(self, obs: dict) -> tuple[np.ndarray, dict[str, float]]:
        t0 = time.perf_counter()
        # Raw obs: the actor reuses the cached prefix and only rebuilds the obs on the recompute
        # fallback. device_get forces the async prefix work to finish for an honest timing.
        feat = self._actor.encode_prefix_features(obs)
        out = np.asarray(jax.device_get(feat), dtype=np.float32).reshape(-1)
        dt = time.perf_counter() - t0
        return out, {"vlm": dt, "total": dt}


class PiPrefixGroupsEncoder(StateEncoder):
    """Per-group mean pools of the prefix -- concat of masked-mean over ``[image, prompt, reasoning,
    subtask]`` (4*D, FAST excluded).

    Same frozen VLM backbone + sampled CoT as :class:`PiPrefixPoolEncoder`, but keeps the four token
    groups separate instead of one global mean, so a trainable (ideally nonlinear) state head can
    learn to weight them -- unlike a single pooled vector, where a linear head is a low-rank reparam.
    Reuses the same cached prefix (one slice + pool per candidate).
    """

    name = "pi_prefix_groups"
    cot_dependent = True

    def __init__(self, steervla_actor):
        self._actor = steervla_actor

    def encode(self, obs: dict) -> np.ndarray:
        return self.encode_timed(obs)[0]

    def encode_timed(self, obs: dict) -> tuple[np.ndarray, dict[str, float]]:
        t0 = time.perf_counter()
        feat = self._actor.encode_prefix_group_features(obs)
        out = np.asarray(jax.device_get(feat), dtype=np.float32).reshape(-1)
        dt = time.perf_counter() - t0
        return out, {"vlm": dt, "total": dt}
