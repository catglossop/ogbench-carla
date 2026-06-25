"""Frozen, mean-pooled SteerVLA prefix feature (the default residual-RL state)."""

from __future__ import annotations

import jax
import numpy as np

from .base import StateEncoder


class PiPrefixPoolEncoder(StateEncoder):
    """Full PaliGemma forward over image + prompt, then mean-pool the prefix.

    Runs the frozen VLM backbone (vision tower + Gemma LLM) on the prefix tokens
    and mean-pools the valid hidden states into one vector (stop-gradient). The
    feature is deterministic given the obs; speed + routing command ride in via
    the prompt, so no separate proprio vector is concatenated.
    """

    name = "pi_prefix"

    def __init__(self, steervla_actor):
        self._actor = steervla_actor

    def encode(self, obs: dict) -> np.ndarray:
        openpi_obs = self._actor.build_observation_batch_numpy(1, raw=obs)
        feat = self._actor.encode_prefix_features(openpi_obs)
        return np.asarray(jax.device_get(feat), dtype=np.float32).reshape(-1)
