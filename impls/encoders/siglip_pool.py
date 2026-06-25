"""Frozen, mean-pooled SigLIP image feature (perception-only, no LLM)."""

from __future__ import annotations

import jax
import numpy as np

from .base import StateEncoder


class SiglipPoolEncoder(StateEncoder):
    """Mean-pool the raw camera image through the PaliGemma vision tower.

    Stops after ``_embed_images`` (the SigLIP vision encoder) and mean-pools the
    image tokens into one vector -- it does **not** run the Gemma LLM. This is a
    perception-only state that bypasses the VLM language backbone, useful as a
    lower-bound ablation against the full-prefix and RLT encoders.
    """

    name = "siglip_pool"

    def __init__(self, steervla_actor):
        self._actor = steervla_actor

    def encode(self, obs: dict) -> np.ndarray:
        openpi_obs = self._actor.build_observation_batch_numpy(1, raw=obs)
        feat = self._actor.encode_image_features(openpi_obs)
        return np.asarray(jax.device_get(feat), dtype=np.float32).reshape(-1)
