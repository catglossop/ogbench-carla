"""Frozen, mean-pooled SigLIP image feature (perception-only, no LLM)."""

from __future__ import annotations

import time

import jax
import numpy as np

from .base import StateEncoder


class SiglipPoolEncoder(StateEncoder):
    """Mean-pool the raw camera image through the PaliGemma vision tower.

    Stops after ``_embed_images`` (the SigLIP vision encoder) and mean-pools the
    image tokens into one vector. This is a perception-only state that bypasses the 
    VLM language backbone, useful as a lower-bound ablation against the full-prefix and RLT encoders.
    """

    name = "siglip_pool"
    cot_dependent = False  # image-only (vision tower, no LLM) -> CoT-independent state.

    def __init__(self, steervla_actor):
        self._actor = steervla_actor

    def encode(self, obs: dict) -> np.ndarray:
        return self.encode_timed(obs)[0]

    def encode_timed(self, obs: dict) -> tuple[np.ndarray, dict[str, float]]:
        t0 = time.perf_counter()
        openpi_obs = self._actor.build_observation_batch_numpy(1, raw=obs)
        t1 = time.perf_counter()
        feat = self._actor.encode_image_features(openpi_obs)
        # device_get forces the (async) SigLIP forward to finish inside the timed region.
        out = np.asarray(jax.device_get(feat), dtype=np.float32).reshape(-1)
        t2 = time.perf_counter()
        return out, {"obs_build": t1 - t0, "vision": t2 - t1, "total": t2 - t0}
