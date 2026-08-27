"""State-encoder seam for residual RL on CARLA / SteerVLA.

An encoder maps a single CARLA gym observation to a 1-D ``float32`` RL state
vector. Encoders are the *only* thing that differs between residual-RL run
variants (``pi_prefix`` / ``siglip_pool`` / ``rl_token`` / ...). The agent,
online loop, and replay buffer are encoder-agnostic: the state width is probed
once at startup by calling :meth:`StateEncoder.encode` on the first obs.
"""

from __future__ import annotations

import abc
import time

import numpy as np


class StateEncoder(abc.ABC):
    """Map a CARLA gym observation to a 1-D ``float32`` RL state vector."""

    #: Short identifier, used in logging / run names.
    name: str = "base"

    #: Does the state depend on the sampled CoT? True for token encoders (pi_prefix / rl_token),
    #: False for perception-only (siglip_pool). EXPO encodes one state per candidate iff True.
    cot_dependent: bool = True

    @abc.abstractmethod
    def encode(self, obs: dict) -> np.ndarray:
        """Return a 1-D ``float32`` state vector for one env observation."""
        raise NotImplementedError

    def encode_timed(self, obs: dict) -> tuple[np.ndarray, dict[str, float]]:
        """Encode + return a per-phase wall-time breakdown (seconds) for speed profiling.

        The default just reports ``{"total": ...}``. Encoders override to expose sub-phases
        (e.g. ``obs_build`` / ``vlm`` / ``ae``) so different encoders can be compared. Sub-phase
        timers must force any async device work to materialize (e.g. ``jax.device_get`` /
        ``.cpu().numpy()``) inside the timed region, or the number is meaningless.
        """
        t0 = time.perf_counter()
        vec = self.encode(obs)
        return vec, {"total": time.perf_counter() - t0}
