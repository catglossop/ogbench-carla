"""Frozen HF SigLIP2 image (+ routing-command) state encoder."""

from __future__ import annotations

import time

import numpy as np

from .base import StateEncoder

# CARLA ego state layout: index 15 is speed (m/s). Matches vlas.steervla and
# main_carla._EGO_STATE_IDX_SPEED.
_STATE_IDX_SPEED = 15


class SiglipPoolEncoder(StateEncoder):
    """Frozen HF SigLIP2 ``[image_embed, prompt_embed]`` state (no Gemma LLM).

    The prompt is the base policy's clean instruction (``"The current speed is X
    m/s. <routing_command>"``); image and text share SigLIP's aligned space. Set
    ``include_prompt=False`` for the image-only lower-bound ablation.
    """

    name = "siglip_pool"
    cot_dependent = False

    def __init__(self, steervla_actor, *, model_id, device, include_prompt=True):
        self._actor = steervla_actor
        self._include_prompt = bool(include_prompt)

        from utils.siglip_encoder import SigLIPEncoder

        self._siglip = SigLIPEncoder(model_id=str(model_id), device=device)
        self._siglip.setup()

    def _prompt_text(self, obs: dict) -> str:
        """Clean routing instruction (mirrors the base policy's prompt_text)."""
        from vlas.steervla import routing_instruction_prompt

        state = np.asarray(obs.get("state", []), dtype=np.float32).reshape(-1)
        speed = float(state[_STATE_IDX_SPEED]) if state.size > _STATE_IDX_SPEED else 0.0
        routing = str(obs.get("routing_command") or "").strip()
        if not routing:
            routing = str(getattr(self._actor, "routing_command", "") or "").strip() or "Follow the route."
        return routing_instruction_prompt(routing_command=routing, current_speed_mps=speed)

    def encode(self, obs: dict) -> np.ndarray:
        return self.encode_timed(obs)[0]

    def encode_timed(self, obs: dict) -> tuple[np.ndarray, dict[str, float]]:
        t0 = time.perf_counter()
        img_embed = self._siglip.encode(obs["image"])
        t1 = time.perf_counter()
        if not self._include_prompt:
            out = np.asarray(img_embed, dtype=np.float32).reshape(-1)
            return out, {"vision": t1 - t0, "total": t1 - t0}
        txt_embed = self._siglip.encode_text(self._prompt_text(obs))
        t2 = time.perf_counter()
        out = np.concatenate([img_embed, txt_embed], axis=-1).astype(np.float32).reshape(-1)
        return out, {"vision": t1 - t0, "text": t2 - t1, "total": t2 - t0}
