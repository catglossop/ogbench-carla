"""State encoders for residual RL: ``config.state_encoder`` -> :class:`StateEncoder`.

The agent / loop / buffer are encoder-agnostic; the state width is probed once at
startup. Builders import their implementation lazily so optional deps are only
pulled in when that encoder is actually selected.
"""

from __future__ import annotations

from .base import StateEncoder


def build_state_encoder(config, steervla_actor) -> StateEncoder:
    """Construct the :class:`StateEncoder` named by ``config.state_encoder``."""
    name = str(config.get("state_encoder", "siglip_pool"))

    if name == "siglip_pool":
        from .siglip_pool import SiglipPoolEncoder

        siglip_cfg = config.get("siglip", {}) or {}
        return SiglipPoolEncoder(
            steervla_actor,
            model_id=str(siglip_cfg.get("model_id", "google/siglip2-so400m-patch14-384")),
            device=str(siglip_cfg.get("device", "cuda")),
            include_prompt=bool(siglip_cfg.get("include_prompt", True)),
        )

    raise ValueError(f"Unknown state_encoder {name!r}; expected: siglip_pool.")


__all__ = ["StateEncoder", "build_state_encoder"]
