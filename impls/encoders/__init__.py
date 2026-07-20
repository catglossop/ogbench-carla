"""State encoders for residual RL: ``config.state_encoder`` -> :class:`StateEncoder`.

Each residual-RL run variant is a different encoder behind one config knob; the
agent / loop / buffer are unchanged. Builders import their implementation lazily
so heavyweight or optional deps (e.g. Torch for ``rl_token``) are only pulled in
when that encoder is actually selected.
"""

from __future__ import annotations

from .base import StateEncoder


def build_state_encoder(config, steervla_actor) -> StateEncoder:
    """Construct the :class:`StateEncoder` named by ``config.state_encoder``."""
    name = str(config.get("state_encoder", "pi_prefix"))

    if name == "pi_prefix":
        from .pi_prefix import PiPrefixPoolEncoder

        return PiPrefixPoolEncoder(steervla_actor)

    if name == "siglip_pool":
        from .siglip_pool import SiglipPoolEncoder

        siglip_cfg = config.get("siglip", {}) or {}
        return SiglipPoolEncoder(
            steervla_actor,
            model_id=str(siglip_cfg.get("model_id", "google/siglip2-so400m-patch14-384")),
            device=str(siglip_cfg.get("device", "cuda")),
            include_prompt=bool(siglip_cfg.get("include_prompt", True)),
        )

    if name == "rl_token":
        from .rlt import RLTokenEncoder

        rlt_cfg = config.get("rl_token", {}) or {}
        return RLTokenEncoder(
            steervla_actor,
            checkpoint_path=str(rlt_cfg.get("checkpoint_path", "")),
            device=str(rlt_cfg.get("device", "cpu")),
        )

    raise ValueError(
        f"Unknown state_encoder {name!r}; expected one of: pi_prefix, siglip_pool, rl_token."
    )


__all__ = ["StateEncoder", "build_state_encoder"]
