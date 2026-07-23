"""RLT state encoder: a separately trained autoencoder over the un-pooled prefix.

Same VLM-backbone tokens as ``pi_prefix``, but instead of a parameter-free
mean-pool we apply a frozen RL-Token autoencoder's encoder to compress the 
``[B, M, D]`` prefix into a single learned ``z_rl`` vector of width ``d_model``.
"""

from __future__ import annotations

import time

import numpy as np

from .base import StateEncoder
from .rl_token_ae import RLTokenAEConfig, RLTokenAutoencoder


def load_autoencoder(checkpoint_path: str, device: str):
    import torch

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        # Older torch (no weights_only) or a checkpoint with non-tensor cfg/args.
        # The path is a local, trusted checkpoint produced by train_rl_token_ae.py.
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "model" not in ckpt:
        raise ValueError(
            f"RLT checkpoint {checkpoint_path!r} has no 'model' state_dict "
            f"(found keys: {sorted(ckpt)[:8]}...)."
        )
    cfg_kwargs = dict(ckpt.get("cfg", {}))
    cfg = RLTokenAEConfig(**cfg_kwargs) if cfg_kwargs else RLTokenAEConfig()
    model = RLTokenAutoencoder(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, cfg


class RLTokenEncoder(StateEncoder):
    """Frozen RL-Token autoencoder encoder over the un-pooled SteerVLA prefix."""

    name = "rl_token"

    def __init__(self, steervla_actor, checkpoint_path: str, *, device: str = "cuda"):
        if not checkpoint_path:
            raise ValueError("rl_token encoder requires config.rl_token.checkpoint_path.")
        self._actor = steervla_actor
        self._device = device
        self._model, self._cfg = load_autoencoder(checkpoint_path, device)

    def encode(self, obs: dict) -> np.ndarray:
        return self.encode_timed(obs)[0]

    def encode_timed(self, obs: dict) -> tuple[np.ndarray, dict[str, float]]:
        """Timed encode split into the two dominant phases:

        - ``vlm``: the SteerVLA prefix forward (full PaliGemma vision tower + Gemma LLM), forced to
          host memory so the async JAX work is captured in the timing.
        - ``ae``: the frozen RL-Token autoencoder's encoder, forced back to host.
        """
        import torch

        t0 = time.perf_counter()
        prefix_out, prefix_mask = self._actor.encode_prefix_tokens(obs)  # f32[1,M,D], bool[1,M]
        # Materialize the prefix on host (blocks the async JAX forward) so ``vlm`` time is real.
        prefix_out = np.ascontiguousarray(prefix_out)
        prefix_mask = np.ascontiguousarray(prefix_mask)
        t1 = time.perf_counter()

        m = int(prefix_out.shape[1])
        d = int(prefix_out.shape[2])
        if d != int(self._cfg.vla_embed_dim):
            raise ValueError(
                f"RLT prefix width {d} != autoencoder vla_embed_dim {self._cfg.vla_embed_dim}; "
                "the AE was trained on a different SteerVLA checkpoint."
            )
        if m > int(self._cfg.max_seq_len):
            raise ValueError(
                f"RLT prefix length {m} exceeds autoencoder max_seq_len {self._cfg.max_seq_len}; "
                "the inference token layout does not match the offline dump."
            )
        z = torch.from_numpy(prefix_out).to(self._device, non_blocking=True).float()
        mask = torch.from_numpy(prefix_mask).to(self._device, non_blocking=True).bool()
        with torch.inference_mode():
            z_rl = self._model.encode(z, mask)
        out = z_rl[0].cpu().numpy().astype(np.float32, copy=False)
        t2 = time.perf_counter()
        return out, {"vlm": t1 - t0, "ae": t2 - t1, "total": t2 - t0}
