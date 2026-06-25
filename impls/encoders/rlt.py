"""RLT state encoder: a separately trained autoencoder over the un-pooled prefix.

Same VLM-backbone tokens as ``pi_prefix``, but instead of a parameter-free
mean-pool we apply a frozen RL-Token autoencoder's encoder (Physical
Intelligence's RLT) to compress the ``[B, M, D]`` prefix into a single
learned ``z_rl`` vector of width ``d_model``.

The autoencoder is trained offline (``train_rl_token_ae.py``) on prefix
embeddings dumped from the *same* frozen SteerVLA checkpoint
(``dump_rl_token_embeddings.py``); :meth:`SteerVLAActor.encode_prefix_tokens`
reproduces that exact token layout at inference time so ``z_rl`` is meaningful.

PyTorch is imported lazily (this stack is otherwise JAX-only); inference runs on
CPU by default to avoid contending with JAX for GPU memory.
"""

from __future__ import annotations

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

    def __init__(self, steervla_actor, checkpoint_path: str, *, device: str = "cpu"):
        if not checkpoint_path:
            raise ValueError("rl_token encoder requires config.rl_token.checkpoint_path.")
        self._actor = steervla_actor
        self._device = device
        self._model, self._cfg = load_autoencoder(checkpoint_path, device)

    def encode(self, obs: dict) -> np.ndarray:
        import torch

        prefix_out, prefix_mask = self._actor.encode_prefix_tokens(obs)  # f32[1,M,D], bool[1,M]
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
        z = torch.from_numpy(np.ascontiguousarray(prefix_out)).to(self._device).float()
        mask = torch.from_numpy(np.ascontiguousarray(prefix_mask)).to(self._device).bool()
        with torch.no_grad():
            z_rl = self._model.encode(z, mask)
        return z_rl[0].cpu().numpy().astype(np.float32, copy=False)
