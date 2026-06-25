"""RLT smoke test: load an RL-Token autoencoder checkpoint and run its encoder.

CPU-only and standalone -- needs neither CARLA, JAX, nor the SteerVLA model. It
validates the pieces most likely to be wrong before a full run:
  * the checkpoint loads and its saved ``cfg`` parses,
  * the vendored ``RLTokenAutoencoder`` matches the saved ``state_dict``,
  * ``encode(z, mask)`` runs and returns ``z_rl`` of width ``d_model``.

This does NOT check token-layout alignment (that needs the real SteerVLA prefix);
for that, do the short end-to-end run printed at the end of this file's docstring.

Usage::

    python impls/scripts/rlt_smoke_test.py \
        --checkpoint ~/steervla-pi/rl_token_ae/traffic_light_large_d4096_nofast/ae_step5000.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_IMPLS_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPLS_ROOT))

from encoders.rlt import load_autoencoder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Path to ae_step*.pt / ae_ckpt.pt.")
    ap.add_argument("--device", default="cpu", help="Torch device (cpu | cuda).")
    ap.add_argument("--batch", type=int, default=2, help="Dummy batch size for the encode call.")
    args = ap.parse_args()

    import torch

    ckpt_path = os.path.expanduser(args.checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"RLT checkpoint not found: {ckpt_path}")

    model, cfg = load_autoencoder(ckpt_path, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[rlt_smoke] loaded {ckpt_path}")
    print(
        f"[rlt_smoke] cfg: vla_embed_dim={cfg.vla_embed_dim} d_model={cfg.d_model} "
        f"max_seq_len={cfg.max_seq_len} enc/dec_layers={cfg.encoder_layers}/{cfg.decoder_layers} "
        f"heads={cfg.num_heads}"
    )
    print(f"[rlt_smoke] params: {n_params / 1e6:.2f}M")

    b, m, d = int(args.batch), int(cfg.max_seq_len), int(cfg.vla_embed_dim)
    rng = np.random.default_rng(0)
    z = torch.from_numpy(rng.standard_normal((b, m, d)).astype(np.float32)).to(args.device)
    mask = torch.ones(b, m, dtype=torch.bool, device=args.device)
    # Mask a random tail per row to mimic prefix padding.
    for i in range(b):
        mask[i, int(rng.integers(m // 2, m)):] = False

    with torch.no_grad():
        z_rl = model.encode(z, mask)

    print(f"[rlt_smoke] encode: z{tuple(z.shape)} mask{tuple(mask.shape)} -> z_rl{tuple(z_rl.shape)}")
    assert z_rl.shape == (b, cfg.d_model), f"expected {(b, cfg.d_model)}, got {tuple(z_rl.shape)}"
    assert torch.isfinite(z_rl).all(), "z_rl contains non-finite values"
    print(f"[rlt_smoke] OK: z_rl finite, width d_model={cfg.d_model}")


if __name__ == "__main__":
    main()
