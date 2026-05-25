"""Frozen SigLIP image/text encoder for DSRL actor/critic observations."""

from __future__ import annotations

import re
import threading
from typing import Optional

import numpy as np

_LOC_TOKEN_RE = re.compile(r"<loc\d+>")


def clean_cot_text(text: str) -> str:
    """Strip Paligemma location tokens and collapse whitespace for SigLIP text input."""
    cleaned = _LOC_TOKEN_RE.sub(" ", str(text or ""))
    return re.sub(r"\s+", " ", cleaned).strip()


class SigLIPEncoder:
    """Compute L2-normalized SigLIP image and text embeddings via HuggingFace transformers."""

    def __init__(
        self,
        model_id: str = "google/siglip2-so400m-patch14-384",
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self._device = device
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._embedding_dim: Optional[int] = None

    def setup(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        device = self._device
        if device is None:
            device = "cpu"
        self._device = device

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id)
        self._model.eval()
        self._model.to(device)

        probe = np.zeros((144, 256, 3), dtype=np.uint8)
        self._embedding_dim = int(self.encode(probe).shape[-1])

    def _normalize(self, embeds):
        return embeds / embeds.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            self.setup()
        return int(self._embedding_dim)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text string into an L2-normalized embedding (zeros if empty)."""
        self.setup()
        cleaned = clean_cot_text(text)
        if not cleaned:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        import torch

        inputs = self._processor(text=[cleaned], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with self._lock:
            with torch.no_grad():
                embeds = self._model.get_text_features(**inputs)
                embeds = self._normalize(embeds)

        return embeds.detach().cpu().numpy()[0].astype(np.float32)

    def encode_observation(
        self,
        image: np.ndarray,
        *,
        prompt: str = "",
        subtask: str = "",
        include_prompt_subtask: bool = False,
    ) -> np.ndarray:
        """Image embedding, optionally concatenated with prompt/subtask text embeddings."""
        parts = [self.encode(image)]
        if include_prompt_subtask:
            parts.append(self.encode_text(prompt))
            parts.append(self.encode_text(subtask))
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def observation_dim(self, *, include_prompt_subtask: bool = False) -> int:
        mult = 3 if include_prompt_subtask else 1
        return self.embedding_dim * mult

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode HWC or BHWC uint8 RGB into L2-normalized embedding(s)."""
        self.setup()
        import torch
        from PIL import Image

        img = np.asarray(image, dtype=np.uint8)
        if img.ndim == 3:
            images = [Image.fromarray(img)]
            squeeze = True
        elif img.ndim == 4:
            images = [Image.fromarray(img[i]) for i in range(img.shape[0])]
            squeeze = False
        else:
            raise ValueError(f"Expected HWC or BHWC uint8 image, got shape {img.shape}")

        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with self._lock:
            with torch.no_grad():
                embeds = self._model.get_image_features(**inputs)
                embeds = self._normalize(embeds)

        out = embeds.detach().cpu().numpy().astype(np.float32)
        return out[0] if squeeze else out
