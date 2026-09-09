from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from mini_vla_cl.config import CLIP_MODEL_ID, EMBED_DIM


class Encoder(Protocol):
    embed_dim: int

    def encode_images(self, rgbs: list[np.ndarray]) -> np.ndarray: ...

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


def _digest_to_vec(data: bytes, embed_dim: int) -> np.ndarray:
    digest = hashlib.sha256(data).digest()
    raw = np.frombuffer((digest * ((embed_dim // len(digest)) + 1))[:embed_dim], dtype=np.uint8)
    vec = raw.astype(np.float32) - 127.5
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


class FakeEncoder:
    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        self.embed_dim = embed_dim

    def encode_images(self, rgbs: list[np.ndarray]) -> np.ndarray:
        return np.stack([_digest_to_vec(rgb.tobytes(), self.embed_dim) for rgb in rgbs])

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack(
            [_digest_to_vec(text.encode("utf-8"), self.embed_dim) for text in texts]
        )


class ClipEncoder:
    def __init__(self, model_id: str | None = None, device: str = "cpu") -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.embed_dim = EMBED_DIM
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_id or CLIP_MODEL_ID)
        self.model = CLIPModel.from_pretrained(model_id or CLIP_MODEL_ID)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._torch = torch

    def encode_images(self, rgbs: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        images = [Image.fromarray(rgb).convert("RGB").resize((224, 224)) for rgb in rgbs]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype(np.float32)
