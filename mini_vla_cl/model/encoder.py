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


def _to_unit_features(feats, projection, torch):
    """Normalize CLIP features across transformers versions.

    Some builds return a tensor; others wrap already-projected 512-d vectors
    in a pooling object. Only run the Linear projection when the last dim
    matches ``projection.in_features``.
    """
    if torch.is_tensor(feats):
        x = feats
    else:
        x = getattr(feats, "image_embeds", None)
        if x is None:
            x = getattr(feats, "text_embeds", None)
        if x is None:
            x = getattr(feats, "pooler_output", None)
        if x is None:
            x = feats[1]
        if not torch.is_tensor(x):
            raise TypeError(f"CLIP features are not a tensor: {type(x)}")
        in_dim = projection.in_features
        out_dim = projection.out_features
        if x.shape[-1] == in_dim:
            x = projection(x)
        elif x.shape[-1] != out_dim:
            raise ValueError(
                f"CLIP feature dim {x.shape[-1]} matches neither "
                f"projection in={in_dim} nor out={out_dim}"
            )
    return x / x.norm(dim=-1, keepdim=True)


class ClipEncoder:
    def __init__(self, model_id: str | None = None, device: str = "cpu") -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.embed_dim = EMBED_DIM
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(
            model_id or CLIP_MODEL_ID, use_fast=False
        )
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
        pixel_values = inputs["pixel_values"].to(self.device)
        with self._torch.no_grad():
            feats = self.model.get_image_features(pixel_values=pixel_values)
            feats = _to_unit_features(feats, self.model.visual_projection, self._torch)
        return feats.detach().cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        kwargs = {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
        }
        with self._torch.no_grad():
            feats = self.model.get_text_features(**kwargs)
            feats = _to_unit_features(feats, self.model.text_projection, self._torch)
        return feats.detach().cpu().numpy().astype(np.float32)
