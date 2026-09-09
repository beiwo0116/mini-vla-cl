from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mini_vla_cl.env.expert import RawTransition
from mini_vla_cl.model.encoder import Encoder


@dataclass
class EmbeddedTransition:
    image_emb: np.ndarray
    text_emb: np.ndarray
    action: int
    skill_id: str


def embed_transitions(
    raw: list[RawTransition], encoder: Encoder, batch_size: int = 16
) -> list[EmbeddedTransition]:
    out: list[EmbeddedTransition] = []
    for i in range(0, len(raw), batch_size):
        chunk = raw[i : i + batch_size]
        images = encoder.encode_images([t.rgb for t in chunk])
        texts = encoder.encode_texts([t.mission for t in chunk])
        for j, t in enumerate(chunk):
            out.append(
                EmbeddedTransition(
                    image_emb=np.asarray(images[j], dtype=np.float32),
                    text_emb=np.asarray(texts[j], dtype=np.float32),
                    action=t.action,
                    skill_id=t.skill_id,
                )
            )
    return out


def save_embedded(path: Path, rows: list[EmbeddedTransition]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        np.savez_compressed(
            path,
            image_emb=np.zeros((0, 0), dtype=np.float32),
            text_emb=np.zeros((0, 0), dtype=np.float32),
            action=np.zeros((0,), dtype=np.int64),
            skill_id=np.array([], dtype=object),
        )
        return
    np.savez_compressed(
        path,
        image_emb=np.stack([r.image_emb for r in rows]),
        text_emb=np.stack([r.text_emb for r in rows]),
        action=np.array([r.action for r in rows], dtype=np.int64),
        skill_id=np.array([r.skill_id for r in rows], dtype=object),
    )


def load_embedded(path: Path) -> list[EmbeddedTransition]:
    data = np.load(path, allow_pickle=True)
    images = data["image_emb"]
    texts = data["text_emb"]
    actions = data["action"]
    skills = data["skill_id"]
    return [
        EmbeddedTransition(
            image_emb=np.asarray(images[i], dtype=np.float32),
            text_emb=np.asarray(texts[i], dtype=np.float32),
            action=int(actions[i]),
            skill_id=str(skills[i]),
        )
        for i in range(len(actions))
    ]
