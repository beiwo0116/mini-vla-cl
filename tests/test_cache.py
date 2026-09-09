from pathlib import Path

import numpy as np

from mini_vla_cl.data.cache import embed_transitions, load_embedded, save_embedded
from mini_vla_cl.env.expert import RawTransition
from mini_vla_cl.model.encoder import FakeEncoder


def test_embed_roundtrip(tmp_path: Path):
    rgb = np.zeros((16, 32, 3), dtype=np.uint8)
    raw = [RawTransition(rgb=rgb, mission="go to red ball", action=2, skill_id="goto")]
    rows = embed_transitions(raw, FakeEncoder())
    assert rows[0].image_emb.shape == (512,)
    p = tmp_path / "goto.npz"
    save_embedded(p, rows)
    loaded = load_embedded(p)
    assert loaded[0].action == 2
    assert np.allclose(loaded[0].image_emb, rows[0].image_emb)
