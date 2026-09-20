from types import SimpleNamespace

import numpy as np
import torch

from mini_vla_cl.model.encoder import FakeEncoder, _to_unit_features


def test_fake_encoder_deterministic_and_shape():
    enc = FakeEncoder(embed_dim=512)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img2 = img.copy()
    img2[0, 0] = 255
    a = enc.encode_images([img, img])
    b = enc.encode_texts(["go to the red ball", "go to the red ball"])
    assert a.shape == (2, 512) and b.shape == (2, 512)
    assert np.allclose(a[0], enc.encode_images([img])[0])
    assert not np.allclose(enc.encode_images([img])[0], enc.encode_images([img2])[0])
    assert np.allclose(b[0], b[1])


def test_to_unit_features_accepts_pooling_output():
    pooled = torch.ones(2, 4)
    proj = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        proj.weight.copy_(torch.eye(4))
    out = _to_unit_features(SimpleNamespace(pooler_output=pooled), proj, torch)
    assert torch.is_tensor(out)
    assert out.shape == (2, 4)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2))
