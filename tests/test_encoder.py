import numpy as np

from mini_vla_cl.model.encoder import FakeEncoder


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
