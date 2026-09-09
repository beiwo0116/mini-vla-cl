import numpy as np

from mini_vla_cl.data.cache import EmbeddedTransition
from mini_vla_cl.data.replay import ReplayBuffer


def _rows(n: int, skill: str) -> list[EmbeddedTransition]:
    z = np.zeros(512, dtype=np.float32)
    return [EmbeddedTransition(z, z, action=0, skill_id=skill) for _ in range(n)]


def test_keep_two_percent():
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(keep_frac=0.02, rng=rng)
    buf.add_task(_rows(100, "goto"))
    mixed = buf.build_trainset(_rows(50, "pickup"))
    assert len(mixed) == 50 + 2
    assert sum(r.skill_id == "goto" for r in mixed) == 2
