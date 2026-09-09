import numpy as np

from mini_vla_cl.data.cache import EmbeddedTransition
from mini_vla_cl.data.replay import ReplayBuffer
from mini_vla_cl.train.sequential import mix_for_skill


def test_naive_vs_replay_mix():
    z = np.zeros(512, np.float32)
    goto = [EmbeddedTransition(z, z, 0, "goto") for _ in range(100)]
    pickup = [EmbeddedTransition(z, z, 1, "pickup") for _ in range(50)]
    naive = mix_for_skill("naive", current=pickup, buffer=ReplayBuffer(0.02))
    assert len(naive) == 50
    buf = ReplayBuffer(0.02, rng=np.random.default_rng(0))
    buf.add_task(goto)
    mixed = mix_for_skill("replay", current=pickup, buffer=buf)
    assert len(mixed) == 52
