from mini_vla_cl.config import EMBED_DIM, N_ACTIONS, REPLAY_KEEP_FRAC, SEED


def test_frozen_hyperparams():
    assert SEED == 42
    assert REPLAY_KEEP_FRAC == 0.02
    assert N_ACTIONS == 7
    assert EMBED_DIM == 512
