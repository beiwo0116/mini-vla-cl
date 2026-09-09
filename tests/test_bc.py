import numpy as np
import torch

from mini_vla_cl.data.cache import EmbeddedTransition
from mini_vla_cl.model.policy import FusionPolicy
from mini_vla_cl.train.bc import train_one_epoch


def _row(action: int) -> EmbeddedTransition:
    return EmbeddedTransition(
        image_emb=np.random.RandomState(0).randn(512).astype(np.float32),
        text_emb=np.random.RandomState(1).randn(512).astype(np.float32),
        action=action,
        skill_id="goto",
    )


def test_one_epoch_loss_decreases():
    torch.manual_seed(0)
    rows = [_row(i % 7) for i in range(64)]
    policy = FusionPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss1 = train_one_epoch(policy, rows, opt, batch_size=16, device="cpu")
    loss2 = train_one_epoch(policy, rows, opt, batch_size=16, device="cpu")
    assert loss2 < loss1
