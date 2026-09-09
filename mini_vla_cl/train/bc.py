import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mini_vla_cl.data.cache import EmbeddedTransition


def transitions_to_tensors(
    rows: list[EmbeddedTransition], device: str
) -> tuple[Tensor, Tensor, Tensor]:
    image = torch.tensor(np.stack([r.image_emb for r in rows]), dtype=torch.float32, device=device)
    text = torch.tensor(np.stack([r.text_emb for r in rows]), dtype=torch.float32, device=device)
    action = torch.tensor([r.action for r in rows], dtype=torch.long, device=device)
    return image, text, action


def train_one_epoch(
    policy: nn.Module,
    rows: list[EmbeddedTransition],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: str,
) -> float:
    policy.train()
    if not rows:
        return 0.0
    order = np.random.permutation(len(rows))
    total = 0.0
    n_batches = 0
    for start in range(0, len(rows), batch_size):
        idx = order[start : start + batch_size]
        batch = [rows[int(i)] for i in idx]
        image, text, action = transitions_to_tensors(batch, device)
        logits = policy(image, text)
        loss = F.cross_entropy(logits, action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n_batches += 1
    return total / max(n_batches, 1)
