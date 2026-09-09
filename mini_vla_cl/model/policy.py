import torch
from torch import Tensor, nn

from mini_vla_cl.config import EMBED_DIM, N_ACTIONS


class FusionPolicy(nn.Module):
    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        hidden: int = 256,
        n_actions: int = N_ACTIONS,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, image_emb: Tensor, text_emb: Tensor) -> Tensor:
        x = torch.cat([image_emb, text_emb], dim=-1)
        return self.net(x)
