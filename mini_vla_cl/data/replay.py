import numpy as np

from mini_vla_cl.config import REPLAY_KEEP_FRAC
from mini_vla_cl.data.cache import EmbeddedTransition


class ReplayBuffer:
    def __init__(
        self,
        keep_frac: float = REPLAY_KEEP_FRAC,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.keep_frac = keep_frac
        self.rng = rng if rng is not None else np.random.default_rng()
        self._pool: list[EmbeddedTransition] = []

    def add_task(self, rows: list[EmbeddedTransition]) -> None:
        if not rows:
            return
        k = max(1, int(round(len(rows) * self.keep_frac)))
        k = min(k, len(rows))
        idx = self.rng.choice(len(rows), size=k, replace=False)
        self._pool.extend(rows[int(i)] for i in idx)

    def build_trainset(
        self, current: list[EmbeddedTransition]
    ) -> list[EmbeddedTransition]:
        return list(current) + list(self._pool)
