from __future__ import annotations

import torch

from mini_vla_cl.config import BATCH_SIZE, LR, N_EPOCHS, N_EVAL_EPISODES
from mini_vla_cl.data.cache import EmbeddedTransition
from mini_vla_cl.data.replay import ReplayBuffer
from mini_vla_cl.eval.metrics import average_success, forgetting, update_history_max
from mini_vla_cl.eval.rollout import rollout_success
from mini_vla_cl.model.policy import FusionPolicy
from mini_vla_cl.train.bc import train_one_epoch


def mix_for_skill(
    method: str,
    current: list[EmbeddedTransition],
    buffer: ReplayBuffer,
) -> list[EmbeddedTransition]:
    if method == "naive":
        return list(current)
    if method == "replay":
        return buffer.build_trainset(current)
    raise ValueError(f"unknown method: {method}")


def run_method(
    method: str,
    skill_ids: list[str],
    demos: dict[str, list[EmbeddedTransition]],
    encoder,
    n_epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    device: str = "cpu",
    eval_episodes: int = N_EVAL_EPISODES,
    seed: int = 42,
) -> dict:
    if method not in {"naive", "replay"}:
        raise ValueError(f"unsupported method for week-4 path: {method}")
    policy = FusionPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    buffer = ReplayBuffer()
    history_max: dict[str, float] = {}
    after_skill: list[dict] = []

    for i, skill_id in enumerate(skill_ids):
        current = demos[skill_id]
        train_rows = mix_for_skill(method, current, buffer)
        for _ in range(n_epochs):
            train_one_epoch(policy, train_rows, optimizer, batch_size, device)
        if method == "replay":
            buffer.add_task(current)

        seen = skill_ids[: i + 1]
        success = {
            sid: rollout_success(
                policy, encoder, sid, eval_episodes, seed=seed + i * 1000, device=device
            )
            for sid in seen
        }
        history_max = update_history_max(history_max, success)
        after_skill.append(
            {
                "just_learned": skill_id,
                "success": success,
                "average": average_success(success),
                "forgetting_goto": (
                    forgetting(history_max, success, "goto") if "goto" in success else None
                ),
            }
        )
    return {"method": method, "after_skill": after_skill, "policy": policy}
