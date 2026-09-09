import numpy as np
import torch

from mini_vla_cl.config import MAX_EPISODE_STEPS
from mini_vla_cl.env.make_env import make_skill_env


def rollout_success(
    policy,
    encoder,
    skill_id: str,
    n_episodes: int,
    seed: int,
    device: str,
) -> float:
    policy.eval()
    wins = 0
    for i in range(n_episodes):
        env = make_skill_env(skill_id)
        obs, _info = env.reset(seed=seed + i)
        try:
            for _ in range(MAX_EPISODE_STEPS):
                rgb = np.asarray(obs["image"])
                mission = str(obs["mission"])
                image = torch.tensor(
                    encoder.encode_images([rgb]), dtype=torch.float32, device=device
                )
                text = torch.tensor(
                    encoder.encode_texts([mission]), dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    action = int(policy(image, text).argmax(dim=-1).item())
                obs, reward, terminated, truncated, _info = env.step(action)
                if terminated:
                    if float(reward) > 0 or not truncated:
                        wins += 1
                    break
                if truncated:
                    break
        finally:
            env.close()
    return wins / n_episodes if n_episodes else 0.0
