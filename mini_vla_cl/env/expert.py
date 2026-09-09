from dataclasses import dataclass

import numpy as np

from mini_vla_cl.env.make_env import make_skill_env


@dataclass
class RawTransition:
    rgb: np.ndarray
    mission: str
    action: int
    skill_id: str


def collect_expert_episode(
    skill_id: str, seed: int, max_steps: int = 240
) -> list[RawTransition] | None:
    from minigrid.utils.baby_ai_bot import BabyAIBot

    env = make_skill_env(skill_id, seed=seed)
    obs, _info = env.reset(seed=seed)
    expert = BabyAIBot(env)
    last_action = None
    traj: list[RawTransition] = []
    try:
        for _ in range(max_steps):
            action = expert.replan(last_action)
            rgb = np.asarray(obs["image"])
            mission = str(obs["mission"])
            next_obs, _reward, terminated, truncated, _info = env.step(action)
            traj.append(
                RawTransition(
                    rgb=rgb, mission=mission, action=int(action), skill_id=skill_id
                )
            )
            last_action = action
            obs = next_obs
            if terminated:
                return traj
            if truncated:
                return None
    finally:
        env.close()
    return None


def collect_skill_demos(
    skill_id: str, n_episodes: int, start_seed: int
) -> list[RawTransition]:
    out: list[RawTransition] = []
    seed = start_seed
    collected = 0
    while collected < n_episodes:
        traj = collect_expert_episode(skill_id, seed=seed)
        seed += 1
        if traj is not None:
            out.extend(traj)
            collected += 1
        if seed > start_seed + 10_000:
            raise RuntimeError(
                f"Could not collect {n_episodes} expert episodes for {skill_id}"
            )
    return out

