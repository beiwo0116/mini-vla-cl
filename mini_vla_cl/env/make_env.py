import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper

from mini_vla_cl.env.skills import SKILLS


def make_skill_env(skill_id: str, seed: int | None = None) -> gym.Env:
    spec = SKILLS[skill_id]
    env = gym.make(spec.env_id, render_mode="rgb_array")
    env = RGBImgObsWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env
