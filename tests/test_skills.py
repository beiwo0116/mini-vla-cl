import numpy as np

from mini_vla_cl.env.make_env import make_skill_env
from mini_vla_cl.env.skills import SKILLS, WEEK4_SKILL_IDS


def test_week4_ids_and_registry():
    assert WEEK4_SKILL_IDS == ("goto", "pickup")
    assert SKILLS["goto"].env_id == "BabyAI-GoToLocal-v0"
    assert SKILLS["pickup"].env_id == "BabyAI-PickupLoc-v0"
    assert SKILLS["open"].phase == 2
    assert SKILLS["putnext"].phase == 2


def test_make_env_rgb_and_mission():
    env = make_skill_env("goto", seed=0)
    obs, info = env.reset(seed=0)
    assert env.action_space.n == 7
    assert obs["image"].ndim == 3 and obs["image"].shape[2] == 3
    assert obs["image"].dtype == np.uint8
    assert isinstance(obs["mission"], str) and len(obs["mission"]) > 0
    env.close()
