from dataclasses import dataclass

from mini_vla_cl.config import N_EVAL_EPISODES


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    env_id: str
    n_train_episodes: int
    n_eval_episodes: int
    phase: int


SKILLS: dict[str, SkillSpec] = {
    "goto": SkillSpec("goto", "BabyAI-GoToLocal-v0", 200, N_EVAL_EPISODES, 1),
    "pickup": SkillSpec("pickup", "BabyAI-PickupLoc-v0", 200, N_EVAL_EPISODES, 1),
    "open": SkillSpec("open", "BabyAI-OpenDoor-v0", 300, N_EVAL_EPISODES, 2),
    "putnext": SkillSpec("putnext", "BabyAI-PutNextLocal-v0", 300, N_EVAL_EPISODES, 2),
}

WEEK4_SKILL_IDS: tuple[str, ...] = ("goto", "pickup")
