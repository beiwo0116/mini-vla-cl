def current_skill_success(success_by_skill: dict[str, float], skill_id: str) -> float:
    return float(success_by_skill[skill_id])


def average_success(success_by_skill: dict[str, float]) -> float:
    if not success_by_skill:
        raise ValueError("success_by_skill is empty")
    return sum(success_by_skill.values()) / len(success_by_skill)


def forgetting(
    history_max: dict[str, float], current: dict[str, float], skill_id: str
) -> float:
    return float(history_max[skill_id] - current[skill_id])


def update_history_max(
    history_max: dict[str, float], current: dict[str, float]
) -> dict[str, float]:
    out = dict(history_max)
    for k, v in current.items():
        out[k] = max(out.get(k, v), v)
    return out
