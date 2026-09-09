from mini_vla_cl.eval.metrics import (
    average_success,
    current_skill_success,
    forgetting,
    update_history_max,
)


def test_current_and_average():
    cur = {"goto": 0.9, "pickup": 0.4}
    assert current_skill_success(cur, "pickup") == 0.4
    assert abs(average_success(cur) - 0.65) < 1e-9


def test_forgetting_and_history():
    history = {"goto": 0.8}
    current = {"goto": 0.5, "pickup": 0.7}
    assert abs(forgetting(history, current, "goto") - 0.3) < 1e-9
    new_hist = update_history_max(history, current)
    assert history == {"goto": 0.8}
    assert new_hist["goto"] == 0.8
    assert new_hist["pickup"] == 0.7
