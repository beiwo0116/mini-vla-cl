from mini_vla_cl.env.expert import collect_expert_episode


def test_collect_one_success_goto():
    traj = collect_expert_episode("goto", seed=0)
    assert traj is not None
    assert len(traj) >= 1
    assert traj[0].skill_id == "goto"
    assert traj[0].rgb.ndim == 3
    assert isinstance(traj[0].mission, str)
    assert 0 <= traj[0].action <= 6
