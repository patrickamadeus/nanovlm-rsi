import numpy as np
import pytest

from envs.base import split_for_seed
from envs.frozenlake import FrozenLakeEnv, bfs, layout


def _tasks(env, n=20, split="train"):
    return env.tasks(split, n, np.random.default_rng(0))


def _play(env, task, actions):
    env.reset(task)
    total, done, info = 0.0, False, {}
    for a in actions:
        _, r, done, info = env.step(a)
        total += r
        if done:
            break
    return total, done, info


def test_every_task_is_solvable_and_the_solver_wins():
    env = FrozenLakeEnv(levels=((4, 3), (5, 6)))
    for t in _tasks(env, 40):
        path = env.solve(t)
        assert path and len(path) <= t.budget
        assert _play(env, t, path)[0] == 1.0


def test_reset_is_deterministic_and_obs_is_an_image():
    env = FrozenLakeEnv()
    t = _tasks(env, 1)[0]
    a, b = env.reset(t), env.reset(t)
    assert np.array_equal(a.rgb, b.rgb) and a.rgb.dtype == np.uint8 and a.rgb.shape == (192, 192, 3)
    assert "hole" in a.text


def test_hole_ends_the_episode_with_no_reward():
    size, holes = 4, frozenset({(0, 1)})
    assert bfs(size, holes, (0, 0), (3, 3)) is not None
    env = FrozenLakeEnv()
    for t in _tasks(env, 50):
        _, start, _, holes = env._world(t)
        for a, (dr, dc) in {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}.items():
            if (start[0] + dr, start[1] + dc) in holes:
                total, done, info = _play(env, t, [a])
                assert total == 0.0 and done and info["events"] == ["fell"]
                return
    pytest.fail("no task with a hole next to the start in 50 draws")


def test_budget_ends_the_episode():
    env = FrozenLakeEnv()
    t = _tasks(env, 1)[0]
    env.reset(t)
    for i in range(t.budget):
        _, _, done, _ = env.step("left" if i % 2 else "right")
        if done:
            break
    assert done


def test_tasks_respect_splits_and_heldout_is_frozen():
    env = FrozenLakeEnv(heldout_n=30)
    assert all(t.split == split_for_seed(t.seed) == "gate" for t in _tasks(env, 10, "gate"))
    a = env.tasks("heldout", 30, np.random.default_rng(1))
    b = env.tasks("heldout", 30, np.random.default_rng(2))
    assert a == b


def test_level_is_carried_on_the_task_and_changes_the_world():
    env = FrozenLakeEnv(levels=((5, 6),))
    t = _tasks(env, 1)[0]
    assert t.level == (5, 6) and env.reset(t).rgb.shape == (240, 240, 3)
    assert len(layout(t.seed, 5, 6)[2]) == 6
    assert env.task_key(t) == "frozenlake:5:6"
    assert env.make_task(t.seed, (6, 8)).level == (6, 8)


def test_relabel_turns_a_failed_attempt_into_a_success_for_where_it_ended():
    env = FrozenLakeEnv()
    for t in _tasks(env, 50):
        path = env.solve(t)
        if len(path) < 3:
            continue
        attempt = path[:-1]                              # stops one step short: a failure
        assert _play(env, t, attempt)[0] == 0.0
        new_task, actions = env.relabel(t, attempt)
        assert new_task.seed == t.seed and new_task.split == t.split and new_task.goal != t.goal
        assert _play(env, new_task, actions)[0] == 1.0
        return
    pytest.fail("no task with an optimal path of 3+ steps")


def test_relabel_returns_none_when_the_attempt_never_moved():
    env = FrozenLakeEnv()
    t = _tasks(env, 1)[0]
    assert env.relabel(t, []) is None


def test_descriptor_is_a_unit_vector_and_fork_is_independent():
    env = FrozenLakeEnv()
    t1, t2 = _tasks(env, 2)
    d = env.descriptor(t1)
    assert d.shape == (27,) and abs(np.linalg.norm(d) - 1) < 1e-5
    other = env.fork()
    env.reset(t1)
    other.reset(t2)
    env.step(env.solve(t1)[0])
    assert other._agent == other._world(t2)[1]


def test_levels_parse_from_yaml_lists_and_cli_strings():
    from configs.config import load_config
    from envs import make_env
    assert FrozenLakeEnv(levels=("5x6",)).levels == ((5, 6),)
    env = make_env(load_config(None, env="frozenlake", frozenlake_levels=[[5, 6], [4, 3]]))
    assert env.levels == ((5, 6), (4, 3))
