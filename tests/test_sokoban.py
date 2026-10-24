import numpy as np
import pytest

from envs.base import split_for_seed
from envs.sokoban import SokobanEnv, generate, move, solve_level


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


def test_generated_levels_are_solvable_and_not_already_solved():
    env = SokobanEnv(levels=((6, 1), (7, 2)))
    for t in _tasks(env, 30):
        world = generate(t.seed, *t.level)
        assert world.boxes != world.targets
        path = env.solve(t)
        assert path and len(path) <= t.budget
        assert _play(env, t, path)[0] == 1.0


def test_generation_is_deterministic_per_seed():
    a, b = generate(123, 6, 1), generate(123, 6, 1)
    assert a == b
    assert generate(124, 6, 1) != a


def test_obs_is_an_image_with_rules():
    env = SokobanEnv()
    t = _tasks(env, 1)[0]
    obs = env.reset(t)
    assert obs.rgb.shape == (288, 288, 3) and obs.rgb.dtype == np.uint8
    assert "box" in obs.text and np.array_equal(obs.rgb, env.reset(t).rgb)


def test_walls_block_boxes_get_pushed_and_never_pulled():
    walls = frozenset({(r, c) for r in range(4) for c in range(6) if r in (0, 3) or c in (0, 5)})
    boxes = frozenset({(1, 2)})
    assert move(walls, (1, 1), boxes, "up") == ((1, 1), boxes)                      # wall
    assert move(walls, (1, 1), boxes, "right") == ((1, 2), frozenset({(1, 3)}))      # push
    assert move(walls, (1, 3), frozenset({(1, 4)}), "right") == ((1, 3), frozenset({(1, 4)}))  # box against wall
    assert move(walls, (1, 3), boxes, "right") == ((1, 4), boxes)                    # walking away pulls nothing
    two = frozenset({(1, 2), (1, 3)})
    assert move(walls, (1, 1), two, "right") == ((1, 1), two)                        # box against box


def test_solving_reports_reward_one_once():
    env = SokobanEnv()
    t = _tasks(env, 1)[0]
    total, done, info = _play(env, t, env.solve(t))
    assert total == 1.0 and done and "solved" in info["events"]


def test_deadlocked_box_in_a_corner_ends_the_episode():
    env = SokobanEnv()
    for t in _tasks(env, 200):
        w = generate(t.seed, *t.level)
        env.reset(t)
        for a in env.actions:
            e = env.fork()
            e.reset(t)
            _, r, done, info = e.step(a)
            if "deadlock" in info["events"]:
                assert done and r == 0.0
                return
    pytest.skip("no one-step deadlock in 200 levels")


def test_splits_and_frozen_heldout():
    env = SokobanEnv(heldout_n=20)
    assert all(t.split == split_for_seed(t.seed) == "gate" for t in _tasks(env, 5, "gate"))
    assert env.tasks("heldout", 20, np.random.default_rng(1)) == env.tasks("heldout", 20, np.random.default_rng(9))


def test_relabel_moves_the_target_to_where_the_pushed_box_ended():
    env = SokobanEnv()
    for t in _tasks(env, 50):
        path = env.solve(t)
        world = generate(t.seed, *t.level)
        # replay until the first push, then stop: the box moved but is not on its target
        e = env.fork()
        e.reset(t)
        prefix = []
        for a in path:
            boxes = e._boxes
            e.step(a)
            prefix.append(a)
            if e._boxes != boxes:
                break
        if e._boxes == world.targets:
            continue
        new_task, actions = env.relabel(t, prefix)
        assert new_task.seed == t.seed and new_task.goal != t.goal
        assert _play(env, new_task, actions)[0] == 1.0
        return
    pytest.fail("no level where the first push leaves the box off target")


def test_relabel_none_without_a_push():
    env = SokobanEnv()
    t = _tasks(env, 1)[0]
    assert env.relabel(t, []) is None


def test_descriptor_unit_vector_and_solver_on_trivial_level():
    env = SokobanEnv()
    t = _tasks(env, 1)[0]
    d = env.descriptor(t)
    assert d.ndim == 1 and abs(np.linalg.norm(d) - 1) < 1e-5
    walls = frozenset({(r, c) for r in range(4) for c in range(5) if r in (0, 3) or c in (0, 4)})
    assert solve_level(walls, (1, 1), frozenset({(1, 2)}), frozenset({(1, 3)})) == ["right"]
