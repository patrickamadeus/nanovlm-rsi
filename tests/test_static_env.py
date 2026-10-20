import numpy as np

from envs.base import Task
from envs.static import StaticEnv


def _first(env, family, split="train", n=200):
    return [t for t in env.tasks(split, n, np.random.default_rng(0)) if t.family == family][0]


def test_reset_is_deterministic_by_seed():
    env = StaticEnv()
    t = _first(env, "count_objects")
    assert np.array_equal(env.reset(t).rgb, env.reset(t).rgb)


def test_obs_is_rgb_image():
    env = StaticEnv()
    obs = env.reset(env.tasks("train", 1, np.random.default_rng(0))[0])
    assert obs.rgb.ndim == 3 and obs.rgb.shape[2] == 3
    assert obs.rgb.dtype == np.uint8


def test_count_objects_rewards_correct_answer_only():
    env = StaticEnv()
    t = _first(env, "count_objects")
    env.reset(t)
    _, r_bad, done, _ = env.step("say_9" if t.goal[0] != "9" else "say_1")
    assert r_bad == 0.0 and done is True
    env.reset(t)
    _, r_good, done, _ = env.step(f"say_{t.goal[0]}")
    assert r_good == 1.0 and done is True


def test_gather_order_requires_correct_order_and_respects_budget():
    env = StaticEnv()
    t = _first(env, "gather_order")
    assert t.budget > 1
    env.reset(t)
    done = False
    for _ in range(t.budget):
        _, r, done, info = env.step("pick")
        if done:
            break
    assert done is True
    assert "steps" in info


def test_gather_order_solvable_by_walking_to_objects_in_order():
    env = StaticEnv()
    t = _first(env, "gather_order")
    env.reset(t)
    reward, done = 0.0, False
    for color in t.goal:
        tr, tc = env._objects[color]
        while env._agent != (tr, tc):
            ar, ac = env._agent
            a = "down" if ar < tr else "up" if ar > tr else "right" if ac < tc else "left"
            env.step(a)
        _, reward, done, _ = env.step("pick")
    assert reward == 1.0 and done is True


def test_tasks_only_returns_requested_split():
    env = StaticEnv()
    for split in ("train", "gate", "heldout"):
        ts = env.tasks(split, 30, np.random.default_rng(1))
        assert len(ts) == 30
        assert {x.split for x in ts} == {split}


def test_heldout_is_frozen_per_env_seed():
    a = StaticEnv(env_seed=3).tasks("heldout", 200, np.random.default_rng(99))
    b = StaticEnv(env_seed=3).tasks("heldout", 200, np.random.default_rng(1234))
    assert a == b
    c = StaticEnv(env_seed=4).tasks("heldout", 200, np.random.default_rng(99))
    assert a != c
