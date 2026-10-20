import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("craftax", reason="install the [craftax] extra")

from envs.base import split_for_seed  # noqa: E402
from envs.craftax import GOALS, CraftaxEnv, achievement_reward  # noqa: E402


@pytest.fixture(scope="module")
def env():
    return CraftaxEnv(env_seed=0, heldout_n=20, budget=8)


def test_jax_is_quarantined_to_the_craftax_module():
    # importing the loop, evaluation and the static env must never load JAX
    out = subprocess.run([sys.executable, "-c", "import rsi, evaluate, sys;"
                          "print(any(m == 'jax' or m.startswith('jax.') for m in sys.modules))"],
                         capture_output=True, text=True, check=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.stdout.strip() == "False"


def test_obs_is_uint8_rgb(env):
    task = env.tasks("train", 1, np.random.default_rng(0))[0]
    obs = env.reset(task)
    assert obs.rgb.dtype == np.uint8 and obs.rgb.ndim == 3 and obs.rgb.shape[2] == 3
    assert task.goal[0].replace("_", " ") in obs.text


def test_reset_is_deterministic_by_seed(env):
    task = env.tasks("train", 1, np.random.default_rng(0))[0]
    assert np.array_equal(env.reset(task).rgb, env.reset(task).rgb)


def test_actions_are_the_17_craftax_classic_actions(env):
    assert len(env.actions) == 17 and "do" in env.actions and "noop" in env.actions


def test_tasks_respect_splits_and_heldout_is_frozen(env):
    for split in ("train", "gate", "heldout"):
        ts = env.tasks(split, 5, np.random.default_rng(3))
        assert {t.split for t in ts} == {split} and all(split_for_seed(t.seed) == split for t in ts)
        assert all(t.goal[0] in GOALS and t.family == "craftax_achieve" for t in ts)
    a = env.tasks("heldout", 20, np.random.default_rng(1))
    assert a == CraftaxEnv(env_seed=0, heldout_n=20, budget=8).tasks("heldout", 20, np.random.default_rng(2))


def test_budget_terminates_the_episode(env):
    env.reset(env.tasks("train", 1, np.random.default_rng(0))[0])
    done = False
    for _ in range(8):
        _, _, done, info = env.step("noop")
    assert done and info["steps"] == 8


def test_reward_is_paid_once_when_the_goal_achievement_first_unlocks():
    before, after = np.zeros(22, bool), np.zeros(22, bool)
    after[0] = True                                    # COLLECT_WOOD
    assert achievement_reward(before, after, 0) == 1.0
    assert achievement_reward(after, after, 0) == 0.0  # already had it
    assert achievement_reward(before, after, 3) == 0.0 # a different achievement
