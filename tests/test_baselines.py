import numpy as np

from configs.config import LoopConfig
from envs.frozenlake import FrozenLakeEnv
from models.baselines import RandomPolicy
from utils.rollout import rollout_group


def test_random_policy_only_takes_legal_actions_and_sometimes_wins():
    env = FrozenLakeEnv()
    tasks = env.tasks("train", 200, np.random.default_rng(0))
    trajs = rollout_group(RandomPolicy(0), env, tasks, LoopConfig())
    assert sum(t.parse_fails for t in trajs) == 0
    rate = np.mean([t.total_reward for t in trajs])
    assert 0.0 < rate < 0.8
