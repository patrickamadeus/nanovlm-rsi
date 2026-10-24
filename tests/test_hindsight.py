import numpy as np

from configs.config import LoopConfig
from envs.frozenlake import FrozenLakeEnv
from envs.sokoban import SokobanEnv
from rsi.hindsight import Hindsight
from rsi.judges import ScoredTrajectory
from tests.test_viskill import play

ENV = FrozenLakeEnv()


def _failure():
    for t in ENV.tasks("train", 50, np.random.default_rng(0)):
        path = ENV.solve(t)
        if len(path) >= 3:
            batch, _ = play([t], [path[:-1]])
            if batch[0].reward == 0.0:
                return t, batch
    raise AssertionError("no short failure found")


def test_a_failure_gains_a_relabelled_success_with_matching_frames():
    t, batch = _failure()
    out = Hindsight(LoopConfig()).signal(batch, ENV)
    assert len(out) == 2 and out[0] is batch[0]
    copy = out[1]
    assert copy.origin == "hindsight" and copy.reward == 1.0 and copy.traj.total_reward == 1.0
    assert copy.traj.task.goal != t.goal and copy.traj.task.seed == t.seed
    first = ENV.fork()
    assert np.array_equal(copy.traj.steps[0]["images"][-1], first.reset(copy.traj.task).rgb)   # the new goal is drawn
    assert copy.traj.steps[0]["token_ids"] == batch[0].traj.steps[0]["token_ids"]


def test_successes_flags_and_copies_are_not_relabelled():
    t, batch = _failure()
    h = Hindsight(LoopConfig())
    won = [ScoredTrajectory(batch[0].traj, 1.0, None, "outcome", 1.0, ())]
    flagged = [ScoredTrajectory(batch[0].traj, 0.0, None, "outcome", 1.0, ("action_loop",))]
    assert h.signal(won, ENV) == won and h.signal(flagged, ENV) == flagged


def test_at_most_hindsight_per_task_copies_per_task():
    t, batch = _failure()
    out = Hindsight(LoopConfig(hindsight_per_task=1)).signal(batch * 4, ENV)
    assert sum(s.origin == "hindsight" for s in out) == 1


def test_grpo_ignores_hindsight_copies(tmp_path):
    from rsi.grpo import GRPO

    class P:
        version = 0
    g = GRPO.__new__(GRPO)
    g.policy, g.cfg = P(), LoopConfig()
    t, batch = _failure()
    copy = Hindsight(LoopConfig()).signal(batch, ENV)[1]
    assert g.prepare([copy, copy]) is None          # only relabelled copies: no attempts, no signal
