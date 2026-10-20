import pytest

from config import LoopConfig
from envs.base import Task
from verify import OutcomeVerifier, Trajectory, make_verifier

TASK = Task("gather_order", 1, ("red", "green", "blue"), 24, "train")


def traj(total_reward=0.0, env_steps=None, parse_fails=0, actions=None):
    actions = actions if actions is not None else ["up"] * (env_steps or 0)
    steps = [{"action": a} for a in actions]
    return Trajectory(task=TASK, steps=steps, total_reward=total_reward,
                      env_steps=len(actions) if env_steps is None else env_steps,
                      parse_fails=parse_fails, tokens_out=0)


def test_binary_success_from_env_reward():
    s = OutcomeVerifier().score(traj(total_reward=1.0, actions=["up", "pick", "left"]))
    assert s.reward == 1.0 and s.verdict_source == "outcome" and s.confidence == 1.0
    assert s.flags == ()


def test_partial_credit_is_passed_through():
    assert OutcomeVerifier().score(traj(total_reward=0.5, actions=["up", "pick"])).reward == 0.5


def test_flags_zero_step_success():
    assert "zero_step_success" in OutcomeVerifier().score(traj(total_reward=1.0, actions=[])).flags


def test_flags_degenerate_action_loop():
    assert "action_loop" in OutcomeVerifier().score(traj(actions=["up"] * 12)).flags


def test_flags_noop_spam_from_parse_fails():
    s = OutcomeVerifier().score(traj(actions=[None] * 9 + ["up"], parse_fails=9))
    assert "noop_spam" in s.flags


def test_judge_verifier_not_yet_available():
    with pytest.raises(NotImplementedError, match="Slice 4"):
        make_verifier("judge", LoopConfig())
