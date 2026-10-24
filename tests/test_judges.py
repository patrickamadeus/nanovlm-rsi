import pytest

from configs.config import LoopConfig
from envs.base import Task
from utils.rollout import Trajectory
from rsi import make_judge
from rsi.judges import OutcomeJudge

TASK = Task("gather_order", 1, ("red", "green", "blue"), 24, "train")


def traj(total_reward=0.0, env_steps=None, parse_fails=0, actions=None):
    actions = actions if actions is not None else ["up"] * (env_steps or 0)
    steps = [{"action": a} for a in actions]
    return Trajectory(task=TASK, steps=steps, total_reward=total_reward,
                      env_steps=len(actions) if env_steps is None else env_steps,
                      parse_fails=parse_fails, tokens_out=0)


def test_binary_success_from_env_reward():
    s = OutcomeJudge().score(traj(total_reward=1.0, actions=["up", "pick", "left"]))
    assert s.reward == 1.0 and s.verdict_source == "outcome" and s.confidence == 1.0
    assert s.flags == ()


def test_partial_credit_is_passed_through():
    assert OutcomeJudge().score(traj(total_reward=0.5, actions=["up", "pick"])).reward == 0.5


def test_flags_zero_step_success():
    assert "zero_step_success" in OutcomeJudge().score(traj(total_reward=1.0, actions=[])).flags


def test_flags_degenerate_action_loop():
    assert "action_loop" in OutcomeJudge().score(traj(actions=["up"] * 12)).flags


def test_flags_noop_spam_from_parse_fails():
    s = OutcomeJudge().score(traj(actions=[None] * 9 + ["up"], parse_fails=9))
    assert "noop_spam" in s.flags


def test_unknown_judge_is_rejected():
    with pytest.raises(ValueError):
        make_judge(LoopConfig(judge="oracle"))


def test_outcome_judge_adds_vagens_per_turn_format_reward():
    from rsi.judges import OutcomeJudge
    steps = [{"actions": ["up"], "format_ok": True}, {"actions": ["up"], "format_ok": False},
             {"actions": ["up"], "format_ok": True}]
    t = Trajectory(TASK, steps, 1.0, 3, 0, 3)
    assert OutcomeJudge(format_reward=0.1).score(t).reward == pytest.approx(1.2)
    assert OutcomeJudge().score(t).reward == 1.0
