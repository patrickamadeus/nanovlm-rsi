import numpy as np
import pytest

from configs.config import LoopConfig, PolicyConfig
from envs.base import Task
from harness.rollout import Trajectory
from judges.outcome import OutcomeJudge
from judges.vlm import CompositeJudge, VLMJudge

IMG = np.zeros((64, 64, 3), np.uint8)
NATIVE = Task("count_objects", 2, ("3",), 1, "train")
PROPOSED = Task("proposed", 5, ("stack the red block on the blue one",), 4, "train")


class FakeJudge:
    """Stands in for the judge VLM: a fixed P(yes), and it records what it was asked."""
    def __init__(self, p_yes):
        self.p_yes, self.prompts = p_yes, []

    def p_yes_given(self, prompt, images):
        self.prompts.append(prompt)
        return self.p_yes, 10


def traj(task=NATIVE, reward=1.0):
    steps = [{"action": "3", "obs_text": "How many blue squares are there?"}]
    return Trajectory(task, steps, reward, 1, 0, 2, final_rgb=IMG)


def test_judge_verdict_comes_from_p_yes_with_confidence():
    s = VLMJudge(FakeJudge(0.8)).score(traj(reward=0.0))
    assert s.reward == 1.0 and s.verdict_source == "judge" and s.confidence == pytest.approx(0.8)
    s = VLMJudge(FakeJudge(0.3)).score(traj(reward=1.0))
    assert s.reward == 0.0 and s.confidence == pytest.approx(0.7)


def test_judge_prompt_shows_goal_context_and_the_agents_actions():
    j = FakeJudge(0.5)
    VLMJudge(j).score(traj())
    assert "How many blue squares" in j.prompts[0] and "3" in j.prompts[0]


def test_composite_routes_env_native_goals_to_outcome():
    s = CompositeJudge(OutcomeJudge(), VLMJudge(FakeJudge(0.0)), {"count_objects"},
                          sample_rate=0.0, seed=0).score(traj(reward=1.0))
    assert s.verdict_source == "outcome" and s.reward == 1.0 and s.judge_verdict is None


def test_composite_routes_goals_without_a_checker_to_the_judge():
    s = CompositeJudge(OutcomeJudge(), VLMJudge(FakeJudge(0.9)), {"count_objects"},
                          sample_rate=0.0, seed=0).score(traj(task=PROPOSED, reward=0.0))
    assert s.verdict_source == "judge" and s.reward == 1.0


def test_agreement_logging_samples_the_judge_on_native_tasks_without_changing_the_reward():
    v = CompositeJudge(OutcomeJudge(), VLMJudge(FakeJudge(0.2)), {"count_objects"},
                          sample_rate=1.0, seed=0)
    s = v.score(traj(reward=1.0))
    assert s.reward == 1.0 and s.verdict_source == "outcome"      # ground truth still decides
    assert s.judge_verdict == 0.0 and s.judge_tokens == 10        # but the disagreement is recorded


def test_sample_rate_controls_how_often_the_judge_runs():
    j = FakeJudge(0.5)
    v = CompositeJudge(OutcomeJudge(), VLMJudge(j), {"count_objects"}, sample_rate=0.1, seed=0)
    for _ in range(500):
        v.score(traj())
    assert 25 <= len(j.prompts) <= 80


def test_real_tiny_judge_returns_a_probability():
    from judges.vlm import YesNo
    j = YesNo(PolicyConfig(tiny=True, device="cpu", dtype="float32"))
    p, tokens = j.p_yes_given("Did the agent succeed?", [IMG])
    assert 0.0 <= p <= 1.0 and tokens > 0
