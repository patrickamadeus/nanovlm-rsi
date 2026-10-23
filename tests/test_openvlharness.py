import numpy as np
import pytest

from configs.config import LoopConfig
from envs.static import StaticEnv
from harness.openvlharness import ovh_rollout


class FakeAgent:
    """Stands in for openvlharness MainAgent: records the question, returns a fixed result."""
    def __init__(self, prediction, tool_calls=("Zoom_In_Tool",)):
        self.prediction, self.tool_calls, self.messages = prediction, tool_calls, []

    def generate(self, message, task_id=None):
        self.messages.append(message)
        turns = [{"turn": 1, "assistant": {"content": "<thinking>zoom</thinking>",
                                           "tool_calls": [{"name": n} for n in self.tool_calls]}},
                 {"turn": 2, "assistant": {"content": f"<answer>{self.prediction}</answer>", "tool_calls": []}}]
        return {"prediction": self.prediction,
                "extra_records": {"trajectory": {"chat_history": turns, "final_answer": self.prediction},
                                  "usage": {"total": {"prompt_tokens": 900, "completion_tokens": 40}}}}


def task_with_goal(env):
    return env.tasks("heldout", 1, np.random.default_rng(0))[0]


def test_a_correct_final_answer_is_stepped_into_the_env_and_rewarded(tmp_path):
    env = StaticEnv(families=("count_objects",))
    task = task_with_goal(env)
    traj = ovh_rollout(FakeAgent(task.goal[0]), env, task, LoopConfig(), image_dir=str(tmp_path))
    assert traj.total_reward == 1.0 and traj.env_steps == 1 and traj.parse_fails == 0
    assert traj.tokens_in == 900 and traj.tokens_out == 40
    assert traj.steps[-1]["action"] == task.goal[0] and traj.steps[-1]["tools"] == ["Zoom_In_Tool"]


def test_question_carries_the_image_the_task_text_and_recalled_memory(tmp_path):
    env = StaticEnv(families=("count_objects",))
    task = task_with_goal(env)
    agent = FakeAgent("3")
    ovh_rollout(agent, env, task, LoopConfig(), memory_text="Skills that worked before:\n- x -> 3",
                image_dir=str(tmp_path))
    (image, text), = agent.messages
    assert image["type"] == "image" and image["value"].endswith(".png")
    assert env.reset(task).text in text["value"] and "Skills that worked before" in text["value"]


def test_no_parseable_answer_is_a_parse_failure_not_an_env_call(tmp_path):
    env = StaticEnv(families=("count_objects",))
    traj = ovh_rollout(FakeAgent("Error: Reached the iteration limit without a final answer."), env,
                       task_with_goal(env), LoopConfig(), image_dir=str(tmp_path))
    assert traj.parse_fails == 1 and traj.env_steps == 0 and traj.total_reward == 0.0


def test_openvlharness_only_runs_single_answer_tasks_and_prompt_level_updates():
    from harness.openvlharness import check_config
    check_config(LoopConfig(harness="openvlharness", update="memory", static_families=("count_objects",)))
    with pytest.raises(ValueError, match="token ids"):
        check_config(LoopConfig(harness="openvlharness", update="grpo", static_families=("count_objects",)))
    with pytest.raises(ValueError, match="single-answer"):
        check_config(LoopConfig(harness="openvlharness", update="memory"))   # gather_order is multi-step
