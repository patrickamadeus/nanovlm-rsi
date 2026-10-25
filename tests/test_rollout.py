import numpy as np
import pytest

from configs.config import LoopConfig
from envs.static import StaticEnv
from utils.rollout import rollout
from tests.conftest import ScriptedPolicy

cfg = LoopConfig()
env = StaticEnv()
_train = env.tasks("train", 200, np.random.default_rng(0))
count_task = next(t for t in _train if t.family == "count_objects")
gather_task = next(t for t in _train if t.family == "gather_order")


def test_rollout_records_one_step_per_action():
    tr = rollout(ScriptedPolicy([f"ACT: {count_task.goal[0]}"]), env, count_task, cfg)
    assert tr.env_steps == 1 and tr.total_reward == 1.0 and tr.parse_fails == 0


def test_parse_failure_is_a_wasted_turn_without_shaping():
    tr = rollout(ScriptedPolicy(["no action here"]), env, gather_task, cfg)
    assert tr.parse_fails >= 1
    assert tr.steps[0]["actions"] == []
    assert tr.steps[0]["shaping"] == 0.0     # VAGEN only withholds the format reward


def test_env_shaping_is_recorded_per_turn():
    from envs.sokoban import SokobanEnv
    sok = SokobanEnv()
    task = sok.tasks("train", 1, np.random.default_rng(0))[0]
    path = sok.solve(task)
    tr = rollout(ScriptedPolicy([f"ACT: {a}" for a in path]), sok, task, LoopConfig(max_turns=0))
    assert tr.total_reward == 1.0
    assert sum(st["shaping"] for st in tr.steps) == pytest.approx(-0.01 * len(path) + 0.1)


def test_illegal_action_counts_as_parse_failure():
    tr = rollout(ScriptedPolicy(["ACT: teleport"]), env, gather_task, cfg)
    assert tr.parse_fails >= 1 and tr.steps[0]["actions"] == []


def test_rollout_never_exceeds_task_budget():
    tr = rollout(ScriptedPolicy(["ACT: up"] * 100), env, gather_task, cfg)
    assert tr.env_steps <= gather_task.budget
    assert len(tr.steps) == gather_task.budget


def test_memory_text_reaches_the_prompt():
    p = ScriptedPolicy([f"ACT: {count_task.goal[0]}"])
    rollout(p, env, count_task, cfg, memory_text="RECALL: count shapes")
    assert "RECALL: count shapes" in p.prompts[0]


class EchoSolver:
    """Answers each prompt with the solver's next move for the task whose step count is in the prompt:
    only correct if rollout_group keeps every row's prompt, env and history lined up."""

    def __init__(self, env, tasks):
        self.plans = {i: env.solve(t) for i, t in enumerate(tasks)}
        self.calls = 0

    def act_batch(self, prompts, images, **kw):
        from models.vlm import RolloutStep
        self.calls += 1
        out = []
        for p in prompts:
            row = int(p.split("ROW=")[1].split()[0])
            t = int(p.split("Step ")[1].split()[0])
            out.append(RolloutStep(f"ACT: {self.plans[row][t]}", [1], np.zeros(1, np.float32), 3))
        return out


def test_rollout_group_runs_rows_in_lockstep_and_drops_finished_ones():
    from envs.frozenlake import FrozenLakeEnv
    from utils.rollout import Context, rollout_group
    fl = FrozenLakeEnv()
    tasks = fl.tasks("train", 6, np.random.default_rng(3))
    policy = EchoSolver(fl, tasks)
    trajs = rollout_group(policy, fl, tasks, cfg, [Context(f"ROW={i}") for i in range(6)])
    assert [t.task for t in trajs] == tasks
    assert all(t.total_reward == 1.0 and t.env_steps == len(fl.solve(t.task)) for t in trajs)
    assert policy.calls == max(len(fl.solve(t)) for t in tasks)          # one batched call per lockstep turn


def test_context_images_come_before_the_observation_and_are_recorded():
    from envs.frozenlake import FrozenLakeEnv
    from utils.rollout import Context, rollout_group
    fl = FrozenLakeEnv()
    task = fl.tasks("train", 1, np.random.default_rng(0))[0]
    seen = []

    class Spy(EchoSolver):
        def act_batch(self, prompts, images, **kw):
            seen.append([im.shape for im in images[0]])
            return super().act_batch(prompts, images, **kw)

    card = np.zeros((10, 20, 3), np.uint8)
    trajs = rollout_group(Spy(fl, [task]), fl, [task], cfg, [Context("ROW=0 [Skill] <image>", [card], ("s1",))])
    assert seen[0] == [(10, 20, 3), (192, 192, 3)]
    assert len(trajs[0].steps[0]["images"]) == 2 and trajs[0].steps[0]["events"] is not None


def test_single_rollout_is_a_group_of_one():
    tr = rollout(ScriptedPolicy([f"ACT: {count_task.goal[0]}"]), env, count_task, cfg)
    assert tr.total_reward == 1.0 and tr.steps[0]["images"][0].shape == (64, 64, 3)
