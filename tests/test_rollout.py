import numpy as np
import pytest

from config import LoopConfig
from envs.static import StaticEnv
from rollout import rollout
from tests.conftest import ScriptedPolicy

cfg = LoopConfig()
env = StaticEnv()
_train = env.tasks("train", 200, np.random.default_rng(0))
count_task = next(t for t in _train if t.family == "count_objects")
gather_task = next(t for t in _train if t.family == "gather_order")


def test_rollout_records_one_step_per_action():
    tr = rollout(ScriptedPolicy([f"ACT: {count_task.goal[0]}"]), env, count_task, cfg)
    assert tr.env_steps == 1 and tr.total_reward == 1.0 and tr.parse_fails == 0


def test_parse_failure_becomes_noop_with_negative_shaping():
    tr = rollout(ScriptedPolicy(["no action here"]), env, gather_task, cfg)
    assert tr.parse_fails >= 1
    assert tr.steps[0]["action"] is None
    assert tr.steps[0]["shaping"] < 0


def test_illegal_action_counts_as_parse_failure():
    tr = rollout(ScriptedPolicy(["ACT: teleport"]), env, gather_task, cfg)
    assert tr.parse_fails >= 1 and tr.steps[0]["action"] is None


def test_rollout_never_exceeds_task_budget():
    tr = rollout(ScriptedPolicy(["ACT: up"] * 100), env, gather_task, cfg)
    assert tr.env_steps <= gather_task.budget
    assert len(tr.steps) == gather_task.budget


def test_memory_text_reaches_the_prompt():
    p = ScriptedPolicy([f"ACT: {count_task.goal[0]}"])
    rollout(p, env, count_task, cfg, memory_text="RECALL: count shapes")
    assert "RECALL: count shapes" in p.prompts[0]
