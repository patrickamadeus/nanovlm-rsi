import math

import numpy as np
import pytest

from configs.config import LoopConfig
from envs.static import StaticEnv
from evaluate import evaluate, make_eval_fn
from rsi.memory import Skill, SkillStore
from utils.log import Trace
from tests.conftest import ScriptedPolicy, fake_embed

env = StaticEnv()
_held = env.tasks("heldout", 200, np.random.default_rng(0))
GOAL3 = [t for t in _held if t.family == "count_objects" and t.goal == ("3",)][:2]
GOAL5 = [t for t in _held if t.family == "count_objects" and t.goal == ("5",)][:2]


@pytest.fixture
def trace(tmp_path):
    tr = Trace(str(tmp_path))
    tr.log_rollout(family="count_objects", goal=["3"], split="train", reward=1.0, env_steps=1)
    return tr


def say3():
    return ScriptedPolicy(iter(lambda: "ACT: 3", None))


def test_evaluate_reports_both_heldout_axes_separately(trace):
    r = evaluate(say3(), env, GOAL3 + GOAL5, trace, LoopConfig())
    assert r.unseen_seed_rate == 1.0          # goal "3" was trained on; these are new worlds
    assert r.unseen_goal_rate == 0.0          # goal "5" never seen in training
    assert r.success_rate == 0.5 and r.n == 4


def test_evaluate_does_not_write_to_the_store(trace, tmp_path):
    store = SkillStore(str(tmp_path / "s.jsonl"), embed_fn=fake_embed)
    store.append(Skill("", "count_objects", "count", ("3",), 1.0, fake_embed("count")))
    before = (tmp_path / "s.jsonl").read_bytes()
    from rsi.memory import Memory
    evaluate(say3(), env, GOAL3, trace, LoopConfig(), methods=[Memory(store, LoopConfig())])
    assert (tmp_path / "s.jsonl").read_bytes() == before


def test_evaluate_logs_an_eval_record_with_cost(trace):
    evaluate(say3(), env, GOAL3, trace, LoopConfig())
    rec = trace.records("eval")[-1]
    assert rec["split"] == "heldout" and rec["cost"]["env_steps"] > 0


def test_evaluate_on_empty_task_list_returns_zeros_not_nan(trace):
    r = evaluate(say3(), env, [], trace, LoopConfig())
    assert r.n == 0 and r.success_rate == 0.0 and not math.isnan(r.success_rate)


def test_eval_fn_for_gate_returns_success_and_parse_fail_rates(trace):
    fn = make_eval_fn(ScriptedPolicy(iter(lambda: "garbage", None)), env, trace, LoopConfig())
    success, parse_fail = fn(GOAL3)
    assert success == 0.0 and parse_fail == 1.0
    assert {r["split"] for r in trace.records("rollout")} >= {"heldout"}


def test_evaluate_shows_memory_to_the_policy_and_logs_success_per_task_key(trace, tmp_path):
    from rsi.memory import Memory
    store = SkillStore(str(tmp_path / "s.jsonl"), embed_fn=fake_embed)
    store.append(Skill("", "count_objects", "How many blue squares", ("3",), 1.0, fake_embed("How many blue squares")))
    policy = say3()
    evaluate(policy, env, GOAL3 + GOAL5, trace, LoopConfig(), methods=[Memory(store, LoopConfig())])
    assert all("Skills that worked before" in p for p in policy.prompts)
    assert trace.records("eval")[-1]["by_key"] == {"count_objects:3": 1.0, "count_objects:5": 0.0}
