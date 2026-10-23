from dataclasses import replace

import pytest

from configs.config import LoopConfig
from envs.base import Task
from envs.static import StaticEnv
from rsi.gate import gate
from rsi.memory import SkillStore
from tests.conftest import fake_embed
from rsi.memory import Memory
from harness.rollout import Trajectory
from judges.outcome import ScoredTrajectory

TASK = Task("count_objects", 2, ("3",), 1, "train")


def evals(*results):
    it = iter(results)
    return lambda tasks: next(it)


improving = lambda: evals((0.2, 0.1), (0.5, 0.1))        # (success_rate, parse_fail_rate) before, after
worsening = lambda: evals((0.5, 0.1), (0.2, 0.1))


class ForgetfulUpdater(Memory):
    def commit(self, update):                            # claims success but writes nothing to disk
        pass


@pytest.fixture
def setup(tmp_path):
    store = SkillStore(str(tmp_path / "skills.jsonl"), embed_fn=fake_embed)
    cfg = LoopConfig(gate_n=4, run_dir=str(tmp_path))
    up = Memory(store, cfg)
    t = Trajectory(TASK, [{"action": "3", "obs_text": "count the squares"}], 1.0, 1, 0, 3)
    update = up.propose([ScoredTrajectory(t, 1.0, None, "outcome", 1.0, ())])
    return up, update, StaticEnv(), cfg, store


def test_all_conditions_pass_for_a_good_update(setup):
    up, u, env, cfg, store = setup
    r = gate(u, up, env, cfg, eval_fn=improving())
    assert r.passed and all(r.conditions.values())
    assert set(r.conditions) == {"proposed", "survives_reset", "gate_gain", "deployment_feedback", "no_regression"}
    assert len(store) == 1                                # committed


def test_empty_provenance_fails_proposed_and_reverts(setup):
    up, u, env, cfg, store = setup
    r = gate(replace(u, provenance=()), up, env, cfg, eval_fn=improving())
    assert not r.passed and r.conditions["proposed"] is False
    assert len(store) == 0                                # reverted


def test_reload_probe_fails_when_update_not_on_disk(setup):
    _, u, env, cfg, store = setup
    r = gate(u, ForgetfulUpdater(store, cfg), env, cfg, eval_fn=improving())
    assert not r.passed and r.conditions["survives_reset"] is False


def test_negative_gate_delta_fails(setup):
    up, u, env, cfg, store = setup
    r = gate(u, up, env, cfg, eval_fn=worsening())
    assert not r.passed and r.conditions["gate_gain"] is False
    assert "gate_gain" in r.reason and len(store) == 0


def test_disallowed_verdict_source_fails(setup):
    up, u, env, cfg, _ = setup
    r = gate(replace(u, signal="oracle"), up, env, cfg, eval_fn=improving())
    assert r.conditions["deployment_feedback"] is False and not r.passed


def test_parse_fail_regression_fails(setup):
    up, u, env, cfg, _ = setup
    r = gate(u, up, env, cfg, eval_fn=evals((0.2, 0.1), (0.5, 0.3)))
    assert r.conditions["no_regression"] is False and not r.passed


def test_gate_evaluates_on_the_gate_split_only(setup):
    up, u, env, cfg, _ = setup
    seen = []
    def spy(tasks):
        seen.extend(tasks)
        return (0.5, 0.0)
    gate(u, up, env, cfg, eval_fn=spy)
    assert seen and {t.split for t in seen} == {"gate"}


def test_disabled_gate_commits_and_identifies_itself(setup):
    up, u, env, cfg, store = setup
    r = gate(u, up, env, replace(cfg, gate_enabled=False), eval_fn=worsening())
    assert r.passed and r.gate_enabled is False and len(store) == 1
