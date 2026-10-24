import numpy as np
import pytest

from configs.config import LoopConfig
from envs.base import split_for_seed
from envs.frozenlake import FrozenLakeEnv
from rsi import make_planner, make_update_planner
from rsi.gate import GateResult
from rsi.judges import ScoredTrajectory
from tests.conftest import ScriptedPolicy
from utils.rollout import Trajectory

ENV = FrozenLakeEnv(levels=((3, 1), (4, 3), (6, 8)))


def outcome(task, won):
    return ScoredTrajectory(Trajectory(task, [], float(won), 1, 0, 1), float(won), None, "outcome", 1.0, ())


def feed(planner, rates, rounds=1):
    """One observation per key with the given success rate (8 attempts each)."""
    for _ in range(rounds):
        batch = []
        for level, rate in rates.items():
            t = ENV.make_task(1, level)
            batch += [outcome(t, i < round(8 * rate)) for i in range(8)]
        planner.observe(batch, {})


def test_tscl_practises_the_key_whose_success_is_moving():
    p = make_planner(ENV, LoopConfig(planner="tscl", curriculum_epsilon=0.0))
    feed(p, {(3, 1): 1.0, (4, 3): 0.2, (6, 8): 0.0})
    feed(p, {(3, 1): 1.0, (4, 3): 0.6, (6, 8): 0.0})                  # only 4x4 is improving
    tasks = p.tasks(16)
    assert sum(t.level == (4, 3) for t in tasks) >= 12 and {t.split for t in tasks} == {"train"}


def test_accel_replays_and_edits_uncertain_levels_and_only_writes_train_seeds():
    cfg = LoopConfig(planner="accel", accel_replay=1.0, accel_edit=0.5)
    p = make_planner(ENV, cfg)
    t = ENV.make_task(next(s for s in range(1000) if split_for_seed(s) == "train"), (4, 3))
    p.observe([outcome(t, i < 4) for i in range(8)], {})                 # p = 0.5: high potential
    tasks = p.tasks(40)
    assert p.last_stats["replay"] > 0 and p.last_stats["edit"] > 0
    assert all(split_for_seed(x.seed) == "train" for x in tasks)
    assert any(x.level != (4, 3) for x in tasks)                         # it wrote levels it was never given


def test_accel_drops_solved_and_hopeless_levels_from_the_buffer_first():
    p = make_planner(ENV, LoopConfig(planner="accel", accel_buffer=1))
    a, b = ENV.make_task(1, (4, 3)), ENV.make_task(2, (4, 3))
    p.observe([outcome(a, True)] * 8 + [outcome(b, i < 4) for i in range(8)], {})
    assert list(p.buffer) == [b]


def test_voyager_lets_the_policy_choose_from_a_menu_that_includes_new_levels():
    policy = ScriptedPolicy(["Next: frozenlake:6:8"])
    p = make_planner(ENV, LoopConfig(planner="voyager"), policy=policy)
    tasks = p.tasks(4)
    assert "never tried" in policy.prompts[0] and p.last_stats["picked"] == "frozenlake:6:8"
    assert all(t.level == (6, 8) and t.split == "train" for t in tasks) and len({t.seed for t in tasks}) == 4


def test_voyager_falls_back_and_counts_it_when_the_answer_names_nothing():
    p = make_planner(ENV, LoopConfig(planner="voyager"), policy=ScriptedPolicy(["Next: the blue one"]))
    assert len(p.tasks(4)) == 4 and p.last_stats["fallbacks"] == 1


def _gate(passed, before, after):
    return GateResult(passed, {}, None, True, (before, 0.0), (after, 0.0))


def test_exp3_learns_to_run_the_method_whose_updates_are_kept():
    p = make_update_planner(LoopConfig(update_planner="exp3", exp3_gamma=0.2), ["grpo", "viskill"])
    for _ in range(60):
        chosen = p.updates(["grpo", "viskill"])
        assert len(chosen) == 1
        p.observe([], {"viskill": _gate(True, 0.2, 0.3), "grpo": _gate(False, 0.2, 0.1)})
    assert p.probs()[1] > 0.8


def test_all_runs_every_method():
    assert make_update_planner(LoopConfig(), ["a", "b"]).updates(["a", "b"]) == ["a", "b"]
