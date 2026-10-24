from collections import Counter

import numpy as np
import pytest

from configs.config import LoopConfig
from envs.base import split_for_seed
from envs.static import StaticEnv
from rsi import make_planner
from rsi.curriculum import frontier_weight
from utils.log import Trace


def test_frontier_weight_prefers_the_informative_band():
    cfg = LoopConfig()
    assert frontier_weight(0.5, cfg) > frontier_weight(1.0, cfg)
    assert frontier_weight(0.5, cfg) > frontier_weight(0.0, cfg)
    assert frontier_weight(None, cfg) >= frontier_weight(0.5, cfg) * 0.5   # unseen tasks still get explored


def _trace_with(tmp_path, rates):
    tr = Trace(str(tmp_path))
    for key, (wins, n) in rates.items():
        for i in range(n):
            tr.log_rollout(key=key, split="train", reward=float(i < wins), iter=0)
    return tr


def test_frontier_planner_samples_the_frontier_far_more_than_mastered_tasks(tmp_path):
    rates = {f"count_objects:{k}": (8, 8) for k in range(1, 10)}           # all counts mastered...
    rates["count_objects:4"] = (4, 8)                                       # ...except 4, on the frontier
    rates["gather_order:red/green/blue"] = (8, 8)
    planner = make_planner(StaticEnv(), LoopConfig(planner="frontier", tasks_per_iter=8, curriculum_epsilon=0.0),
                           _trace_with(tmp_path, rates))
    counts = Counter()
    for _ in range(30):
        for t in planner.tasks(8):
            if t.family == "count_objects":
                counts[t.goal[0]] += 1
    others = max(v for k, v in counts.items() if k != "4")
    assert counts["4"] > 3 * others


def test_frontier_planner_learns_from_what_it_observes():
    from rsi.judges import ScoredTrajectory
    from utils.rollout import Trajectory
    env = StaticEnv()
    planner = make_planner(env, LoopConfig(planner="frontier"))
    t = [x for x in env.tasks("train", 100, np.random.default_rng(0)) if x.family == "count_objects"][0]
    batch = [ScoredTrajectory(Trajectory(t, [], r, 1, 0, 1), r, None, "outcome", 1.0, ()) for r in (1.0, 0.0)]
    planner.observe(batch, {})
    assert planner.table.rate(env.task_key(t)) == 0.5


def test_frontier_planner_only_ever_emits_train_tasks(tmp_path):
    planner = make_planner(StaticEnv(), LoopConfig(planner="frontier"), Trace(str(tmp_path)))
    for _ in range(10):
        assert all(split_for_seed(t.seed) == "train" and t.split == "train" for t in planner.tasks(32))


def test_frontier_planner_reports_coverage_and_entropy():
    planner = make_planner(StaticEnv(), LoopConfig(planner="frontier"))
    planner.tasks(8)
    stats = planner.last_stats
    assert 0.0 <= stats["frontier_frac"] <= 1.0 and stats["entropy"] >= 0.0 and stats["n_keys"] >= 1


def test_success_table_reads_the_trace_once(tmp_path):
    tr = Trace(str(tmp_path))
    tr.log_rollout(family="count_objects", goal=["3"], split="train", reward=1.0)
    tr.log_rollout(family="count_objects", goal=["3"], split="train", reward=0.0)
    assert tr.success_table()[("count_objects", ("3",))] == (1, 2)


def test_uniform_planner_returns_only_train_tasks():
    tasks = make_planner(StaticEnv(), LoopConfig(planner="uniform")).tasks(16)
    assert len(tasks) == 16 and {t.split for t in tasks} == {"train"}


def test_planner_can_never_emit_gate_or_heldout_seeds():
    env = StaticEnv()
    planner = make_planner(env, LoopConfig(planner="uniform"))
    heldout_seeds = {t.seed for t in env.tasks("heldout", 200, np.random.default_rng(0))}
    for _ in range(20):
        for t in planner.tasks(64):
            assert split_for_seed(t.seed) == "train" and t.seed not in heldout_seeds


def test_successive_calls_draw_fresh_tasks():
    planner = make_planner(StaticEnv(), LoopConfig(planner="uniform"))
    assert [t.seed for t in planner.tasks(8)] != [t.seed for t in planner.tasks(8)]


def test_a_planner_that_leaks_non_train_seeds_is_stopped():
    from envs.base import Task
    from rsi.base import train_only
    gate_seed = next(s for s in range(100) if split_for_seed(s) == "gate")
    with pytest.raises(RuntimeError):
        train_only([Task("f", gate_seed, ("g",), 1, "train")])


def test_unknown_planner_is_rejected():
    with pytest.raises(ValueError):
        make_planner(StaticEnv(), LoopConfig(planner="oracle"))
