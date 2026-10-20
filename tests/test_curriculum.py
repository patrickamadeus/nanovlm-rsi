from collections import Counter

import numpy as np

from config import LoopConfig
from envs.base import split_for_seed
from envs.static import StaticEnv
from infra.trace import Trace
from propose import frontier_weight, make_proposer


def test_frontier_weight_prefers_the_informative_band():
    cfg = LoopConfig()
    assert frontier_weight(0.5, cfg) > frontier_weight(1.0, cfg)
    assert frontier_weight(0.5, cfg) > frontier_weight(0.0, cfg)
    assert frontier_weight(None, cfg) >= frontier_weight(0.5, cfg) * 0.5   # unseen tasks still get explored


def _trace_with(tmp_path, rates):
    tr = Trace(str(tmp_path))
    for (family, goal), (wins, n) in rates.items():
        for i in range(n):
            tr.log_rollout(family=family, goal=list(goal), split="train", reward=float(i < wins))
    return tr


def test_self_proposer_samples_the_frontier_far_more_than_mastered_tasks(tmp_path):
    rates = {("count_objects", (str(k),)): (8, 8) for k in range(1, 10)}       # all counts mastered...
    rates[("count_objects", ("4",))] = (4, 8)                                   # ...except 4, on the frontier
    rates.update({("gather_order", g): (8, 8) for g in [("red", "green", "blue")]})
    tr = _trace_with(tmp_path, rates)
    prop = make_proposer("self", StaticEnv(), LoopConfig(tasks_per_iter=8, proposer_epsilon=0.0))
    counts = Counter()
    for _ in range(30):
        for t in prop(None, tr):
            if t.family == "count_objects":
                counts[t.goal[0]] += 1
    others = max(v for k, v in counts.items() if k != "4")
    assert counts["4"] > 3 * others


def test_self_proposer_only_ever_emits_train_tasks(tmp_path):
    prop = make_proposer("self", StaticEnv(), LoopConfig(tasks_per_iter=32))
    tr = Trace(str(tmp_path))
    for _ in range(10):
        assert all(split_for_seed(t.seed) == "train" and t.split == "train" for t in prop(None, tr))


def test_self_proposer_reports_coverage_and_entropy(tmp_path):
    prop = make_proposer("self", StaticEnv(), LoopConfig(tasks_per_iter=8))
    prop(None, Trace(str(tmp_path)))
    stats = prop.last_stats
    assert 0.0 <= stats["frontier_frac"] <= 1.0 and stats["entropy"] >= 0.0 and stats["n_keys"] >= 1


def test_success_table_reads_the_trace_once(tmp_path):
    tr = _trace_with(tmp_path, {("count_objects", ("3",)): (1, 2)})
    table = tr.success_table()
    assert table[("count_objects", ("3",))] == (1, 2)
