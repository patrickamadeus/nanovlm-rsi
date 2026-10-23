from collections import Counter

import numpy as np
import pytest

from configs.config import LoopConfig
from envs.base import split_for_seed
from envs.static import StaticEnv
from utils.log import Trace
from rsi.curriculum import frontier_weight, make_curriculum


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


def test_frontier_curriculum_samples_the_frontier_far_more_than_mastered_tasks(tmp_path):
    rates = {("count_objects", (str(k),)): (8, 8) for k in range(1, 10)}       # all counts mastered...
    rates[("count_objects", ("4",))] = (4, 8)                                   # ...except 4, on the frontier
    rates.update({("gather_order", g): (8, 8) for g in [("red", "green", "blue")]})
    tr = _trace_with(tmp_path, rates)
    prop = make_curriculum(StaticEnv(), LoopConfig(curriculum="frontier", tasks_per_iter=8, curriculum_epsilon=0.0))
    counts = Counter()
    for _ in range(30):
        for t in prop(None, tr):
            if t.family == "count_objects":
                counts[t.goal[0]] += 1
    others = max(v for k, v in counts.items() if k != "4")
    assert counts["4"] > 3 * others


def test_frontier_curriculum_only_ever_emits_train_tasks(tmp_path):
    prop = make_curriculum(StaticEnv(), LoopConfig(curriculum="frontier", tasks_per_iter=32))
    tr = Trace(str(tmp_path))
    for _ in range(10):
        assert all(split_for_seed(t.seed) == "train" and t.split == "train" for t in prop(None, tr))


def test_frontier_curriculum_reports_coverage_and_entropy(tmp_path):
    prop = make_curriculum(StaticEnv(), LoopConfig(curriculum="frontier", tasks_per_iter=8))
    prop(None, Trace(str(tmp_path)))
    stats = prop.last_stats
    assert 0.0 <= stats["frontier_frac"] <= 1.0 and stats["entropy"] >= 0.0 and stats["n_keys"] >= 1


def test_success_table_reads_the_trace_once(tmp_path):
    tr = _trace_with(tmp_path, {("count_objects", ("3",)): (1, 2)})
    table = tr.success_table()
    assert table[("count_objects", ("3",))] == (1, 2)


def test_uniform_curriculum_returns_only_train_tasks():
    tasks = make_curriculum(StaticEnv(), LoopConfig(curriculum="uniform", tasks_per_iter=16))(None, None)
    assert len(tasks) == 16
    assert {t.split for t in tasks} == {"train"}


def test_curriculum_can_never_emit_gate_or_heldout_seeds():
    env = StaticEnv()
    prop = make_curriculum(env, LoopConfig(curriculum="uniform", tasks_per_iter=64))
    heldout_seeds = {t.seed for t in env.tasks("heldout", 200, np.random.default_rng(0))}
    for _ in range(20):
        for t in prop(None, None):
            assert split_for_seed(t.seed) == "train"
            assert t.seed not in heldout_seeds


def test_successive_calls_draw_fresh_tasks():
    prop = make_curriculum(StaticEnv(), LoopConfig(curriculum="uniform", tasks_per_iter=8))
    assert [t.seed for t in prop(None, None)] != [t.seed for t in prop(None, None)]


def test_unknown_curriculum_is_rejected():
    with pytest.raises(ValueError):
        make_curriculum(StaticEnv(), LoopConfig(curriculum="oracle"))
