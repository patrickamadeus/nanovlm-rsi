import numpy as np
import pytest

from config import LoopConfig
from envs.base import split_for_seed
from envs.static import StaticEnv
from propose import make_proposer


def test_fixed_proposer_returns_only_train_tasks():
    tasks = make_proposer("fixed", StaticEnv(), LoopConfig(tasks_per_iter=16))(None, None)
    assert len(tasks) == 16
    assert {t.split for t in tasks} == {"train"}


def test_proposer_can_never_emit_gate_or_heldout_seeds():
    env = StaticEnv()
    prop = make_proposer("fixed", env, LoopConfig(tasks_per_iter=64))
    heldout_seeds = {t.seed for t in env.tasks("heldout", 200, np.random.default_rng(0))}
    for _ in range(20):
        for t in prop(None, None):
            assert split_for_seed(t.seed) == "train"
            assert t.seed not in heldout_seeds


def test_successive_calls_draw_fresh_tasks():
    prop = make_proposer("fixed", StaticEnv(), LoopConfig(tasks_per_iter=8))
    assert [t.seed for t in prop(None, None)] != [t.seed for t in prop(None, None)]


def test_self_proposer_not_yet_available():
    with pytest.raises(NotImplementedError, match="Slice 5"):
        make_proposer("self", StaticEnv(), LoopConfig())
