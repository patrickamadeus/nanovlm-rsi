"""Proposers choose what to practice next. They see only the train split — enforced here, not by convention.

The self-curriculum (Slice 5) samples tasks at the competence frontier, where the
estimated success rate from Trace.success_rate is in [0.2, 0.8]: that band is
where GRPO groups have non-zero advantage, so the curriculum is what makes sparse
binary rewards sample-efficient.
"""
from collections.abc import Callable

import numpy as np

from config import LoopConfig
from envs.base import Env, Task, stamp_split


def _train_only(tasks: list[Task]) -> list[Task]:
    stamped = [stamp_split(t) for t in tasks]
    leaked = [t.seed for t in stamped if t.split != "train"]
    if leaked:   # a proposer reaching gate/heldout space would silently contaminate every reported number
        raise RuntimeError(f"proposer produced non-train seeds {leaked[:5]}")
    return stamped


def make_proposer(name: str, env: Env, cfg: LoopConfig) -> Callable[..., list[Task]]:
    rng = np.random.default_rng(cfg.seed)
    if name == "fixed":
        def fixed_proposer(policy, trace) -> list[Task]:
            return _train_only(env.tasks("train", cfg.tasks_per_iter, rng))
        return fixed_proposer
    if name == "self":
        raise NotImplementedError("self proposer lands in Slice 5")
    raise ValueError(f"unknown proposer {name!r}")
