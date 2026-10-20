"""Proposers choose what to practice next. They see only the train split — enforced here, not by convention.

  fixed  uniform over the train split
  self   competence-frontier curriculum: tasks whose running success rate sits in proposer_band
         (default 0.2-0.8) are preferred, mastered and hopeless ones are down-weighted, unseen ones
         explored. That band is where a GRPO group has non-zero advantage, so with sparse binary
         rewards the curriculum is what keeps the weights carrier sample-efficient.
"""
from collections import Counter
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


def frontier_weight(rate: float | None, cfg: LoopConfig) -> float:
    if rate is None:
        return cfg.proposer_unseen_weight
    lo, hi = cfg.proposer_band
    return 1.0 if lo <= rate <= hi else cfg.proposer_off_band_weight


class SelfProposer:
    def __init__(self, env: Env, cfg: LoopConfig, rng: np.random.Generator):
        self.env, self.cfg, self.rng = env, cfg, rng
        self.last_stats: dict = {}

    def __call__(self, policy, trace) -> list[Task]:
        n = self.cfg.tasks_per_iter
        pool = self.env.tasks("train", n * self.cfg.proposer_pool, self.rng)
        table = trace.success_table() if trace is not None else {}
        rates = [(lambda wn: wn[0] / wn[1] if wn else None)(table.get((t.family, t.goal))) for t in pool]
        w = np.array([frontier_weight(r, self.cfg) for r in rates])
        p = (1 - self.cfg.proposer_epsilon) * w / w.sum() + self.cfg.proposer_epsilon / len(pool)
        idx = self.rng.choice(len(pool), size=min(n, len(pool)), replace=False, p=p / p.sum())
        chosen = [pool[i] for i in idx]
        keys = Counter((t.family, t.goal) for t in chosen)
        probs = np.array(list(keys.values()), float) / len(chosen)
        lo, hi = self.cfg.proposer_band
        self.last_stats = {"n_keys": len(keys), "entropy": float(-(probs * np.log(probs)).sum()),
                           "frontier_frac": float(np.mean([r is not None and lo <= r <= hi for r in
                                                           (rates[i] for i in idx)]))}
        return _train_only(chosen)


def make_proposer(name: str, env: Env, cfg: LoopConfig) -> Callable[..., list[Task]]:
    rng = np.random.default_rng(cfg.seed)
    if name == "fixed":
        def fixed_proposer(policy, trace) -> list[Task]:
            return _train_only(env.tasks("train", cfg.tasks_per_iter, rng))
        return fixed_proposer
    if name == "self":
        return SelfProposer(env, cfg, rng)
    raise ValueError(f"unknown proposer {name!r}")
