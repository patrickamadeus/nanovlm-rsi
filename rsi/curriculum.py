"""The curriculum chooses what to practice next. It sees only the train split — enforced here, not by convention.

  uniform   uniform over the train split
  frontier  tasks whose running success rate sits in frontier_band (default 0.2-0.8) are preferred,
            mastered and hopeless ones are down-weighted, unseen ones explored. That band is where a
            GRPO group has non-zero advantage, so with sparse binary rewards it keeps GRPO sample-efficient.
"""
from collections import Counter
from collections.abc import Callable

import numpy as np

from configs.config import LoopConfig
from envs.base import Env, Task, stamp_split


def _train_only(tasks: list[Task]) -> list[Task]:
    stamped = [stamp_split(t) for t in tasks]
    leaked = [t.seed for t in stamped if t.split != "train"]
    if leaked:   # a curriculum reaching gate/heldout space would silently contaminate every reported number
        raise RuntimeError(f"curriculum produced non-train seeds {leaked[:5]}")
    return stamped


def frontier_weight(rate: float | None, cfg: LoopConfig) -> float:
    if rate is None:
        return cfg.frontier_unseen_weight
    lo, hi = cfg.frontier_band
    return 1.0 if lo <= rate <= hi else cfg.frontier_off_band_weight


class Frontier:
    def __init__(self, env: Env, cfg: LoopConfig, rng: np.random.Generator):
        self.env, self.cfg, self.rng = env, cfg, rng
        self.last_stats: dict = {}

    def __call__(self, policy, log) -> list[Task]:
        n = self.cfg.tasks_per_iter
        pool = self.env.tasks("train", n * self.cfg.curriculum_pool, self.rng)
        table = log.success_table() if log is not None else {}
        rates = [(lambda wn: wn[0] / wn[1] if wn else None)(table.get((t.family, t.goal))) for t in pool]
        w = np.array([frontier_weight(r, self.cfg) for r in rates])
        p = (1 - self.cfg.curriculum_epsilon) * w / w.sum() + self.cfg.curriculum_epsilon / len(pool)
        idx = self.rng.choice(len(pool), size=min(n, len(pool)), replace=False, p=p / p.sum())
        chosen = [pool[i] for i in idx]
        keys = Counter((t.family, t.goal) for t in chosen)
        probs = np.array(list(keys.values()), float) / len(chosen)
        lo, hi = self.cfg.frontier_band
        self.last_stats = {"n_keys": len(keys), "entropy": float(-(probs * np.log(probs)).sum()),
                           "frontier_frac": float(np.mean([r is not None and lo <= r <= hi for r in
                                                           (rates[i] for i in idx)]))}
        return _train_only(chosen)


def make_curriculum(env: Env, cfg: LoopConfig) -> Callable[..., list[Task]]:
    rng = np.random.default_rng(cfg.seed)
    if cfg.curriculum == "uniform":
        def uniform(policy, log) -> list[Task]:
            return _train_only(env.tasks("train", cfg.tasks_per_iter, rng))
        return uniform
    if cfg.curriculum == "frontier":
        return Frontier(env, cfg, rng)
    raise ValueError(f"unknown curriculum {cfg.curriculum!r}")
