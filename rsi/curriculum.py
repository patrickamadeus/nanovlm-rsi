"""The baseline planners: which train tasks to practise next. Smarter planners live in their own files.

  uniform   uniform over the train split
  frontier  task keys (env.task_key) whose running success rate sits in frontier_band (default 0.2-0.8) are
            preferred, mastered and hopeless ones are down-weighted, unseen ones explored. That band is where
            a GRPO group has non-zero advantage, so with sparse binary rewards it keeps GRPO sample-efficient.

Every planner returns tasks through train_only(): gate and held-out seeds can never be practised.
"""
from collections import Counter

import numpy as np

from configs.config import LoopConfig
from envs.base import Env, Task
from rsi.base import Planner, train_only


class Uniform(Planner):
    name = "uniform"

    def __init__(self, env: Env, cfg: LoopConfig, rng: np.random.Generator, log=None):
        self.env, self.rng = env, rng
        self.last_stats: dict = {}

    def tasks(self, n: int) -> list[Task]:
        return train_only(self.env.tasks("train", n, self.rng))


def frontier_weight(rate: float | None, cfg: LoopConfig) -> float:
    if rate is None:
        return cfg.frontier_unseen_weight
    lo, hi = cfg.frontier_band
    return 1.0 if lo <= rate <= hi else cfg.frontier_off_band_weight


class SuccessTable:
    """Running (wins, attempts) per task key, from the planner's own observations. Seeded from the log on resume."""

    def __init__(self, log=None):
        self.cells: dict[str, list[int]] = {}
        for r in (log.records("rollout") if log is not None else []):
            if r.get("split") == "train" and "key" in r and r.get("iter") is not None:
                self.add(r["key"], r.get("reward", 0) >= 1.0)

    def add(self, key: str, won: bool) -> None:
        cell = self.cells.setdefault(key, [0, 0])
        cell[0] += int(won)
        cell[1] += 1

    def rate(self, key: str) -> float | None:
        """None means never attempted — distinct from always failed."""
        wins, n = self.cells.get(key, (0, 0))
        return wins / n if n else None


class Frontier(Planner):
    name = "frontier"

    def __init__(self, env: Env, cfg: LoopConfig, rng: np.random.Generator, log=None):
        self.env, self.cfg, self.rng = env, cfg, rng
        self.table = SuccessTable(log)
        self.last_stats: dict = {}

    def tasks(self, n: int) -> list[Task]:
        pool = self.env.tasks("train", n * self.cfg.curriculum_pool, self.rng)
        rates = [self.table.rate(self.env.task_key(t)) for t in pool]
        w = np.array([frontier_weight(r, self.cfg) for r in rates])
        p = (1 - self.cfg.curriculum_epsilon) * w / w.sum() + self.cfg.curriculum_epsilon / len(pool)
        idx = self.rng.choice(len(pool), size=min(n, len(pool)), replace=False, p=p / p.sum())
        chosen = [pool[i] for i in idx]
        keys = Counter(self.env.task_key(t) for t in chosen)
        probs = np.array(list(keys.values()), float) / len(chosen)
        lo, hi = self.cfg.frontier_band
        self.last_stats = {"n_keys": len(keys), "entropy": float(-(probs * np.log(probs)).sum()),
                           "frontier_frac": float(np.mean([r is not None and lo <= r <= hi for r in
                                                           (rates[i] for i in idx)]))}
        return train_only(chosen)

    def observe(self, batch: list, results: dict) -> None:
        for s in batch:
            if s.origin == "attempt":          # a hindsight copy is not a success at the task that was practised
                self.table.add(self.env.task_key(s.traj.task), s.traj.total_reward >= 1.0)
