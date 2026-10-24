"""Teacher-Student Curriculum Learning (Matiisen et al., 2017), online variant: practise where success is
*changing* fastest, not where it is highest or lowest.

Per task key (env.task_key: a level, an achievement), Q tracks the recent slope of the success rate:
    Q[k] <- alpha * (rate_now[k] - rate_before[k]) + (1 - alpha) * Q[k]
and each slot of the iteration picks a key epsilon-greedily on |Q| (unseen keys first). |Q| is used because
forgetting (a falling rate) also deserves practice. With one key (a single-level env) this is uniform.
"""
import numpy as np

from rsi.base import Planner, train_only


class TSCL(Planner):
    name = "tscl"

    def __init__(self, env, cfg, rng, log=None):
        self.env, self.cfg, self.rng = env, cfg, rng
        self.q: dict[str, float] = {}
        self.last: dict[str, float] = {}
        self.last_stats: dict = {}

    def pick(self, keys: list[str]) -> str:
        unseen = [k for k in keys if k not in self.last]
        if unseen:
            return unseen[int(self.rng.integers(len(unseen)))]
        if self.rng.random() < self.cfg.curriculum_epsilon:
            return keys[int(self.rng.integers(len(keys)))]
        return max(keys, key=lambda k: (abs(self.q.get(k, 0.0)), self.rng.random()))

    def tasks(self, n: int) -> list:
        pool = self.env.tasks("train", n * self.cfg.curriculum_pool, self.rng)
        by_key: dict[str, list] = {}
        for t in pool:
            by_key.setdefault(self.env.task_key(t), []).append(t)
        chosen = []
        for _ in range(n):
            keys = [k for k, ts in by_key.items() if ts]
            chosen.append(by_key[self.pick(keys)].pop())
        self.last_stats = {"q": {k: round(v, 4) for k, v in self.q.items()}}
        return train_only(chosen)

    def observe(self, batch: list, results: dict) -> None:
        wins: dict[str, list[float]] = {}
        for s in batch:
            if s.origin == "attempt":
                wins.setdefault(self.env.task_key(s.traj.task), []).append(float(s.traj.total_reward >= 1.0))
        for k, w in wins.items():
            rate = float(np.mean(w))
            if k in self.last:
                a = self.cfg.tscl_alpha
                self.q[k] = a * (rate - self.last[k]) + (1 - a) * self.q.get(k, 0.0)
            self.last[k] = rate
