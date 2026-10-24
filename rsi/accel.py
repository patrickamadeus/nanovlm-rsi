"""ACCEL (Parker-Holder et al., 2022): the agent writes its own levels. A buffer keeps levels scored by
learning potential; each slot either replays a high-scoring level, *edits* one into a new level
(env.edit: one step harder or easier, fresh layout), or draws a new random one.

Learning potential is p * (1 - p) for the level's success rate p within its group of attempts: highest when
the outcome is uncertain, zero when it is always solved or never. (ACCEL scores by positive value loss; with
binary rewards and a group of attempts, p(1 - p) is the same idea and is exactly where GRPO has signal.)
Edited levels are verifiable by construction: the env generated them, and its solver checks they are solvable.
Gate and held-out tasks stay the fixed distribution: what the agent writes changes what it practises, never
what it is measured on.
"""
import numpy as np

from rsi.base import Planner, train_only


class ACCEL(Planner):
    name = "accel"

    def __init__(self, env, cfg, rng, log=None):
        self.env, self.cfg, self.rng = env, cfg, rng
        self.buffer: dict = {}               # task -> learning potential
        self.last_stats: dict = {}

    def replay(self):
        ranked = sorted(self.buffer, key=lambda t: -self.buffer[t])
        w = 1 / np.arange(1, len(ranked) + 1)           # rank-based prioritisation, as in PLR
        return ranked[int(self.rng.choice(len(ranked), p=w / w.sum()))]

    def tasks(self, n: int) -> list:
        out, kinds = [], {"new": 0, "replay": 0, "edit": 0}
        for _ in range(n):
            if self.buffer and self.rng.random() < self.cfg.accel_replay:
                level = self.replay()
                edited = self.env.edit(level, self.rng) if self.rng.random() < self.cfg.accel_edit else None
                kind, task = ("edit", edited) if edited is not None else ("replay", level)
            else:
                kind, task = "new", self.env.tasks("train", 1, self.rng)[0]
            kinds[kind] += 1
            out.append(task)
        self.last_stats = {**kinds, "buffer": len(self.buffer),
                           "levels": sorted({"x".join(map(str, t.level)) for t in out})}
        return train_only(out)

    def observe(self, batch: list, results: dict) -> None:
        groups: dict = {}
        for s in batch:
            if s.origin == "attempt":
                groups.setdefault(s.traj.task, []).append(float(s.traj.total_reward >= 1.0))
        for task, wins in groups.items():
            p = float(np.mean(wins))
            self.buffer[task] = p * (1 - p)
        if len(self.buffer) > self.cfg.accel_buffer:
            for t in sorted(self.buffer, key=lambda t: self.buffer[t])[:len(self.buffer) - self.cfg.accel_buffer]:
                del self.buffer[t]
