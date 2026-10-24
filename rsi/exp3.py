"""Exp3 (Auer et al., 2002), the adversarial bandit Graves et al. (2017) use for automated curricula, here
choosing *which updating method runs* each iteration: the agent learns how to learn from its own gate results.

Arms are the updating methods in cfg.methods. Each iteration exp3_k of them are drawn from
    p = (1 - gamma) * softmax(w) + gamma / K
and each drawn arm is paid r = 1 if its update was kept and raised gate success, 0.5 if kept without a
gain, 0 if undone or if it found no signal; w[arm] += gamma * r / (K * p[arm]) (importance-weighted).
"""
import numpy as np

from rsi.base import Planner


class Exp3(Planner):
    name = "exp3"

    def __init__(self, cfg, names: list[str], rng: np.random.Generator):
        self.names, self.cfg, self.rng = list(names), cfg, rng
        self.w = np.zeros(len(self.names))
        self.chosen: list[str] = []
        self.p = np.full(len(self.names), 1 / max(len(self.names), 1))

    def probs(self) -> np.ndarray:
        e = np.exp(self.w - self.w.max())
        g = self.cfg.exp3_gamma
        return (1 - g) * e / e.sum() + g / len(self.names)

    def updates(self, names: list[str]) -> list[str]:
        if not self.names:
            return []
        self.p = self.probs()
        k = min(self.cfg.exp3_k, len(self.names))
        idx = self.rng.choice(len(self.names), size=k, replace=False, p=self.p)
        self.chosen = [self.names[i] for i in idx]
        return [n for n in names if n in self.chosen]

    def observe(self, batch: list, results: dict) -> None:
        g, K = self.cfg.exp3_gamma, len(self.names)
        for name in self.chosen:
            r = results.get(name)
            pay = 0.0
            if r is not None and r.passed:
                gain = (r.after[0] - r.before[0]) if r.after and r.before else 0.0
                pay = 1.0 if gain > 0 else 0.5
            i = self.names.index(name)
            self.w[i] += g * pay / (K * self.p[i])

    @property
    def last_stats(self) -> dict:
        return {"exp3_p": dict(zip(self.names, np.round(self.probs(), 3).tolist())), "exp3_chosen": self.chosen}
