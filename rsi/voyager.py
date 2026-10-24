"""Voyager's automatic curriculum (Wang et al., 2023): the agent itself proposes what to practise next.

Each iteration the policy is shown its own record per task key (solved / tried) and a menu: the keys it has
met plus *edited* ones (env.edit: nearby harder or easier levels it has never seen), and answers with one.
All tasks of the iteration come from that key. An answer that names nothing on the menu falls back to TSCL,
and the log counts how often that happens: a planner that never parses is not planning.
"""
import numpy as np

from rsi.base import Planner, train_only
from rsi.tscl import TSCL

PROMPT = ("You are practising to get better at these puzzles. Your record so far:\n{record}\n"
          "Pick the one to practise next: not so easy that you learn nothing, not so hard that you never succeed. "
          "Answer with its name exactly.")


class Voyager(Planner):
    name = "voyager"

    def __init__(self, env, cfg, rng, policy, log=None):
        self.env, self.cfg, self.rng, self.policy = env, cfg, rng, policy
        self.fallback = TSCL(env, cfg, rng)
        self.record: dict[str, list[int]] = {}
        self.examples: dict[str, object] = {}              # a task of each key, to edit and to rebuild from
        self.last_stats: dict = {}
        self.fallbacks = 0

    def menu(self, pool: list) -> dict:
        options = {}
        for t in pool + list(self.examples.values()):
            options.setdefault(self.env.task_key(t), t)
            edited = self.env.edit(t, self.rng)
            if edited is not None:
                options.setdefault(self.env.task_key(edited), edited)
        return options

    def tasks(self, n: int) -> list:
        pool = self.env.tasks("train", n, self.rng)
        options = self.menu(pool)
        lines = []
        for k in sorted(options):
            won, tried = self.record.get(k, (0, 0))
            lines.append(f"- {k}: solved {won} of {tried}" if tried else f"- {k}: never tried")
        answer = self.policy.act(PROMPT.format(record="\n".join(lines)), [], prefix="Next:").text
        picked = next((k for k in sorted(options, key=len, reverse=True) if k in answer), None)
        if picked is None:
            self.fallbacks += 1
            self.last_stats = {"picked": None, "fallbacks": self.fallbacks, "answer": answer[:80]}
            return self.fallback.tasks(n)
        out = [self.relayout(options[picked]) for _ in range(n)]     # fresh layouts of the chosen level
        self.last_stats = {"picked": picked, "fallbacks": self.fallbacks}
        return train_only(out)

    def relayout(self, task):
        from envs.base import train_seed
        if hasattr(self.env, "make_task") and task.level:
            return self.env.make_task(train_seed(self.rng), task.level)
        same = [t for t in self.env.tasks("train", 50, self.rng) if self.env.task_key(t) == self.env.task_key(task)]
        return same[0] if same else task

    def observe(self, batch: list, results: dict) -> None:
        self.fallback.observe(batch, results)
        for s in batch:
            if s.origin == "attempt":
                k = self.env.task_key(s.traj.task)
                cell = self.record.setdefault(k, [0, 0])
                cell[0] += int(s.traj.total_reward >= 1.0)
                cell[1] += 1
                self.examples.setdefault(k, s.traj.task)
