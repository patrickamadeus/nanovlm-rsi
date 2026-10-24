"""ExpeL (Zhao et al., 2024): learn *rules* by comparing a success with a failure on the same task.

When a task's group has both, the policy is shown the two move lists and how the failure ended, and writes
one sentence: the rule that explains the difference. Rules are global per task family, not retrieved by
similarity: the top cfg.expel_top_k by utility (EMA of success while the rule was shown) go in every prompt,
which plays the role of ExpeL's upvote / downvote. Rules live in the same exactly-revertible event log as
skills (rsi/memory.py), in their own file.

This is the first method here that learns from failures directly: a failure is half of every rule.
"""
import os

import numpy as np

from rsi.memory import Memory, SkillStore, clean
from rsi.gate import Update
from utils.log import Cost
from utils.rollout import Context

PAIR_PROMPT = ("Two attempts at the same level.\n"
               "Succeeded with: {win}.\n"
               "Failed with: {lose} ({ending}).\n"
               "In one short sentence, state a general rule that explains why one worked and the other did not.")
GLOBAL = np.ones(1, np.float32)        # rules are not retrieved by similarity: every rule has the same key


def moves(s) -> str:
    return ", ".join(st["action"] for st in s.traj.steps if st["action"] is not None) or "no valid moves"


def ending(s) -> str:
    events = s.traj.steps[-1].get("events", []) if s.traj.steps else []
    if events:
        return "ended: " + ", ".join(events)
    return "ran out of steps" if len(s.traj.steps) >= s.traj.task.budget else "stopped"


class ExpeL(Memory):
    name = "expel"

    def __init__(self, cfg, policy, run_dir: str):
        store = SkillStore(os.path.join(run_dir, "rules.jsonl"), embed_fn=lambda text: GLOBAL,
                           max_size=cfg.mem_max_size, dedupe_sim=1.01)       # dedupe is by exact text, below
        super().__init__(store, cfg, None, policy)

    def context(self, task, env) -> Context:
        rules = self.store.retrieve(GLOBAL, self.cfg.expel_top_k, alpha=0.0, family=task.family)
        if not rules:
            return Context()
        text = "Rules learned from your own successes and failures:\n" + "\n".join(f"- {r.strategy}" for r in rules)
        return Context(text, (), tuple(r.id for r in rules))

    def propose(self, batch) -> Update | None:
        groups: dict = {}
        for s in batch:
            if s.origin == "attempt" and not s.flags:
                groups.setdefault((s.traj.task.family, s.traj.task.seed), []).append(s)
        known = {r.strategy for r in self.store.skills()}
        rules, used = [], []
        for group in groups.values():
            wins = [s for s in group if s.reward >= 1.0]
            fails = [s for s in group if s.reward < 1.0 and any(st["action"] for st in s.traj.steps)]
            if not wins or not fails:
                continue
            win, lose = min(wins, key=lambda s: s.traj.env_steps), fails[0]
            prompt = PAIR_PROMPT.format(win=moves(win), lose=moves(lose), ending=ending(lose))
            rule = self.policy.act(prompt, [win.traj.steps[0]["images"][-1]], prefix="Rule:").text
            rule = clean(rule.removeprefix("Rule:"))
            if rule and rule not in known:
                known.add(rule)
                rules.append({"family": win.traj.task.family, "situation": "", "actions": [], "reward": 1.0,
                              "key": GLOBAL.tolist(), "strategy": rule})
                used += [win, lose]
        if not rules:
            return None
        cost = Cost(sum(s.traj.env_steps for s in batch), sum(s.traj.tokens_out for s in batch))
        return Update(self.name, "outcome", {"skills": rules, "revert_offset": self.store.offset()}, cost,
                      tuple(f"{s.traj.task.family}:{s.traj.task.seed}" for s in used))
