"""Reflexion (Shinn et al., 2023): after a failure, the policy writes in words what went wrong and what to
do instead; the reflection is shown the next time it faces a similar task.

Reflexion retries the *same* task; tasks here rarely repeat, so a reflection is keyed like a skill (the env's
layout descriptor, or the first observation's text) and retrieved for the most similar new task. One
reflection per failed task per iteration, from its first failed attempt. Same revertible store as skills.
"""
import os

from rsi.expel import ending, moves
from rsi.gate import Update
from rsi.memory import Memory, SkillStore
from utils.log import Cost
from utils.rollout import Context

REFLECT_PROMPT = ("You failed this level. Your moves: {moves} ({ending}).\n"
                  "In one short sentence, say what went wrong and what you will do differently next time.")


class Reflexion(Memory):
    name = "reflexion"

    def __init__(self, cfg, policy, env, embed_fn):
        store = SkillStore(os.path.join(cfg.run_dir, "reflections.jsonl"), embed_fn=embed_fn,
                           max_size=cfg.mem_max_size, dedupe_sim=1.01)
        super().__init__(store, cfg, env, policy)

    def context(self, task, env) -> Context:
        if not len(self.store):
            return Context()
        found = self.store.retrieve(self.key(task, env), self.cfg.mem_top_k, 1.0, self.cfg.mem_retrieve_min,
                                    task.family)
        if not found:
            return Context()
        text = "Lessons from your failures on similar levels:\n" + "\n".join(f"- {r.strategy}" for r in found)
        return Context(text, (), tuple(r.id for r in found))

    def propose(self, batch) -> Update | None:
        seen, notes, used = set(), [], []
        for s in batch:
            k = (s.traj.task.family, s.traj.task.seed)
            if s.reward >= 1.0 or s.flags or s.origin != "attempt" or k in seen or not s.traj.steps:
                continue
            seen.add(k)
            prompt = REFLECT_PROMPT.format(moves=moves(s), ending=ending(s))
            note = self.policy.act(prompt, [s.traj.steps[0]["images"][-1]], prefix="Reflection:").text
            note = note.removeprefix("Reflection:").strip()
            if note:
                key = self.key(s.traj.task, self.env)
                notes.append({"family": s.traj.task.family, "situation": s.traj.steps[0].get("obs_text", ""),
                              "actions": [], "reward": 0.0, "strategy": note,
                              "key": key.tolist() if not isinstance(key, str) else None})
                used.append(s)
        if not notes:
            return None
        cost = Cost(sum(s.traj.env_steps for s in batch), sum(s.traj.tokens_out for s in batch))
        return Update(self.name, "outcome", {"skills": notes, "revert_offset": self.store.offset()}, cost,
                      tuple(f"{s.traj.task.family}:{s.traj.task.seed}" for s in used))
