"""The skill store: an append-only event log of add / use / drop records.

State is the replay of the log. Because nothing is ever rewritten in place,
truncating to a byte offset reverts *everything* after it — new skills, usage
counts and expirations alike — which is what makes the gate's revert exact.
"""
import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from infra.trace import heal_tail


@dataclass(frozen=True)
class Skill:
    id: str
    family: str
    situation: str
    actions: tuple[str, ...]
    reward: float
    embedding: np.ndarray
    uses: int = 0
    uses_before_success: int = 0   # use-attribution: was this skill in context when the episode succeeded?


class SkillStore:
    def __init__(self, path: str, embed_fn: Callable[[str], np.ndarray], max_size: int = 500,
                 dedupe_sim: float = 0.95):
        self.path, self.embed_fn, self.max_size, self.dedupe_sim = path, embed_fn, max_size, dedupe_sim
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        open(path, "a").close()
        heal_tail(path)
        self._load()

    def _load(self) -> None:
        self._skills: dict[str, Skill] = {}
        with open(self.path) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:   # partial last line from a killed run
                    continue
                self._apply(ev)

    def _apply(self, ev: dict) -> None:
        if ev["op"] == "add":
            s = ev["skill"]
            self._skills[s["id"]] = Skill(s["id"], s["family"], s["situation"], tuple(s["actions"]),
                                          s["reward"], np.asarray(s["embedding"], np.float32))
        elif ev["op"] == "use":
            for sid in ev["ids"]:
                if sid in self._skills:
                    s = self._skills[sid]
                    self._skills[sid] = replace(s, uses=s.uses + 1,
                                                uses_before_success=s.uses_before_success + int(ev["succeeded"]))
        elif ev["op"] == "drop":
            self._skills.pop(ev["id"], None)

    def _emit(self, ev: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(ev) + "\n")
        self._apply(ev)

    def offset(self) -> int:
        return os.path.getsize(self.path)

    def truncate(self, offset: int) -> None:
        with open(self.path, "r+") as f:
            f.truncate(offset)
        self._load()

    def __len__(self) -> int:
        return len(self._skills)

    def skills(self) -> list[Skill]:
        return list(self._skills.values())

    def _twin(self, skill: Skill, emb: np.ndarray) -> tuple[Skill | None, float]:
        """Nearest stored skill that would replay the same actions — only that counts as a duplicate."""
        best, best_sim = None, -1.0
        for s in self._skills.values():
            if s.family != skill.family or s.actions != tuple(skill.actions):
                continue
            sim = float(np.dot(s.embedding, emb))
            if sim > best_sim:
                best, best_sim = s, sim
        return best, best_sim

    def append(self, skill: Skill) -> str | None:
        """Add a skill. A duplicate — same family, same actions, near-identical situation — is kept only if it
        beats the incumbent's reward. Same situation with *different* actions is a different skill. Returns id or None."""
        emb = np.asarray(skill.embedding, np.float32)
        twin, sim = self._twin(skill, emb)
        if twin is not None and sim >= self.dedupe_sim:
            if skill.reward <= twin.reward:
                return None
            self._emit({"op": "drop", "id": twin.id})
        sid = uuid.uuid4().hex[:10]           # never reused after a revert, so trace references stay unambiguous
        self._emit({"op": "add", "skill": {"id": sid, "family": skill.family, "situation": skill.situation,
                                           "actions": list(skill.actions), "reward": skill.reward,
                                           "embedding": emb.tolist()}})
        return sid

    def retrieve(self, situation: str, k: int) -> list[Skill]:
        if not self._skills or k <= 0:
            return []
        q = np.asarray(self.embed_fn(situation), np.float32)
        ranked = sorted(self._skills.values(), key=lambda s: -float(np.dot(s.embedding, q)))
        return ranked[:k]

    def note_use(self, skill_ids: list[str], succeeded: bool) -> None:
        if skill_ids:
            self._emit({"op": "use", "ids": list(skill_ids), "succeeded": bool(succeeded)})

    def expire(self) -> int:
        """Drop the least useful skills until the store fits max_size."""
        excess = len(self._skills) - self.max_size
        if excess <= 0:
            return 0
        rank = sorted(self._skills.values(), key=lambda s: (s.uses_before_success / max(s.uses, 1), s.reward))
        for s in rank[:excess]:
            self._emit({"op": "drop", "id": s.id})
        return excess

    @staticmethod
    def as_text(skills: list[Skill]) -> str:
        if not skills:
            return ""
        lines = [f"- {s.situation} -> {' '.join(s.actions)}" for s in skills]
        return "Skills that worked before:\n" + "\n".join(lines)
