"""Memory updates: successful attempts become retrievable skills (the survey's context evolution).

What a skill shows the policy is the format (cfg.skill_format), separate from how it is stored:
  action       the moves that worked                    "Moves that solved a similar level: down right right"
  text         a one-line strategy the policy wrote itself about its own success
  text+action  both
  visual / text+visual   a rendered skill card (rsi/viskill.py)

Retrieval key: env.descriptor(task) when the env has one (a layout vector, as in ViSkill), otherwise the
embedding of the first observation's text. Retrieval score is alpha * similarity + (1 - alpha) * utility,
utility being an EMA of "the attempt succeeded while this skill was in context" (ViSkill's u).

The skill store is an append-only event log of add / use / drop records. State is the replay of the log.
Because nothing is ever rewritten in place, truncating to a byte offset reverts *everything* after it —
new skills, usage counts and expirations alike — which is what makes the gate's revert exact.
"""
import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from rsi.base import Method
from rsi.gate import Update
from utils.log import Cost, heal_tail
from utils.rollout import Context

FORMATS = ("action", "text", "text+action", "visual", "text+visual")


@dataclass(frozen=True)
class Skill:
    id: str
    family: str
    situation: str
    actions: tuple[str, ...]
    reward: float
    embedding: np.ndarray           # the retrieval key: a descriptor vector or a text embedding
    uses: int = 0
    uses_before_success: int = 0    # use-attribution: was this skill in context when the episode succeeded?
    strategy: str = ""              # the policy's own one-line account of why it worked
    card: str = ""                  # path of the rendered skill card (rsi/viskill.py)
    utility: float = 0.5            # EMA of success when used; ViSkill starts every skill at 0.5


class SkillStore:
    def __init__(self, path: str, embed_fn: Callable[[str], np.ndarray], max_size: int = 500,
                 dedupe_sim: float = 0.95, utility_beta: float = 0.1):
        self.path, self.embed_fn, self.max_size, self.dedupe_sim = path, embed_fn, max_size, dedupe_sim
        self.beta = utility_beta
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
            self._skills[s["id"]] = Skill(s["id"], s["family"], s["situation"], tuple(s["actions"]), s["reward"],
                                          np.asarray(s["embedding"], np.float32), strategy=s.get("strategy", ""),
                                          card=s.get("card", ""))
        elif ev["op"] == "use":
            for sid in ev["ids"]:
                if sid in self._skills:
                    s, won = self._skills[sid], int(ev["succeeded"])
                    self._skills[sid] = replace(s, uses=s.uses + 1, uses_before_success=s.uses_before_success + won,
                                                utility=(1 - self.beta) * s.utility + self.beta * won)
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
            if s.family != skill.family or s.actions != tuple(skill.actions) or s.embedding.shape != emb.shape:
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
        sid = skill.id or uuid.uuid4().hex[:10]   # never reused after a revert, so trace references stay unambiguous
        self._emit({"op": "add", "skill": {"id": sid, "family": skill.family, "situation": skill.situation,
                                           "actions": list(skill.actions), "reward": skill.reward,
                                           "embedding": emb.tolist(), "strategy": skill.strategy,
                                           "card": skill.card}})
        return sid

    def retrieve(self, query, k: int, alpha: float = 1.0, min_score: float = -np.inf,
                 family: str | None = None) -> list[Skill]:
        """Top k by alpha * cosine + (1 - alpha) * utility, among skills keyed like the query. `query` is
        a key vector, or text to embed."""
        if not self._skills or k <= 0:
            return []
        q = np.asarray(self.embed_fn(query) if isinstance(query, str) else query, np.float32)
        pool = [s for s in self._skills.values()
                if s.embedding.shape == q.shape and (family is None or s.family == family)]
        scored = [(alpha * float(np.dot(s.embedding, q)) + (1 - alpha) * s.utility, s) for s in pool]
        ranked = sorted((x for x in scored if x[0] >= min_score), key=lambda x: -x[0])
        return [s for _, s in ranked[:k]]

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
    def as_text(skills: list[Skill], fmt: str = "action", keyed_by_text: bool = True) -> str:
        if not skills:
            return ""
        lines = []
        for s in skills:
            moves = " ".join(s.actions)
            if fmt == "action":
                lines.append(f"- {s.situation} -> {moves}" if keyed_by_text else f"- moves that worked: {moves}")
            elif fmt == "text":
                lines.append(f"- {s.strategy}")
            elif fmt == "text+action":
                lines.append(f"- {s.strategy} Moves that worked: {moves}")
        return "Skills that worked before:\n" + "\n".join(lines)


def situation(scored) -> str:
    # what the policy saw, never task.goal: for count_objects the goal is the answer itself
    return scored.traj.steps[0].get("obs_text", "") if scored.traj.steps else ""


STRATEGY_PROMPT = ("You reached the goal in this level with these moves: {moves}.\n"
                   "In one short sentence, state the strategy that worked, so you can reuse it on a similar level.")


class Memory(Method):
    name, changes = "memory", "memory"

    def __init__(self, store: SkillStore, cfg, env=None, policy=None):
        self.store, self.cfg, self.env, self.policy = store, cfg, env, policy
        self.fmt = getattr(cfg, "skill_format", "action")
        if self.fmt not in FORMATS:
            raise ValueError(f"unknown skill_format {self.fmt!r}; one of {FORMATS}")
        self._added: dict[int, list[str]] = {}    # revert_offset -> skill ids that commit actually wrote

    def key(self, task, env) -> np.ndarray | str:
        """The env's layout descriptor if it has one, else the first observation's text (embedded by the store)."""
        d = env.descriptor(task) if env is not None else None
        return d if d is not None else (env.fork().reset(task).text or "")

    def context(self, task, env) -> Context:
        """The k skills most similar to this task, keyed exactly as propose() wrote them."""
        if not len(self.store) or self.cfg.mem_top_k <= 0:
            return Context()
        key = self.key(task, env)
        skills = self.store.retrieve(key, self.cfg.mem_top_k, self.cfg.mem_retrieve_alpha, self.cfg.mem_retrieve_min,
                                     task.family)
        return Context(self.store.as_text(skills, self.fmt, isinstance(key, str)), (), tuple(s.id for s in skills))

    def observe(self, batch) -> None:
        """Use attribution: was the skill in context when the attempt succeeded? Utility and expiry read it."""
        for s in batch:
            self.store.note_use(list(s.traj.context_ids), succeeded=s.reward >= 1.0)

    def admissible(self, s) -> bool:
        """A success, unflagged, and (if the env has a solver) within mem_max_excess_steps of optimal:
        ViSkill admits only optimal-length successes (0), so the library never teaches a detour."""
        if s.reward < self.cfg.mem_write_threshold or s.flags:
            return False
        if self.env is None or self.cfg.mem_max_excess_steps < 0:
            return True
        best = self.env.solve(s.traj.task)
        return best is None or s.traj.env_steps <= len(best) + self.cfg.mem_max_excess_steps

    def strategy(self, s) -> str:
        """The policy's own sentence about why this attempt worked (ViSkill's d_k), for the text formats."""
        if "text" not in self.fmt or self.policy is None:
            return ""
        moves = ", ".join(st["action"] for st in s.traj.steps if st["action"] is not None)
        first = s.traj.steps[0]["images"][-1:] if s.traj.steps and s.traj.steps[0].get("images") else []
        out = self.policy.act(STRATEGY_PROMPT.format(moves=moves), list(first), prefix="Strategy:")
        return out.text.removeprefix("Strategy:").strip()

    def skill_record(self, s) -> dict:
        key = self.key(s.traj.task, self.env) if self.env is not None else situation(s)
        return {"family": s.traj.task.family, "situation": situation(s),
                "actions": [st["action"] for st in s.traj.steps if st["action"] is not None], "reward": s.reward,
                "key": key.tolist() if isinstance(key, np.ndarray) else None, "strategy": self.strategy(s)}

    def propose(self, batch) -> Update | None:
        keep, seen = [], set()
        for s in batch:                       # one skill per task: the group's attempts are the same level
            k = (s.traj.task.family, s.traj.task.seed, s.traj.task.goal)
            if k not in seen and self.admissible(s):
                keep.append(s)
                seen.add(k)
        if not keep:
            return None
        skills = [self.skill_record(s) for s in keep]
        cost = Cost(sum(s.traj.env_steps for s in batch), sum(s.traj.tokens_out for s in batch))
        provenance = tuple(f"{s.traj.task.family}:{s.traj.task.seed}" for s in keep)
        signal = keep[0].verdict_source if len({s.verdict_source for s in keep}) == 1 else "mixed"
        return Update(self.name, signal, {"skills": skills, "revert_offset": self.store.offset()}, cost, provenance)

    def to_skill(self, s: dict) -> Skill:
        emb = np.asarray(s["key"], np.float32) if s.get("key") is not None else self.store.embed_fn(s["situation"])
        return Skill(s.get("id", ""), s["family"], s["situation"], tuple(s["actions"]), s["reward"], emb,
                     strategy=s.get("strategy", ""), card=s.get("card", ""))

    def commit(self, update: Update) -> None:
        added = [self.store.append(self.to_skill(s)) for s in update.payload["skills"]]
        self.store.expire()
        self._added[update.payload["revert_offset"]] = [sid for sid in added if sid is not None]

    def revert(self, update: Update) -> None:
        self.store.truncate(update.payload["revert_offset"])

    def persisted(self, update: Update) -> bool:
        """True only if this commit wrote at least one skill and all of them reload from disk.
        A commit that deduped to nothing changed nothing durable, so it did not survive anything."""
        added = self._added.get(update.payload["revert_offset"], [])
        fresh = SkillStore(self.store.path, self.store.embed_fn, self.store.max_size, self.store.dedupe_sim)
        stored = {s.id for s in fresh.skills()}
        return bool(added) and all(sid in stored for sid in added)
