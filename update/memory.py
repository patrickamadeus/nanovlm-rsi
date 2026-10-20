"""Memory carrier (A3): successful trajectories become retrievable skills."""
from infra.memory import Skill, SkillStore
from infra.trace import Cost
from update.base import Update, Updater


def situation(scored) -> str:
    # what the policy saw, never task.goal: for count_objects the goal is the answer itself
    return scored.traj.steps[0].get("obs_text", "") if scored.traj.steps else ""


class MemoryUpdater(Updater):
    def __init__(self, store: SkillStore, cfg):
        self.store, self.cfg = store, cfg

    def propose(self, batch) -> Update | None:
        keep = [s for s in batch if s.reward >= self.cfg.mem_write_threshold and not s.flags]
        if not keep:
            return None
        skills = [{"family": s.traj.task.family, "situation": situation(s),
                   "actions": [st["action"] for st in s.traj.steps if st["action"] is not None],
                   "reward": s.reward} for s in keep]
        cost = Cost(sum(s.traj.env_steps for s in batch), sum(s.traj.tokens_out for s in batch))
        provenance = tuple(f"{s.traj.task.family}:{s.traj.task.seed}" for s in keep)
        signal = keep[0].verdict_source if len({s.verdict_source for s in keep}) == 1 else "mixed"
        return Update("A3", "B0", "C3", "between_episodes", signal,
                      {"skills": skills, "revert_offset": self.store.offset()}, cost, provenance)

    def commit(self, update: Update) -> None:
        for s in update.payload["skills"]:
            self.store.append(Skill("", s["family"], s["situation"], tuple(s["actions"]), s["reward"],
                                    self.store.embed_fn(s["situation"])))
        self.store.expire()

    def revert(self, update: Update) -> None:
        self.store.truncate(update.payload["revert_offset"])

    def persisted(self, update: Update) -> bool:
        fresh = SkillStore(self.store.path, self.store.embed_fn, self.store.max_size, self.store.dedupe_sim)
        stored = {s.situation for s in fresh.skills()}
        return all(s["situation"] in stored for s in update.payload["skills"])
