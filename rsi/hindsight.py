"""Hindsight relabelling (HER; Andrychowicz et al., 2017): a failed attempt is a success for the goal it did
reach. env.relabel(task, actions) names that goal; the turns are replayed on the relabelled task, so the
prompts and frames match the new goal, and the copy joins the batch as a success with origin="hindsight".

A signal-only method: it changes nothing itself. Learners that imitate successes use the copies (rsi/rest.py,
the skill library in rsi/memory.py and rsi/viskill.py); GRPO skips them, because a copy would join its own
original's group and fake an advantage. Put it before those methods in cfg.methods.
"""
from dataclasses import replace

from rsi.base import Method
from rsi.judges import ScoredTrajectory
from utils.rollout import moves, replay


class Hindsight(Method):
    name = "hindsight"

    def __init__(self, cfg):
        self.cfg = cfg

    def signal(self, batch: list, env) -> list:
        extra, seen = [], set()
        for s in batch:
            if s.reward >= 1.0 or s.flags or s.origin != "attempt":
                continue
            acted = [st for st in s.traj.steps if st.get("actions") or st.get("action")]
            rel = env.relabel(s.traj.task, moves(s.traj))
            if rel is None:
                continue
            task, actions = rel
            key = (task.family, task.seed, task.goal)
            per_task = sum(1 for k in seen if k[:2] == key[:2])
            if key in seen or per_task >= self.cfg.hindsight_per_task:
                continue
            turns, n = [], 0
            for st in acted:                     # the turns that hold the first len(actions) moves
                if n >= len(actions):
                    break
                turns.append(st)
                n += len(st.get("actions") or [st["action"]])
            traj = replay(env, task, turns, self.cfg, s.traj.context)   # ends where the new goal is reached
            if traj.total_reward >= 1.0:
                seen.add(key)
                extra.append(replace(s, traj=traj, reward=1.0, origin="hindsight", per_step_credit=None))
        return batch + extra
