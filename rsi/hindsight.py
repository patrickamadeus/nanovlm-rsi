"""Hindsight relabelling (HER; Andrychowicz et al., 2017): a failed attempt is a success for the goal it did
reach. env.relabel(task, actions) names that goal; the turns are replayed on the relabelled task, so the
prompts and frames match the new goal, and the copy joins the batch as a success with origin="hindsight".

A signal-only method: it changes nothing itself. Learners that imitate successes use the copies (rsi/rest.py,
the skill library in rsi/memory.py and rsi/viskill.py); GRPO skips them, because a copy would join its own
original's group and fake an advantage. Put it before those methods in cfg.methods.

With reasoning (answer_format "answer"), a copy's original thoughts are about the *old* goal, and training on
them would teach reasoning that contradicts the image. So each relabelled turn keeps only its actions, under a
neutral thought: "<think>Moving towards the goal.</think><answer>Down, Right</answer>", the same filler VAGEN
uses for a turn without reasoning, with its tokens rebuilt from that text.
"""
from dataclasses import replace

from rsi.base import Method
from rsi.judges import ScoredTrajectory
from utils.rollout import moves, replay


NEUTRAL = "Moving towards the goal."


class Hindsight(Method):
    name = "hindsight"

    def __init__(self, cfg, tokenizer=None):
        self.cfg, self.tokenizer = cfg, tokenizer

    def neutral(self, traj):
        """Swap each turn's reasoning for NEUTRAL, keeping what it did."""
        steps = []
        for st in traj.steps:
            text = f"<think>{NEUTRAL}</think><answer>{', '.join(a.capitalize() for a in st['actions'])}</answer>"
            ids = self.tokenizer(text, add_special_tokens=False).input_ids
            steps.append({**st, "text": text, "token_ids": ids, "logprobs": None})
        return replace(traj, steps=steps, tokens_out=sum(len(st["token_ids"]) for st in steps))

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
                if self.cfg.answer_format == "answer":
                    traj = self.neutral(traj)
                seen.add(key)
                extra.append(replace(s, traj=traj, reward=1.0, origin="hindsight", per_step_credit=None))
        return batch + extra
