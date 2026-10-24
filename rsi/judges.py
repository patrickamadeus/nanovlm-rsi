"""Judges turn a trajectory into a reward. Each has one method: score(traj) -> ScoredTrajectory.

  OutcomeJudge    the environment's ground truth: it already knows whether the goal was reached
  VLMJudge        a separate, frozen VLM asked "did the agent achieve the goal?" — for goals with no checker
  CompositeJudge  ground truth where the env has it, the VLM elsewhere. On ground-truth tasks it also runs
                  the VLM at judge_sample_rate *without* letting it change the reward, so the log collects
                  judge-vs-truth disagreements: the data a judge-improvement mode would learn from.

The reward says where it came from (verdict_source), which the gate checks: an update learned from a
signal you won't have at deployment is refused. The VLM judge is a different model from the policy on
purpose: one sharing the policy's weights shares its blind spots.
"""
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from utils.rollout import Trajectory


@dataclass(frozen=True)
class ScoredTrajectory:
    traj: Trajectory
    reward: float
    per_step_credit: list[float] | None
    verdict_source: str        # "outcome" | "judge" — where the signal came from
    confidence: float
    flags: tuple[str, ...]     # reward-hacking and degeneracy tripwires
    judge_verdict: float | None = None    # sampled VLM opinion on a ground-truth task (agreement logging)
    judge_tokens: int = 0
    origin: str = "attempt"    # "attempt" | "hindsight": a relabelled copy of a failed attempt (rsi/hindsight.py)


def hack_flags(traj: Trajectory) -> tuple[str, ...]:
    flags = []
    if traj.total_reward > 0 and traj.env_steps == 0:
        flags.append("zero_step_success")            # almost always an env bug, never a skill
    actions = [s["action"] for s in traj.steps if s.get("action") is not None]
    if len(actions) >= 8 and Counter(actions).most_common(1)[0][1] >= 0.8 * len(actions):
        flags.append("action_loop")
    if traj.parse_fails / max(traj.env_steps, 1) > 0.5:
        flags.append("noop_spam")
    return tuple(flags)


class OutcomeJudge:
    """Never re-derives success, only reads it from the env."""

    def score(self, traj: Trajectory) -> ScoredTrajectory:
        return ScoredTrajectory(traj, traj.total_reward, None, "outcome", 1.0, hack_flags(traj))


class YesNo:
    """P(yes) from a frozen VLM, read off the logprobs of ' yes' vs ' no' after a prefilled 'Answer:'."""

    def __init__(self, policy_cfg):
        from models.vlm import Policy
        self.policy = Policy(replace(policy_cfg, response_prefix="Answer:"))
        tok = self.policy.processor.tokenizer
        self.yes = tok(" yes", add_special_tokens=False).input_ids
        self.no = tok(" no", add_special_tokens=False).input_ids

    def p_yes_given(self, prompt: str, images: list[np.ndarray]) -> tuple[float, int]:
        import torch
        with torch.no_grad():
            lp_yes = self.policy.logprobs(prompt, images, self.yes).sum().item()
            lp_no = self.policy.logprobs(prompt, images, self.no).sum().item()
        tokens = int(self.policy._inputs(prompt, images)["input_ids"].shape[1])
        return float(1 / (1 + np.exp(lp_no - lp_yes))), tokens


class VLMJudge:
    def __init__(self, yes_no):
        self.yes_no = yes_no

    def score(self, traj: Trajectory) -> ScoredTrajectory:
        context = traj.steps[0].get("obs_text") if traj.steps else None
        actions = [s["action"] for s in traj.steps if s.get("action") is not None][-8:]
        prompt = (f"An agent was given this task: {context or ' '.join(traj.task.goal)}\n"
                  f"Its actions were: {', '.join(actions) or 'none'}.\n"
                  "The image shows the final state. Did the agent achieve the goal? Answer yes or no.")
        images = [traj.final_rgb] if traj.final_rgb is not None else []
        p, tokens = self.yes_no.p_yes_given(prompt, images)
        verdict = float(p >= 0.5)
        return ScoredTrajectory(traj, verdict, None, "judge", max(p, 1 - p), hack_flags(traj), verdict, tokens)


class CompositeJudge:
    def __init__(self, outcome: OutcomeJudge, vlm: VLMJudge, native_families: set[str], sample_rate: float,
                 seed: int = 0):
        self.outcome, self.vlm, self.native, self.rate = outcome, vlm, set(native_families), sample_rate
        self.rng = np.random.default_rng(seed)

    def score(self, traj: Trajectory) -> ScoredTrajectory:
        if traj.task.family not in self.native:
            return self.vlm.score(traj)
        scored = self.outcome.score(traj)
        if self.rng.random() < self.rate:    # agreement logging: record the VLM, never let it decide
            j = self.vlm.score(traj)
            scored = replace(scored, judge_verdict=j.reward, judge_tokens=j.judge_tokens)
        return scored
