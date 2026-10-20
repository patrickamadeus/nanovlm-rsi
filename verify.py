"""Verifiers turn a trajectory into a reward plus where that verdict came from.

  OutcomeVerifier    environment ground truth
  JudgeVerifier      a separate, frozen VLM asked "did the agent achieve the goal?" — for goals with no checker
  CompositeVerifier  routes env-native goals to outcome and everything else to the judge. On native tasks it
                     also runs the judge at judge_sample_rate *without* letting it change the reward, so the
                     trace accumulates judge-vs-truth disagreements: the training data update/verifier.py
                     would learn from.

The judge is a different model from the policy on purpose: one sharing the policy's weights shares its blind spots.
"""
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from config import LoopConfig
from envs.base import Task


@dataclass(frozen=True)
class Trajectory:
    task: Task
    steps: list[dict]          # one dict per budget step; action None means a parse failure no-op
    total_reward: float        # unmodified environment reward — shaping is kept out of it
    env_steps: int
    parse_fails: int
    tokens_out: int
    tokens_in: int = 0         # prompt tokens: memory text makes prompts longer, so this is real cost
    model_seconds: float = 0.0 # wall time inside policy calls, CUDA-synchronized
    wall_ms: float = 0.0
    final_rgb: np.ndarray | None = None   # what the judge looks at


@dataclass(frozen=True)
class ScoredTrajectory:
    traj: Trajectory
    reward: float
    per_step_credit: list[float] | None
    verdict_source: str        # axis E: where the signal came from
    confidence: float
    flags: tuple[str, ...]     # reward-hacking and degeneracy tripwires
    judge_verdict: float | None = None    # sampled judge opinion on a ground-truth task (agreement logging)
    judge_tokens: int = 0


class Verifier(ABC):
    @abstractmethod
    def score(self, traj: Trajectory) -> ScoredTrajectory: ...


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


class OutcomeVerifier(Verifier):
    """The env is ground truth; this verifier never re-derives success, it only reads it."""

    def score(self, traj: Trajectory) -> ScoredTrajectory:
        return ScoredTrajectory(traj, traj.total_reward, None, "outcome", 1.0, hack_flags(traj))


class VLMJudge:
    """P(yes) from a frozen VLM, read off the logprobs of ' yes' vs ' no' after a prefilled 'Answer:'."""

    def __init__(self, cfg):
        from policy.vlm import Policy
        self.policy = Policy(replace(cfg, response_prefix="Answer:"))
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


class JudgeVerifier(Verifier):
    def __init__(self, judge):
        self.judge = judge

    def score(self, traj: Trajectory) -> ScoredTrajectory:
        context = traj.steps[0].get("obs_text") if traj.steps else None
        actions = [s["action"] for s in traj.steps if s.get("action") is not None][-8:]
        prompt = (f"An agent was given this task: {context or ' '.join(traj.task.goal)}\n"
                  f"Its actions were: {', '.join(actions) or 'none'}.\n"
                  "The image shows the final state. Did the agent achieve the goal? Answer yes or no.")
        images = [traj.final_rgb] if traj.final_rgb is not None else []
        p, tokens = self.judge.p_yes_given(prompt, images)
        verdict = float(p >= 0.5)
        return ScoredTrajectory(traj, verdict, None, "judge", max(p, 1 - p), hack_flags(traj), verdict, tokens)


class CompositeVerifier(Verifier):
    def __init__(self, outcome: Verifier, judge: JudgeVerifier, native_families: set[str], sample_rate: float,
                 seed: int = 0):
        self.outcome, self.judge, self.native, self.rate = outcome, judge, set(native_families), sample_rate
        self.rng = np.random.default_rng(seed)

    def score(self, traj: Trajectory) -> ScoredTrajectory:
        if traj.task.family not in self.native:
            return self.judge.score(traj)
        scored = self.outcome.score(traj)
        if self.rng.random() < self.rate:    # agreement logging: record the judge, never let it decide
            j = self.judge.score(traj)
            scored = replace(scored, judge_verdict=j.reward, judge_tokens=j.judge_tokens)
        return scored


def make_verifier(name: str, cfg: LoopConfig, env=None) -> Verifier:
    if name == "outcome":
        return OutcomeVerifier()
    if name in ("judge", "composite"):
        from config import PolicyConfig
        judge = JudgeVerifier(VLMJudge(PolicyConfig(model_id=cfg.judge_model_id, tiny=cfg.policy.tiny,
                                                    device=cfg.policy.device, dtype=cfg.policy.dtype)))
        if name == "judge":
            return judge
        return CompositeVerifier(OutcomeVerifier(), judge, getattr(env, "families", set()),
                                 cfg.judge_sample_rate, cfg.seed)
    raise ValueError(f"unknown verifier {name!r}")
