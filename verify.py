"""Verifiers turn a trajectory into a reward plus where that verdict came from.

OutcomeVerifier reads environment ground truth. JudgeVerifier (a separate small
frozen VLM) and CompositeVerifier land in Slice 4, together with agreement
logging: on env-native tasks the judge also runs at judge_sample_rate, and the
trace fields `judge_verdict` and `verdict_source` accumulate the disagreement
data that update/verifier.py will eventually train on.
"""
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ScoredTrajectory:
    traj: Trajectory
    reward: float
    per_step_credit: list[float] | None
    verdict_source: str        # axis E: where the signal came from
    confidence: float
    flags: tuple[str, ...]     # reward-hacking and degeneracy tripwires


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


def make_verifier(name: str, cfg: LoopConfig) -> Verifier:
    if name == "outcome":
        return OutcomeVerifier()
    if name in ("judge", "composite"):
        raise NotImplementedError(f"{name} verifier lands in Slice 4")
    raise ValueError(f"unknown verifier {name!r}")
