"""The ground-truth judge: the environment already knows whether the goal was reached.

Every judge returns a ScoredTrajectory — the reward plus where it came from (verdict_source),
which the gate checks: an update learned from a signal you won't have at deployment is refused.
"""
from collections import Counter
from dataclasses import dataclass

from harness.rollout import Trajectory


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
