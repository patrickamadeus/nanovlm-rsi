"""A new self-improvement method or planner, step by step. Copy this file to rsi/<method name>.py (named after
the method or paper, e.g. reflexion.py), keep the hooks you need, fill in each NotImplementedError, then add one
branch to make_method() / make_planner() in rsi/__init__.py and a test file.

What the gate relies on (tests/test_template.py checks it for every registered method):
  - propose() never changes the agent; it only describes a change (the gate decides whether it happens)
  - commit() writes the change to disk; revert() restores the exact previous state, on disk and in memory
  - persisted() rebuilds from disk and is False if the commit changed nothing
  - Update.signal says where the evidence came from ("outcome" = the env's verifier); the gate refuses
    signals not in cfg.gate_allowed_signals, so a method can't learn from something unavailable at deployment

Each scored attempt in `batch` is a ScoredTrajectory (rsi/judges.py): .reward, .flags, and .traj with
.task, .steps (per turn: action, prompt, images, token_ids, logprobs, events) and .total_reward.
"""
from envs.base import Task
from rsi.base import Method, Planner
from rsi.gate import Update
from utils.log import Cost
from utils.rollout import Context


class TemplateMethod(Method):
    name = "template"
    changes = "memory"                           # "weights" | "memory" | None if it never updates

    def __init__(self, policy, cfg, run_dir: str):
        self.policy, self.cfg, self.run_dir = policy, cfg, run_dir

    def context(self, task: Task, env) -> Context:
        """Optional. Return text (and images, referenced with <image>) to show the policy before this task.
        Example: retrieve the k most similar stored skills (rsi/memory.py, rsi/viskill.py)."""
        raise NotImplementedError

    def signal(self, batch: list, env) -> list:
        """Optional. Return a new batch: change rewards (rsi/viskill.py adds a progress reward to failures)
        or add attempts (rsi/hindsight.py relabels failures as successes for the goal they did reach)."""
        raise NotImplementedError

    def propose(self, batch: list) -> Update | None:
        """Describe one change, or None if the batch carries no usable signal. Put everything commit() and
        revert() need in the payload (a weight snapshot, a file offset)."""
        payload: dict = ...
        raise NotImplementedError
        return Update(self.name, "outcome", payload, Cost(), tuple(f"{s.traj.task.family}:{s.traj.task.seed}"
                                                                   for s in batch))

    def commit(self, update: Update) -> None:
        """Apply the change and write it to disk."""
        raise NotImplementedError

    def revert(self, update: Update) -> None:
        """Undo the change exactly, on disk and in memory."""
        raise NotImplementedError

    def persisted(self, update: Update) -> bool:
        """Reload from disk: True only if the change is there and actually changed something."""
        raise NotImplementedError

    def observe(self, batch: list) -> None:
        """Optional. Bookkeeping after the gate, e.g. which stored items were in context when an attempt won."""
        raise NotImplementedError


class TemplatePlanner(Planner):
    name = "template"

    def __init__(self, env, cfg, rng):
        self.env, self.cfg, self.rng = env, cfg, rng

    def tasks(self, n: int) -> list[Task]:
        """Pick n train tasks. Draw from self.env.tasks("train", ...) or build new ones with
        self.env.make_task(seed, level); return them through rsi.base.train_only()."""
        raise NotImplementedError

    def updates(self, names: list[str]) -> list[str]:
        """Optional. Which updating methods run this iteration (rsi/exp3.py learns this from gate results)."""
        raise NotImplementedError

    def observe(self, batch: list, results: dict) -> None:
        """Optional. Learn from the scored attempts and from results = {method name: GateResult}."""
        raise NotImplementedError
