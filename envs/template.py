"""A new environment, step by step. Copy this file to envs/<name>.py, fill in every method that raises
NotImplementedError, then add one branch to make_env() in envs/__init__.py and a test file.

The contract every env must keep (tests/test_template.py checks it on each real env):
  - a task is a pure function of its seed (and level): reset(task) twice gives the same first frame
  - task.split == split_for_seed(task.seed); held-out tasks are frozen per env_seed (SeededTasks does both)
  - reset() *rebinds* episode state instead of mutating it, so fork() can be a shallow copy
  - reward is the verifier: 1.0 when the goal is reached, so OutcomeJudge never has to re-derive success

Optional hooks have defaults in envs/base.py; implement them to unlock methods:
  solve       -> solver baselines and ViSkill's length check   (frozenlake.py: BFS)
  descriptor  -> ViSkill skill retrieval                        (sokoban.py: player/box/target offsets)
  relabel     -> rsi/hindsight.py                               (both grid envs)
  task_key    -> what planners track success in                (craftax.py: one key per achievement)
"""
import numpy as np

from envs.base import Env, Obs, SeededTasks, Task, split_for_seed


class TemplateEnv(SeededTasks, Env):
    families = {"template"}                       # task families whose reward is ground truth

    def __init__(self, env_seed: int = 0, heldout_n: int = 200, levels: tuple = ((),)):
        self.env_seed, self.heldout_n, self.levels = env_seed, heldout_n, levels

    def make_task(self, seed: int, level: tuple | None = None) -> Task:
        """Draw the goal from the seed. Keep in task.goal whatever the verifier reads (and relabel may change)."""
        level = tuple(level or self.levels[seed % len(self.levels)])
        goal: tuple[str, ...] = ...               # e.g. the target cell, the answer, the achievement name
        raise NotImplementedError
        return Task("template", seed, goal, 16, split_for_seed(seed), level)

    @property
    def actions(self) -> list[str]:
        """The legal action strings. They appear verbatim in the prompt, so make them words a VLM would say."""
        raise NotImplementedError

    def reset(self, task: Task) -> Obs:
        """Build the world from task.seed (and task.goal, task.level) and return the first observation."""
        self._task, self._t = task, 0
        raise NotImplementedError

    def step(self, action: str) -> tuple[Obs, float, bool, dict]:
        """Apply one legal action. Return (obs, reward, done, info); done at the goal, on failure, or at
        task.budget. Put milestones in info["events"] (e.g. "pushed", "fell"): methods and logs read them."""
        self._t += 1
        raise NotImplementedError

    def _obs(self) -> Obs:
        """An [H, W, 3] uint8 image (mandatory) and optional text: the rules and the step count, never the answer."""
        raise NotImplementedError

    # optional: delete the ones you don't implement, the defaults in envs/base.py return None

    def solve(self, task: Task) -> list[str] | None:
        raise NotImplementedError

    def descriptor(self, task: Task) -> np.ndarray | None:
        raise NotImplementedError

    def relabel(self, task: Task, actions: list[str]) -> tuple[Task, list[str]] | None:
        raise NotImplementedError
