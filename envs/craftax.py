"""Craftax-Classic (Matthews et al., 2024) behind the four-method Env contract. The only file that imports JAX.

A task is (world seed, one target achievement, step budget). Reward is 1.0 the step the target
achievement first unlocks, which also ends the episode. Pixel observations cross the JAX -> numpy
boundary here and nowhere else.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # let JAX and PyTorch share one GPU

import jax  # noqa: E402
import numpy as np  # noqa: E402
from craftax.craftax_classic.constants import Achievement, Action  # noqa: E402
from craftax.craftax_env import make_craftax_env_from_name  # noqa: E402

from envs.base import Env, Obs, SeededTasks, Task, split_for_seed  # noqa: E402

# early-game achievements a small policy can plausibly reach inside a short budget
GOALS = ("collect_wood", "collect_sapling", "collect_drink", "place_plant", "eat_cow",
         "place_table", "defeat_zombie", "collect_stone", "make_wood_pickaxe", "make_wood_sword")
ACHIEVEMENT_INDEX = {a.name.lower(): a.value for a in Achievement}
ACTIONS = [a.name.lower() for a in Action]
ITEMS = ("wood", "stone", "coal", "iron", "diamond", "sapling", "wood_pickaxe", "stone_pickaxe",
         "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword")


def achievement_reward(before: np.ndarray, after: np.ndarray, goal_index: int) -> float:
    return float(bool(after[goal_index]) and not bool(before[goal_index]))


class CraftaxEnv(SeededTasks, Env):
    families = {"craftax_achieve"}
    _jax_env = _reset = _step = None             # compiled once per process, shared by every instance

    def __init__(self, env_seed: int = 0, heldout_n: int = 200, budget: int = 64):
        self.env_seed, self.heldout_n, self.budget = env_seed, heldout_n, budget
        if CraftaxEnv._jax_env is None:
            CraftaxEnv._jax_env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
            CraftaxEnv._reset = jax.jit(CraftaxEnv._jax_env.reset)
            CraftaxEnv._step = jax.jit(CraftaxEnv._jax_env.step)
        self.params = CraftaxEnv._jax_env.default_params

    def make_task(self, seed: int) -> Task:
        goal = GOALS[int(np.random.default_rng(seed).integers(len(GOALS)))]
        return Task("craftax_achieve", seed, (goal,), self.budget, split_for_seed(seed))

    def task_key(self, task: Task) -> str:
        return f"{task.family}:{task.goal[0]}"          # one key per achievement

    @property
    def actions(self) -> list[str]:
        return ACTIONS

    def reset(self, task: Task) -> Obs:
        self._task, self._t = task, 0
        self._key = jax.random.PRNGKey(task.seed)
        obs, self._state = CraftaxEnv._reset(self._key, self.params)
        return self._obs(obs)

    def step(self, action: str) -> tuple[Obs, float, bool, dict]:
        self._t += 1
        before = np.asarray(self._state.achievements)
        key = jax.random.fold_in(self._key, self._t)
        obs, self._state, _, env_done, _ = CraftaxEnv._step(key, self._state, ACTIONS.index(action), self.params)
        reward = achievement_reward(before, np.asarray(self._state.achievements),
                                    ACHIEVEMENT_INDEX[self._task.goal[0]])
        done = bool(reward) or bool(env_done) or self._t >= self._task.budget
        return self._obs(obs), reward, done, {"steps": self._t}

    def _obs(self, obs) -> Obs:
        rgb = (np.asarray(obs) * 255).clip(0, 255).astype(np.uint8)
        inv = self._state.inventory
        held = [f"{name} {int(getattr(inv, name))}" for name in ITEMS if int(getattr(inv, name)) > 0]
        text = (f"Goal: {self._task.goal[0].replace('_', ' ')}. Inventory: {', '.join(held) or 'empty'}. "
                f"Step {self._t} of {self._task.budget}.")
        return Obs(rgb, text)
