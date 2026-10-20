"""From-scratch procedural verifiable tasks, pure numpy. Runs on CPU in seconds; the CI env.

Two families, so the act loop is real even without a simulator:
  count_objects  single step  — count the squares, answer say_N
  gather_order   multi step   — walk an 8x8 grid and pick 3 objects in a required order
Every task is a pure function of its seed, so a Task fully reproduces its world.
"""
import numpy as np

from envs.base import Env, Obs, SeededTasks, Task, split_for_seed

GRID, CELL = 8, 8                      # 8x8 cells of 8px -> 64x64 image
COLORS = {"red": (220, 40, 40), "green": (40, 200, 60), "blue": (50, 90, 230),
          "yellow": (230, 210, 40), "purple": (160, 60, 200)}
BACKGROUND, AGENT = (30, 30, 30), (255, 255, 255)
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
GATHER_BUDGET = 24


def make_task(seed: int) -> Task:
    rng = np.random.default_rng(seed)
    if seed % 2 == 0:
        n = int(rng.integers(1, 10))
        return Task("count_objects", seed, (str(n),), 1, split_for_seed(seed))
    order = tuple(str(c) for c in rng.choice(list(COLORS), size=3, replace=False))
    return Task("gather_order", seed, order, GATHER_BUDGET, split_for_seed(seed))


def render(cells: dict[tuple[int, int], tuple[int, int, int]], agent: tuple[int, int] | None = None) -> np.ndarray:
    img = np.empty((GRID * CELL, GRID * CELL, 3), np.uint8)
    img[:] = BACKGROUND
    for (r, c), color in cells.items():
        img[r * CELL + 1:(r + 1) * CELL - 1, c * CELL + 1:(c + 1) * CELL - 1] = color
    if agent is not None:                      # a white frame, so the object under the agent stays visible
        r, c = agent
        cell = img[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
        cell[0, :] = cell[-1, :] = cell[:, 0] = cell[:, -1] = AGENT
    return img


class StaticEnv(SeededTasks, Env):
    def __init__(self, env_seed: int = 0, heldout_n: int = 200):
        self.env_seed, self.heldout_n = env_seed, heldout_n
        self._task: Task | None = None

    def make_task(self, seed: int) -> Task:
        return make_task(seed)

    @property
    def actions(self) -> list[str]:
        return [f"say_{i}" for i in range(10)] + list(MOVES) + ["pick"]

    def reset(self, task: Task) -> Obs:
        self._task, self._steps = task, 0
        rng = np.random.default_rng([task.seed, 1])   # layout stream, independent of the goal stream in make_task
        cells = [(int(i) // GRID, int(i) % GRID) for i in rng.permutation(GRID * GRID)]
        if task.family == "count_objects":
            self._squares = {cell: COLORS["blue"] for cell in cells[:int(task.goal[0])]}
            self._agent = None
        else:
            self._objects = {color: cells[i] for i, color in enumerate(task.goal)}
            self._agent, self._collected = cells[3], []
        return self._obs()

    def step(self, action: str) -> tuple[Obs, float, bool, dict]:
        task = self._task
        self._steps += 1
        reward, done = 0.0, False
        if task.family == "count_objects":
            reward, done = float(action == f"say_{task.goal[0]}"), True
        elif action in MOVES:
            dr, dc = MOVES[action]
            r, c = self._agent
            self._agent = (min(max(r + dr, 0), GRID - 1), min(max(c + dc, 0), GRID - 1))
        elif action == "pick":
            here = [color for color, pos in self._objects.items() if pos == self._agent]
            if here:                           # picking an empty cell is a wasted step, not a failure
                want = task.goal[len(self._collected)]
                if here[0] != want:
                    done = True
                else:
                    self._collected.append(want)
                    del self._objects[want]
                    if len(self._collected) == len(task.goal):
                        reward, done = 1.0, True
        if self._steps >= task.budget:
            done = True
        return self._obs(), reward, done, {"steps": self._steps}

    def _obs(self) -> Obs:
        task = self._task
        if task.family == "count_objects":
            return Obs(render(self._squares), "How many blue squares are there? Answer with say_N.")
        cells = {pos: COLORS[color] for color, pos in self._objects.items()}
        got = ", ".join(self._collected) or "nothing"
        text = (f"You are the white frame. Pick up objects in this order: {', '.join(task.goal)}. "
                f"Collected so far: {got}.")
        return Obs(render(cells, self._agent), text)
