"""FrozenLake (Gym's toy text env, non-slippery, as in VAGEN and ViSkill), drawn as pixels.

A level is (size, holes). Each seed places the start, the goal and the holes, and is redrawn until
BFS finds a path, so every task is solvable and has a known optimal length. Stepping into a hole ends
the episode with reward 0; reaching the goal ends it with reward 1.

The goal cell lives in task.goal, not only in the seed: hindsight relabelling moves it to where an
attempt actually ended, and reset() redraws the frames with the goal there.
"""
from collections import deque
from dataclasses import replace

import numpy as np

from envs.base import Env, Obs, SeededTasks, Task, parse_levels, split_for_seed

MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
CELL = 48
ICE, HOLE, GOAL, AGENT, GRIDLINE = (200, 225, 245), (20, 40, 90), (40, 180, 70), (220, 50, 50), (150, 180, 210)
RULES = ("You are the red circle on a frozen lake. Reach the green square. "
         "Dark blue squares are holes: stepping into one ends the game.")


def _cell(text: str) -> tuple[int, int]:
    r, c = text.split(",")
    return int(r), int(c)


def bfs(size: int, holes: frozenset, start: tuple[int, int], goal: tuple[int, int]) -> list[str] | None:
    """Shortest action list from start to goal avoiding holes, or None."""
    prev = {start: None}
    queue = deque([start])
    while queue:
        pos = queue.popleft()
        if pos == goal:
            path = []
            while prev[pos] is not None:
                pos, action = prev[pos]
                path.append(action)
            return path[::-1]
        for action, (dr, dc) in MOVES.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if 0 <= nxt[0] < size and 0 <= nxt[1] < size and nxt not in holes and nxt not in prev:
                prev[nxt] = (pos, action)
                queue.append(nxt)
    return None


def layout(seed: int, size: int, n_holes: int) -> tuple[tuple[int, int], tuple[int, int], frozenset]:
    """(start, goal, holes) for a seed: redrawn on a sub-stream until solvable."""
    for attempt in range(1000):
        rng = np.random.default_rng([seed, size, n_holes, attempt])
        cells = [(int(i) // size, int(i) % size) for i in rng.permutation(size * size)]
        start, goal, holes = cells[0], cells[1], frozenset(cells[2:2 + n_holes])
        if bfs(size, holes, start, goal):
            return start, goal, holes
    raise RuntimeError(f"no solvable FrozenLake layout for seed {seed}, level {(size, n_holes)}")


def render(size: int, holes: frozenset, goal: tuple[int, int], agent: tuple[int, int]) -> np.ndarray:
    img = np.empty((size * CELL, size * CELL, 3), np.uint8)
    img[:] = ICE
    img[::CELL, :] = img[:, ::CELL] = GRIDLINE
    for (r, c), color in [*((h, HOLE) for h in holes), (goal, GOAL)]:
        img[r * CELL + 4:(r + 1) * CELL - 4, c * CELL + 4:(c + 1) * CELL - 4] = color
    yy, xx = np.mgrid[:CELL, :CELL]
    disk = (yy - CELL / 2) ** 2 + (xx - CELL / 2) ** 2 <= (CELL * 0.35) ** 2
    r, c = agent
    img[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL][disk] = AGENT
    return img


class FrozenLakeEnv(SeededTasks, Env):
    families = {"frozenlake"}
    cell_px = CELL                   # lets a skill card draw the path (rsi/viskill.py)

    def __init__(self, env_seed: int = 0, heldout_n: int = 200, levels: tuple[tuple[int, int], ...] = ((4, 3),)):
        self.env_seed, self.heldout_n, self.levels = env_seed, heldout_n, parse_levels(levels)

    def make_task(self, seed: int, level: tuple[int, int] | None = None) -> Task:
        level = tuple(level or self.levels[seed % len(self.levels)])
        _, goal, _ = layout(seed, *level)
        return Task("frozenlake", seed, (f"{goal[0]},{goal[1]}",), 3 * level[0], split_for_seed(seed), level)

    @property
    def actions(self) -> list[str]:
        return list(MOVES)

    def _world(self, task: Task):
        size, n_holes = task.level
        start, _, holes = layout(task.seed, size, n_holes)
        return size, start, _cell(task.goal[0]), holes

    def reset(self, task: Task) -> Obs:
        self._task, self._t = task, 0
        self._size, self._agent, self._goal, self._holes = self._world(task)
        return self._obs()

    def step(self, action: str) -> tuple[Obs, float, bool, dict]:
        self._t += 1
        dr, dc = MOVES[action]
        r, c = self._agent
        self._agent = (min(max(r + dr, 0), self._size - 1), min(max(c + dc, 0), self._size - 1))
        fell, won = self._agent in self._holes, self._agent == self._goal
        done = fell or won or self._t >= self._task.budget
        events = ["fell"] if fell else ["goal"] if won else []
        return self._obs(), float(won), done, {"steps": self._t, "pos": self._agent, "events": events}

    def _obs(self) -> Obs:
        return Obs(render(self._size, self._holes, self._goal, self._agent),
                   f"{RULES} Step {self._t} of {self._task.budget}.")

    def solve(self, task: Task) -> list[str] | None:
        size, start, goal, holes = self._world(task)
        return bfs(size, holes, start, goal)

    def descriptor(self, task: Task) -> np.ndarray:
        """Goal offset from the start, plus the holes in a 5x5 window around the start (ViSkill's
        rule-based extractor, adapted): similar vectors mean the same opening moves work."""
        size, (sr, sc), (gr, gc), holes = self._world(task)
        window = [float((sr + dr, sc + dc) in holes or not (0 <= sr + dr < size and 0 <= sc + dc < size))
                  for dr in range(-2, 3) for dc in range(-2, 3)]
        v = np.array([4 * (gr - sr) / size, 4 * (gc - sc) / size, *window], np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def relabel(self, task: Task, actions: list[str]) -> tuple[Task, list[str]] | None:
        """Hindsight: the last safe cell this attempt stood on becomes the goal."""
        size, pos, _, holes = self._world(task)
        start, safe, reached = pos, [], pos
        for a in actions:
            dr, dc = MOVES[a]
            nxt = (min(max(pos[0] + dr, 0), size - 1), min(max(pos[1] + dc, 0), size - 1))
            if nxt in holes:
                break
            pos = nxt
            safe.append(a)
            reached = pos
        if reached == start:
            return None
        # cut the path where it first stands on the new goal: reaching it ends the episode there
        pos, cut = start, 0
        for i, a in enumerate(safe):
            dr, dc = MOVES[a]
            pos = (min(max(pos[0] + dr, 0), size - 1), min(max(pos[1] + dc, 0), size - 1))
            if pos == reached:
                cut = i + 1
                break
        return replace(task, goal=(f"{reached[0]},{reached[1]}",)), safe[:cut]
