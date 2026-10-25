"""Sokoban (as in VAGEN, ViSkill and gym-sokoban), procedurally generated, drawn as pixels.

A level is (size, boxes). Generation follows gym-sokoban: carve a room by random walk, put each box on
its target, then play *backwards* (the player walks and may pull a box behind it), so every level is
solvable by construction, and no box may start on its target. BFS then gives the optimal length. A box pushed into a corner off its target
can never move again, so that ends the episode early instead of wasting the rest of the budget.

The reward is 1.0 for solving, so success reads the same in every env. gym-sokoban's dense reward
(-0.1 per step, +1 / -1 for a box onto / off a target, +10 for solving), divided by 10 to match, is
reported per step as info["shaping"]; the judge adds it, so GRPO sees progress short of a solve.

The targets live in task.goal: hindsight relabelling moves them to where the boxes actually ended.
"""
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from envs.base import Env, Obs, SeededTasks, Task, parse_levels, split_for_seed, train_seed

MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
CELL = 48
WALL, MORTAR, FLOOR = (110, 110, 110), (70, 70, 70), (35, 35, 35)
TARGET, BOX, BOX_DONE, PLAYER = (220, 50, 50), (190, 140, 60), (60, 180, 80), (60, 120, 230)
# The rules are in prompts/sokoban.txt (VAGEN's); this says how *our* drawing shows them, since it is not gym's art.
LEGEND = ("In the image: the blue circle is you, brown squares with an X are boxes, red outlines are targets, "
          "a green box is on its target, grey bricks are walls.")


@dataclass(frozen=True)
class World:
    walls: frozenset
    player: tuple[int, int]
    boxes: frozenset
    targets: frozenset


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _cells(text: tuple[str, ...]) -> frozenset:
    return frozenset(tuple(int(x) for x in t.split(",")) for t in text)


def move(walls: frozenset, player, boxes: frozenset, action: str):
    """One step: walk, or push the box in front if the cell behind it is free. Blocked moves change nothing."""
    d = MOVES[action]
    nxt = _add(player, d)
    if nxt in walls:
        return player, boxes
    if nxt in boxes:
        beyond = _add(nxt, d)
        if beyond in walls or beyond in boxes:
            return player, boxes
        return nxt, (boxes - {nxt}) | {beyond}
    return nxt, boxes


def solve_level(walls: frozenset, player, boxes: frozenset, targets: frozenset,
                max_states: int = 200_000) -> list[str] | None:
    """Shortest action list that puts every box on a target, by BFS over (player, boxes)."""
    start = (player, boxes)
    prev = {start: None}
    queue = deque([start])
    while queue and len(prev) < max_states:
        state = queue.popleft()
        if state[1] == targets:
            path = []
            while prev[state] is not None:
                state, action = prev[state]
                path.append(action)
            return path[::-1]
        for action in MOVES:
            nxt = move(walls, *state, action)
            if nxt not in prev:
                prev[nxt] = (state, action)
                queue.append(nxt)
    return None


def cornered(walls: frozenset, box, targets: frozenset) -> bool:
    if box in targets:
        return False
    up, down = _add(box, (-1, 0)) in walls, _add(box, (1, 0)) in walls
    left, right = _add(box, (0, -1)) in walls, _add(box, (0, 1)) in walls
    return (up or down) and (left or right)


def generate(seed: int, size: int, n_boxes: int) -> World:
    for attempt in range(1000):
        rng = np.random.default_rng([seed, size, n_boxes, attempt])
        interior = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
        pos = interior[int(rng.integers(len(interior)))]
        floor = {pos}
        while len(floor) < 0.75 * len(interior):                    # carve the room by random walk
            d = list(MOVES.values())[int(rng.integers(4))]
            nxt = _add(pos, d)
            if 1 <= nxt[0] < size - 1 and 1 <= nxt[1] < size - 1:
                pos = nxt
                floor.add(pos)
        walls = frozenset((r, c) for r in range(size) for c in range(size)) - floor
        cells = sorted(floor)
        picks = rng.permutation(len(cells))[:n_boxes + 1]
        targets = frozenset(cells[i] for i in picks[:n_boxes])
        player, boxes = cells[picks[-1]], targets
        for _ in range(int(rng.integers(6 * size, 12 * size))):     # reverse play: walk, often pull
            d = list(MOVES.values())[int(rng.integers(4))]
            nxt, behind = _add(player, d), _add(player, (-d[0], -d[1]))
            if nxt in walls or nxt in boxes:
                continue
            if behind in boxes and rng.random() < 0.8:
                boxes = (boxes - {behind}) | {player}
            player = nxt
        if boxes & targets or any(cornered(walls, b, targets) for b in boxes):
            continue
        path = solve_level(walls, player, boxes, targets)
        if path and len(path) >= 3:
            return World(walls, player, boxes, targets)
    raise RuntimeError(f"no Sokoban level for seed {seed}, level {(size, n_boxes)}")


def render(size: int, walls: frozenset, targets: frozenset, boxes: frozenset, player) -> np.ndarray:
    img = np.empty((size * CELL, size * CELL, 3), np.uint8)
    img[:] = FLOOR
    for r, c in walls:
        cell = img[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
        cell[:] = WALL
        cell[::CELL // 3, :] = MORTAR                                  # bricks, so walls read as walls
        cell[:, ::CELL // 2] = MORTAR
    for r, c in targets:
        cell = img[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
        cell[6:-6, 6:12] = cell[6:-6, -12:-6] = cell[6:12, 6:-6] = cell[-12:-6, 6:-6] = TARGET
    for r, c in boxes:
        cell = img[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
        cell[8:-8, 8:-8] = BOX_DONE if (r, c) in targets else BOX
        idx = np.arange(8, CELL - 8)
        cell[idx, idx] = cell[idx, CELL - 1 - idx] = MORTAR             # an X, so a box is not just a square
    yy, xx = np.mgrid[:CELL, :CELL]
    disk = (yy - CELL / 2) ** 2 + (xx - CELL / 2) ** 2 <= (CELL * 0.33) ** 2
    r, c = player
    img[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL][disk] = PLAYER
    return img


class SokobanEnv(SeededTasks, Env):
    families = {"sokoban"}
    cell_px = CELL

    def __init__(self, env_seed: int = 0, heldout_n: int = 200, levels: tuple[tuple[int, int], ...] = ((6, 1),)):
        self.env_seed, self.heldout_n, self.levels = env_seed, heldout_n, parse_levels(levels)

    def make_task(self, seed: int, level: tuple[int, int] | None = None) -> Task:
        level = tuple(level or self.levels[seed % len(self.levels)])
        world = generate(seed, *level)
        goal = tuple(f"{r},{c}" for r, c in sorted(world.targets))
        return Task("sokoban", seed, goal, 4 * level[0] - 4, split_for_seed(seed), level)

    @property
    def actions(self) -> list[str]:
        return list(MOVES)

    def _start(self, task: Task) -> World:
        return replace(generate(task.seed, *task.level), targets=_cells(task.goal))

    def reset(self, task: Task) -> Obs:
        self._task, self._t = task, 0
        self._world = self._start(task)
        self._player, self._boxes = self._world.player, self._world.boxes
        self._boxes_at_reset = self._boxes
        return self._obs()

    def step(self, action: str) -> tuple[Obs, float, bool, dict]:
        self._t += 1
        w = self._world
        before = self._boxes
        self._player, self._boxes = move(w.walls, self._player, self._boxes, action)
        solved = self._boxes == w.targets
        stuck = self._boxes != before and any(cornered(w.walls, b, w.targets) for b in self._boxes)
        shaping = (-0.1 + len(self._boxes & w.targets) - len(before & w.targets)) / 10
        events = (["pushed"] if self._boxes != before else []) + (["solved"] if solved else []) + \
                 (["deadlock"] if stuck and not solved else [])
        done = solved or stuck or self._t >= self._task.budget
        return self._obs(), float(solved), done, {"steps": self._t, "pos": self._player, "events": events,
                                                     "shaping": shaping}

    def _obs(self) -> Obs:
        w = self._world
        return Obs(render(self._task.level[0], w.walls, w.targets, self._boxes, self._player),
                   f"{LEGEND} Step {self._t} of {self._task.budget}.")

    def edit(self, task: Task, rng: np.random.Generator) -> Task | None:
        """ACCEL's editor: one step bigger or smaller, one box more or fewer, on a fresh train layout."""
        size, boxes = task.level
        size = int(np.clip(size + rng.integers(-1, 2), 5, 9))
        boxes = int(np.clip(boxes + rng.integers(-1, 2), 1, 3))
        return self.make_task(train_seed(rng), (size, boxes))

    def solve(self, task: Task) -> list[str] | None:
        w = self._start(task)
        return solve_level(w.walls, w.player, w.boxes, w.targets)

    def descriptor(self, task: Task) -> np.ndarray:
        """Per box: offset from the player, offset to its nearest target, walls around it (ViSkill's
        player-box, box-target and wall relations)."""
        w, size = self._start(task), task.level[0]
        parts = []
        for b in sorted(w.boxes):
            t = min(w.targets, key=lambda t: abs(t[0] - b[0]) + abs(t[1] - b[1]))
            around = [float((b[0] + dr, b[1] + dc) in w.walls) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
            parts += [4 * (b[0] - w.player[0]) / size, 4 * (b[1] - w.player[1]) / size,
                      4 * (t[0] - b[0]) / size, 4 * (t[1] - b[1]) / size, *around]
        v = np.array(parts, np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def relabel(self, task: Task, actions: list[str]) -> tuple[Task, list[str]] | None:
        """Hindsight: wherever the boxes ended up become the targets, if the attempt pushed anything and
        left no box cornered: on a real level that push is the deadlock, so it is never a goal to imitate."""
        w = self._start(task)
        player, boxes, states = w.player, w.boxes, []
        for a in actions:
            player, boxes = move(w.walls, player, boxes, a)
            states.append(boxes)
        if boxes == w.boxes or any(cornered(w.walls, b, w.targets) for b in boxes):
            return None
        cut = states.index(boxes) + 1               # the episode ends the first time the boxes sit there
        return replace(task, goal=tuple(f"{r},{c}" for r, c in sorted(boxes))), list(actions[:cut])
