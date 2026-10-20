"""The environment contract. Every env — numpy or JAX — fits these four methods."""
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Task:
    family: str            # "count_objects" | "gather_order" | ...
    seed: int              # drives procedural generation, fully reproducible
    goal: tuple[str, ...]  # goal spec the verifier reads
    budget: int            # max env steps — part of matched-cost accounting
    split: str             # "train" | "gate" | "heldout" — set by the harness, never by a proposer


@dataclass(frozen=True)
class Obs:
    rgb: np.ndarray        # [H,W,3] uint8 — mandatory, the policy is genuinely multimodal
    text: str | None = None


def split_for_seed(seed: int) -> str:
    # crc32, not hash(): builtin hash is salted per process, which would move seeds between splits across runs
    bucket = zlib.crc32(str(seed).encode()) % 10
    if bucket <= 6:
        return "train"
    if bucket == 7:
        return "gate"      # touched only by accept/reject decisions
    return "heldout"       # touched only by reporting


def stamp_split(task: Task) -> Task:
    """The harness, not the caller, decides which split a task belongs to."""
    return replace(task, split=split_for_seed(task.seed))


def draw_tasks(make_task, split: str, n: int, rng: np.random.Generator) -> list[Task]:
    """Sample n tasks of `split` by rejection on the seed hash, so the split is a property of the seed."""
    out = []
    for _ in range(n * 100):
        seed = int(rng.integers(0, 2**31 - 1))
        if split_for_seed(seed) == split:
            out.append(make_task(seed))
            if len(out) == n:
                return out
    raise RuntimeError(f"could not draw {n} {split!r} tasks; split buckets may be misconfigured")


class SeededTasks:
    """Mixin for envs whose tasks are a pure function of a seed: train/gate drawn from rng, held-out frozen."""
    env_seed: int
    heldout_n: int
    _heldout: list[Task] | None = None

    def make_task(self, seed: int) -> Task:
        raise NotImplementedError

    def tasks(self, split: str, n: int, rng: np.random.Generator) -> list[Task]:
        if split == "heldout":
            if self._heldout is None:      # frozen: drawn once per env_seed, ignores the caller's rng
                self._heldout = draw_tasks(self.make_task, "heldout", self.heldout_n,
                                           np.random.default_rng(self.env_seed))
            if n > len(self._heldout):
                raise ValueError(f"asked for {n} held-out tasks, only {len(self._heldout)} are frozen")
            return self._heldout[:n]
        return draw_tasks(self.make_task, split, n, rng)


class Env(ABC):
    @abstractmethod
    def reset(self, task: Task) -> Obs:
        """Build the world for `task` deterministically from task.seed."""

    @abstractmethod
    def step(self, action: str) -> tuple[Obs, float, bool, dict]:
        """Apply one legal action string. Returns (obs, reward, done, info)."""

    @abstractmethod
    def tasks(self, split: str, n: int, rng: np.random.Generator) -> list[Task]:
        """n tasks from `split`. Held-out is frozen per env_seed and ignores rng."""

    @property
    @abstractmethod
    def actions(self) -> list[str]:
        """Legal action strings, shown to the policy in the prompt."""
