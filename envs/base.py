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
