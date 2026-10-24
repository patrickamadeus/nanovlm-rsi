"""Reference policies with the same surface as models/vlm.py, so they run through the same harness and
evaluation: a number from a VLM means little without what chance gets on the same tasks."""
import re

import numpy as np

from models.vlm import RolloutStep

LEGAL = re.compile(r"Actions available: (.*)")


class RandomPolicy:
    """A uniformly random legal action every turn, read from the prompt's action list."""

    version = 0

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act_batch(self, prompts: list[str], images, **kw) -> list[RolloutStep]:
        out = []
        for p in prompts:
            actions = [a.strip() for a in LEGAL.search(p).group(1).split(",")]
            out.append(RolloutStep(f"ACT: {actions[int(self.rng.integers(len(actions)))]}", [0],
                                   np.zeros(1, np.float32), 0))
        return out
