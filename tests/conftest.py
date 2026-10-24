"""Shared test doubles. They satisfy the Policy surface so loop tests need no model."""
import zlib
from collections.abc import Iterable

import numpy as np

from models.vlm import RolloutStep


class ScriptedPolicy:
    def __init__(self, generations: Iterable[str]):
        self._gen = iter(generations)
        self.prompts: list[str] = []

    def act(self, prompt, images, **kw):
        self.prompts.append(prompt)
        text = next(self._gen, "")
        n = max(len(text.split()), 1)
        return RolloutStep(text, [0] * n, np.zeros(n, np.float32), len(prompt.split()))

    def embed(self, text, images):
        return fake_embed(text)


def fake_embed(text: str) -> np.ndarray:
    v = np.zeros(64)
    for word in text.lower().split():
        v[zlib.crc32(word.encode()) % 64] += 1.0
    return v / (np.linalg.norm(v) or 1.0)
