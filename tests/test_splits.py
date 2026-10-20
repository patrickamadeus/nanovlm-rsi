import os
import subprocess
import sys
from collections import Counter

from envs.base import Task, split_for_seed, stamp_split


def test_split_buckets_match_spec():
    seen = {split_for_seed(s) for s in range(2000)}
    assert seen == {"train", "gate", "heldout"}
    counts = Counter(split_for_seed(s) for s in range(10000))
    assert 0.6 < counts["train"] / 10000 < 0.8
    assert 0.05 < counts["gate"] / 10000 < 0.15
    assert 0.15 < counts["heldout"] / 10000 < 0.25


def test_split_is_stable_across_processes():
    # builtin hash() is salted per process; splits must not be
    out = subprocess.run(
        [sys.executable, "-c",
         "from envs.base import split_for_seed;"
         "print(','.join(split_for_seed(s) for s in range(50)))"],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONHASHSEED": "random", "PYTHONPATH": os.getcwd()},
    )
    local = ",".join(split_for_seed(s) for s in range(50))
    assert out.stdout.strip() == local


def test_stamp_split_overrides_claimed_split():
    t = Task(family="f", seed=1, goal=("g",), budget=1, split="train")
    assert stamp_split(t).split == split_for_seed(1)
