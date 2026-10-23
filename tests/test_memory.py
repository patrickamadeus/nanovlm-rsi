import numpy as np
import pytest

from configs.config import LoopConfig
from envs.base import Task
from rsi.memory import SkillStore
from tests.conftest import ScriptedPolicy, fake_embed
from rsi import make_update
from rsi.memory import Memory
from utils.rollout import Trajectory
from rsi.judges import ScoredTrajectory

COUNT = Task("count_objects", 2, ("3",), 1, "train")
OBS_TEXT = "How many blue squares are there? Answer with the number."


def scored(reward=1.0, parse_fail=False, flags=(), task=COUNT):
    step = {"action": None if parse_fail else "3", "obs_text": OBS_TEXT}
    t = Trajectory(task, [step], reward, 0 if parse_fail else 1, int(parse_fail), 4)
    return ScoredTrajectory(t, reward, None, "outcome", 1.0, tuple(flags))


@pytest.fixture
def store(tmp_path):
    return SkillStore(str(tmp_path / "skills.jsonl"), embed_fn=fake_embed)


def test_update_carries_taxonomy_labels(store):
    u = Memory(store, LoopConfig()).propose([scored(reward=1.0)])
    assert u.mode == "memory"
    assert u.signal == "outcome" and u.provenance
    assert u.cost.env_steps == 1


def test_proposes_only_successes_above_threshold(store):
    assert Memory(store, LoopConfig()).propose([scored(reward=0.4)]) is None
    assert Memory(store, LoopConfig()).propose([scored(reward=1.0)]) is not None


def test_degenerate_batch_returns_none(store):
    up = Memory(store, LoopConfig())
    assert up.propose([]) is None
    assert up.propose([scored(reward=0.0, parse_fail=True) for _ in range(8)]) is None


def test_flagged_trajectories_are_never_written(store):
    assert Memory(store, LoopConfig()).propose([scored(flags=("zero_step_success",))]) is None


def test_skill_situation_comes_from_observation_not_goal(store):
    # for count_objects the goal *is* the answer; keying memory on it would leak labels into the prompt
    u = Memory(store, LoopConfig()).propose([scored()])
    situation = u.payload["skills"][0]["situation"]
    assert situation == OBS_TEXT


def test_commit_then_revert_restores_store(store, tmp_path):
    path = tmp_path / "skills.jsonl"
    up, before = Memory(store, LoopConfig()), path.read_bytes()
    u = up.propose([scored(reward=1.0)])
    up.commit(u)
    assert len(store) == 1
    up.revert(u)
    assert path.read_bytes() == before and len(store) == 0


def test_make_update_memory_builds_store_in_run_dir(tmp_path):
    up = make_update(LoopConfig(update="memory", run_dir=str(tmp_path)), ScriptedPolicy([]))
    assert isinstance(up, Memory) and up.store.path.startswith(str(tmp_path))


def test_unknown_update_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        make_update(LoopConfig(update="scaffold", run_dir=str(tmp_path)), ScriptedPolicy([]))
