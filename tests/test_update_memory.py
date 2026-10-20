import numpy as np
import pytest

from config import LoopConfig
from envs.base import Task
from infra.memory import SkillStore
from tests.conftest import ScriptedPolicy, fake_embed
from update import make_updater
from update.memory import MemoryUpdater
from verify import ScoredTrajectory, Trajectory

COUNT = Task("count_objects", 2, ("3",), 1, "train")
OBS_TEXT = "How many blue squares are there? Answer with say_N."


def scored(reward=1.0, parse_fail=False, flags=(), task=COUNT):
    step = {"action": None if parse_fail else "say_3", "obs_text": OBS_TEXT}
    t = Trajectory(task, [step], reward, 0 if parse_fail else 1, int(parse_fail), 4)
    return ScoredTrajectory(t, reward, None, "outcome", 1.0, tuple(flags))


@pytest.fixture
def store(tmp_path):
    return SkillStore(str(tmp_path / "skills.jsonl"), embed_fn=fake_embed)


def test_update_carries_taxonomy_labels(store):
    u = MemoryUpdater(store, LoopConfig()).propose([scored(reward=1.0)])
    assert (u.carrier, u.role, u.horizon) == ("A3", "B0", "C3")
    assert u.signal == "outcome" and u.provenance
    assert u.cost.env_steps == 1


def test_proposes_only_successes_above_threshold(store):
    assert MemoryUpdater(store, LoopConfig()).propose([scored(reward=0.4)]) is None
    assert MemoryUpdater(store, LoopConfig()).propose([scored(reward=1.0)]) is not None


def test_degenerate_batch_returns_none(store):
    up = MemoryUpdater(store, LoopConfig())
    assert up.propose([]) is None
    assert up.propose([scored(reward=0.0, parse_fail=True) for _ in range(8)]) is None


def test_flagged_trajectories_are_never_written(store):
    assert MemoryUpdater(store, LoopConfig()).propose([scored(flags=("zero_step_success",))]) is None


def test_skill_situation_comes_from_observation_not_goal(store):
    # for count_objects the goal *is* the answer; keying memory on it would leak labels into the prompt
    u = MemoryUpdater(store, LoopConfig()).propose([scored()])
    situation = u.payload["skills"][0]["situation"]
    assert situation == OBS_TEXT


def test_commit_then_revert_restores_store(store, tmp_path):
    path = tmp_path / "skills.jsonl"
    up, before = MemoryUpdater(store, LoopConfig()), path.read_bytes()
    u = up.propose([scored(reward=1.0)])
    up.commit(u)
    assert len(store) == 1
    up.revert(u)
    assert path.read_bytes() == before and len(store) == 0


def test_make_updater_memory_builds_store_in_run_dir(tmp_path):
    up = make_updater("memory", ScriptedPolicy([]), LoopConfig(run_dir=str(tmp_path)))
    assert isinstance(up, MemoryUpdater) and up.store.path.startswith(str(tmp_path))


def test_scaffold_carrier_not_yet_available(tmp_path):
    cfg = LoopConfig(run_dir=str(tmp_path))
    with pytest.raises(NotImplementedError, match="Slice 6"):
        make_updater("scaffold", ScriptedPolicy([]), cfg)
