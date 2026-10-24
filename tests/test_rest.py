import numpy as np
import pytest

from configs.config import LoopConfig, PolicyConfig
from envs.base import Task
from rsi.judges import ScoredTrajectory
from rsi.rest import ReST
from utils.rollout import Trajectory

IMG = np.zeros((64, 64, 3), np.uint8)


@pytest.fixture
def setup(tmp_path):
    from models.vlm import Policy
    policy = Policy(PolicyConfig(tiny=True, device="cpu", dtype="float32", max_new_tokens=6))
    return policy, ReST(policy, LoopConfig(run_dir=str(tmp_path), rest_lr=5e-2))


def _scored(policy, reward, origin="attempt"):
    ids = policy.act("count the squares", [IMG]).token_ids
    st = {"action": "3", "prompt": "count the squares", "images": [IMG], "token_ids": ids, "logprobs": None,
          "policy_version": policy.version}
    return ScoredTrajectory(Trajectory(Task("count_objects", 2, ("3",), 1, "train"), [st], reward, 1, 0, len(ids)),
                            reward, None, "outcome", 1.0, (), origin=origin)


def test_no_success_no_update(setup):
    policy, rest = setup
    assert rest.propose([_scored(policy, 0.0)]) is None


def test_a_single_success_is_enough_and_becomes_more_likely(setup):
    policy, rest = setup
    s = _scored(policy, 1.0)
    st = s.traj.steps[0]
    before = policy.logprobs(st["prompt"], st["images"], st["token_ids"]).sum().item()
    u = rest.propose([s])
    rest.commit(u)
    after = policy.logprobs(st["prompt"], st["images"], st["token_ids"]).sum().item()
    assert after > before and rest.persisted(u)
    rest.revert(u)
    assert policy.logprobs(st["prompt"], st["images"], st["token_ids"]).sum().item() == pytest.approx(before, abs=1e-5)


def test_hindsight_copies_count_as_successes(setup):
    policy, rest = setup
    u = rest.propose([_scored(policy, 1.0, origin="hindsight")])
    assert u is not None and u.payload["hindsight"] == 1
