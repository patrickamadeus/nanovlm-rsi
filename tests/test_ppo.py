import numpy as np
import pytest
import torch

from configs.config import LoopConfig, PolicyConfig
from rsi.ppo import PPO
from tests.test_rest import _scored


@pytest.fixture
def setup(tmp_path):
    from models.vlm import Policy
    policy = Policy(PolicyConfig(tiny=True, device="cpu", dtype="float32", max_new_tokens=6))
    return policy, PPO(policy, LoopConfig(run_dir=str(tmp_path), grpo_lr=5e-2, ppo_value_lr=0.1))


def test_a_single_attempt_is_enough_signal(setup):
    policy, ppo = setup
    u = ppo.propose([_scored(policy, 1.0)])
    assert u is not None and u.payload["value_loss"] == pytest.approx(1.0)     # V starts at 0, R = 1


def test_commit_moves_the_value_toward_the_return_and_revert_undoes_it(setup, tmp_path):
    policy, ppo = setup
    batch = [_scored(policy, 1.0), _scored(policy, 0.0), _scored(policy, 1.0)]
    u = ppo.propose(batch)
    ppo.commit(u)
    assert ppo.persisted(u) and (tmp_path / "value_head.safetensors").exists()
    fresh = [_scored(policy, 1.0), _scored(policy, 0.0)]          # sampled after the step, as the loop does
    assert ppo.propose(fresh).payload["mean_value"] > 0.0
    ppo.revert(u)
    assert ppo.head is None and not (tmp_path / "value_head.safetensors").exists()


def test_rewarded_completion_becomes_more_likely(setup):
    policy, ppo = setup
    good, bad = _scored(policy, 1.0), _scored(policy, 0.0)
    st = good.traj.steps[0]
    before = policy.logprobs(st["prompt"], st["images"], st["token_ids"]).sum().item()
    ppo.commit(ppo.propose([good, bad]))
    assert policy.logprobs(st["prompt"], st["images"], st["token_ids"]).sum().item() > before or \
        good.traj.steps[0]["token_ids"] == bad.traj.steps[0]["token_ids"]
