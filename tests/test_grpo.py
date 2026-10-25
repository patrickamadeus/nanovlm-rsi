import os

import numpy as np
import pytest
import torch

from configs.config import LoopConfig, PolicyConfig
from envs.base import Task
from rsi.grpo import GRPO, group_advantages, grpo_token_loss
from utils.rollout import Trajectory
from rsi.judges import ScoredTrajectory


def test_group_advantages_are_relative_within_each_task_group():
    adv = group_advantages([1.0, 0.0, 0.0, 1.0, 1.0, 0.0], ["a", "a", "b", "b", "c", "c"])
    np.testing.assert_allclose(adv, [1, -1, -1, 1, 1, -1], atol=1e-4)


def test_all_equal_groups_are_discarded_as_nan():
    adv = group_advantages([1.0, 1.0, 0.0, 1.0], ["a", "a", "b", "b"])
    assert np.isnan(adv[:2]).all() and not np.isnan(adv[2:]).any()


def test_unclipped_gradient_matches_the_analytic_policy_gradient():
    lp_new = torch.tensor([-1.0, -2.0], requires_grad=True)
    lp_old = lp_new.detach().clone()
    loss = grpo_token_loss(lp_new, lp_old, None, adv=0.5, clip=0.2, beta=0.0).sum()
    loss.backward()
    torch.testing.assert_close(lp_new.grad, torch.tensor([-0.5, -0.5]))   # d(-ratio*A)/d lp at ratio 1


def test_clipped_ratio_has_no_gradient_for_positive_advantage():
    lp_old = torch.tensor([-2.0])
    lp_new = (lp_old + 1.0).requires_grad_(True)                         # ratio e^1 > 1.2
    grpo_token_loss(lp_new, lp_old, None, adv=1.0, clip=0.2, beta=0.0).sum().backward()
    assert lp_new.grad.abs().item() == 0.0


def test_k3_kl_is_zero_at_reference_and_positive_away_from_it():
    lp = torch.tensor([-1.0, -2.0])
    same = grpo_token_loss(lp, lp, lp, adv=0.0, clip=0.2, beta=1.0)
    away = grpo_token_loss(lp, lp, lp - 0.5, adv=0.0, clip=0.2, beta=1.0)
    assert torch.allclose(same, torch.zeros(2)) and (away > 0).all()


IMG = np.zeros((64, 64, 3), np.uint8)


@pytest.fixture
def setup(tmp_path):
    from models.vlm import Policy
    policy = Policy(PolicyConfig(tiny=True, device="cpu", dtype="float32", max_new_tokens=6))
    cfg = LoopConfig(run_dir=str(tmp_path), grpo_lr=5e-2, grpo_beta=0.0, minibatches=1)
    return policy, GRPO(policy, cfg), cfg


def _traj(policy, task, reward, prompt="count the squares", token_ids=None):
    if token_ids is None:
        token_ids = policy.act(prompt, [IMG]).token_ids
    logprobs = policy.logprobs(prompt, [IMG], token_ids).detach().cpu().numpy()
    s = {"action": "3", "obs_text": prompt, "prompt": prompt, "images": [IMG],
         "token_ids": token_ids, "logprobs": logprobs}
    t = Trajectory(task, [s], reward, 1, 0, len(token_ids))
    return ScoredTrajectory(t, reward, None, "outcome", 1.0, ())


def _batch(policy):
    """A rewarded and a penalized completion that share no tokens, so pushing one down can't drag the
    other with it — two independent samples from a tiny model often overlap, which made this flaky."""
    task = Task("count_objects", 2, ("3",), 1, "train")
    good = _traj(policy, task, 1.0)
    vocab = policy.model.get_output_embeddings().weight.shape[0]
    used = set(good.traj.steps[0]["token_ids"]) | set(policy.never)
    bad_ids = [i for i in range(10, vocab) if i not in used][:len(good.traj.steps[0]["token_ids"])]
    return [good, _traj(policy, task, 0.0, token_ids=bad_ids)]


def test_lora_is_enabled_and_only_adapters_train(setup):
    policy, up, _ = setup
    trainable = [n for n, p in policy.model.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith((".A", ".B")) and "text_model" in n for n in trainable)


def test_update_is_labelled_grpo(setup):
    policy, up, _ = setup
    u = up.propose(_batch(policy))
    assert u.mode == "grpo" and u.provenance


def test_degenerate_batch_proposes_nothing(setup):
    policy, up, _ = setup
    task = Task("count_objects", 2, ("3",), 1, "train")
    assert up.propose([_traj(policy, task, 0.0), _traj(policy, task, 0.0)]) is None
    assert up.propose([]) is None


def test_one_grpo_step_makes_the_rewarded_completion_more_likely(setup):
    policy, up, _ = setup
    batch = _batch(policy)
    good = batch[0].traj.steps[0]
    before = policy.logprobs(good["prompt"], [IMG], good["token_ids"]).sum().item()
    up.commit(up.propose(batch))
    after = policy.logprobs(good["prompt"], [IMG], good["token_ids"]).sum().item()
    assert after > before


def test_commit_persists_and_revert_restores_exactly(setup):
    policy, up, cfg = setup
    snap = policy.snapshot()
    u = up.propose(_batch(policy))
    up.commit(u)
    assert os.path.exists(os.path.join(cfg.run_dir, "weights.safetensors"))
    assert up.persisted(u)
    up.revert(u)
    for n, v in policy.snapshot().items():
        torch.testing.assert_close(v, snap[n])
    assert not up.persisted(u)             # what is on disk is the pre-update state again


def test_commit_takes_one_step_per_minibatch_and_reverts_all_of_them(setup):
    policy, up, cfg = setup
    up.cfg.minibatches = 2
    snap = policy.snapshot()
    u = up.propose(_batch(policy))
    assert all(torch.equal(v, snap[n]) for n, v in policy.snapshot().items())   # propose changes nothing
    up.commit(u)
    assert u.payload["steps"] == 2 and np.isfinite(u.payload["loss"])
    up.revert(u)
    assert all(torch.equal(v, snap[n]) for n, v in policy.snapshot().items())


def test_pi_old_is_scored_with_the_same_batching_as_the_step(setup):
    policy, up, cfg = setup
    batch = _batch(policy)
    u = up.propose(batch)
    seen = []
    real = up.item_loss

    def spy(turn, adv, lp, old, ref):
        seen.append((lp.detach() - old).abs().max().item())
        return real(turn, adv, lp, old, ref)
    up.item_loss = spy
    up.commit(u)
    assert max(seen[:len(seen) // up.cfg.minibatches or 1]) < 1e-6         # first step: ratio exactly 1


def test_commit_and_revert_move_the_policy_version(setup):
    policy, grpo, cfg = setup
    task = Task("count_objects", 2, ("3",), 1, "train")
    batch = [_traj(policy, task, r) for r in (1.0, 0.0, 1.0, 0.0)]
    v = policy.version
    u = grpo.propose(batch)
    grpo.commit(u)
    assert policy.version == v + 1
    grpo.revert(u)
    assert policy.version == v + 2


def test_parse_failure_turns_are_not_trained(setup):
    policy, up, _ = setup
    batch = _batch(policy)
    good = batch[0]
    junk = {**good.traj.steps[0], "action": None, "actions": []}
    t = Trajectory(good.traj.task, [junk, *good.traj.steps], 1.0, 1, 1, 0)
    prepared = up.prepare([ScoredTrajectory(t, 1.0, None, "outcome", 1.0, ()), batch[1]])
    assert all(st.get("actions") or st.get("action") for st, _ in prepared["items"])
    assert len(prepared["items"]) == 2
