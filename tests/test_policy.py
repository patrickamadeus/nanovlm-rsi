import numpy as np
import pytest
import torch

from config import PolicyConfig
from policy.vlm import Policy

IMG = [np.zeros((64, 64, 3), np.uint8)]


@pytest.fixture(scope="module")
def tiny():
    return Policy(PolicyConfig(tiny=True, max_new_tokens=8, device="cpu", dtype="float32"))


def test_act_returns_aligned_tokens_and_logprobs(tiny):
    step = tiny.act("hello", IMG)
    assert isinstance(step.text, str)
    assert len(step.token_ids) == len(step.logprobs) > 0
    assert step.prompt_tokens > 0


def test_logprobs_rescores_consistently(tiny):
    step = tiny.act("hello", IMG)
    again = tiny.logprobs("hello", IMG, step.token_ids)
    assert again.shape[0] == len(step.token_ids)
    np.testing.assert_allclose(again.detach().cpu().numpy(), step.logprobs, rtol=1e-3, atol=1e-3)


def test_embed_returns_fixed_width_unit_vector(tiny):
    v = tiny.embed("a situation", IMG)
    assert v.ndim == 1 and np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)
    assert v.shape == tiny.embed("other", [np.ones((64, 64, 3), np.uint8)]).shape


def test_snapshot_restore_roundtrip(tiny):
    p = next(tiny.model.parameters())
    p.requires_grad_(True)                 # stand-in for a trainable adapter (Slice 2)
    ids = tiny.act("hello", IMG).token_ids
    before = tiny.logprobs("hello", IMG, ids).detach()
    state = tiny.snapshot()
    with torch.no_grad():
        p.add_(1.0)
    tiny.restore(state)
    after = tiny.logprobs("hello", IMG, ids).detach()
    p.requires_grad_(False)
    torch.testing.assert_close(before, after)
