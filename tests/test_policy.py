import numpy as np
import pytest
import torch

from configs.config import PolicyConfig
from models.vlm import Policy

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


def test_prompt_is_wrapped_in_the_chat_template_with_response_prefix(tiny):
    text = tiny._format("hello <image> there", n_images=1)
    assert text.count("<image>") == 1
    assert text.rstrip().endswith("Assistant: ACT:")


def test_act_text_includes_the_prefix_so_parsing_sees_it(tiny):
    assert tiny.act("hello", IMG).text.startswith("ACT:")


def test_empty_prefix_restores_free_reasoning(tiny):
    tiny.cfg.response_prefix = ""
    try:
        assert tiny._format("hello", n_images=1).rstrip().endswith("Assistant:")
    finally:
        tiny.cfg.response_prefix = "ACT:"


def test_generation_stops_at_the_end_of_the_action_line(tiny):
    nl = tiny.processor.tokenizer("\n", add_special_tokens=False).input_ids[0]
    assert nl in tiny.stop_ids
    tiny.cfg.max_new_tokens = 40
    try:
        for _ in range(5):
            ids = tiny.act("hello", IMG).token_ids
            assert nl not in ids[:-1]          # nothing is generated past the first newline
    finally:
        tiny.cfg.max_new_tokens = 8


def test_small_env_images_are_one_tile_not_upscaled_and_split():
    from transformers import AutoProcessor
    from models.vlm import configure_images
    proc = AutoProcessor.from_pretrained(PolicyConfig().model_id)
    configure_images(proc, 512)
    from PIL import Image
    x = proc(text=["<image>hi"], images=[[Image.fromarray(np.zeros((64, 64, 3), np.uint8))]], return_tensors="pt")
    assert x["pixel_values"].shape[1] == 1          # the processor default was 17 tiles for a 64px image


def test_act_and_rescore_share_one_processed_input(tiny, monkeypatch):
    calls = []
    real = tiny._inputs
    monkeypatch.setattr(tiny, "_inputs", lambda *a, **k: calls.append(1) or real(*a, **k))
    tiny.act("hello", IMG)
    assert len(calls) == 1


def test_cpu_policy_caps_torch_threads():
    # small matmuls on many threads thrash: 11 s/action at 96 threads vs 0.04 s at 8
    import torch
    torch.set_num_threads(64)
    Policy(PolicyConfig(tiny=True, device="cpu", dtype="float32"))
    assert torch.get_num_threads() <= 8


def test_greedy_decoding_is_deterministic_and_takes_a_prefix_override(tiny):
    a = tiny.act("Which?", IMG, prefix="Answer:", greedy=True)
    b = tiny.act("Which?", IMG, prefix="Answer:", greedy=True)
    assert a.text == b.text and a.text.startswith("Answer:")


def test_policy_never_emits_image_placeholder_tokens(tiny):
    # a sampled <image> token makes the re-scoring forward pass count more image slots than images
    image_id = tiny.processor.tokenizer.convert_tokens_to_ids("<image>")
    head = tiny.model.get_output_embeddings()
    with torch.no_grad():
        saved = head.weight[image_id].clone()
        head.weight[image_id] = 50.0 * head.weight.abs().max()     # make <image> the argmax everywhere
    try:
        ids = tiny.act("hello", IMG, greedy=True).token_ids
        assert ids and image_id not in ids
    finally:
        with torch.no_grad():
            head.weight[image_id] = saved


def test_act_batch_returns_one_step_per_prompt_scored_like_logprobs(tiny):
    prompts = ["hello", "a much longer prompt that needs left padding in the batch", "hi"]
    images = [IMG, IMG, IMG + IMG]                       # rows may carry different numbers of images
    steps = tiny.act_batch(prompts, images)
    assert len(steps) == 3
    for p, im, s in zip(prompts, images, steps):
        assert len(s.token_ids) == len(s.logprobs) > 0 and s.text.startswith("ACT:")
        np.testing.assert_allclose(tiny.logprobs(p, im, s.token_ids).detach().numpy(), s.logprobs,
                                   rtol=1e-3, atol=1e-3)


def test_greedy_batch_matches_greedy_one_at_a_time(tiny):
    prompts = ["hello", "a much longer prompt that needs left padding in the batch"]
    batch = tiny.act_batch(prompts, [IMG, IMG], greedy=True)
    for p, s in zip(prompts, batch):
        assert s.token_ids == tiny.act(p, IMG, greedy=True).token_ids


def test_a_completion_stops_at_its_first_stop_token(tiny):
    for s in tiny.act_batch(["x"] * 4, [IMG] * 4):
        stops = [i for i, t in enumerate(s.token_ids) if t in tiny.stop_ids]
        assert not stops or stops[0] == len(s.token_ids) - 1


def test_image_placeholder_tokens_are_never_sampled(tiny):
    assert tiny.never                                   # found from the tokenizer, not a hard-coded model list
