import numpy as np
import torch
from PIL import Image

from configs.config import LoopConfig, PolicyConfig
from evaluate import load_run
from utils.lmms import answer
from utils.log import Trace
from tests.conftest import ScriptedPolicy


def test_answer_feeds_the_task_image_and_prompt_and_cuts_at_stop_strings():
    seen = {}

    class Spy(ScriptedPolicy):
        def act(self, prompt, images, **kw):
            seen.update(prompt=prompt, n=len(images), shape=images[0].shape, **kw)
            return super().act(prompt, images)

    policy = Spy(["B\n\nbecause the cat"])
    policy.cfg = PolicyConfig()
    docs = {"mmstar": {"val": [{"img": Image.new("RGB", (32, 24))}]}}
    args = ("Which?\nA. x\nB. y", {"until": ["\n\n"], "max_new_tokens": 8}, lambda d: [d["img"]], 0, "mmstar", "val")
    assert answer(policy, args, docs) == "B"
    assert seen["prompt"] == "Which?\nA. x\nB. y" and seen["n"] == 1 and seen["shape"] == (24, 32, 3)
    assert seen["greedy"] is True and seen["prefix"] == ""          # measure the policy, no ACT: prefill
    assert policy.cfg.max_new_tokens == 8


def test_load_run_restores_config_and_committed_lora_weights(tmp_path):
    from dataclasses import asdict

    from models.vlm import Policy
    from safetensors.torch import save_file
    cfg = LoopConfig(run_dir=str(tmp_path), lora_r=2, lora_alpha=4,
                     policy=PolicyConfig(tiny=True, device="cpu", dtype="float32", max_new_tokens=4))
    Trace(str(tmp_path)).log_run(config=asdict(cfg))
    p = Policy(cfg.policy)
    p.enable_lora(2, 4)
    state = {k: torch.randn_like(v) for k, v in p.snapshot().items()}
    save_file(state, str(tmp_path / "weights.safetensors"))
    loaded_cfg, loaded = load_run(str(tmp_path))
    assert loaded_cfg.lora_r == 2 and loaded_cfg.policy.tiny
    assert all(torch.equal(loaded.snapshot()[k], v) for k, v in state.items())


def test_benchmark_policy_settings_override_the_env_ones(tmp_path):
    from dataclasses import asdict
    cfg = LoopConfig(run_dir=str(tmp_path), policy=PolicyConfig(tiny=True, device="cpu", dtype="float32"))
    Trace(str(tmp_path)).log_run(config=asdict(cfg))
    _, p = load_run(str(tmp_path), response_prefix="", stop_at_newline=False)
    assert p.cfg.response_prefix == "" and p.cfg.stop_at_newline is False
