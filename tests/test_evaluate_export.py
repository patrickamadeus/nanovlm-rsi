import numpy as np
import torch

from configs.config import LoopConfig, PolicyConfig
from evaluate import export_checkpoint, lmms_eval_command, load_run
from harness.log import Trace


def test_lmms_eval_command_points_at_the_checkpoint():
    cmd = lmms_eval_command("runs/x/eval/checkpoint", "mmstar", "runs/x/eval/lmms_eval")
    assert cmd[1:3] == ["-m", "lmms_eval"]
    assert ["--model", "huggingface"] == cmd[3:5] and "pretrained=runs/x/eval/checkpoint" in cmd
    assert "--limit" not in cmd                              # reported numbers use every item
    assert lmms_eval_command("c", "mmstar", "o", limit=20)[-2:] == ["--limit", "20"]


def test_exported_checkpoint_reloads_as_a_plain_hf_model_with_the_trained_weights(tmp_path):
    from transformers import AutoModelForImageTextToText

    from models.smolvlm import Policy
    p = Policy(PolicyConfig(tiny=True, device="cpu", dtype="float32", max_new_tokens=4))
    p.enable_lora(2, 4)
    for n, w in p.model.named_parameters():
        if n.endswith(".B"):
            torch.nn.init.normal_(w)                            # a trained adapter, not the zero init
    inputs = p._inputs("hello", [np.zeros((64, 64, 3), np.uint8)])
    with torch.no_grad():
        want = p.model(**inputs).logits
    export_checkpoint(p, str(tmp_path / "ckpt"))
    reloaded = AutoModelForImageTextToText.from_pretrained(str(tmp_path / "ckpt"), dtype=torch.float32)
    assert not any("lora" in n or n.endswith((".A", ".B")) for n, _ in reloaded.named_parameters())
    with torch.no_grad():
        assert torch.allclose(reloaded(**inputs).logits, want, atol=1e-4)


def test_load_run_restores_config_and_committed_lora_weights(tmp_path):
    from dataclasses import asdict

    from models.smolvlm import Policy
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
