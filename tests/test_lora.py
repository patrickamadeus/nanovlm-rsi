import torch
import torch.nn as nn

from models.lora import LoRALinear, adapters_disabled, apply_lora


def test_lora_starts_as_identity_on_the_base_layer():
    base = nn.Linear(6, 4)
    x = torch.randn(3, 6)
    torch.testing.assert_close(LoRALinear(base, r=2, alpha=4)(x), base(x))


def test_only_adapter_params_are_trainable():
    layer = LoRALinear(nn.Linear(6, 4), r=2, alpha=4)
    trainable = {n for n, p in layer.named_parameters() if p.requires_grad}
    assert trainable == {"A", "B"}


def test_adapter_contributes_scaled_low_rank_update():
    base = nn.Linear(6, 4, bias=False)
    layer = LoRALinear(base, r=2, alpha=4)
    with torch.no_grad():
        layer.B.fill_(1.0)
    x = torch.randn(3, 6)
    expected = base(x) + (x @ layer.A.T @ layer.B.T) * (4 / 2)
    torch.testing.assert_close(layer(x), expected)


def test_apply_lora_targets_language_model_q_v_only():
    model = nn.ModuleDict({
        "text_model": nn.ModuleDict({"q_proj": nn.Linear(4, 4), "v_proj": nn.Linear(4, 4), "k_proj": nn.Linear(4, 4)}),
        "vision_model": nn.ModuleDict({"q_proj": nn.Linear(4, 4)}),
    })
    wrapped = apply_lora(model, r=2, alpha=4)
    assert sorted(wrapped) == ["text_model.q_proj", "text_model.v_proj"]
    assert isinstance(model["text_model"]["q_proj"], LoRALinear)
    assert not isinstance(model["vision_model"]["q_proj"], LoRALinear)


def test_adapters_disabled_recovers_the_base_model_exactly():
    model = nn.ModuleDict({"text_model": nn.ModuleDict({"q_proj": nn.Linear(4, 4)})})
    x = torch.randn(2, 4)
    before = model["text_model"]["q_proj"](x)
    apply_lora(model, r=2, alpha=4)
    with torch.no_grad():
        model["text_model"]["q_proj"].B.fill_(0.5)
    assert not torch.allclose(model["text_model"]["q_proj"](x), before)
    with adapters_disabled(model):
        torch.testing.assert_close(model["text_model"]["q_proj"](x), before)


def test_adapters_stay_float32_on_a_bfloat16_base():
    # bf16 has ~3 significant digits: an AdamW step of 1e-4 on a bf16 adapter mostly rounds to nothing
    base = nn.Linear(6, 4).to(torch.bfloat16)
    layer = LoRALinear(base, r=2, alpha=4)
    assert layer.A.dtype == layer.B.dtype == torch.float32
    out = layer(torch.randn(3, 6, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16


def test_merge_lora_folds_adapters_into_plain_linear_layers():
    import torch.nn as nn
    from models.lora import merge_lora
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 6))
    model.text_model = nn.Module()
    model.text_model.q_proj = nn.Linear(6, 4)
    apply_lora(model, r=2, alpha=4)
    nn.init.normal_(model.text_model.q_proj.B)            # a trained adapter, not the zero init
    x = torch.randn(3, 6)
    want = model.text_model.q_proj(x)
    merge_lora(model)
    assert type(model.text_model.q_proj) is nn.Linear
    assert torch.allclose(model.text_model.q_proj(x), want, atol=1e-5)
