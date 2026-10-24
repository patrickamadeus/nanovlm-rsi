"""LoRA from scratch (Hu et al., 2021): W x + (alpha / r) * B A x, with A random, B zero, base frozen.

B starts at zero, so a freshly wrapped model is exactly the pretrained one. Disabling the
adapters recovers it again — which is how GRPO gets pi_init for the KL term without a second copy.
"""
import math
from contextlib import contextmanager

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16):
        super().__init__()
        self.base, self.scale, self.enabled = base, alpha / r, True
        base.requires_grad_(False)
        w = base.weight
        # adapters train in fp32 even on a bf16 base: bf16 would round most optimizer steps away
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=w.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=w.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            out = out + ((x.to(self.A.dtype) @ self.A.T @ self.B.T) * self.scale).to(out.dtype)
        return out


def apply_lora(model: nn.Module, r: int = 8, alpha: float = 16, targets: tuple[str, ...] = ("q_proj", "v_proj"),
               scope: str | None = None) -> list[str]:
    """Wrap the language model's attention projections. The vision tower stays frozen and untouched.
    The language model is `text_model` in SmolVLM and `language_model` in Qwen-VL; None picks whichever exists."""
    if scope is None:
        names = [n for n, _ in model.named_modules()]
        scope = next((s for s in ("text_model", "language_model") if any(s in n for n in names)), "")
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if child_name in targets and scope in full and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha))
                wrapped.append(full)
    return wrapped


@contextmanager
def adapters_disabled(model: nn.Module):
    layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    for m in layers:
        m.enabled = False
    try:
        yield
    finally:
        for m in layers:
            m.enabled = True
