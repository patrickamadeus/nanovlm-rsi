"""The policy: a small HF VLM behind two capabilities GRPO needs — act() and logprobs().

transformers owns weight loading and the forward pass. It owns nothing in the
control flow: sampling settings, logprob bookkeeping and state snapshots live here.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from config import PolicyConfig
from policy.lora import adapters_disabled, apply_lora

IMAGE = "<image>"


@dataclass(frozen=True)
class RolloutStep:
    text: str
    token_ids: list[int]    # completion tokens only, never the prompt
    logprobs: np.ndarray    # log pi(token) under the sampling distribution, one per completion token
    prompt_tokens: int


def _tiny(cfg, processor):
    """Shrink every dimension so the whole loop runs on CPU in seconds, through the same code as the real model."""
    v, t = cfg.vision_config, cfg.text_config
    v.hidden_size, v.intermediate_size, v.num_hidden_layers, v.num_attention_heads, v.image_size = 32, 64, 2, 2, 64
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 32, 64, 2
    t.num_attention_heads = t.num_key_value_heads = 2
    cfg.scale_factor = 2
    ip = processor.image_processor
    ip.do_image_splitting = False
    ip.size = ip.max_image_size = {"longest_edge": 64}
    processor.image_seq_len = (64 // v.patch_size) ** 2 // cfg.scale_factor ** 2
    return AutoModelForImageTextToText.from_config(cfg)


class Policy:
    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        dtype = getattr(torch, cfg.dtype)
        if cfg.tiny:
            self.model = _tiny(AutoConfig.from_pretrained(cfg.model_id), self.processor).to(dtype)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, dtype=dtype)
        self.model.to(self.device).eval().requires_grad_(False)   # nothing trains unless a carrier opts in
        self.mode, self.reference = "frozen", None
        tok = self.processor.tokenizer
        self.stop_ids = [tok.eos_token_id] + (tok("\n", add_special_tokens=False).input_ids if cfg.stop_at_newline else [])

    def enable_lora(self, r: int, alpha: float) -> list[str]:
        self.mode = "lora"
        return apply_lora(self.model, r, alpha)

    def enable_full(self, keep_reference: bool) -> None:
        """Train every language-model weight. The KL term then needs a frozen copy of the initial model."""
        import copy
        self.mode = "full"
        if keep_reference:
            self.reference = copy.deepcopy(self.model).requires_grad_(False)
        for n, p in self.model.named_parameters():
            p.requires_grad_("text_model" in n)

    @torch.no_grad()
    def reference_logprobs(self, prompt: str, images: list[np.ndarray], token_ids: list[int]) -> torch.Tensor:
        """Logprobs under pi_init: LoRA adapters switched off, or the frozen copy in full mode."""
        if self.mode == "lora":
            with adapters_disabled(self.model):
                return self.logprobs(prompt, images, token_ids)
        model, self.model = self.model, self.reference
        try:
            return self.logprobs(prompt, images, token_ids)
        finally:
            self.model = model

    def _format(self, prompt: str, n_images: int, prefix: str | None = None) -> str:
        """Wrap the rendered scaffold as one user turn in the model's chat template, then prefill the reply."""
        prompt = IMAGE * max(n_images - prompt.count(IMAGE), 0) + prompt
        content = []
        for i, part in enumerate(prompt.split(IMAGE)):
            if i:
                content.append({"type": "image"})
            if part.strip():
                content.append({"type": "text", "text": part})
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        prefix = self.cfg.response_prefix if prefix is None else prefix
        return f"{text} {prefix}" if prefix else text

    def _inputs(self, prompt: str, images: list[np.ndarray], chat: bool = True, prefix: str | None = None) -> dict:
        if chat:
            text = self._format(prompt, len(images), prefix)
        else:
            text = IMAGE * max(len(images) - prompt.count(IMAGE), 0) + prompt
        pil = [Image.fromarray(im) for im in images]
        batch = self.processor(text=[text], images=[pil] if pil else None, return_tensors="pt")
        return {k: (v.to(self.device, self.model.dtype) if v.is_floating_point() else v.to(self.device))
                for k, v in batch.items()}

    @torch.no_grad()
    def act(self, prompt: str, images: list[np.ndarray], *, prefix: str | None = None,
            greedy: bool = False) -> RolloutStep:
        """Sample a completion. `prefix` overrides cfg.response_prefix for this call; `greedy` is for
        benchmarks, where we measure the policy rather than draw from it (logprobs are then not pi_old)."""
        inputs = self._inputs(prompt, images, prefix=prefix)
        n_prompt = inputs["input_ids"].shape[1]
        sampling = (dict(do_sample=False) if greedy else
                    # top_k=0, top_p=1: the sampling distribution is exactly softmax(logits / T)
                    dict(do_sample=True, temperature=self.cfg.temperature, top_k=0, top_p=1.0))
        out = self.model.generate(**inputs, **sampling, max_new_tokens=self.cfg.max_new_tokens,
                                  eos_token_id=self.stop_ids, return_dict_in_generate=True,
                                  pad_token_id=self.processor.tokenizer.pad_token_id)
        ids = out.sequences[0, n_prompt:].tolist()
        # Re-score with one full forward pass rather than trusting the decode-time logits: in bf16 the
        # KV-cache decode drifts from a full pass by up to ~0.27 nats, which would put GRPO ratios
        # past the clip before any parameter moved. pi_old must be exactly what logprobs() computes.
        lp = self.logprobs(prompt, images, ids, prefix=prefix).float()
        text = self.processor.tokenizer.decode(ids, skip_special_tokens=True)
        prefix = self.cfg.response_prefix if prefix is None else prefix
        if prefix:                     # the prefilled tokens are prompt, not policy output, but parsing needs them
            text = f"{prefix} {text.strip()}"
        return RolloutStep(text, ids, lp.cpu().numpy(), n_prompt)

    def logprobs(self, prompt: str, images: list[np.ndarray], token_ids: list[int],
                 prefix: str | None = None) -> torch.Tensor:
        """Re-score an existing completion under current params; differentiable when params require grad."""
        inputs = self._inputs(prompt, images, prefix=prefix)
        n_prompt = inputs["input_ids"].shape[1]
        completion = torch.tensor([token_ids], device=self.device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], completion], dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        logits = self.model(**inputs).logits[0, n_prompt - 1:-1].float() / self.cfg.temperature
        return F.log_softmax(logits, dim=-1).gather(-1, completion[0][:, None])[:, 0]

    @torch.no_grad()
    def embed(self, text: str, images: list[np.ndarray]) -> np.ndarray:
        """Mean-pooled last hidden state, L2-normalized. The memory store's embedding: no second model needed."""
        inputs = self._inputs(text, images, chat=False)
        h = self.model(**inputs, output_hidden_states=True).hidden_states[-1][0].float()
        mask = inputs["attention_mask"][0, :, None].float()
        v = (h * mask).sum(0) / mask.sum()
        return F.normalize(v, dim=0).cpu().numpy()

    def snapshot(self) -> dict:
        """Trainable state only — tiny for LoRA, which is what makes a pre-update snapshot free."""
        return {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def restore(self, state: dict) -> None:
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, v in state.items():
                params[n].copy_(v.to(params[n].device))
