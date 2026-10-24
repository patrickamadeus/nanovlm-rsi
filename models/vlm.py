"""The policy: any HF image-text-to-text VLM (SmolVLM, Qwen2.5-VL, ...) behind the capabilities the
methods need — act() / act_batch(), logprobs(), embed(), snapshot() / restore().

transformers owns weight loading and the forward pass. It owns nothing in the
control flow: sampling settings, logprob bookkeeping and state snapshots live here.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from configs.config import PolicyConfig
from models.lora import adapters_disabled, apply_lora

IMAGE = "<image>"


@dataclass(frozen=True)
class RolloutStep:
    text: str
    token_ids: list[int]    # completion tokens only, never the prompt
    logprobs: np.ndarray    # log pi(token) under the sampling distribution, one per completion token
    prompt_tokens: int


def configure_images(processor, side: int) -> None:
    """One tile of at most `side` px, never upscaled-and-split: env frames are small and the default is for documents.
    Only SmolVLM/Idefics3 tiles images; Qwen-VL already encodes a frame at its native size, so it is left alone."""
    ip = processor.image_processor
    if not hasattr(ip, "do_image_splitting"):
        return
    ip.do_image_splitting = False
    ip.size = ip.max_image_size = {"longest_edge": side}


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
        if self.device == "cpu":                                   # many threads on tiny matmuls thrash (~300x slower)
            torch.set_num_threads(min(torch.get_num_threads(), 8))
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        dtype = getattr(torch, cfg.dtype)
        if cfg.image_side and not cfg.tiny:
            configure_images(self.processor, cfg.image_side)
        if cfg.tiny:
            self.model = _tiny(AutoConfig.from_pretrained(cfg.model_id), self.processor).to(dtype)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, dtype=dtype)
        self.model.to(self.device).eval().requires_grad_(False)   # nothing trains unless a carrier opts in
        self.mode, self.reference = "frozen", None
        self.version = 0                    # bumped by every weight change; tells GRPO whether pi_old == pi
        tok = self.processor.tokenizer
        self.stop_ids = [tok.eos_token_id] + (tok("\n", add_special_tokens=False).input_ids if cfg.stop_at_newline else [])
        # image placeholders are never valid output: one sampled <image> breaks the re-scoring forward pass
        markers = ("image", "img", "vision", "video")
        self.never = sorted({i for t, i in tok.get_added_vocab().items() if any(m in t for m in markers)})

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

    @torch.no_grad()
    def reference_logprobs_batch(self, prompts, images, token_ids) -> list[torch.Tensor]:
        if self.mode == "lora":
            with adapters_disabled(self.model):
                return self.logprobs_batch(prompts, images, token_ids)
        model, self.model = self.model, self.reference
        try:
            return self.logprobs_batch(prompts, images, token_ids)
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
    def act_batch(self, prompts: list[str], images: list[list[np.ndarray]], *, prefix: str | None = None,
                  greedy: bool = False) -> list[RolloutStep]:
        """act() for many prompts at once: one left-padded generate call, which is where the time goes in a
        multi-step episode. Each completion is then re-scored alone, unpadded, exactly as logprobs() will
        score it in training, so pi_old never depends on what else happened to be in the batch."""
        out = []
        for i in range(0, len(prompts), self.cfg.batch_size):
            out += self._act_chunk(prompts[i:i + self.cfg.batch_size], images[i:i + self.cfg.batch_size],
                                   prefix, greedy)
        return out

    def _act_chunk(self, prompts, images, prefix, greedy) -> list[RolloutStep]:
        tok = self.processor.tokenizer
        texts = [self._format(p, len(im), prefix) for p, im in zip(prompts, images)]
        pil = [[Image.fromarray(x) for x in im] for im in images]
        side, tok.padding_side = tok.padding_side, "left"
        try:
            batch = self.processor(text=texts, images=pil if any(pil) else None, padding=True, return_tensors="pt")
        finally:
            tok.padding_side = side
        batch = {k: (v.to(self.device, self.model.dtype) if v.is_floating_point() else v.to(self.device))
                 for k, v in batch.items()}
        n_prompt = batch["input_ids"].shape[1]
        seqs = self.model.generate(**batch, **self._sampling(greedy), max_new_tokens=self.cfg.max_new_tokens,
                                   eos_token_id=self.stop_ids, suppress_tokens=self.never,
                                   pad_token_id=tok.pad_token_id)[:, n_prompt:].tolist()
        prefix = self.cfg.response_prefix if prefix is None else prefix
        completions = []
        for ids in seqs:
            stop = next((j for j, t in enumerate(ids) if t in self.stop_ids), None)
            completions.append(ids[:stop + 1] if stop is not None else ids)
        lps = self._score_batch(batch, completions)
        n_real = batch["attention_mask"].sum(1).tolist()
        return [RolloutStep(f"{prefix} {t.strip()}" if prefix else t, ids, lp.float().cpu().numpy(), int(n))
                for ids, lp, n, t in zip(completions, lps, n_real,
                                         (tok.decode(c, skip_special_tokens=True) for c in completions))]

    def _score_batch(self, batch: dict, completions: list[list[int]], hidden: bool = False):
        """Score each row's completion in one forward pass. The left-padded prompts are re-packed to start at
        position 0 with the completion right after and padding only at the end: causal attention never sees
        right padding, so each row is scored as if alone (up to bf16 kernel noise)."""
        mask = batch["attention_mask"].bool()
        rows = [batch["input_ids"][i][mask[i]].tolist() + c for i, c in enumerate(completions)]
        width = max(map(len, rows))
        pad = self.processor.tokenizer.pad_token_id or 0
        packed = dict(batch)
        packed["input_ids"] = torch.tensor([r + [pad] * (width - len(r)) for r in rows], device=self.device)
        packed["attention_mask"] = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows],
                                                device=self.device)
        for k, v in batch.items():           # per-token side inputs (Qwen's mm_token_type_ids) move with their tokens
            if k not in ("input_ids", "attention_mask") and torch.is_tensor(v) and v.shape[:2] == mask.shape:
                out = torch.zeros((len(rows), width, *v.shape[2:]), dtype=v.dtype, device=v.device)
                for i in range(len(rows)):
                    n = int(mask[i].sum())
                    out[i, :n] = v[i][mask[i]]
                packed[k] = out
        result = self.model(**packed, output_hidden_states=hidden)
        logits = result.logits.float() / self.cfg.temperature
        out, feats = [], []
        for i, c in enumerate(completions):
            n = int(mask[i].sum())
            lp = F.log_softmax(logits[i, n - 1:n - 1 + len(c)], dim=-1)
            out.append(lp.gather(-1, torch.tensor(c, device=self.device)[:, None])[:, 0])
            if hidden:
                feats.append(result.hidden_states[-1][i, n - 1].detach().float())
        return (out, torch.stack(feats)) if hidden else out

    def _sampling(self, greedy: bool) -> dict:
        # top_k=0, top_p=1: the sampling distribution is exactly softmax(logits / T)
        return (dict(do_sample=False) if greedy else
                dict(do_sample=True, temperature=self.cfg.temperature, top_k=0, top_p=1.0))

    @torch.no_grad()
    def act(self, prompt: str, images: list[np.ndarray], *, prefix: str | None = None,
            greedy: bool = False) -> RolloutStep:
        """Sample a completion. `prefix` overrides cfg.response_prefix for this call; `greedy` is for
        benchmarks, where we measure the policy rather than draw from it (logprobs are then not pi_old)."""
        inputs = self._inputs(prompt, images, prefix=prefix)
        n_prompt = inputs["input_ids"].shape[1]
        out = self.model.generate(**inputs, **self._sampling(greedy), max_new_tokens=self.cfg.max_new_tokens,
                                  eos_token_id=self.stop_ids, suppress_tokens=self.never, return_dict_in_generate=True,
                                  pad_token_id=self.processor.tokenizer.pad_token_id)
        ids = out.sequences[0, n_prompt:].tolist()
        # Re-score with one full forward pass rather than trusting the decode-time logits: in bf16 the
        # KV-cache decode drifts from a full pass by up to ~0.27 nats, which would put GRPO ratios
        # past the clip before any parameter moved. pi_old must be exactly what logprobs() computes.
        lp = self._score(inputs, ids).float()
        text = self.processor.tokenizer.decode(ids, skip_special_tokens=True)
        prefix = self.cfg.response_prefix if prefix is None else prefix
        if prefix:                     # the prefilled tokens are prompt, not policy output, but parsing needs them
            text = f"{prefix} {text.strip()}"
        return RolloutStep(text, ids, lp.cpu().numpy(), n_prompt)

    def logprobs(self, prompt: str, images: list[np.ndarray], token_ids: list[int],
                 prefix: str | None = None) -> torch.Tensor:
        """Re-score an existing completion under current params; differentiable when params require grad."""
        return self._score(self._inputs(prompt, images, prefix=prefix), token_ids)

    def logprobs_batch(self, prompts: list[str], images: list[list[np.ndarray]], token_ids: list[list[int]],
                       prefix: str | None = None, hidden: bool = False):
        """logprobs() for several completions in one forward pass; differentiable. Training uses this.
        hidden=True also returns each row's last-prompt-token hidden state (detached): the state a value head reads."""
        texts = [self._format(p, len(im), prefix) for p, im in zip(prompts, images)]
        pil = [[Image.fromarray(x) for x in im] for im in images]
        batch = self.processor(text=texts, images=pil if any(pil) else None, padding=True, return_tensors="pt")
        batch = {k: (v.to(self.device, self.model.dtype) if v.is_floating_point() else v.to(self.device))
                 for k, v in batch.items()}
        return self._score_batch(batch, token_ids, hidden)

    def _score(self, inputs: dict, token_ids: list[int]) -> torch.Tensor:
        inputs = dict(inputs)
        n_prompt = inputs["input_ids"].shape[1]
        completion = torch.tensor([token_ids], device=self.device)
        for k, v in inputs.items():          # per-token side inputs (Qwen's mm_token_type_ids): text, so zeros
            if k not in ("input_ids", "attention_mask") and torch.is_tensor(v) and v.shape[:2] == (1, n_prompt):
                inputs[k] = torch.cat([v, torch.zeros_like(v[:, :1]).expand(1, len(token_ids), *v.shape[2:])], 1)
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
        self.version += 1
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, v in state.items():
                params[n].copy_(v.to(params[n].device))
