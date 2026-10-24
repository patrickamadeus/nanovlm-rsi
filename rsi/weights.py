"""What every weight-updating method (grpo.py, ppo.py, rest.py) shares: the LoRA adapters, one file on disk,
and the propose / commit / revert / persisted cycle the gate drives.

A subclass only says how to turn a batch into gradients: backward(batch) accumulates them and returns
stats (or None when the batch has no signal). propose() snapshots the adapters and optimizer first, so
revert() is exact; commit() takes the optimizer step and writes the adapters to disk. LoRA state is
small, so the snapshot is cheap.

Several weight methods in one run share one set of adapters (Weights) but keep their own optimizers.
"""
import copy
import os

import torch
from safetensors.torch import load_file, save_file

from rsi.base import Method
from rsi.gate import Update
from utils.log import Cost


class Weights:
    def __init__(self, policy, cfg):
        if cfg.grpo_mode == "lora":
            policy.enable_lora(cfg.lora_r, cfg.lora_alpha)
        elif cfg.grpo_mode == "full":
            policy.enable_full(keep_reference=cfg.grpo_beta > 0)
        else:
            raise ValueError(f"unknown grpo_mode {cfg.grpo_mode!r}")
        self.policy = policy
        self.params = [p for p in policy.model.parameters() if p.requires_grad]
        self.path = os.path.join(cfg.run_dir, "weights.safetensors")
        os.makedirs(cfg.run_dir, exist_ok=True)
        if cfg.resume and os.path.exists(self.path):
            policy.restore(load_file(self.path))

    def save(self) -> None:
        save_file(self.policy.snapshot(), self.path)


class WeightMethod(Method):
    changes = "weights"

    def __init__(self, policy, cfg, weights: Weights | None = None, lr: float | None = None):
        self.policy, self.cfg = policy, cfg
        self.weights = weights or Weights(policy, cfg)
        self.params, self.path = self.weights.params, self.weights.path
        self.opt = torch.optim.AdamW(self.params, lr=lr or cfg.grpo_lr, weight_decay=0.0)

    def backward(self, batch: list) -> dict | None:
        raise NotImplementedError

    def propose(self, batch: list) -> Update | None:
        snapshot, opt_state = self.policy.snapshot(), copy.deepcopy(self.opt.state_dict())
        self.opt.zero_grad(set_to_none=True)
        stats = self.backward(batch)
        if stats is None:
            self.opt.zero_grad(set_to_none=True)
            return None
        stats["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grpo_max_grad_norm).item()
        used = stats.pop("used")
        cost = Cost(sum(s.traj.env_steps for s in batch), sum(s.traj.tokens_in + s.traj.tokens_out for s in batch))
        signal = used[0].verdict_source if len({s.verdict_source for s in used}) == 1 else "mixed"
        provenance = tuple(f"{s.traj.task.family}:{s.traj.task.seed}" for s in used)
        return Update(self.name, signal, {"snapshot": snapshot, "opt_state": opt_state, **stats}, cost, provenance)

    def commit(self, update: Update) -> None:
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.policy.version += 1
        self.weights.save()

    def revert(self, update: Update) -> None:
        self.policy.restore(update.payload["snapshot"])
        self.opt.load_state_dict(update.payload["opt_state"])
        save_file(update.payload["snapshot"], self.path)

    def persisted(self, update: Update) -> bool:
        """The weights on disk are the current weights, and they differ from the pre-update snapshot."""
        if not os.path.exists(self.path):
            return False
        disk, now, before = load_file(self.path), self.policy.snapshot(), update.payload["snapshot"]
        same_as_now = all(torch.equal(disk[n], now[n]) for n in now)
        changed = any(not torch.equal(now[n], before[n]) for n in now)
        return same_as_now and changed


def old_logprobs(policy, step: dict, lp_new: torch.Tensor) -> torch.Tensor:
    """pi_old for a ratio. If the weights haven't moved since the step was sampled (one gradient step per
    batch, the usual case), pi_old *is* the current policy: use lp_new itself, detached, so the ratio is
    exactly 1 rather than off by batched-vs-single bf16 noise. Otherwise use what was stored at sampling."""
    if step.get("policy_version") == getattr(policy, "version", 0):
        return lp_new.detach()
    return torch.as_tensor(step["logprobs"], device=lp_new.device, dtype=lp_new.dtype)
