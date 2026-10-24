"""PPO (Schulman et al., 2017) with a small value head, the weight update ViSkill trains with.

GRPO's baseline is the mean reward of the task's group; PPO's is a learned value V(s), so a single attempt
carries signal and no group is needed. Rewards here arrive at the end of an episode and gamma = lambda = 1
(as in ViSkill), so every turn's return is the episode's reward R, and per turn

    A_t = R - V(s_t),  normalised over the batch          loss = clipped surrogate (same as GRPO) + (V - R)^2

V reads the policy's hidden state at the last prompt token, detached: the head learns, the trunk is moved only
by the policy loss. That is a cheaper critic than ViSkill's separate one (a full copy of the model), on purpose.
Value targets come from the same batch, before the step: one gradient step per batch, like every weight method.
"""
import copy
import os

import torch
from safetensors.torch import save_file

from rsi.grpo import grpo_token_loss
from rsi.weights import WeightMethod, old_logprobs


class PPO(WeightMethod):
    name = "ppo"

    def __init__(self, policy, cfg, weights=None):
        super().__init__(policy, cfg, weights)
        self.head = self.head_opt = None
        self.head_path = os.path.join(cfg.run_dir, "value_head.safetensors")

    def _make_head(self, width: int) -> None:
        self.head = torch.nn.Linear(width, 1).to(self.policy.device)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.head_opt = torch.optim.AdamW(self.head.parameters(), lr=self.cfg.ppo_value_lr, weight_decay=0.0)
        if self.cfg.resume and os.path.exists(self.head_path):
            from safetensors.torch import load_file
            self.head.load_state_dict({k: v.to(self.policy.device) for k, v in load_file(self.head_path).items()})

    def backward(self, batch: list) -> dict | None:
        runs = [s for s in batch if not s.flags and s.origin == "attempt"]
        items = [(st, s.reward) for s in runs for st in s.traj.steps if st["token_ids"]]
        if not items:
            return None
        chunks = [items[i:i + self.cfg.micro_batch] for i in range(0, len(items), self.cfg.micro_batch)]
        feats = []
        with torch.no_grad():                    # pass 1: every turn's state, so advantages normalise over the batch
            for chunk in chunks:
                feats.append(self.policy.logprobs_batch(*self._args(chunk), hidden=True)[1])
        feats = torch.cat(feats)
        if self.head is None:
            self._make_head(feats.shape[1])
        returns = torch.tensor([r for _, r in items], device=feats.device, dtype=torch.float32)
        self.head_opt.zero_grad(set_to_none=True)
        values = self.head(feats).squeeze(-1)
        value_loss = ((values - returns) ** 2).mean()
        value_loss.backward()
        adv = returns - values.detach()
        if len(adv) > 1 and adv.std() > 0:
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        n_tokens = sum(len(st["token_ids"]) for st, _ in items)
        total, k = 0.0, 0
        for chunk in chunks:                     # pass 2: the policy loss, with gradients
            lps = self.policy.logprobs_batch(*self._args(chunk))
            loss = 0.0
            for (st, _), lp in zip(chunk, lps):
                loss = loss + grpo_token_loss(lp, old_logprobs(self.policy, st, lp), None, adv=float(adv[k]),
                                              clip=self.cfg.grpo_clip, beta=0.0).sum()
                k += 1
            loss = loss / n_tokens
            loss.backward()
            total += loss.item()
        return {"used": runs, "loss": total, "value_loss": value_loss.item(), "n_tokens": n_tokens,
                "mean_value": values.mean().item()}

    @staticmethod
    def _args(chunk):
        return ([st["prompt"] for st, _ in chunk], [st["images"] for st, _ in chunk],
                [st["token_ids"] for st, _ in chunk])

    def propose(self, batch):
        head = (copy.deepcopy(self.head.state_dict()), copy.deepcopy(self.head_opt.state_dict())) if self.head else None
        update = super().propose(batch)
        if update is not None:
            update.payload["head_before"] = head
        return update

    def commit(self, update) -> None:
        super().commit(update)
        self.head_opt.step()
        save_file({k: v.detach().cpu() for k, v in self.head.state_dict().items()}, self.head_path)

    def revert(self, update) -> None:
        super().revert(update)
        before = update.payload["head_before"]
        if before is None:                       # the head was born in this update: forget it entirely
            self.head = self.head_opt = None
            if os.path.exists(self.head_path):
                os.remove(self.head_path)
        else:
            self.head.load_state_dict(before[0])
            self.head_opt.load_state_dict(before[1])
            save_file({k: v.detach().cpu() for k, v in self.head.state_dict().items()}, self.head_path)
