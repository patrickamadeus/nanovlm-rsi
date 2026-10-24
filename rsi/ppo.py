"""PPO (Schulman et al., 2017) with a small value head, the weight update ViSkill trains with.

GRPO's baseline is the mean reward of the task's group; PPO's is a learned value V(s), so a single attempt
carries signal and no group is needed. Rewards here arrive at the end of an episode and gamma = lambda = 1
(as in ViSkill), so every turn's return is the episode's reward R, and per turn

    A_t = R - V(s_t),  normalised over the batch          loss = clipped surrogate (same as GRPO) + (V - R)^2

V reads the policy's hidden state at the last prompt token, detached: the head learns, the trunk is moved only
by the policy loss. That is a cheaper critic than ViSkill's separate one (a full copy of the model), on purpose.
Advantages use the values from before the update; the head takes one step per minibatch alongside the policy.
"""
import copy
import os

import torch
from safetensors.torch import save_file

from rsi.grpo import grpo_token_loss
from rsi.weights import WeightMethod, _args


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

    def prepare(self, batch: list) -> dict | None:
        runs = [s for s in batch if not s.flags and s.origin == "attempt"]
        turns = [(st, s.reward) for s in runs for st in s.traj.steps if st["token_ids"]]
        if not turns:
            return None
        feats = []
        with torch.no_grad():                # every turn's state, so advantages normalise over the batch
            for i in range(0, len(turns), self.cfg.micro_batch):
                chunk = [t for t, _ in turns[i:i + self.cfg.micro_batch]]
                feats.append(self.policy.logprobs_batch(*_args(chunk), hidden=True)[1])
            feats = torch.cat(feats)
            if self.head is None:
                self._make_head(feats.shape[1])
            returns = torch.tensor([r for _, r in turns], device=feats.device, dtype=torch.float32)
            values = self.head(feats).squeeze(-1)
        adv = returns - values
        if len(adv) > 1 and adv.std() > 0:
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        head = (copy.deepcopy(self.head.state_dict()), copy.deepcopy(self.head_opt.state_dict()))
        return {"items": [(t, float(a)) for (t, _), a in zip(turns, adv)], "used": runs, "feats": feats,
                "returns": returns, "head_before": head, "value_loss": float(((values - returns) ** 2).mean()),
                "mean_value": float(values.mean())}

    def item_loss(self, turn, adv, lp, old, ref):
        return grpo_token_loss(lp, old, None, adv=adv, clip=self.cfg.grpo_clip, beta=0.0).sum()

    def after_step(self, update, minibatch: list[int]) -> None:
        feats, returns = update.payload["feats"][minibatch], update.payload["returns"][minibatch]
        self.head_opt.zero_grad(set_to_none=True)
        ((self.head(feats).squeeze(-1) - returns) ** 2).mean().backward()
        self.head_opt.step()

    def commit(self, update) -> None:
        super().commit(update)
        save_file({k: v.detach().cpu() for k, v in self.head.state_dict().items()}, self.head_path)

    def revert(self, update) -> None:
        super().revert(update)
        self.head.load_state_dict(update.payload["head_before"][0])
        self.head_opt.load_state_dict(update.payload["head_before"][1])
        save_file({k: v.detach().cpu() for k, v in self.head.state_dict().items()}, self.head_path)
