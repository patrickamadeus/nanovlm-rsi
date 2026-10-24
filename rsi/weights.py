"""What every weight-updating method (grpo.py, ppo.py, rest.py) shares: the LoRA adapters, one file on disk,
and the propose / commit / revert / persisted cycle the gate drives.

A subclass says two things:
    prepare(batch) -> {"items": [(turn, coef), ...], "used": [...], **stats} | None   which turns, how weighted
    item_loss(turn, coef, lp_new, lp_old, lp_ref) -> tensor                          the loss of one turn

propose() only prepares and snapshots the adapters and optimizer, so it changes nothing. commit() does the
optimisation, as in PPO-style trainers (and ViSkill): the turns are shuffled into cfg.minibatches minibatches,
pi_old is scored once for all of them before any step, with exactly the batching each minibatch will use,
so each ratio moves only because the parameters did; then one optimizer step per minibatch. revert()
restores the snapshot exactly. LoRA state is small, so the snapshot is cheap.

Several weight methods in one run share one set of adapters (Weights) but keep their own optimizers.
"""
import copy
import os

import numpy as np
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


def _args(turns):
    return [t["prompt"] for t in turns], [t["images"] for t in turns], [t["token_ids"] for t in turns]


class WeightMethod(Method):
    changes = "weights"
    needs_old = True                 # ratio methods score pi_old; plain likelihood (ReST) does not
    uses_ref = False                 # GRPO's KL term scores pi_ref

    def __init__(self, policy, cfg, weights: Weights | None = None, lr: float | None = None):
        self.policy, self.cfg = policy, cfg
        self.weights = weights or Weights(policy, cfg)
        self.params, self.path = self.weights.params, self.weights.path
        self.opt = torch.optim.AdamW(self.params, lr=lr or cfg.grpo_lr, weight_decay=0.0)

    def prepare(self, batch: list) -> dict | None:
        raise NotImplementedError

    def item_loss(self, turn: dict, coef: float, lp, old, ref) -> torch.Tensor:
        raise NotImplementedError

    def after_step(self, update: Update, minibatch: list[int]) -> None:
        """Optional: anything else that steps with each minibatch (PPO's value head)."""

    def propose(self, batch: list) -> Update | None:
        prep = self.prepare(batch)
        if prep is None or not prep["items"]:
            return None
        used = prep.pop("used")
        cost = Cost(sum(s.traj.env_steps for s in batch), sum(s.traj.tokens_in + s.traj.tokens_out for s in batch))
        signal = used[0].verdict_source if len({s.verdict_source for s in used}) == 1 else "mixed"
        provenance = tuple(f"{s.traj.task.family}:{s.traj.task.seed}" for s in used)
        prep["n_tokens"] = sum(len(t["token_ids"]) for t, _ in prep["items"])
        payload = {"snapshot": self.policy.snapshot(), "opt_state": copy.deepcopy(self.opt.state_dict()), **prep}
        return Update(self.name, signal, payload, cost, provenance)

    def chunks(self, items: list) -> list[list[list[int]]]:
        """Minibatches of micro-batches of item indices, shuffled reproducibly."""
        order = np.random.default_rng([self.cfg.seed, self.policy.version]).permutation(len(items))
        mbs = [mb for mb in np.array_split(order, min(self.cfg.minibatches, len(items))) if len(mb)]
        return [[mb[i:i + self.cfg.micro_batch].tolist() for i in range(0, len(mb), self.cfg.micro_batch)]
                for mb in mbs]

    def score(self, items: list, idx: list[int], ref: bool = False) -> list[torch.Tensor]:
        fn = self.policy.reference_logprobs_batch if ref else self.policy.logprobs_batch
        return fn(*_args([items[i][0] for i in idx]))

    def commit(self, update: Update) -> None:
        items = update.payload["items"]
        plan = self.chunks(items)
        old, ref = {}, {}
        with torch.no_grad():                # pi_old (and pi_ref), batched exactly as the steps will be
            for mb in plan:
                for idx in mb:
                    if self.needs_old:
                        old.update(zip(idx, self.score(items, idx)))
                    if self.uses_ref and self.cfg.grpo_beta > 0:
                        ref.update(zip(idx, self.score(items, idx, ref=True)))
        losses, norms, kls, clipped, n_seen = [], [], 0.0, 0.0, 0
        for mb in plan:
            self.opt.zero_grad(set_to_none=True)
            n_tok = sum(len(items[i][0]["token_ids"]) for idx in mb for i in idx)
            total = 0.0
            for idx in mb:
                lps = self.score(items, idx)
                loss = sum(self.item_loss(items[i][0], items[i][1], lp, old.get(i), ref.get(i))
                           for i, lp in zip(idx, lps)) / n_tok
                with torch.no_grad():        # how far each minibatch's policy has moved from pi_old
                    for i, lp in zip(idx, lps):
                        if i in old:
                            d = (old[i] - lp).float()
                            kls += float(d.sum())
                            clipped += float(((torch.exp(-d) - 1).abs() > self.cfg.grpo_clip).sum())
                            n_seen += d.numel()
                loss.backward()
                total += loss.item()
            norms.append(torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grpo_max_grad_norm).item())
            self.opt.step()
            self.after_step(update, [i for idx in mb for i in idx])
            losses.append(total)
        self.opt.zero_grad(set_to_none=True)
        self.policy.version += 1
        self.weights.save()
        update.payload.update(loss=float(np.mean(losses)), grad_norm=float(np.mean(norms)), steps=len(plan),
                              kl=kls / max(n_seen, 1), clip_frac=clipped / max(n_seen, 1))

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
