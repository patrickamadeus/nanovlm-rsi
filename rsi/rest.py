"""ReST / STaR-style self-training (Gulcehre et al., 2023; Zelikman et al., 2022): keep the policy's own
successful attempts and fine-tune on them, i.e. maximise the log-likelihood of every turn of every success:

    loss = - sum over success turns and tokens of log pi(token)  /  number of tokens

Unlike GRPO it needs no group with mixed outcomes, so one success is enough signal, and hindsight copies
(rsi/hindsight.py) count as successes: a failure relabelled to the goal it reached becomes training data.
"""
from rsi.weights import WeightMethod


class ReST(WeightMethod):
    name = "rest"

    def __init__(self, policy, cfg, weights=None):
        super().__init__(policy, cfg, weights, lr=cfg.rest_lr)

    needs_old = False

    def prepare(self, batch: list) -> dict | None:
        wins = [s for s in batch if s.reward >= 1.0 and not s.flags]
        items = [(st, 1.0) for s in wins for st in s.traj.steps if st["token_ids"] and st["action"] is not None]
        if not items:
            return None
        return {"items": items, "used": wins, "hindsight": sum(s.origin == "hindsight" for s in wins)}

    def item_loss(self, turn, coef, lp, old, ref):
        return -lp.sum()
