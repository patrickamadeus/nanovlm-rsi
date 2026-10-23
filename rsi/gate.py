"""The acceptance bar, as code. An update is kept only if all five conditions hold.

Every update mode (rsi/grpo.py, rsi/memory.py) has the same four methods:
    propose(batch) -> Update | None    turn scored trajectories into a candidate change
    commit(update)                     apply it durably (to disk)
    revert(update)                     undo it exactly — the gate depends on this
    persisted(update) -> bool          rebuild from disk: is the change there, and did it change anything?

The gate owns commit/revert: measuring a gain and probing persistence both need the update
applied, so the sequence is  eval before -> commit -> probe -> eval after -> revert on failure.
It evaluates on the gate split only; held-out data never influences an accept/reject decision.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from configs.config import LoopConfig
from envs.base import Env, Task
from utils.log import Cost

EvalFn = Callable[[list[Task]], tuple[float, float]]   # tasks -> (success_rate, parse_fail_rate)


@dataclass(frozen=True)
class Update:
    mode: str                  # "grpo" | "memory"
    signal: str                # where the evidence came from; checked against gate_allowed_signals
    payload: dict = field(default_factory=dict)
    cost: Cost = field(default_factory=Cost)
    provenance: tuple[str, ...] = ()   # the trajectories that produced this update


@dataclass(frozen=True)
class GateResult:
    passed: bool
    conditions: dict[str, bool | None]
    reason: str | None
    gate_enabled: bool
    before: tuple[float, float] | None = None
    after: tuple[float, float] | None = None


def gate(update: Update, updater, env: Env, cfg: LoopConfig, *, eval_fn: EvalFn) -> GateResult:
    names = ("proposed", "survives_reset", "gate_gain", "deployment_feedback", "no_regression")
    if not cfg.gate_enabled:       # allowed for ablations — and self-identifying, so it can't pass as gated
        updater.commit(update)
        return GateResult(True, dict.fromkeys(names), None, False)

    tasks = env.tasks("gate", cfg.gate_n, np.random.default_rng(cfg.seed))   # same gate set every decision
    before = eval_fn(tasks)
    updater.commit(update)
    after = eval_fn(tasks)
    conditions = {
        "proposed": bool(update.provenance),
        "survives_reset": updater.persisted(update),                # reload the organ from disk, look for it
        "gate_gain": after[0] - before[0] >= 0,
        "deployment_feedback": update.signal in cfg.gate_allowed_signals,
        "no_regression": after[1] - before[1] <= cfg.gate_parse_fail_delta,
    }
    failed = [n for n, ok in conditions.items() if not ok]
    if failed:
        updater.revert(update)
    return GateResult(not failed, conditions, f"failed: {', '.join(failed)}" if failed else None, True, before, after)
