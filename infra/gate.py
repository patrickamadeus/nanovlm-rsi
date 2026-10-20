"""The acceptance bar, as code. An update is kept only if all five conditions hold.

The gate owns commit/revert: measuring a gain and probing persistence both need
the update applied, so the sequence is  eval before -> commit -> probe -> eval
after -> revert on failure. It evaluates on the gate split only; held-out data
never influences an accept/reject decision.
"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from config import LoopConfig
from envs.base import Env, Task
from update.base import Update, Updater

EvalFn = Callable[[list[Task]], tuple[float, float]]   # tasks -> (success_rate, parse_fail_rate)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    conditions: dict[str, bool | None]
    reason: str | None
    gate_enabled: bool
    before: tuple[float, float] | None = None
    after: tuple[float, float] | None = None


def gate(update: Update, updater: Updater, env: Env, cfg: LoopConfig, *, eval_fn: EvalFn) -> GateResult:
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
        "deployment_feedback": update.signal in cfg.gate_allowed_verdict_sources,
        "no_regression": after[1] - before[1] <= cfg.gate_parse_fail_delta,
    }
    failed = [n for n, ok in conditions.items() if not ok]
    if failed:
        updater.revert(update)
    return GateResult(not failed, conditions, f"failed: {', '.join(failed)}" if failed else None, True, before, after)
