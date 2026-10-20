"""nanoVLM-RSI: the loop. Read this file first.

    propose tasks -> roll out a group per task -> verify -> carrier proposes an update
                  -> gate commits or reverts it -> trace everything -> evaluate on held-out

One carrier per run (update_rule), one proposer (proposer). Swapping either is one config line.
"""
import argparse
import os
import subprocess
from dataclasses import fields

import numpy as np

from config import LoopConfig, PolicyConfig
from envs.base import Env
from envs.static import StaticEnv
from evaluate import EvalResult, evaluate, make_eval_fn
from infra.gate import gate
from infra.trace import Trace
from propose import make_proposer
from rollout import recall, rollout, rollout_record
from update import make_updater
from verify import make_verifier


def make_env(cfg: LoopConfig) -> Env:
    if cfg.env == "static":
        return StaticEnv(env_seed=cfg.env_seed, heldout_n=cfg.heldout_n)
    if cfg.env == "craftax":
        from envs.craftax import CraftaxEnv          # imported lazily: JAX loads only when Craftax is used
        return CraftaxEnv(env_seed=cfg.env_seed, heldout_n=cfg.heldout_n, budget=cfg.craftax_budget)
    raise ValueError(f"unknown env {cfg.env!r}")


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def check_run_dir(cfg: LoopConfig) -> None:
    """A second run in the same directory would inherit memory, cost and trained goals, so the
    'pre-RSI baseline' would not be one. Refuse unless the caller is resuming on purpose."""
    used = [f for f in ("trace.jsonl", "skills.jsonl")
            if os.path.exists(os.path.join(cfg.run_dir, f)) and os.path.getsize(os.path.join(cfg.run_dir, f))]
    if used and not cfg.resume:
        raise FileExistsError(f"{cfg.run_dir} already holds {used}; pick a new --run_dir or pass --resume true")


def rsi(cfg: LoopConfig, policy=None) -> EvalResult:
    check_run_dir(cfg)
    if policy is None:
        from policy.vlm import Policy
        policy = Policy(cfg.policy)
    env      = make_env(cfg)
    verifier = make_verifier(cfg.verifier, cfg)
    updater  = make_updater(cfg.update_rule, policy, cfg)     # weights | memory | scaffold
    propose  = make_proposer(cfg.proposer, env, cfg)
    trace    = Trace(cfg.run_dir)
    store    = getattr(updater, "store", None)                # only the memory carrier has one
    heldout  = env.tasks("heldout", cfg.eval_n, np.random.default_rng(cfg.seed))   # frozen per env_seed
    trace.log_run(cfg_hash=cfg.hash(), git_sha=git_sha(), seeds={"seed": cfg.seed, "env_seed": cfg.env_seed},
                  carrier=cfg.update_rule, proposer=cfg.proposer, gate_enabled=cfg.gate_enabled)

    for it in range(cfg.iters):
        if it % cfg.eval_every == 0:                          # it == 0 is the pre-RSI baseline
            evaluate(policy, env, heldout, trace, cfg, store=store)

        tasks = propose(policy, trace)                        # train split only
        batch = []
        for task in tasks:
            memory_text, skill_ids, recall_s = recall(store, env, task, cfg.mem_top_k)   # once per task
            for g in range(cfg.group_size):                   # GRPO needs a group; memory just gets more tries
                traj = rollout(policy, env, task, cfg, memory_text=memory_text)
                scored = verifier.score(traj)
                trace.log_rollout(**rollout_record(traj, skill_ids, scored.verdict_source,
                                                   extra_seconds=recall_s if g == 0 else 0.0),
                                  flags=scored.flags, iter=it)
                if store is not None:                         # use-attribution: was the skill there when it worked?
                    store.note_use(skill_ids, succeeded=scored.reward >= 1.0)
                batch.append(scored)

        update = updater.propose(batch)
        if update is None:
            trace.log_update(iter=it, committed=False, gate_enabled=cfg.gate_enabled, reason="no usable signal")
            continue
        result = gate(update, updater, env, cfg, eval_fn=make_eval_fn(policy, env, trace, cfg, store))
        trace.log_update(iter=it, carrier=update.carrier, role=update.role, horizon=update.horizon,
                         clock=update.clock, signal=update.signal, cost=update.cost,
                         provenance=list(update.provenance), committed=result.passed,
                         gate_enabled=result.gate_enabled, conditions=result.conditions,
                         reason=result.reason, gate_before=result.before, gate_after=result.after)

    return evaluate(policy, env, heldout, trace, cfg, store=store, final=True)


def main():
    ap = argparse.ArgumentParser(description="Run one RSI loop. Every LoopConfig field is a flag.")
    for f in fields(LoopConfig):
        if f.type is bool:
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"))
        elif f.type in (int, float, str):
            ap.add_argument(f"--{f.name}", type=f.type)
    ap.add_argument("--tiny", action="store_true", help="randomly-initialized tiny policy (CPU smoke run)")
    args = vars(ap.parse_args())
    tiny = args.pop("tiny")
    cfg = LoopConfig(**{k: v for k, v in args.items() if v is not None})
    if tiny:
        cfg.policy = PolicyConfig(tiny=True, device="cpu", dtype="float32", max_new_tokens=8)
    r = rsi(cfg)
    print(f"held-out success {r.success_rate:.3f}  (unseen seed {r.unseen_seed_rate:.3f}, "
          f"unseen goal {r.unseen_goal_rate:.3f})  parse_fail {r.parse_fail_rate:.3f}  "
          f"env_steps {r.cost.env_steps}  tokens {r.cost.policy_tokens}  -> {cfg.run_dir}/trace.jsonl")


if __name__ == "__main__":
    main()
