"""Evaluation with learning disabled. Reports both held-out axes separately, always against cumulative cost.

  unseen_seed_rate  tasks whose goal appeared in training: new world, known skill
  unseen_goal_rate  tasks whose goal never appeared in training: compositional generalization
"""
import argparse
from dataclasses import dataclass

import numpy as np

from config import LoopConfig
from envs.base import Env, Task
from infra.trace import Cost, Trace
from rollout import recall, rollout, rollout_record


@dataclass(frozen=True)
class EvalResult:
    split: str
    n: int
    success_rate: float
    unseen_seed_rate: float
    unseen_goal_rate: float
    parse_fail_rate: float
    cost: Cost


def _rate(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_tasks(policy, env: Env, tasks: list[Task], trace: Trace, cfg: LoopConfig, store=None) -> list:
    """Roll out each task once, read-only on memory, logging every rollout as cost."""
    trajs = []
    for task in tasks:
        memory_text, ids, secs = recall(store, env, task, cfg.mem_top_k)
        traj = rollout(policy, env, task, cfg, memory_text=memory_text)
        trace.log_rollout(**rollout_record(traj, ids, extra_seconds=secs))
        trajs.append(traj)
    return trajs


def evaluate(policy, env: Env, tasks: list[Task], trace: Trace, cfg: LoopConfig, *, store=None,
             final: bool = False) -> EvalResult:
    trained_goals = {(r.get("family"), tuple(r.get("goal", ()))) for r in trace.records("rollout")
                     if r.get("split") == "train"}
    trajs = run_tasks(policy, env, tasks, trace, cfg, store)
    won = [float(t.total_reward >= 1.0) for t in trajs]
    seen = [w for w, t in zip(won, trajs) if (t.task.family, t.task.goal) in trained_goals]
    unseen = [w for w, t in zip(won, trajs) if (t.task.family, t.task.goal) not in trained_goals]
    turns = sum(len(t.steps) for t in trajs)
    split = tasks[0].split if tasks else "heldout"
    result = EvalResult(split, len(trajs), _rate(won), _rate(seen), _rate(unseen),
                        sum(t.parse_fails for t in trajs) / turns if turns else 0.0, trace.cumulative_cost)
    trace.log_eval(split=result.split, n=result.n, success_rate=result.success_rate,
                   unseen_seed_rate=result.unseen_seed_rate, unseen_goal_rate=result.unseen_goal_rate,
                   n_unseen_seed=len(seen), n_unseen_goal=len(unseen),
                   parse_fail_rate=result.parse_fail_rate, cost=result.cost, final=final)
    return result


def make_eval_fn(policy, env: Env, trace: Trace, cfg: LoopConfig, store=None):
    """The gate's measuring stick: tasks -> (success_rate, parse_fail_rate). Gate rollouts count as cost too."""
    def eval_fn(tasks: list[Task]) -> tuple[float, float]:
        trajs = run_tasks(policy, env, tasks, trace, cfg, store)
        turns = sum(len(t.steps) for t in trajs)
        return (_rate([float(t.total_reward >= 1.0) for t in trajs]),
                sum(t.parse_fails for t in trajs) / turns if turns else 0.0)
    return eval_fn


def main():
    from policy.vlm import Policy
    from rsi import make_env

    ap = argparse.ArgumentParser(description="Evaluate a policy with learning disabled.")
    ap.add_argument("--split", default="heldout", choices=["heldout", "gate"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--run_dir", default="runs/eval")
    ap.add_argument("--tiny", action="store_true", help="randomly-initialized tiny policy (CPU smoke)")
    ap.add_argument("--benchmarks", action="store_true", help="MMStar regression tripwire")
    args = ap.parse_args()
    cfg = LoopConfig(run_dir=args.run_dir)
    cfg.policy.tiny = args.tiny
    if args.benchmarks:
        from evals.benchmarks import load_mmstar, log_bench, run_mmstar
        from policy.vlm import Policy as _Policy
        r = run_mmstar(_Policy(cfg.policy), load_mmstar(args.n if args.n != 200 else None))
        log_bench(Trace(cfg.run_dir), r, when="standalone")
        print(f"MMStar n={r.n} accuracy={r.accuracy:.3f}{'  (SUBSET: debug only)' if r.subset else ''}")
        for c, a in r.per_category.items():
            print(f"  {c:<28}{a:.3f}")
        return
    env = make_env(cfg)
    r = evaluate(Policy(cfg.policy), env, env.tasks(args.split, args.n, np.random.default_rng(cfg.seed)),
                 Trace(cfg.run_dir), cfg, final=True)
    print(f"{'split':<10}{'n':>5}{'success':>10}{'unseen_seed':>13}{'unseen_goal':>13}{'parse_fail':>12}{'env_steps':>11}")
    print(f"{r.split:<10}{r.n:>5}{r.success_rate:>10.3f}{r.unseen_seed_rate:>13.3f}{r.unseen_goal_rate:>13.3f}"
          f"{r.parse_fail_rate:>12.3f}{r.cost.env_steps:>11}")


if __name__ == "__main__":
    main()
