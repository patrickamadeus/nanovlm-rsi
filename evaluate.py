"""Evaluation with learning off.

  held-out   our own frozen env tasks — the in-domain claim. Two axes, reported separately:
             unseen_seed_rate (goal seen in training, new image) and unseen_goal_rate (goal never trained on)
  benchmarks standard VLM benchmarks through lmms-eval (MMStar by default) — the forgetting check.
             lmms-eval runs our policy directly (utils/lmms.py), trained LoRA included.

    python evaluate.py --run runs/count_grpo                          # held-out, the run's final weights
    python evaluate.py --run runs/count_grpo --benchmarks mmstar      # needs: pip install -e ".[eval]"
    python evaluate.py --benchmarks mmstar                            # the untrained base model, for comparison
"""
import argparse
import os
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from configs.config import LoopConfig, load_config
from envs.base import Env, Task
from utils.log import Cost, Trace
from utils.rollout import rollout_group, rollout_record, timed


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


def contexts_for(methods, env: Env, tasks: list[Task]) -> list:
    """Every method's addition to each task's prompt, merged. Built once per task, never per attempt."""
    from rsi.base import merge
    return [merge([m.context(t, env) for m in methods]) for t in tasks]


def run_tasks(policy, env: Env, tasks: list[Task], log: Trace, cfg: LoopConfig, methods=(), **fields) -> list:
    """Roll out each task once in one lockstep batch, read-only on every method, logging every rollout as cost."""
    contexts, secs = timed(contexts_for, methods, env, tasks)
    trajs = rollout_group(policy, env, tasks, cfg, contexts, greedy=cfg.eval_greedy)
    for traj in trajs:
        log.log_rollout(**rollout_record(traj, extra_seconds=secs / max(len(trajs), 1), key=env.task_key(traj.task)),
                        **fields)
    return trajs


def evaluate(policy, env: Env, tasks: list[Task], log: Trace, cfg: LoopConfig, *, methods=(),
             final: bool = False, env_name: str | None = None) -> EvalResult:
    trained_goals = {(r.get("family"), tuple(r.get("goal", ()))) for r in log.records("rollout")
                     if r.get("split") == "train"}
    with paired_randomness(cfg.seed + 1):     # every held-out point of a run, and every run, sees the same draws
        trajs = run_tasks(policy, env, tasks, log, cfg, methods)
    if trajs:
        log.log_sample(env_name or cfg.env, trajs[0])
    won = [float(t.total_reward >= 1.0) for t in trajs]
    seen = [w for w, t in zip(won, trajs) if (t.task.family, t.task.goal) in trained_goals]
    unseen = [w for w, t in zip(won, trajs) if (t.task.family, t.task.goal) not in trained_goals]
    by_key: dict[str, list[float]] = {}
    for w, t in zip(won, trajs):
        by_key.setdefault(env.task_key(t.task), []).append(w)
    turns = sum(len(t.steps) for t in trajs)
    split = tasks[0].split if tasks else "heldout"
    result = EvalResult(split, len(trajs), _rate(won), _rate(seen), _rate(unseen),
                        sum(t.parse_fails for t in trajs) / turns if turns else 0.0, log.cumulative_cost)
    log.log_eval(split=result.split, n=result.n, success_rate=result.success_rate,
                 unseen_seed_rate=result.unseen_seed_rate, unseen_goal_rate=result.unseen_goal_rate,
                 n_unseen_seed=len(seen), n_unseen_goal=len(unseen), by_key={k: _rate(v) for k, v in by_key.items()},
                 parse_fail_rate=result.parse_fail_rate, cost=result.cost, final=final, env=env_name or cfg.env)
    return result


@contextmanager
def paired_randomness(seed: int):
    """Sample with a fixed seed, then put the caller's random state back. The gate measures before and after
    an update with the same draws, so the comparison sees the update rather than sampling noise, and the
    training rollouts that follow are as random as if no evaluation had happened."""
    cpu = torch.get_rng_state()
    gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if gpu is not None:
            torch.cuda.set_rng_state_all(gpu)


def make_eval_fn(policy, env: Env, log: Trace, cfg: LoopConfig, methods=()):
    """The gate's measuring stick: tasks -> (success_rate, parse_fail_rate). Gate rollouts count as cost too."""
    def eval_fn(tasks: list[Task]) -> tuple[float, float]:
        with paired_randomness(cfg.seed):
            trajs = run_tasks(policy, env, tasks, log, cfg, methods)
        turns = sum(len(t.steps) for t in trajs)
        return (_rate([float(t.total_reward >= 1.0) for t in trajs]),
                sum(t.parse_fails for t in trajs) / turns if turns else 0.0)
    return eval_fn


def load_run(run_dir: str, **policy_overrides):
    """Rebuild a finished run's config and policy, with its committed LoRA weights if it trained any."""
    from dataclasses import replace

    from models import make_policy
    run = Trace(run_dir).records("run")[0]
    saved = {k: v for k, v in run["config"].items() if k != "policy"}
    if "update" in saved:                     # runs from before methods/planner replaced update/curriculum
        saved["methods"], saved["planner"] = [saved.pop("update")], saved.pop("curriculum", "uniform")
    cfg = load_config(None, **saved, policy=run["config"]["policy"])
    policy = make_policy(replace(cfg.policy, **policy_overrides))
    weights = os.path.join(run_dir, "weights.safetensors")
    if os.path.exists(weights):
        from safetensors.torch import load_file
        policy.enable_lora(cfg.lora_r, cfg.lora_alpha)
        policy.restore(load_file(weights))
    return cfg, policy


# benchmarks see the model's standard image pipeline and free-form answers, not our env settings
BENCHMARK_POLICY = dict(image_side=0, response_prefix="", stop_at_newline=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="a finished run directory; omit to evaluate an untrained model")
    ap.add_argument("--config", help="without --run: the env and settings to evaluate on (a configs/yaml name)")
    ap.add_argument("--model", help="without --run: a HF model id, or 'random' for the random-action baseline")
    ap.add_argument("--out", help="where to write the eval trace (default: <run>/eval or runs/base)")
    ap.add_argument("--split", default="heldout", choices=["heldout", "gate"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--benchmarks", help="comma-separated lmms-eval tasks, e.g. mmstar")
    ap.add_argument("--limit", type=int, help="lmms-eval items per task (debug only)")
    args = ap.parse_args()

    from dataclasses import replace

    overrides = BENCHMARK_POLICY if args.benchmarks else {}
    methods = []
    if args.run:
        cfg, policy = load_run(args.run, **overrides)
        from rsi import WEIGHT_METHODS, make_method
        run_cfg = replace(cfg, run_dir=args.run, resume=True)            # memory is read from the run, never written
        methods = [make_method(m, run_cfg, policy) for m in cfg.methods if m not in WEIGHT_METHODS]
    else:
        cfg = load_config(args.config, heldout_n=max(args.n, LoopConfig.heldout_n))
        if args.model == "random":
            from models.baselines import RandomPolicy
            policy = RandomPolicy(cfg.seed)
        else:
            from models import make_policy
            policy_cfg = replace(cfg.policy, **({"model_id": args.model} if args.model else {}), **overrides)
            policy = make_policy(policy_cfg)
    out = args.out or os.path.join(args.run or "runs/base", "eval")

    if args.benchmarks:
        from utils.lmms import run_benchmarks
        scores = run_benchmarks(policy, args.benchmarks.split(","), args.limit)
        Trace(out).log_eval(split="benchmarks", scores=scores, limit=args.limit, run=args.run)
        for task, metrics in scores.items():
            print(task, " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return

    from envs import make_env
    env = make_env(cfg)
    r = evaluate(policy, env, env.tasks(args.split, args.n, np.random.default_rng(cfg.seed)), Trace(out), cfg,
                 methods=methods, final=True)
    print(f"{'split':<10}{'n':>5}{'success':>10}{'unseen_seed':>13}{'unseen_goal':>13}{'parse_fail':>12}")
    print(f"{r.split:<10}{r.n:>5}{r.success_rate:>10.3f}{r.unseen_seed_rate:>13.3f}{r.unseen_goal_rate:>13.3f}"
          f"{r.parse_fail_rate:>12.3f}")


if __name__ == "__main__":
    main()
