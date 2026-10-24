"""picoRSI: the self-improvement loop. Read this file first.

    planner picks tasks -> each method adds context -> a group of attempts per task, in one batch
        -> judge scores them -> methods reshape the signal -> each updating method proposes a change,
        and the gate keeps or undoes it -> methods and planner observe -> held-out eval every eval_every

    python train.py --config count_grpo            # the headline run
    python train.py --config tiny                  # CPU smoke run, ~1 min
    python train.py --config count_grpo --iters 10 --run_dir runs/short    # flags override the YAML
"""
import argparse
import os
import subprocess
from dataclasses import asdict, fields

import numpy as np

from configs.config import LoopConfig, load_config
from envs import make_env
from evaluate import EvalResult, contexts_for, evaluate, make_eval_fn
from models import make_policy
from rsi import make_judge, make_methods, make_planner, make_update_planner
from rsi.gate import gate
from utils.log import Trace
from utils.rollout import rollout_group, rollout_record, timed


def train(cfg: LoopConfig, policy=None) -> EvalResult:
    check_run_dir(cfg)
    policy   = policy or make_policy(cfg.policy)
    env      = make_env(cfg)
    log      = Trace(cfg.run_dir)
    judge    = make_judge(cfg, env)
    methods  = make_methods(cfg, policy, env)                # grpo, memory, ... in config order
    planner  = make_planner(env, cfg, log, policy)           # which tasks
    chooser  = make_update_planner(cfg, [m.name for m in methods if m.changes])   # which methods update
    heldout  = env.tasks("heldout", cfg.eval_n, np.random.default_rng(cfg.seed))   # frozen per env_seed
    eval_fn  = make_eval_fn(policy, env, log, cfg, methods)
    log.log_run(cfg_hash=cfg.hash(), git_sha=git_sha(), methods=list(cfg.methods), planner=cfg.planner,
                update_planner=cfg.update_planner, judge=cfg.judge, env=cfg.env, gate_enabled=cfg.gate_enabled,
                config=asdict(cfg))

    for it in range(cfg.iters):
        if it % cfg.eval_every == 0:                         # it == 0 is the pre-RSI baseline
            evaluate(policy, env, heldout, log, cfg, methods=methods)

        tasks = planner.tasks(cfg.tasks_per_iter)            # train split only
        contexts, ctx_s = timed(contexts_for, methods, env, tasks)
        group = [t for t in tasks for _ in range(cfg.group_size)]
        trajs = rollout_group(policy, env, group, cfg, [c for c in contexts for _ in range(cfg.group_size)])
        batch = []
        for traj in trajs:
            scored = judge.score(traj)
            log.log_rollout(**rollout_record(traj, scored.verdict_source, ctx_s / len(trajs), env.task_key(traj.task)),
                            flags=scored.flags, iter=it, judge_verdict=scored.judge_verdict,
                            judge_tokens=scored.judge_tokens)
            batch.append(scored)
        for m in methods:                                    # hindsight, skill rewards: reshape before anyone learns
            batch = m.signal(batch, env)

        results, before = {}, None                           # each update is gated on its own
        names = chooser.updates([m.name for m in methods if m.changes])
        for m in (m for m in methods if m.name in names):
            update = m.propose(batch)
            stats = {**(getattr(planner, "last_stats", None) or {}), **(getattr(chooser, "last_stats", None) or {})}
            if update is None:
                log.log_update(iter=it, method=m.name, committed=False, gate_enabled=cfg.gate_enabled,
                               reason="no usable signal", planner_stats=stats)
                continue
            result = gate(update, m, env, cfg, eval_fn=eval_fn, before=before)
            before = result.after if result.passed else result.before    # the next gate starts from here
            results[m.name] = result
            log.log_update(iter=it, method=m.name, mode=update.mode, changes=m.changes, signal=update.signal,
                           cost=update.cost, provenance=list(update.provenance), committed=result.passed,
                           gate_enabled=result.gate_enabled, conditions=result.conditions, reason=result.reason,
                           gate_before=result.before, gate_after=result.after, planner_stats=stats,
                           **{k: v for k, v in update.payload.items() if isinstance(v, (int, float))})
        for m in methods:
            m.observe(batch)
        planner.observe(batch, results)
        chooser.observe(batch, results)

    return evaluate(policy, env, heldout, log, cfg, methods=methods, final=True)


def check_run_dir(cfg: LoopConfig) -> None:
    """A second run in the same directory would inherit memory, cost and trained goals, so the
    'pre-RSI baseline' would not be one. Refuse unless the caller is resuming on purpose."""
    used = [f for f in ("trace.jsonl", "skills.jsonl")
            if os.path.exists(os.path.join(cfg.run_dir, f)) and os.path.getsize(os.path.join(cfg.run_dir, f))]
    if used and not cfg.resume:
        raise FileExistsError(f"{cfg.run_dir} already holds {used}; pick a new --run_dir or pass --resume true")


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_cfg(argv=None) -> LoopConfig:
    ap = argparse.ArgumentParser(description="Run one self-improvement loop.")
    ap.add_argument("--config", default=None, help="a name in configs/yaml/ (count_grpo, tiny, ...) or a YAML path")
    for f in fields(LoopConfig):
        if str(f.type).startswith("tuple"):
            cast = float if "float" in str(f.type) else str
            ap.add_argument(f"--{f.name}", type=lambda s, cast=cast: tuple(cast(x) for x in s.split(",") if x))
        elif f.type is bool:
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"))
        elif f.type in (int, float, str):
            ap.add_argument(f"--{f.name}", type=f.type)
    args = vars(ap.parse_args(argv))
    name = args.pop("config")
    return load_config(name, **{k: v for k, v in args.items() if v is not None})


def main():
    cfg = parse_cfg()
    r = train(cfg)
    print(f"held-out success {r.success_rate:.3f}  (unseen seed {r.unseen_seed_rate:.3f}, "
          f"unseen goal {r.unseen_goal_rate:.3f})  parse_fail {r.parse_fail_rate:.3f}  "
          f"env_steps {r.cost.env_steps}  tokens {r.cost.policy_tokens}  -> {cfg.run_dir}/trace.jsonl")


if __name__ == "__main__":
    main()
