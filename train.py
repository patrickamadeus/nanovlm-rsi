"""picoRSI: the self-improvement loop. Read this file first.

    curriculum picks tasks -> harness rolls out a group per task -> judge scores them
        -> update mode proposes a change -> gate commits or reverts it -> log everything
        -> evaluate on held-out tasks every eval_every iterations

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
from evaluate import EvalResult, evaluate, make_eval_fn
from harness.log import Trace
from harness.rollout import recall, rollout, rollout_record
from judges import make_judge
from models import make_policy
from rsi import make_update
from rsi.curriculum import make_curriculum
from rsi.gate import gate


def train(cfg: LoopConfig, policy=None) -> EvalResult:
    check_run_dir(cfg)
    policy     = policy or make_policy(cfg.policy)
    env        = make_env(cfg)
    judge      = make_judge(cfg, env)
    updater    = make_update(cfg, policy)                     # grpo | memory
    curriculum = make_curriculum(env, cfg)
    log        = Trace(cfg.run_dir)
    store      = getattr(updater, "store", None)              # only memory has one
    heldout    = env.tasks("heldout", cfg.eval_n, np.random.default_rng(cfg.seed))   # frozen per env_seed
    log.log_run(cfg_hash=cfg.hash(), git_sha=git_sha(), update=cfg.update, curriculum=cfg.curriculum,
                judge=cfg.judge, env=cfg.env, gate_enabled=cfg.gate_enabled, config=asdict(cfg))

    for it in range(cfg.iters):
        if it % cfg.eval_every == 0:                          # it == 0 is the pre-RSI baseline
            evaluate(policy, env, heldout, log, cfg, store=store)

        batch = []
        for task in curriculum(policy, log):                  # train split only
            memory_text, skill_ids, recall_s = recall(store, env, task, cfg.mem_top_k)   # once per task
            for g in range(cfg.group_size):                   # GRPO needs a group; memory just gets more tries
                traj = rollout(policy, env, task, cfg, memory_text=memory_text)
                scored = judge.score(traj)
                log.log_rollout(**rollout_record(traj, skill_ids, scored.verdict_source,
                                                 extra_seconds=recall_s if g == 0 else 0.0),
                                flags=scored.flags, iter=it, judge_verdict=scored.judge_verdict,
                                judge_tokens=scored.judge_tokens)
                if store is not None:                         # use-attribution: was the skill there when it worked?
                    store.note_use(skill_ids, succeeded=scored.reward >= 1.0)
                batch.append(scored)

        update = updater.propose(batch)
        stats = getattr(curriculum, "last_stats", None)
        if update is None:
            log.log_update(iter=it, committed=False, gate_enabled=cfg.gate_enabled, reason="no usable signal",
                           curriculum_stats=stats)
            continue
        result = gate(update, updater, env, cfg, eval_fn=make_eval_fn(policy, env, log, cfg, store))
        log.log_update(iter=it, mode=update.mode, signal=update.signal, cost=update.cost,
                       provenance=list(update.provenance), committed=result.passed,
                       gate_enabled=result.gate_enabled, conditions=result.conditions, reason=result.reason,
                       gate_before=result.before, gate_after=result.after, curriculum_stats=stats,
                       **{k: v for k, v in update.payload.items() if isinstance(v, (int, float))})

    return evaluate(policy, env, heldout, log, cfg, store=store, final=True)


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
    ap.add_argument("--config", default=None, help="a name in configs/ (count_grpo, tiny, ...) or a YAML path")
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
