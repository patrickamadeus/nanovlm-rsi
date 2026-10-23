"""Evaluation with learning off.

  held-out   our own frozen env tasks — the in-domain claim. Two axes, reported separately:
             unseen_seed_rate (goal seen in training, new image) and unseen_goal_rate (goal never trained on)
  benchmarks standard VLM benchmarks through lmms-eval (MMStar by default) — the forgetting check.
             The trained LoRA is merged into the base weights and saved as a plain HF checkpoint first.

    python evaluate.py --run runs/count_grpo                          # held-out, the run's final weights
    python evaluate.py --run runs/count_grpo --benchmarks mmstar      # needs: pip install -e ".[eval]"
    python evaluate.py --benchmarks mmstar                            # the untrained base model, for comparison
"""
import argparse
import os
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

from configs.config import LoopConfig, PolicyConfig, load_config
from envs.base import Env, Task
from harness.log import Cost, Trace
from harness.rollout import recall, rollout, rollout_record


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


def run_tasks(policy, env: Env, tasks: list[Task], log: Trace, cfg: LoopConfig, store=None) -> list:
    """Roll out each task once, read-only on memory, logging every rollout as cost."""
    trajs = []
    for task in tasks:
        memory_text, ids, secs = recall(store, env, task, cfg.mem_top_k)
        traj = rollout(policy, env, task, cfg, memory_text=memory_text)
        log.log_rollout(**rollout_record(traj, ids, extra_seconds=secs))
        trajs.append(traj)
    return trajs


def evaluate(policy, env: Env, tasks: list[Task], log: Trace, cfg: LoopConfig, *, store=None,
             final: bool = False) -> EvalResult:
    trained_goals = {(r.get("family"), tuple(r.get("goal", ()))) for r in log.records("rollout")
                     if r.get("split") == "train"}
    trajs = run_tasks(policy, env, tasks, log, cfg, store)
    won = [float(t.total_reward >= 1.0) for t in trajs]
    seen = [w for w, t in zip(won, trajs) if (t.task.family, t.task.goal) in trained_goals]
    unseen = [w for w, t in zip(won, trajs) if (t.task.family, t.task.goal) not in trained_goals]
    turns = sum(len(t.steps) for t in trajs)
    split = tasks[0].split if tasks else "heldout"
    result = EvalResult(split, len(trajs), _rate(won), _rate(seen), _rate(unseen),
                        sum(t.parse_fails for t in trajs) / turns if turns else 0.0, log.cumulative_cost)
    log.log_eval(split=result.split, n=result.n, success_rate=result.success_rate,
                 unseen_seed_rate=result.unseen_seed_rate, unseen_goal_rate=result.unseen_goal_rate,
                 n_unseen_seed=len(seen), n_unseen_goal=len(unseen),
                 parse_fail_rate=result.parse_fail_rate, cost=result.cost, final=final)
    return result


def make_eval_fn(policy, env: Env, log: Trace, cfg: LoopConfig, store=None):
    """The gate's measuring stick: tasks -> (success_rate, parse_fail_rate). Gate rollouts count as cost too."""
    def eval_fn(tasks: list[Task]) -> tuple[float, float]:
        trajs = run_tasks(policy, env, tasks, log, cfg, store)
        turns = sum(len(t.steps) for t in trajs)
        return (_rate([float(t.total_reward >= 1.0) for t in trajs]),
                sum(t.parse_fails for t in trajs) / turns if turns else 0.0)
    return eval_fn


def load_run(run_dir: str):
    """Rebuild a finished run's config and policy, with its committed LoRA weights if it trained any."""
    from models import make_policy
    run = Trace(run_dir).records("run")[0]
    cfg = load_config(None, **{k: v for k, v in run["config"].items() if k != "policy"}, policy=run["config"]["policy"])
    policy = make_policy(cfg.policy)
    weights = os.path.join(run_dir, "weights.safetensors")
    if os.path.exists(weights):
        from safetensors.torch import load_file
        policy.enable_lora(cfg.lora_r, cfg.lora_alpha)
        policy.restore(load_file(weights))
    return cfg, policy


def export_checkpoint(policy, out_dir: str) -> str:
    """Merge LoRA into a copy of the model and save it as a plain HF checkpoint. The processor saved is the
    model's original one, not our single-tile env setting, so benchmarks see the standard image pipeline —
    exactly what the untrained base model is scored with."""
    import copy

    from transformers import AutoProcessor

    from models.lora import merge_lora
    model = merge_lora(copy.deepcopy(policy.model))
    model.save_pretrained(out_dir)
    AutoProcessor.from_pretrained(policy.cfg.model_id).save_pretrained(out_dir)
    return out_dir


def lmms_eval_command(checkpoint: str, tasks: str, out_dir: str, model: str = "huggingface", batch_size: int = 1,
                      limit: int | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "lmms_eval", "--model", model, "--model_args", f"pretrained={checkpoint}",
           "--tasks", tasks, "--batch_size", str(batch_size), "--output_path", out_dir]
    return cmd + (["--limit", str(limit)] if limit else [])   # --limit is for debugging; reported numbers use all items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="a finished run directory; omit to evaluate the untrained base model")
    ap.add_argument("--split", default="heldout", choices=["heldout", "gate"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--benchmarks", help="comma-separated lmms-eval tasks, e.g. mmstar")
    ap.add_argument("--lmms_model", default="huggingface", help="lmms-eval model class that loads the checkpoint")
    ap.add_argument("--limit", type=int, help="lmms-eval items per task (debug only)")
    args = ap.parse_args()

    if args.run:
        cfg, policy = load_run(args.run)
    else:
        from models import make_policy
        cfg = LoopConfig()
        policy = make_policy(cfg.policy)
    out = os.path.join(args.run or "runs/base", "eval")

    if args.benchmarks:            # a trained run is exported; the base model is scored straight from the Hub
        ckpt = export_checkpoint(policy, os.path.join(out, "checkpoint")) if args.run else cfg.policy.model_id
        subprocess.run(lmms_eval_command(ckpt, args.benchmarks, os.path.join(out, "lmms_eval"), args.lmms_model,
                                         limit=args.limit), check=True)
        return

    from envs import make_env
    env = make_env(cfg)
    r = evaluate(policy, env, env.tasks(args.split, args.n, np.random.default_rng(cfg.seed)), Trace(out), cfg,
                 final=True)
    print(f"{'split':<10}{'n':>5}{'success':>10}{'unseen_seed':>13}{'unseen_goal':>13}{'parse_fail':>12}")
    print(f"{r.split:<10}{r.n:>5}{r.success_rate:>10.3f}{r.unseen_seed_rate:>13.3f}{r.unseen_goal_rate:>13.3f}"
          f"{r.parse_fail_rate:>12.3f}")


if __name__ == "__main__":
    main()
