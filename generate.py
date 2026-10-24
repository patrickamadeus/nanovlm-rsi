"""Watch the policy act on one task, learning off: prompt, reply, parsed action and reward, step by step.

    python generate.py --config tiny                         # CPU, random weights: shows the plumbing
    python generate.py --run runs/count_grpo --split heldout --index 3      # a trained run
"""
import argparse

import numpy as np

from configs.config import load_config
from envs import make_env
from utils.prompt import load_scaffold, parse_action, render
from utils.rollout import SYSTEM


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="a name in configs/yaml/ or a YAML path")
    ap.add_argument("--run", help="a finished run directory: use its config and trained weights")
    ap.add_argument("--split", default="train", choices=["train", "gate", "heldout"])
    ap.add_argument("--index", type=int, default=0, help="which task of the split to play")
    args = ap.parse_args()

    if args.run:
        from evaluate import load_run
        cfg, policy = load_run(args.run)
    else:
        from models import make_policy
        cfg = load_config(args.config)
        policy = make_policy(cfg.policy)
    env = make_env(cfg)
    task = env.tasks(args.split, args.index + 1, np.random.default_rng(cfg.seed))[args.index]
    template = load_scaffold(cfg.policy.prompt_path)
    obs = env.reset(task)
    print(f"task {task.family} seed={task.seed} goal={task.goal} budget={task.budget}\n")
    for t in range(task.budget):
        prompt = render(template, system=SYSTEM, actions=env.actions, memory="", task=obs.text or "")
        gen = policy.act(prompt, [obs.rgb])
        action = parse_action(gen.text, env.actions)
        print(f"--- step {t}\n{prompt}\n>>> {gen.text!r}\n>>> action={action}")
        if action is None:
            continue
        obs, reward, done, _ = env.step(action)
        print(f">>> reward={reward} done={done}")
        if done:
            break


if __name__ == "__main__":
    main()
