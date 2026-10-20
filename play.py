"""Watch a policy act on one task, learning disabled. The debug path: prompt, generation, reward per step."""
import argparse

import numpy as np

from config import LoopConfig, PolicyConfig
from policy.prompt import load_scaffold, parse_action, render
from rollout import SYSTEM
from rsi import make_env


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train", choices=["train", "gate", "heldout"])
    ap.add_argument("--index", type=int, default=0, help="which task of the split to play")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    from policy.vlm import Policy
    cfg = LoopConfig()
    if args.tiny:
        cfg.policy = PolicyConfig(tiny=True, device="cpu", dtype="float32", max_new_tokens=8)
    policy, env = Policy(cfg.policy), make_env(cfg)
    task = env.tasks(args.split, args.index + 1, np.random.default_rng(cfg.seed))[args.index]
    template = load_scaffold(cfg.policy.scaffold_path)
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
