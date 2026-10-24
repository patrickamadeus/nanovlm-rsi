"""The harness: one episode. Render the scaffold, let the policy act, step the env until done or out of budget."""
import time
from dataclasses import dataclass

import numpy as np
import torch

from configs.config import LoopConfig
from envs.base import Env, Task
from utils.prompt import load_scaffold, parse_action, render

SYSTEM = "You are an agent acting in a visual environment. Look at the image and reach the goal."
PARSE_FAIL_SHAPING = -0.01     # recorded per step, kept out of total_reward so judges see raw env signal


@dataclass(frozen=True)
class Trajectory:
    task: Task
    steps: list[dict]          # one dict per budget step; action None means a parse failure no-op
    total_reward: float        # unmodified environment reward — shaping is kept out of it
    env_steps: int
    parse_fails: int
    tokens_out: int
    tokens_in: int = 0         # prompt tokens: memory text makes prompts longer, so this is real cost
    model_seconds: float = 0.0 # wall time inside policy calls, CUDA-synchronized
    wall_ms: float = 0.0
    final_rgb: np.ndarray | None = None   # what a VLM judge looks at
    context_ids: tuple[str, ...] = ()     # memory items that were in the prompt, for use attribution


def timed(fn, *args):
    """Run a model call and return (result, seconds), synchronizing CUDA so GPU time is actually counted."""
    t0 = time.perf_counter()
    out = fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


@dataclass(frozen=True)
class Context:
    """What the methods add to a task's prompt: text for the {memory} slot, images it refers to with
    <image> (shown before the observation), and the ids of the memory items used, for attribution."""
    text: str = ""
    images: tuple = ()
    ids: tuple[str, ...] = ()


def act_all(policy, prompts: list[str], images: list[list[np.ndarray]]):
    """One batched call when the policy has one (models/vlm.py), else one call per prompt (test doubles)."""
    if hasattr(policy, "act_batch"):
        return policy.act_batch(prompts, images)
    return [policy.act(p, im) for p, im in zip(prompts, images)]


def rollout_group(policy, env: Env, tasks: list[Task], cfg: LoopConfig,
                  contexts: list[Context] | None = None) -> list[Trajectory]:
    """Run one episode per task in lockstep: every turn, one batched policy call for all unfinished rows.
    Each row has its own fork of the env. A row leaves the batch when its episode ends."""
    t_start = time.perf_counter()
    template = load_scaffold(cfg.policy.scaffold_path)
    version = getattr(policy, "version", 0)
    contexts = contexts or [Context()] * len(tasks)
    rows = []
    for task, ctx in zip(tasks, contexts):
        e = env.fork()
        rows.append({"env": e, "task": task, "ctx": ctx, "obs": e.reset(task), "steps": [], "total": 0.0,
                     "env_steps": 0, "fails": 0, "tin": 0, "tout": 0, "secs": 0.0, "done": False})
    for _ in range(max((t.budget for t in tasks), default=0)):
        live = [r for r in rows if not r["done"] and len(r["steps"]) < r["task"].budget]
        if not live:
            break
        prompts = [render(template, system=SYSTEM, actions=r["env"].actions, memory=r["ctx"].text,
                          task=r["obs"].text or "") for r in live]
        images = [[*r["ctx"].images, r["obs"].rgb] for r in live]
        gens, secs = timed(act_all, policy, prompts, images)
        for r, prompt, ims, gen in zip(live, prompts, images, gens):
            r["secs"] += secs / len(live)
            r["tin"] += gen.prompt_tokens
            r["tout"] += len(gen.token_ids)
            action = parse_action(gen.text, r["env"].actions)
            replay = {"prompt": prompt, "images": ims, "token_ids": gen.token_ids, "logprobs": gen.logprobs,
                      "text": gen.text, "obs_text": r["obs"].text, "policy_version": version}
            if action is None:     # a failed parse is a wasted turn, never an env call
                r["fails"] += 1
                r["steps"].append({"action": None, "reward": 0.0, "shaping": PARSE_FAIL_SHAPING, "events": [],
                                   **replay})
                continue
            r["obs"], reward, done, info = r["env"].step(action)
            r["env_steps"] += 1
            r["total"] += reward
            r["done"] = done
            r["steps"].append({"action": action, "reward": reward, "shaping": 0.0,
                               "events": list(info.get("events", [])), **replay})
    wall = (time.perf_counter() - t_start) * 1000 / max(len(rows), 1)
    return [Trajectory(r["task"], r["steps"], r["total"], r["env_steps"], r["fails"], r["tout"], r["tin"],
                       r["secs"], wall, r["obs"].rgb, tuple(r["ctx"].ids)) for r in rows]


def rollout(policy, env: Env, task: Task, cfg: LoopConfig, *, memory_text: str = "") -> Trajectory:
    return rollout_group(policy, env, [task], cfg, [Context(memory_text)])[0]


def rollout_record(traj: Trajectory, verdict_source: str = "outcome", extra_seconds: float = 0.0,
                   key: str | None = None) -> dict:
    t, retrieved_ids = traj.task, list(traj.context_ids)
    return dict(task_id=f"{t.family}:{t.seed}", family=t.family, seed=t.seed, split=t.split, goal=list(t.goal),
                reward=traj.total_reward, verdict_source=verdict_source, parse_fail=traj.parse_fails > 0,
                parse_fails=traj.parse_fails, retrieved_skill_ids=retrieved_ids, env_steps=traj.env_steps,
                tokens_in=traj.tokens_in, tokens_out=traj.tokens_out,
                gpu_seconds=traj.model_seconds + extra_seconds, wall_ms=traj.wall_ms,
                key=key or t.family, level=list(t.level))
