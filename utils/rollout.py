"""The harness: one episode. Render the scaffold, let the policy act, step the env until done or out of budget."""
import time
from dataclasses import dataclass

import numpy as np
import torch

from configs.config import LoopConfig
from envs.base import Env, Task
from utils.prompt import load_scaffold, parse_turn, render

SYSTEM = "You are an agent acting in a visual environment. Look at the image and reach the goal."


@dataclass(frozen=True)
class Trajectory:
    task: Task
    steps: list[dict]          # one dict per turn; actions [] means a parse failure, a wasted turn
    total_reward: float        # unmodified environment reward; dense shaping stays per step, for the judge
    env_steps: int
    parse_fails: int
    tokens_out: int
    tokens_in: int = 0         # prompt tokens: memory text makes prompts longer, so this is real cost
    model_seconds: float = 0.0 # wall time inside policy calls, CUDA-synchronized
    wall_ms: float = 0.0
    final_rgb: np.ndarray | None = None   # what a VLM judge looks at
    context_ids: tuple[str, ...] = ()     # memory items that were in the prompt, for use attribution
    context: "Context | None" = None      # what the methods added to the prompt, so a replay can rebuild it


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


def act_all(policy, prompts: list[str], images: list[list[np.ndarray]], greedy: bool = False):
    """One batched call when the policy has one (models/vlm.py), else one call per prompt (test doubles)."""
    if hasattr(policy, "act_batch"):
        return policy.act_batch(prompts, images, greedy=greedy)
    return [policy.act(p, im) for p, im in zip(prompts, images)]


def moves(traj) -> list[str]:
    """Every action an episode executed, in order (a turn may execute several)."""
    out = []
    for st in traj.steps:
        out += st.get("actions") or ([st["action"]] if st.get("action") else [])
    return out


def feedback(last: list[str] | None) -> str:
    """VAGEN's turn header: the first frame, or what the previous answer actually did."""
    if last is None:
        return "[Initial Observation]:"
    return f"After your answer, the extracted valid action is {', '.join(last) or 'none'}.\nAfter that, the observation is:"


def prompt_for(template: str, cfg: LoopConfig, env: Env, ctx: Context, obs, last) -> str:
    return render(template, system=SYSTEM, actions=env.actions, memory=ctx.text, task=obs.text or "",
                  state=obs.text or "", feedback=feedback(last), max_actions=cfg.max_actions_per_turn,
                  sep=cfg.action_sep)


def execute(env: Env, actions: list[str], budget_left: int):
    """Apply a turn's actions in order until the episode ends or the step budget runs out."""
    total, shaping, done, done_actions, events, path, obs = 0.0, 0.0, False, [], [], [], None
    for a in actions[:max(budget_left, 0)]:
        obs, reward, done, info = env.step(a)
        total += reward
        shaping += info.get("shaping", 0.0)
        done_actions.append(a)
        events += list(info.get("events", []))
        path.append(info.get("pos"))
        if done:
            break
    return obs, total, done, done_actions, events, path, shaping


def rollout_group(policy, env: Env, tasks: list[Task], cfg: LoopConfig,
                  contexts: list[Context] | None = None, greedy: bool = False) -> list[Trajectory]:
    """Run one episode per task in lockstep: every turn, one batched policy call for all unfinished rows.
    Each row has its own fork of the env and leaves the batch when its episode ends. A turn may carry
    up to cfg.max_actions_per_turn actions; an episode has at most cfg.max_turns turns (0: one per step)
    and task.budget env steps."""
    t_start = time.perf_counter()
    template = load_scaffold(cfg.policy.prompt_path)
    version = getattr(policy, "version", 0)
    contexts = contexts or [Context()] * len(tasks)
    rows = []
    for task, ctx in zip(tasks, contexts):
        e = env.fork()
        rows.append({"env": e, "task": task, "ctx": ctx, "obs": e.reset(task), "steps": [], "total": 0.0,
                     "env_steps": 0, "fails": 0, "tin": 0, "tout": 0, "secs": 0.0, "done": False, "last": None})
    turns = lambda r: cfg.max_turns or r["task"].budget                          # noqa: E731
    while True:
        live = [r for r in rows if not r["done"] and len(r["steps"]) < turns(r) and r["env_steps"] < r["task"].budget]
        if not live:
            break
        prompts = [prompt_for(template, cfg, r["env"], r["ctx"], r["obs"], r["last"]) for r in live]
        images = [[*r["ctx"].images, r["obs"].rgb] for r in live]
        gens, secs = timed(act_all, policy, prompts, images, greedy)
        for r, prompt, ims, gen in zip(live, prompts, images, gens):
            r["secs"] += secs / len(live)
            r["tin"] += gen.prompt_tokens
            r["tout"] += len(gen.token_ids)
            actions, ok = parse_turn(gen.text, r["env"].actions, cfg.answer_format, cfg.max_actions_per_turn,
                                     cfg.action_sep)
            turn = {"prompt": prompt, "images": ims, "token_ids": gen.token_ids, "logprobs": gen.logprobs,
                    "text": gen.text, "obs_text": r["obs"].text, "policy_version": version, "format_ok": ok}
            if not actions:        # a failed parse is a wasted turn, never an env call
                r["fails"] += 1
                r["last"] = []
                r["steps"].append({"actions": [], "reward": 0.0, "shaping": 0.0, "events": [],
                                   "pos": None, "path": [], **turn})
                continue
            obs, reward, done, done_actions, events, path, shaping = execute(r["env"], actions,
                                                                             r["task"].budget - r["env_steps"])
            r["obs"], r["last"] = obs, done_actions
            r["env_steps"] += len(done_actions)
            r["total"] += reward
            r["done"] = done
            r["steps"].append({"actions": done_actions, "reward": reward, "shaping": shaping, "events": events,
                               "pos": path[-1] if path else None, "path": path, **turn})
    wall = (time.perf_counter() - t_start) * 1000 / max(len(rows), 1)
    return [Trajectory(r["task"], r["steps"], r["total"], r["env_steps"], r["fails"], r["tout"], r["tin"],
                       r["secs"], wall, r["obs"].rgb, tuple(r["ctx"].ids), r["ctx"]) for r in rows]


def replay(env: Env, task: Task, steps: list[dict], cfg: LoopConfig, ctx: "Context | None" = None) -> Trajectory:
    """Re-run already generated turns on (possibly another) task, without the policy: each turn keeps its
    completion tokens and text, and gets the prompt and frame that task would have shown. Hindsight uses
    this to turn a failed attempt into a success for the goal it did reach, with consistent images."""
    ctx = ctx or Context()
    template = load_scaffold(cfg.policy.prompt_path)
    e = env.fork()
    obs, out, total, last, n = e.reset(task), [], 0.0, None, 0
    for st in steps:
        prompt = prompt_for(template, cfg, e, ctx, obs, last)
        images, seen = [*ctx.images, obs.rgb], obs.text
        acts = st.get("actions") or ([st["action"]] if st.get("action") else [])
        new_obs, reward, done, done_actions, events, path, shaping = execute(e, acts, task.budget - n)
        obs = new_obs if new_obs is not None else obs
        total, last, n = total + reward, done_actions, n + len(done_actions)
        out.append({**st, "prompt": prompt, "images": images, "obs_text": seen, "reward": reward, "shaping": shaping, "actions": done_actions,
                    "pos": path[-1] if path else None, "path": path, "events": events, "policy_version": -1})
        if done:
            break
    return Trajectory(task, out, total, n, 0, sum(len(st["token_ids"]) for st in out), 0, 0.0, 0.0, obs.rgb,
                      tuple(ctx.ids), ctx)


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
