"""One episode: render the scaffold, let the policy think and act, step the env until done or out of budget."""
import time

import torch

from config import LoopConfig
from envs.base import Env, Task
from policy.prompt import load_scaffold, parse_action, render
from verify import Trajectory

SYSTEM = "You are an agent acting in a visual environment. Look at the image and reach the goal."
PARSE_FAIL_SHAPING = -0.01     # recorded per step, kept out of total_reward so verifiers see raw env signal


def timed(fn, *args):
    """Run a model call and return (result, seconds), synchronizing CUDA so GPU time is actually counted."""
    t0 = time.perf_counter()
    out = fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def rollout(policy, env: Env, task: Task, cfg: LoopConfig, *, memory_text: str = "") -> Trajectory:
    t_start = time.perf_counter()
    template = load_scaffold(cfg.policy.scaffold_path)
    obs = env.reset(task)
    steps, total, env_steps, parse_fails, tokens_in, tokens_out, model_s = [], 0.0, 0, 0, 0, 0, 0.0
    for _ in range(task.budget):
        prompt = render(template, system=SYSTEM, actions=env.actions, memory=memory_text, task=obs.text or "")
        gen, secs = timed(policy.act, prompt, [obs.rgb])
        model_s += secs
        tokens_in += gen.prompt_tokens
        tokens_out += len(gen.token_ids)
        action = parse_action(gen.text, env.actions)
        if action is None:     # a failed parse is a wasted turn, never an env call
            parse_fails += 1
            steps.append({"action": None, "reward": 0.0, "shaping": PARSE_FAIL_SHAPING, "text": gen.text,
                          "obs_text": obs.text})
            continue
        seen = obs.text
        obs, reward, done, _ = env.step(action)
        env_steps += 1
        total += reward
        steps.append({"action": action, "reward": reward, "shaping": 0.0, "text": gen.text, "obs_text": seen})
        if done:
            break
    return Trajectory(task, steps, total, env_steps, parse_fails, tokens_out, tokens_in, model_s,
                      (time.perf_counter() - t_start) * 1000)


def recall(store, env: Env, task: Task, k: int) -> tuple[str, list[str], float]:
    """Retrieve skills keyed exactly the way the memory carrier wrote them: by the first observation's text.
    Returns (memory_text, skill_ids, seconds) — the embedding forward pass is cost like any other."""
    if store is None or len(store) == 0 or k <= 0:
        return "", [], 0.0
    skills, secs = timed(store.retrieve, env.reset(task).text or "", k)
    return store.as_text(skills), [s.id for s in skills], secs


def rollout_record(traj: Trajectory, retrieved_ids: list[str], verdict_source: str = "outcome",
                   extra_seconds: float = 0.0) -> dict:
    t = traj.task
    return dict(task_id=f"{t.family}:{t.seed}", family=t.family, seed=t.seed, split=t.split, goal=list(t.goal),
                reward=traj.total_reward, verdict_source=verdict_source, parse_fail=traj.parse_fails > 0,
                parse_fails=traj.parse_fails, retrieved_skill_ids=retrieved_ids, env_steps=traj.env_steps,
                tokens_in=traj.tokens_in, tokens_out=traj.tokens_out,
                gpu_seconds=traj.model_seconds + extra_seconds, wall_ms=traj.wall_ms)
