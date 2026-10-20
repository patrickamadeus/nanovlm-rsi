"""One episode: render the scaffold, let the policy think and act, step the env until done or out of budget."""
from config import LoopConfig
from envs.base import Env, Task
from policy.prompt import load_scaffold, parse_action, render
from verify import Trajectory

SYSTEM = "You are an agent acting in a visual environment. Look at the image and reach the goal."
PARSE_FAIL_SHAPING = -0.01     # recorded per step, kept out of total_reward so verifiers see raw env signal


def rollout(policy, env: Env, task: Task, cfg: LoopConfig, *, memory_text: str = "") -> Trajectory:
    template = load_scaffold(cfg.policy.scaffold_path)
    obs = env.reset(task)
    steps, total, env_steps, parse_fails, tokens_out = [], 0.0, 0, 0, 0
    for _ in range(task.budget):
        prompt = render(template, system=SYSTEM, actions=env.actions, memory=memory_text, task=obs.text or "")
        gen = policy.act(prompt, [obs.rgb])
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
    return Trajectory(task, steps, total, env_steps, parse_fails, tokens_out)


def recall(store, env: Env, task: Task, k: int) -> tuple[str, list[str]]:
    """Retrieve skills keyed exactly the way the memory carrier wrote them: by the first observation's text."""
    if store is None or len(store) == 0 or k <= 0:
        return "", []
    skills = store.retrieve(env.reset(task).text or "", k)
    return store.as_text(skills), [s.id for s in skills]


def rollout_record(traj: Trajectory, retrieved_ids: list[str], verdict_source: str = "outcome") -> dict:
    t = traj.task
    return dict(task_id=f"{t.family}:{t.seed}", family=t.family, seed=t.seed, split=t.split, goal=list(t.goal),
                reward=traj.total_reward, verdict_source=verdict_source, parse_fail=traj.parse_fails > 0,
                parse_fails=traj.parse_fails, retrieved_skill_ids=retrieved_ids, env_steps=traj.env_steps,
                tokens_out=traj.tokens_out)
