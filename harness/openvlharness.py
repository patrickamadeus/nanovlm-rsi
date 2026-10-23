"""Rollouts through OpenVLHarness (uclanlp/OpenVLHarness, arXiv 2610.26860) instead of our pico loop.

The agent there is question -> model/tool turns -> final answer, with zoom, coding and other tools and a
multimodal memory. One agent.generate() call is one episode; its final answer is the single env action.
So it fits tasks answered once (count_objects), and prompt-level updates (memory): the model sits behind
an HTTP endpoint, and a 2B model's trajectories carry no token ids, so GRPO is not available here yet.

    pip install "openvlharness @ git+https://github.com/uclanlp/OpenVLHarness.git"
    serve the model (vLLM) and start the tool router: see docs/openvlharness.md
"""
import os
import re

from PIL import Image

from harness.rollout import Trajectory

UPSCALE = 4                    # env frames are 64px; the model and its zoom tool want something to look at
_agents: dict[str, object] = {}


def check_config(cfg) -> None:
    if cfg.update != "memory":
        raise ValueError("harness=openvlharness supports update=memory only: GRPO needs token ids, "
                         "which the served model's trajectories do not carry")
    if cfg.env != "static" or set(cfg.static_families) != {"count_objects"}:
        raise ValueError("harness=openvlharness runs single-answer tasks only (static count_objects)")


def get_agent(config_path: str):
    if config_path not in _agents:
        from openvlharness.bootstrap import build_agent
        _agents[config_path] = build_agent(config_path)
    return _agents[config_path]


def ovh_rollout(agent, env, task, cfg, *, memory_text: str = "", image_dir: str | None = None) -> Trajectory:
    obs = env.reset(task)
    image_dir = image_dir or os.path.join(cfg.run_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    path = os.path.join(image_dir, f"{task.seed}.png")
    h, w = obs.rgb.shape[:2]
    Image.fromarray(obs.rgb).resize((w * UPSCALE, h * UPSCALE), Image.NEAREST).save(path)
    question = "\n".join(p for p in (memory_text, obs.text, "Answer with a single digit.") if p)

    result = agent.generate([{"type": "image", "value": path}, {"type": "text", "value": question}],
                            task_id=str(task.seed))
    turns = result["extra_records"]["trajectory"]["chat_history"]
    usage = result["extra_records"].get("usage", {}).get("total", {})
    digits = re.findall(r"\d", result["prediction"] or "")
    action = digits[-1] if digits and digits[-1] in env.actions else None
    step = {"action": action, "reward": 0.0, "shaping": 0.0, "text": result["prediction"], "obs_text": obs.text,
            "tools": [c["name"] for t in turns for c in t["assistant"]["tool_calls"]], "turns": len(turns),
            "token_ids": [], "logprobs": []}
    env_steps, total, final_rgb = 0, 0.0, obs.rgb
    if action is not None:
        obs, total, _, _ = env.step(action)
        env_steps, step["reward"], final_rgb = 1, total, obs.rgb
    return Trajectory(task, [step], total, env_steps, int(action is None), usage.get("completion_tokens", 0),
                      usage.get("prompt_tokens", 0), final_rgb=final_rgb)
