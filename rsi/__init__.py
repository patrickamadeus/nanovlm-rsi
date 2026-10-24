"""The self-improvement mechanisms, one file per method or planner, each named after it.

    methods   what changes and how: grpo (weights), memory (skill library), ...   -> rsi/base.py Method
    planners  what to practise and which methods update: uniform, frontier, ...   -> rsi/base.py Planner
    judges    how an attempt is scored (judges.py) · gate: what is kept (gate.py)

make_methods / make_planner map a config name to a file, so every choice is one line of YAML.
To add one, copy rsi/template.py.
"""
import os

import numpy as np

WEIGHT_METHODS = {"grpo", "ppo", "rest"}             # they share one set of LoRA adapters; the rest change memory or signals


def make_methods(cfg, policy, env=None) -> list:
    shared = {}
    return [make_method(name, cfg, policy, env, shared) for name in cfg.methods]


def make_method(name: str, cfg, policy, env=None, shared: dict | None = None):
    shared = {} if shared is None else shared

    def weights():                     # one set of LoRA adapters, however many weight methods use it
        if "weights" not in shared:
            from rsi.weights import Weights
            shared["weights"] = Weights(policy, cfg)
        return shared["weights"]

    if name == "grpo":
        from rsi.grpo import GRPO
        return GRPO(policy, cfg, weights())
    if name == "ppo":
        from rsi.ppo import PPO
        return PPO(policy, cfg, weights())
    if name == "rest":
        from rsi.rest import ReST
        return ReST(policy, cfg, weights())
    if name == "memory":
        from rsi.memory import Memory, SkillStore
        return Memory(skill_store(cfg, policy, "skills.jsonl"), cfg, env, policy)
    if name == "hindsight":
        from rsi.hindsight import Hindsight
        return Hindsight(cfg)
    if name == "expel":
        from rsi.expel import ExpeL
        return ExpeL(cfg, policy, cfg.run_dir)
    if name == "reflexion":
        from rsi.reflexion import Reflexion
        return Reflexion(cfg, policy, env, lambda text: policy.embed(text, []))
    if name == "viskill":
        from rsi.viskill import ViSkill
        return ViSkill(skill_store(cfg, policy, "cards.jsonl"), cfg, env, policy)
    raise ValueError(f"unknown method {name!r}")


def skill_store(cfg, policy, filename: str):
    """One file per method: two methods truncating one log on revert would undo each other's commits."""
    from rsi.memory import SkillStore
    return SkillStore(os.path.join(cfg.run_dir, filename), embed_fn=lambda text: policy.embed(text, []),
                      max_size=cfg.mem_max_size, dedupe_sim=cfg.mem_dedupe_sim)


def make_planner(env, cfg, log=None, policy=None):
    rng = np.random.default_rng(cfg.seed)
    if cfg.planner == "tscl":
        from rsi.tscl import TSCL
        return TSCL(env, cfg, rng, log)
    if cfg.planner == "accel":
        from rsi.accel import ACCEL
        return ACCEL(env, cfg, rng, log)
    if cfg.planner == "voyager":
        from rsi.voyager import Voyager
        return Voyager(env, cfg, rng, policy, log)
    if cfg.planner == "uniform":
        from rsi.curriculum import Uniform
        return Uniform(env, cfg, rng, log)
    if cfg.planner == "frontier":
        from rsi.curriculum import Frontier
        return Frontier(env, cfg, rng, log)
    raise ValueError(f"unknown planner {cfg.planner!r}")


def make_update_planner(cfg, names: list[str]):
    """Which updating methods run each iteration. "all" runs every one, in config order."""
    if cfg.update_planner == "all":
        from rsi.base import Planner
        return Planner()
    if cfg.update_planner == "exp3":
        from rsi.exp3 import Exp3
        return Exp3(cfg, names, np.random.default_rng(cfg.seed + 1))
    raise ValueError(f"unknown update planner {cfg.update_planner!r}")


def make_judge(cfg, env=None, families: set | None = None):
    """families: the task families whose reward is ground truth (every stage's, in a lifelong run)."""
    from rsi.judges import CompositeJudge, OutcomeJudge, VLMJudge, YesNo
    if cfg.judge == "outcome":
        return OutcomeJudge()
    if cfg.judge in ("vlm", "composite"):
        from configs.config import PolicyConfig
        p = cfg.policy
        vlm = VLMJudge(YesNo(PolicyConfig(model_id=cfg.judge_model_id, tiny=p.tiny, device=p.device, dtype=p.dtype)))
        if cfg.judge == "vlm":
            return vlm
        native = families if families is not None else getattr(env, "families", set())
        return CompositeJudge(OutcomeJudge(), vlm, native, cfg.judge_sample_rate, cfg.seed)
    raise ValueError(f"unknown judge {cfg.judge!r}")
