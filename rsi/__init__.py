"""The self-improvement mechanisms: what changes (grpo | memory), what to practice (curriculum),
how it's scored (judges), and what is kept (gate)."""
import os


def make_update(cfg, policy):
    if cfg.update == "grpo":
        from rsi.grpo import GRPO
        return GRPO(policy, cfg)
    if cfg.update == "memory":
        from rsi.memory import Memory, SkillStore
        store = SkillStore(os.path.join(cfg.run_dir, "skills.jsonl"), embed_fn=lambda text: policy.embed(text, []),
                           max_size=cfg.mem_max_size, dedupe_sim=cfg.mem_dedupe_sim)
        return Memory(store, cfg)
    raise ValueError(f"unknown update {cfg.update!r}")


def make_judge(cfg, env=None):
    from rsi.judges import CompositeJudge, OutcomeJudge, VLMJudge, YesNo
    if cfg.judge == "outcome":
        return OutcomeJudge()
    if cfg.judge in ("vlm", "composite"):
        from configs.config import PolicyConfig
        p = cfg.policy
        vlm = VLMJudge(YesNo(PolicyConfig(model_id=cfg.judge_model_id, tiny=p.tiny, device=p.device, dtype=p.dtype)))
        if cfg.judge == "vlm":
            return vlm
        return CompositeJudge(OutcomeJudge(), vlm, getattr(env, "families", set()), cfg.judge_sample_rate, cfg.seed)
    raise ValueError(f"unknown judge {cfg.judge!r}")
