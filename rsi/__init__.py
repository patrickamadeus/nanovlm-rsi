"""The self-improvement mechanisms: what changes (grpo | memory), what to practice (curriculum), what is kept (gate)."""
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
