import os

from infra.memory import SkillStore
from update.base import Update, Updater
from update.memory import MemoryUpdater
from update.scaffold import ScaffoldUpdater
from update.verifier import VerifierUpdater
from update.weights import WeightsUpdater


def make_updater(name: str, policy, cfg) -> Updater:
    if name == "memory":
        store = SkillStore(os.path.join(cfg.run_dir, "skills.jsonl"), embed_fn=lambda text: policy.embed(text, []),
                           max_size=cfg.mem_max_size, dedupe_sim=cfg.mem_dedupe_sim)
        return MemoryUpdater(store, cfg)
    carriers = {"weights": WeightsUpdater, "scaffold": ScaffoldUpdater, "verifier": VerifierUpdater}
    if name not in carriers:
        raise ValueError(f"unknown update rule {name!r}")
    return carriers[name](policy, cfg)
