"""Environments: reset(task) -> Obs, step(action) -> (Obs, reward, done, info), tasks(split, n, rng)."""


def make_env(cfg):
    if cfg.env == "static":
        from envs.static import StaticEnv
        return StaticEnv(env_seed=cfg.env_seed, heldout_n=cfg.heldout_n, families=cfg.static_families)
    if cfg.env == "craftax":
        from envs.craftax import CraftaxEnv          # imported lazily: JAX loads only when Craftax is used
        return CraftaxEnv(env_seed=cfg.env_seed, heldout_n=cfg.heldout_n, budget=cfg.craftax_budget)
    if cfg.env == "cauldron":
        from envs.cauldron import CauldronEnv, load
        items, image_of = load(cfg.cauldron_repo, cfg.cauldron_subset, cfg.cauldron_revision, cfg.cauldron_sources)
        return CauldronEnv(items, image_of, env_seed=cfg.env_seed, heldout_n=cfg.heldout_n)
    raise ValueError(f"unknown env {cfg.env!r}")
