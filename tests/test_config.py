from config import LoopConfig, PolicyConfig


def test_config_hash_is_stable_and_sensitive():
    a, b = LoopConfig(), LoopConfig()
    assert a.hash() == b.hash()
    assert LoopConfig(iters=5).hash() != LoopConfig(iters=6).hash()


def test_defaults_match_spec():
    cfg = LoopConfig()
    assert cfg.update_rule == "memory"
    assert cfg.proposer == "fixed"
    assert cfg.group_size == 8
    assert cfg.gate_enabled is True
    assert cfg.heldout_n == 200
    assert isinstance(cfg.policy, PolicyConfig)
