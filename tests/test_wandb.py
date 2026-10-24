"""W&B mirrors the trace: same numbers, charted. A fake wandb module stands in for the real one."""
import sys
import types
from itertools import cycle

from configs.config import LoopConfig, PolicyConfig
from tests.conftest import ScriptedPolicy
from train import train
from utils.log import Trace


def fake_wandb(monkeypatch):
    calls = {"init": [], "log": [], "finish": 0}
    mod = types.SimpleNamespace(
        init=lambda **kw: calls["init"].append(kw) or types.SimpleNamespace(id="x"),
        log=lambda data, step=None: calls["log"].append((step, data)),
        finish=lambda: calls.__setitem__("finish", calls["finish"] + 1),
        Image=lambda arr, caption=None: ("image", getattr(arr, "shape", None), caption),
        Html=lambda s: ("html", s))
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return calls


def test_a_run_is_charted_in_detail_and_the_trace_keeps_the_same_numbers(tmp_path, monkeypatch):
    calls = fake_wandb(monkeypatch)
    cfg = LoopConfig(env="frozenlake", iters=2, tasks_per_iter=2, group_size=2, eval_every=1, eval_n=2, gate_n=2,
                     heldout_n=8, run_dir=str(tmp_path), methods=("grpo",), wandb_project="picorsi-test",
                     policy=PolicyConfig(tiny=True, device="cpu", dtype="float32", max_new_tokens=4))
    train(cfg)
    assert calls["init"][0]["project"] == "picorsi-test" and calls["init"][0]["config"]["iters"] == 2
    keys = set().union(*(d.keys() for _, d in calls["log"]))
    for k in ("train/success", "train/reward", "train/format_ok", "train/parse_fail", "train/tokens_per_turn",
              "train/groups_with_signal", "cost/policy_tokens", "time/iter_seconds",
              "eval/frozenlake/success", "eval/frozenlake/parse_fail"):
        assert k in keys, k
    assert any(k.startswith("sample/frozenlake/") for k in keys)
    proposed = [u for u in Trace(str(tmp_path)).records("update") if u.get("mode")]
    assert all(k in keys for k in ("update/grpo/loss", "update/grpo/kl")) if proposed else True
    steps = [s for s, _ in calls["log"]]
    assert steps == sorted(steps)                                     # W&B needs a non-decreasing step
    assert len(Trace(str(tmp_path)).records("metrics")) == 2            # one record per iteration, in the trace too
    assert calls["finish"] == 1


def test_without_a_project_nothing_touches_wandb(tmp_path, monkeypatch):
    calls = fake_wandb(monkeypatch)
    cfg = LoopConfig(iters=1, tasks_per_iter=1, group_size=2, eval_n=2, gate_n=2, heldout_n=8, run_dir=str(tmp_path),
                     methods=("memory",))
    train(cfg, policy=ScriptedPolicy(cycle(["ACT: 3"])))
    assert calls["init"] == [] and calls["log"] == []
    assert Trace(str(tmp_path)).records("metrics")                        # the trace still has them


def test_weight_updates_report_kl_and_clip_fraction(tmp_path):
    import numpy as np
    from envs.base import Task
    from models.vlm import Policy
    from rsi.grpo import GRPO
    from tests.test_grpo import _batch
    policy = Policy(PolicyConfig(tiny=True, device="cpu", dtype="float32", max_new_tokens=6))
    g = GRPO(policy, LoopConfig(run_dir=str(tmp_path), grpo_lr=5e-2, grpo_beta=0.0, minibatches=2))
    u = g.propose(_batch(policy))
    g.commit(u)
    assert {"kl", "clip_frac"} <= set(u.payload) and 0.0 <= u.payload["clip_frac"] <= 1.0
