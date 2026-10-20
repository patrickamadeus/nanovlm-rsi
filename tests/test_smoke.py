import time
from itertools import cycle

from config import LoopConfig, PolicyConfig
from evaluate import EvalResult
from infra.trace import Trace
from rsi import rsi
from tests.conftest import ScriptedPolicy


def small(tmp_path, **kw):
    base = dict(iters=2, tasks_per_iter=2, group_size=2, eval_every=1, eval_n=4, gate_n=4, heldout_n=8,
                run_dir=str(tmp_path), env="static", update_rule="memory")
    return LoopConfig(**{**base, **kw})


def test_smoke_closes_the_loop_with_a_scripted_policy(tmp_path):
    result = rsi(small(tmp_path), policy=ScriptedPolicy(cycle(["ACT: say_3", "ACT: pick"])))
    assert isinstance(result, EvalResult)
    kinds = {r["kind"] for r in Trace(str(tmp_path)).records()}
    assert kinds == {"run", "rollout", "update", "eval"}


def test_memory_carrier_commits_a_skill_when_rollouts_succeed(tmp_path):
    # say_3 is right for every count_objects task whose answer is 3, so some groups succeed
    rsi(small(tmp_path, iters=4, tasks_per_iter=6, gate_enabled=False),
        policy=ScriptedPolicy(cycle(["ACT: say_3"])))
    updates = Trace(str(tmp_path)).records("update")
    assert any(u["committed"] for u in updates)
    assert all(u["gate_enabled"] is False for u in updates if u.get("carrier"))


def test_smoke_runs_under_60s_on_cpu_with_the_real_tiny_policy(tmp_path):
    t0 = time.monotonic()
    cfg = small(tmp_path, iters=1, tasks_per_iter=1, eval_n=2, gate_n=2,
                policy=PolicyConfig(tiny=True, device="cpu", dtype="float32", max_new_tokens=8))
    assert isinstance(rsi(cfg), EvalResult)
    assert time.monotonic() - t0 < 60


def test_loop_survives_iterations_with_no_usable_signal(tmp_path):
    result = rsi(small(tmp_path), policy=ScriptedPolicy(cycle(["garbage"])))
    assert isinstance(result, EvalResult) and result.parse_fail_rate == 1.0
    assert not any(r["committed"] for r in Trace(str(tmp_path)).records("update"))


def test_run_record_captures_provenance(tmp_path):
    rsi(small(tmp_path, iters=1), policy=ScriptedPolicy(cycle(["ACT: pick"])))
    rec = Trace(str(tmp_path)).records("run")[0]
    assert rec["carrier"] == "memory" and rec["proposer"] == "fixed"
    assert rec["cfg_hash"] and "git_sha" in rec


def test_composite_verifier_logs_judge_agreement_in_the_loop(tmp_path):
    cfg = small(tmp_path, iters=1, tasks_per_iter=1, verifier="composite", judge_sample_rate=1.0,
                policy=PolicyConfig(tiny=True, device="cpu", dtype="float32"))
    rsi(cfg, policy=ScriptedPolicy(cycle(["ACT: say_3"])))
    train = [r for r in Trace(str(tmp_path)).records("rollout") if r.get("iter") is not None]
    assert train and all(r["verdict_source"] == "outcome" and r["judge_verdict"] in (0.0, 1.0) for r in train)
    assert Trace(str(tmp_path)).cumulative_cost.judge_tokens > 0


def test_self_proposer_runs_in_the_loop_and_logs_its_stats(tmp_path):
    rsi(small(tmp_path, proposer="self", gate_enabled=False, tasks_per_iter=4),
        policy=ScriptedPolicy(cycle(["ACT: say_3"])))
    committed = [u for u in Trace(str(tmp_path)).records("update") if u.get("carrier")]
    assert committed and all("frontier_frac" in u["proposer_stats"] for u in committed)


def test_weights_carrier_closes_the_loop_with_the_real_tiny_policy(tmp_path):
    cfg = small(tmp_path, iters=2, tasks_per_iter=2, group_size=4, update_rule="weights", gate_enabled=False,
                policy=PolicyConfig(tiny=True, device="cpu", dtype="float32", max_new_tokens=4,
                                    response_prefix="ACT: say_"))
    assert isinstance(rsi(cfg), EvalResult)
    updates = Trace(str(tmp_path)).records("update")
    assert any(u.get("carrier") == "A5" and "loss" in u for u in updates) or \
        all(u["reason"] == "no usable signal" for u in updates)
