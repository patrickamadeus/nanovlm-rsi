import time

import pytest
from itertools import cycle

from configs.config import LoopConfig, PolicyConfig
from evaluate import EvalResult
from utils.log import Trace
from train import train
from tests.conftest import ScriptedPolicy


def small(tmp_path, **kw):
    base = dict(iters=2, tasks_per_iter=2, group_size=2, eval_every=1, eval_n=4, gate_n=4, heldout_n=8,
                run_dir=str(tmp_path), env="static", methods=("memory",))
    return LoopConfig(**{**base, **kw})


def test_smoke_closes_the_loop_with_a_scripted_policy(tmp_path):
    result = train(small(tmp_path), policy=ScriptedPolicy(cycle(["ACT: 3", "ACT: pick"])))
    assert isinstance(result, EvalResult)
    kinds = {r["kind"] for r in Trace(str(tmp_path)).records()}
    assert kinds == {"run", "rollout", "update", "eval", "metrics"}


def test_memory_update_commits_a_skill_when_rollouts_succeed(tmp_path):
    # 3 is right for every count_objects task whose answer is 3, so some groups succeed
    train(small(tmp_path, iters=4, tasks_per_iter=6, gate_enabled=False),
        policy=ScriptedPolicy(cycle(["ACT: 3"])))
    updates = Trace(str(tmp_path)).records("update")
    assert any(u["committed"] for u in updates)
    assert all(u["gate_enabled"] is False for u in updates if u.get("mode"))


def test_smoke_runs_under_60s_on_cpu_with_the_real_tiny_policy(tmp_path):
    t0 = time.monotonic()
    cfg = small(tmp_path, iters=1, tasks_per_iter=1, eval_n=2, gate_n=2,
                policy=PolicyConfig(tiny=True, device="cpu", dtype="float32", max_new_tokens=8))
    assert isinstance(train(cfg), EvalResult)
    assert time.monotonic() - t0 < 60


def test_loop_survives_iterations_with_no_usable_signal(tmp_path):
    result = train(small(tmp_path), policy=ScriptedPolicy(cycle(["garbage"])))
    assert isinstance(result, EvalResult) and result.parse_fail_rate == 1.0
    assert not any(r["committed"] for r in Trace(str(tmp_path)).records("update"))


def test_run_record_captures_provenance(tmp_path):
    train(small(tmp_path, iters=1), policy=ScriptedPolicy(cycle(["ACT: pick"])))
    rec = Trace(str(tmp_path)).records("run")[0]
    assert rec["methods"] == ["memory"] and rec["planner"] == "uniform"
    assert rec["config"]["group_size"] == 2                     # the full config travels with the run
    assert rec["cfg_hash"] and "git_sha" in rec


def test_composite_judge_logs_judge_agreement_in_the_loop(tmp_path):
    cfg = small(tmp_path, iters=1, tasks_per_iter=1, judge="composite", judge_sample_rate=1.0,
                policy=PolicyConfig(tiny=True, device="cpu", dtype="float32"))
    train(cfg, policy=ScriptedPolicy(cycle(["ACT: 3"])))
    rows = [r for r in Trace(str(tmp_path)).records("rollout") if r.get("iter") is not None]
    assert rows and all(r["verdict_source"] == "outcome" and r["judge_verdict"] in (0.0, 1.0) for r in rows)
    assert Trace(str(tmp_path)).cumulative_cost.judge_tokens > 0


def test_frontier_curriculum_runs_in_the_loop_and_logs_its_stats(tmp_path):
    train(small(tmp_path, planner="frontier", gate_enabled=False, tasks_per_iter=4),
        policy=ScriptedPolicy(cycle(["ACT: 3"])))
    updates = Trace(str(tmp_path)).records("update")
    assert updates and all("frontier_frac" in u["planner_stats"] for u in updates)   # every iteration, update or not


def test_grpo_closes_the_loop_with_the_real_tiny_policy(tmp_path):
    cfg = small(tmp_path, iters=2, tasks_per_iter=2, group_size=4, methods=("grpo",), gate_enabled=False,
                policy=PolicyConfig(tiny=True, device="cpu", dtype="float32", max_new_tokens=4,
                                    response_prefix="ACT:"))
    assert isinstance(train(cfg), EvalResult)
    updates = Trace(str(tmp_path)).records("update")
    assert any(u.get("mode") == "grpo" and "loss" in u for u in updates) or \
        all(u["reason"] == "no usable signal" for u in updates)


def test_config_flag_loads_a_yaml_preset_and_flags_override_it():
    from train import parse_cfg
    cfg = parse_cfg(["--config", "tiny", "--iters", "3"])
    assert cfg.policy.tiny and cfg.policy.device == "cpu" and cfg.iters == 3 and cfg.eval_n == 16
    assert parse_cfg(["--config", "count_grpo"]).static_families == ("count_objects",)   # YAML list -> tuple
    assert parse_cfg(["--frontier_band", "0.1,0.9"]).frontier_band == (0.1, 0.9)
    assert parse_cfg([]).eval_n == LoopConfig().eval_n          # no config: the dataclass defaults


def test_unknown_config_key_is_an_error(tmp_path):
    import pytest
    from configs.config import load_config
    bad = tmp_path / "bad.yaml"
    bad.write_text("iterz: 3\n")
    with pytest.raises(TypeError):
        load_config(str(bad))


@pytest.mark.parametrize("planner,update_planner", [("accel", "exp3"), ("voyager", "all"), ("tscl", "all")])
def test_every_method_runs_together_in_the_loop(tmp_path, planner, update_planner):
    cfg = LoopConfig(env="frozenlake", frozenlake_levels=((3, 1), (4, 3)), iters=2, tasks_per_iter=2, group_size=2,
                     eval_every=1, eval_n=4, gate_n=4, heldout_n=8, run_dir=str(tmp_path), gate_enabled=False,
                     methods=("hindsight", "viskill", "expel", "reflexion", "grpo", "ppo", "rest"),
                     planner=planner, update_planner=update_planner, mem_top_k=1,
                     policy=PolicyConfig(tiny=True, device="cpu", dtype="float32", max_new_tokens=4))
    assert isinstance(train(cfg), EvalResult)
    updates = Trace(str(tmp_path)).records("update")
    assert {u["method"] for u in updates} <= {"viskill", "expel", "reflexion", "grpo", "ppo", "rest"}


def test_lifelong_stages_switch_envs_and_evaluate_every_stage_each_time(tmp_path):
    cfg = LoopConfig(env="frozenlake>sokoban", iters=4, tasks_per_iter=1, group_size=2, eval_every=2, eval_n=2,
                     gate_n=2, heldout_n=8, run_dir=str(tmp_path), methods=("viskill",), mem_top_k=1)
    train(cfg, policy=ScriptedPolicy(cycle(["ACT: up", "ACT: right"])))
    tr = Trace(str(tmp_path))
    stages = [r["stage"] for r in tr.records("rollout") if r.get("iter") is not None]
    assert stages[0] == "frozenlake" and stages[-1] == "sokoban"
    evals = tr.records("eval")
    assert [e["env"] for e in evals] == ["frozenlake", "sokoban"] * 3      # it 0, it 2, final
