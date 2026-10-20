import pytest

from infra.trace import Cost, Trace


def test_records_roundtrip_by_kind(tmp_path):
    tr = Trace(str(tmp_path))
    tr.log_run(cfg_hash="abc", git_sha="def", seeds={"seed": 0},
               carrier="memory", proposer="fixed", gate_enabled=True)
    tr.log_rollout(task_id="t1", family="count_objects", seed=1, split="train",
                   reward=1.0, verdict_source="outcome", parse_fail=False,
                   retrieved_skill_ids=[], env_steps=1, tokens_out=5)
    assert len(tr.records("run")) == 1
    assert tr.records("rollout")[0]["family"] == "count_objects"
    assert tr.records()[0]["kind"] == "run"


def test_cumulative_cost_accumulates(tmp_path):
    tr = Trace(str(tmp_path))
    tr.log_rollout(task_id="a", env_steps=3, tokens_out=10)
    tr.log_rollout(task_id="b", env_steps=4, tokens_out=5)
    assert tr.cumulative_cost.env_steps == 7
    assert tr.cumulative_cost.policy_tokens == 15


def test_eval_record_separates_heldout_axes(tmp_path):
    tr = Trace(str(tmp_path))
    tr.log_eval(split="heldout", n=200, success_rate=0.4,
                unseen_seed_rate=0.5, unseen_goal_rate=0.3, cost=Cost(10, 20, 0, 1.0))
    rec = tr.records("eval")[0]
    assert rec["unseen_seed_rate"] == 0.5 and rec["unseen_goal_rate"] == 0.3
    assert rec["cost"]["env_steps"] == 10


def test_reader_skips_truncated_final_line(tmp_path):
    tr = Trace(str(tmp_path))
    tr.log_rollout(task_id="t1", reward=1.0)
    with (tmp_path / "trace.jsonl").open("a") as f:
        f.write('{"kind": "rollout", "rew')
    assert len(Trace(str(tmp_path)).records("rollout")) == 1


def test_success_rate_tracks_family_goal(tmp_path):
    tr = Trace(str(tmp_path))
    for r in (1.0, 0.0, 1.0):
        tr.log_rollout(family="count_objects", goal=["3"], split="train", reward=r)
    assert tr.success_rate("count_objects", ("3",)) == pytest.approx(2 / 3)
    assert tr.success_rate("count_objects", ("9",)) is None
