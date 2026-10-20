"""Regression tests for the final whole-branch review findings."""
import json

import numpy as np
import pytest

from config import LoopConfig, PolicyConfig
from envs.base import Task
from envs.static import StaticEnv
from infra.gate import gate
from infra.memory import Skill, SkillStore
from infra.trace import Trace
from rollout import rollout, rollout_record
from rsi import rsi
from tests.conftest import ScriptedPolicy, fake_embed
from update.memory import MemoryUpdater
from verify import ScoredTrajectory, Trajectory

OBS = "How many blue squares are there? Answer with the number."


def _scored(answer, seed):
    t = Task("count_objects", seed, (answer,), 1, "train")
    traj = Trajectory(t, [{"action": f"{answer}", "obs_text": OBS}], 1.0, 1, 0, 3)
    return ScoredTrajectory(traj, 1.0, None, "outcome", 1.0, ())


# C1 — same situation, different actions, equal reward: both must be stored
def test_distinct_answers_for_the_same_situation_are_both_kept(tmp_path):
    store = SkillStore(str(tmp_path / "s.jsonl"), embed_fn=fake_embed)
    up = MemoryUpdater(store, LoopConfig())
    up.commit(up.propose([_scored("3", 2)]))
    up.commit(up.propose([_scored("5", 4)]))
    assert len(store) == 2


# C1 — a commit that adds nothing must not pass the gate as committed
def test_noop_update_fails_the_gate(tmp_path):
    store = SkillStore(str(tmp_path / "s.jsonl"), embed_fn=fake_embed)
    cfg = LoopConfig(gate_n=2, run_dir=str(tmp_path))
    up = MemoryUpdater(store, cfg)
    up.commit(up.propose([_scored("3", 2)]))
    dup = up.propose([_scored("3", 6)])                  # exact duplicate of what is stored
    r = gate(dup, up, StaticEnv(), cfg, eval_fn=lambda tasks: (0.5, 0.0))
    assert not r.passed and r.conditions["survives_reset"] is False
    assert len(store) == 1


# Minor-2 upgraded — a reverted skill's id is never reused
def test_skill_ids_are_not_reused_after_revert(tmp_path):
    store = SkillStore(str(tmp_path / "s.jsonl"), embed_fn=fake_embed)
    off = store.offset()
    a = store.append(Skill("", "f", "alpha", ("x",), 1.0, fake_embed("alpha")))
    store.truncate(off)
    b = store.append(Skill("", "f", "beta", ("y",), 1.0, fake_embed("beta")))
    assert a != b


# I2 — reusing a run_dir is refused unless resuming; records carry a run_id
def test_reusing_a_run_dir_requires_resume(tmp_path):
    cfg = LoopConfig(iters=1, tasks_per_iter=1, group_size=1, eval_n=2, gate_n=2, heldout_n=8, run_dir=str(tmp_path))
    rsi(cfg, policy=ScriptedPolicy(iter(lambda: "ACT: pick", None)))
    with pytest.raises(FileExistsError):
        rsi(cfg, policy=ScriptedPolicy(iter(lambda: "ACT: pick", None)))
    cfg.resume = True
    rsi(cfg, policy=ScriptedPolicy(iter(lambda: "ACT: pick", None)))
    recs = Trace(str(tmp_path)).records()
    assert all("run_id" in r for r in recs)
    assert len({r["run_id"] for r in recs}) == 2


# I3 — act() logprobs equal a re-score in the configured (bf16) dtype
def test_act_logprobs_match_rescore_in_bfloat16():
    from policy.vlm import Policy
    p = Policy(PolicyConfig(tiny=True, device="cpu", dtype="bfloat16", max_new_tokens=8))
    img = [np.zeros((64, 64, 3), np.uint8)]
    step = p.act("hello", img)
    again = p.logprobs("hello", img, step.token_ids).float().numpy()
    np.testing.assert_allclose(step.logprobs, again, atol=1e-6)


# I4 — a truncated tail must not swallow the next record
def test_trace_write_after_truncated_tail_is_readable(tmp_path):
    Trace(str(tmp_path)).log_rollout(task_id="a")
    with (tmp_path / "trace.jsonl").open("a") as f:
        f.write('{"kind": "rollout", "rew')
    tr = Trace(str(tmp_path))
    tr.log_rollout(task_id="b")
    assert [r["task_id"] for r in Trace(str(tmp_path)).records("rollout")] == ["a", "b"]


def test_store_write_after_truncated_tail_survives_reload(tmp_path):
    path = tmp_path / "s.jsonl"
    SkillStore(str(path), embed_fn=fake_embed).append(Skill("", "f", "alpha", ("x",), 1.0, fake_embed("alpha")))
    with path.open("a") as f:
        f.write('{"op": "add", "sk')
    SkillStore(str(path), embed_fn=fake_embed).append(Skill("", "f", "beta", ("y",), 1.0, fake_embed("beta")))
    assert len(SkillStore(str(path), embed_fn=fake_embed)) == 2


# I5 — prompt tokens and model time are counted; recall runs once per task, not per group member
def test_rollout_cost_counts_prompt_tokens_and_model_time():
    env = StaticEnv()
    task = env.tasks("train", 1, np.random.default_rng(0))[0]
    rec = rollout_record(rollout(ScriptedPolicy(["ACT: 3"] * 30), env, task, LoopConfig()), [])
    assert rec["tokens_in"] > 0 and rec["gpu_seconds"] > 0 and "wall_ms" in rec


def test_cumulative_policy_tokens_include_prompt_tokens(tmp_path):
    tr = Trace(str(tmp_path))
    tr.log_rollout(tokens_in=100, tokens_out=5, gpu_seconds=0.5)
    assert tr.cumulative_cost.policy_tokens == 105 and tr.cumulative_cost.gpu_seconds == 0.5


def test_recall_runs_once_per_task_not_per_group_member(tmp_path):
    calls = []

    class CountingPolicy(ScriptedPolicy):
        def embed(self, text, images):
            calls.append(text)
            return fake_embed(text)

    cfg = LoopConfig(iters=1, tasks_per_iter=2, group_size=4, eval_n=2, gate_n=2, heldout_n=8,
                     eval_every=100, run_dir=str(tmp_path))
    store_path = tmp_path / "skills.jsonl"
    SkillStore(str(store_path), embed_fn=fake_embed).append(Skill("", "f", "seed skill", ("pick",), 1.0,
                                                                  fake_embed("seed skill")))
    cfg.resume = True
    policy = CountingPolicy(iter(lambda: "ACT: pick", None))
    rsi(cfg, policy=policy)
    train_recalls = len(calls)
    assert train_recalls <= 2 + 2 + 2 * 2      # train tasks + final eval + gate before/after — never x group_size
