import numpy as np

from configs.config import LoopConfig
from envs.frozenlake import FrozenLakeEnv
from rsi.expel import ExpeL, ending
from rsi.reflexion import Reflexion
from tests.conftest import fake_embed
from tests.test_viskill import Solver, play

ENV = FrozenLakeEnv()


class Writer(Solver):
    def __init__(self, text):
        super().__init__([], plans=[[]])
        self.text, self.prompts = text, []

    def act(self, prompt, images, **kw):
        from models.vlm import RolloutStep
        self.prompts.append(prompt)
        return RolloutStep(self.text, [1], np.zeros(1, np.float32), 3)


def _mixed_group():
    for t in ENV.tasks("train", 50, np.random.default_rng(0)):
        win, _ = play([t], [ENV.solve(t)])
        for wander in ("up", "left", "down", "right"):
            lose, _ = play([t], [[wander]])                    # one move, repeated: never the solver's path
            if lose[0].reward == 0.0:
                return t, win + lose
    raise AssertionError("no task with a failing single-direction attempt")


def test_expel_writes_a_rule_from_a_success_failure_pair_and_shows_it_everywhere(tmp_path):
    t, batch = _mixed_group()
    writer = Writer("Rule: Avoid stopping before the green square.")
    m = ExpeL(LoopConfig(run_dir=str(tmp_path)), writer, str(tmp_path))
    u = m.propose(batch)
    assert u.payload["skills"][0]["strategy"] == "Avoid stopping before the green square."
    assert "Succeeded with" in writer.prompts[0] and "Failed with" in writer.prompts[0]
    m.commit(u)
    other = ENV.tasks("train", 5, np.random.default_rng(9))[-1]
    ctx = m.context(other, ENV)                                   # rules are global, not retrieved by layout
    assert "Avoid stopping before the green square." in ctx.text and len(ctx.ids) == 1
    m.revert(u)
    assert m.context(other, ENV).text == ""


def test_expel_needs_both_a_success_and_a_failure(tmp_path):
    t, batch = _mixed_group()
    m = ExpeL(LoopConfig(run_dir=str(tmp_path)), Writer("Rule: x"), str(tmp_path))
    assert m.propose(batch[:1]) is None and m.propose(batch[1:]) is None


def test_expel_does_not_store_the_same_rule_twice(tmp_path):
    t, batch = _mixed_group()
    m = ExpeL(LoopConfig(run_dir=str(tmp_path)), Writer("Rule: same"), str(tmp_path))
    m.commit(m.propose(batch))
    assert m.propose(batch) is None


def test_reflexion_writes_one_note_per_failed_task_and_retrieves_it_for_that_layout(tmp_path):
    t, batch = _mixed_group()
    writer = Writer("Reflection: I stopped one step early; keep going to the green square.")
    m = Reflexion(LoopConfig(run_dir=str(tmp_path), mem_top_k=1), writer, ENV, fake_embed)
    u = m.propose(batch + batch)
    assert len(u.payload["skills"]) == 1 and "You failed" in writer.prompts[0]
    m.commit(u)
    assert "keep going to the green square" in m.context(t, ENV).text


def test_ending_says_how_an_attempt_ended():
    t, batch = _mixed_group()
    assert ending(batch[1]) in ("ran out of steps", "stopped") or ending(batch[1]).startswith("ended:")
