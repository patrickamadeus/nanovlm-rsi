import numpy as np
import pytest

from configs.config import LoopConfig
from envs.frozenlake import FrozenLakeEnv
from rsi.judges import ScoredTrajectory
from rsi.memory import SkillStore
from rsi.viskill import ViSkill, render_card, skill_reward
from tests.conftest import ScriptedPolicy, fake_embed
from utils.rollout import Context, rollout_group

ENV = FrozenLakeEnv()


class Solver:
    """Plays the solver's moves (or a fixed list), so trajectories are real env trajectories."""
    version = 0

    def __init__(self, tasks, plans=None, strategy="Go down first, then right."):
        self.plans = plans or [ENV.solve(t) for t in tasks]
        self.row = {}
        self.strategy = strategy

    def act_batch(self, prompts, images, **kw):
        from models.vlm import RolloutStep
        out = []
        for i, p in enumerate(prompts):
            t = int(p.split("Step ")[1].split()[0])
            plan = self.plans[self.row_of[i]]
            out.append(RolloutStep(f"ACT: {plan[min(t, len(plan) - 1)]}", [1], np.zeros(1, np.float32), 3))
        return out

    def act(self, prompt, images, **kw):
        from models.vlm import RolloutStep
        return RolloutStep(f"Strategy: {self.strategy}", [1], np.zeros(1, np.float32), 3)


def play(tasks, plans=None, contexts=None):
    policy = Solver(tasks, plans)
    trajs = []
    for i, t in enumerate(tasks):                    # one row at a time keeps the scripted plan lookup simple
        policy.row_of = {0: i}
        trajs += rollout_group(policy, ENV, [t], LoopConfig(), [contexts[i]] if contexts else None)
    return [ScoredTrajectory(tr, tr.total_reward, None, "outcome", 1.0, ()) for tr in trajs], policy


@pytest.fixture
def vs(tmp_path):
    cfg = LoopConfig(run_dir=str(tmp_path), mem_top_k=1, skill_format="visual", mem_max_excess_steps=0)
    store = SkillStore(str(tmp_path / "skills.jsonl"), embed_fn=fake_embed)
    tasks = [t for t in ENV.tasks("train", 40, np.random.default_rng(0)) if len(ENV.solve(t)) >= 3][:3]
    batch, policy = play(tasks)
    return ViSkill(store, cfg, ENV, policy), batch, tasks


def test_skill_reward_follows_viskill():
    ref = ["down", "down", "right", "right"]
    assert skill_reward(["down", "down", "right"], ref) == pytest.approx(3 / 4)       # 3 along the path, no detour
    assert skill_reward(["down", "left", "left", "left"], ref) == 0.0                  # 1 along, 3 off: clipped
    assert skill_reward(["down", "down", "up"], ref) == pytest.approx((2 - 1) / 4)
    assert skill_reward([], ref) == 0.0


def test_card_is_one_image_with_every_step(vs):
    _, batch, _ = vs
    traj = batch[0].traj
    card = render_card(traj, "Go down first.", cell_px=48)
    assert card.dtype == np.uint8 and card.ndim == 3 and card.shape[1] > card.shape[0]


def test_commit_writes_cards_and_context_shows_one_as_an_image(vs, tmp_path):
    m, batch, tasks = vs
    u = m.propose(batch)
    assert len(u.payload["skills"]) == 3 and all(s["strategy"] == "Go down first, then right." for s in u.payload["skills"])
    m.commit(u)
    assert m.persisted(u) and len(list((tmp_path / "cards").glob("*.png"))) == 3
    ctx = m.context(tasks[0], ENV)                   # the same layout retrieves its own card
    assert len(ctx.images) == 1 and ctx.text.count("<image>") == 1 and "Skill Reference" in ctx.text
    m.revert(u)
    assert len(m.store) == 0 and m.context(tasks[0], ENV).images == ()


def test_only_optimal_successes_are_admitted(vs):
    m, _, tasks = vs
    path = ENV.solve(tasks[0])
    detour = ["left", "right"] + path                 # a success two steps longer than optimal ... if no hole
    batch, _ = play([tasks[0]], [detour])
    if batch[0].reward == 1.0:
        assert m.propose(batch) is None


def test_signal_adds_progress_reward_to_failures_that_had_a_skill(vs):
    m, batch, tasks = vs
    m.commit(m.propose(batch))
    ctx = m.context(tasks[0], ENV)
    ref = list(m.store.skills()[0].actions) if len(m.store) == 1 else ENV.solve(tasks[0])
    fail, _ = play([tasks[0]], [ref[:-1] + ["up"] * 20], [ctx])       # follows the card, then wanders off
    assert fail[0].reward == 0.0
    acts = [st["action"] for st in fail[0].traj.steps if st["action"]]
    card = next(s for s in m.store.skills() if s.id == ctx.ids[0])
    assert m.signal(fail, ENV)[0].reward == pytest.approx(0.8 * skill_reward(acts, list(card.actions)))
    no_ctx, _ = play([tasks[0]], [ref[:-1] + ["up"] * 20])
    assert m.signal(no_ctx, ENV)[0].reward == no_ctx[0].reward       # no skill in context: nothing to follow


def test_lambda_zero_turns_the_skill_reward_off(vs, tmp_path):
    m, batch, tasks = vs
    m.cfg.viskill_lambda = 0.0
    m.commit(m.propose(batch))
    ctx = m.context(tasks[0], ENV)
    fail, _ = play([tasks[0]], [ENV.solve(tasks[0])[:-1] + ["up"] * 20], [ctx])
    assert m.signal(fail, ENV)[0].reward == fail[0].reward
