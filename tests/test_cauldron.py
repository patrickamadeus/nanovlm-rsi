import numpy as np
import pytest

from configs.config import LoopConfig
from envs.base import split_for_seed
from envs.cauldron import CauldronEnv, answer_key, items_from_rows
from rsi import make_planner
from tests.conftest import ScriptedPolicy
from utils.rollout import rollout


@pytest.mark.parametrize("text, key", [
    ("Three.", ("num", "3")), ("5.", ("num", "5")), ("20", ("num", "20")),
    ("Answer: D", ("mc", "D")), ("C", ("mc", "C")), ("No.", ("yn", "no")), ("Yes", ("yn", "yes")),
    ("Under clock.", None), ("25", None), ("a", None), ("1.5", None),
])
def test_answer_key_keeps_only_answers_exact_match_can_check(text, key):
    assert answer_key(text) == key


def _row(source, *turns, num_imgs=1):
    return {"source": source, "num_imgs": num_imgs, "texts": [{"user": u, "assistant": a} for u, a in turns]}


ROWS = [
    _row("dvqa", ("Describe the chart.", "A bar chart."), ("How many bars?", "Three.")),
    _row("raven", ("Choose the figure.", "C")),
    _row("vqav2", ("Is it sunny?", "Yes.")),
    _row("vqav2", ("What is this?", "A dog.")),                     # nothing checkable: no task
    _row("mmimg", ("Same object?", "Yes."), num_imgs=2),            # one image per task only
]


def test_one_task_per_image_from_its_first_checkable_turn():
    items = items_from_rows(ROWS)
    assert [(i.seed, i.source, i.question, i.kind, i.answer) for i in items] == [
        (0, "dvqa", "How many bars?", "num", "3"), (1, "raven", "Choose the figure.", "mc", "C"),
        (2, "vqav2", "Is it sunny?", "yn", "yes")]
    assert [i.source for i in items_from_rows(ROWS, sources=("vqav2",))] == ["vqav2"]


def _env(n=300, heldout_n=20):
    rows = [_row("tallyqa", (f"How many cats? #{i}", str(i % 10))) for i in range(n)]
    image = lambda seed: np.full((40, 30, 3), seed % 256, np.uint8)
    return CauldronEnv(items_from_rows(rows), image, heldout_n=heldout_n)


def test_reset_shows_the_rows_image_and_question_and_step_checks_the_answer():
    env = _env()
    task = env.tasks("train", 1, np.random.default_rng(0))[0]
    obs = env.reset(task)
    assert obs.rgb.shape == (40, 30, 3) and obs.rgb[0, 0, 0] == task.seed % 256
    assert obs.text == f"How many cats? #{task.seed}" and task.goal == (str(task.seed % 10),)
    assert env.actions == [str(i) for i in range(21)]
    _, reward, done, _ = env.step(task.goal[0])
    assert reward == 1.0 and done
    env.reset(task)
    assert env.step("11")[1] == 0.0


def test_actions_follow_the_answer_kind():
    env = CauldronEnv(items_from_rows(ROWS), lambda s: np.zeros((8, 8, 3), np.uint8), heldout_n=0)
    kinds = {}
    for item in env.items:
        env.reset(env.task(item))
        kinds[item.kind] = env.actions
    assert kinds["mc"] == list("ABCDEFGH") and kinds["yn"] == ["yes", "no", "Yes", "No"]


def test_splits_follow_the_row_hash_and_held_out_is_frozen():
    env = _env()
    for split in ("train", "gate", "heldout"):
        tasks = env.tasks(split, 10, np.random.default_rng(1))
        assert all(t.split == split == split_for_seed(t.seed) for t in tasks)
    a = env.tasks("heldout", 20, np.random.default_rng(1))
    assert a == env.tasks("heldout", 20, np.random.default_rng(2))
    with pytest.raises(ValueError):
        env.tasks("heldout", 21, np.random.default_rng(0))


def test_the_planners_accept_cauldron_train_tasks():
    env = _env()
    for planner in ("uniform", "frontier"):
        tasks = make_planner(env, LoopConfig(planner=planner)).tasks(4)
        assert len(tasks) == 4 and {t.split for t in tasks} == {"train"}


def test_a_rollout_answers_once_and_scores_the_answer():
    env = _env()
    task = env.tasks("train", 1, np.random.default_rng(0))[0]
    traj = rollout(ScriptedPolicy([f"ACT: {task.goal[0]}"]), env, task, LoopConfig())
    assert traj.total_reward == 1.0 and traj.env_steps == 1 and traj.parse_fails == 0


def test_make_env_builds_cauldron_from_the_configured_subset(monkeypatch):
    import envs.cauldron
    from envs import make_env
    seen = {}

    def fake_load(repo, subset, revision, sources):
        seen.update(repo=repo, subset=subset, sources=sources)
        return items_from_rows(ROWS), lambda s: np.zeros((8, 8, 3), np.uint8)
    monkeypatch.setattr(envs.cauldron, "load", fake_load)
    env = make_env(LoopConfig(env="cauldron", cauldron_sources=("vqav2",), heldout_n=0))
    assert isinstance(env, CauldronEnv) and len(env.items) == 3
    assert seen == {"repo": "patrickamadeus/the_cauldron", "subset": "sample_1pct", "sources": ("vqav2",)}


def test_a_capitalized_yes_is_a_correct_answer_not_a_parse_failure():
    rows = [_row("vqav2", (f"Is it sunny? #{i}", "Yes.")) for i in range(50)]
    env = CauldronEnv(items_from_rows(rows), lambda s: np.zeros((8, 8, 3), np.uint8), heldout_n=0)
    task = env.tasks("train", 1, np.random.default_rng(0))[0]
    traj = rollout(ScriptedPolicy(["ACT: Yes."]), env, task, LoopConfig())
    assert traj.parse_fails == 0 and traj.total_reward == 1.0
