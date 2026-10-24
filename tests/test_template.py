"""The templates are the documentation for extending picoRSI: these tests keep them honest, and check
that every real env and method keeps the contract the templates describe."""
import numpy as np
import pytest

from configs.config import LoopConfig
from envs.base import Env, split_for_seed
from envs.frozenlake import FrozenLakeEnv
from envs.sokoban import SokobanEnv
from envs.static import StaticEnv
from envs.template import TemplateEnv
from rsi.base import Method, Planner
from rsi.template import TemplateMethod, TemplatePlanner

ENVS = [StaticEnv, FrozenLakeEnv, SokobanEnv]


def test_the_env_template_is_an_env_that_asks_to_be_filled_in():
    env = TemplateEnv()
    assert isinstance(env, Env)
    for call in (lambda: env.make_task(1), lambda: env.actions, lambda: env.solve(None)):
        with pytest.raises(NotImplementedError):
            call()


def test_the_method_and_planner_templates_ask_to_be_filled_in():
    m, p = TemplateMethod(None, LoopConfig(), "runs/x"), TemplatePlanner(None, LoopConfig(), None)
    assert isinstance(m, Method) and isinstance(p, Planner)
    for call in (lambda: m.context(None, None), lambda: m.propose([]), lambda: m.commit(None),
                 lambda: m.revert(None), lambda: m.persisted(None), lambda: p.tasks(1)):
        with pytest.raises(NotImplementedError):
            call()


def test_base_hooks_default_to_doing_nothing():
    m = Method()
    assert m.context(None, None).text == "" and m.signal([1], None) == [1] and m.propose([]) is None
    assert Planner().updates(["a", "b"]) == ["a", "b"]


@pytest.mark.parametrize("make", ENVS)
def test_every_env_keeps_the_contract(make):
    env = make()
    tasks = env.tasks("train", 5, np.random.default_rng(0))
    assert all(t.split == split_for_seed(t.seed) == "train" for t in tasks)
    assert env.tasks("heldout", 5, np.random.default_rng(1)) == env.tasks("heldout", 5, np.random.default_rng(2))
    first = env.reset(tasks[0])
    assert first.rgb.dtype == np.uint8 and first.rgb.ndim == 3
    fork = env.fork()
    fork.reset(tasks[1])
    obs, reward, done, info = fork.step(fork.actions[0])
    assert isinstance(reward, float) and isinstance(done, bool) and isinstance(info, dict)
    assert np.array_equal(env.reset(tasks[0]).rgb, first.rgb)          # reset is a pure function of the task
    assert isinstance(env.task_key(tasks[0]), str)


@pytest.mark.parametrize("name", ["grpo", "memory"])
def test_every_registered_method_is_a_method(name, tmp_path):
    from models.vlm import Policy
    from configs.config import PolicyConfig
    from rsi import make_method
    policy = Policy(PolicyConfig(tiny=True, device="cpu", dtype="float32"))
    m = make_method(name, LoopConfig(run_dir=str(tmp_path)), policy)
    assert isinstance(m, Method) and m.name == name and m.changes in ("weights", "memory")


def test_two_weight_methods_share_one_set_of_adapters(tmp_path):
    from models.vlm import Policy
    from configs.config import PolicyConfig
    from rsi import make_methods
    policy = Policy(PolicyConfig(tiny=True, device="cpu", dtype="float32"))
    a, b = make_methods(LoopConfig(methods=("grpo", "grpo"), run_dir=str(tmp_path)), policy)
    assert a.weights is b.weights and a.opt is not b.opt
