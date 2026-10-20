import os

import numpy as np
import pytest

from config import LoopConfig, PolicyConfig
from evals.benchmarks import MMSTAR_SIZE, parse_choice, run_mmstar
from infra.trace import Trace
from tests.conftest import ScriptedPolicy

IMG = np.zeros((32, 32, 3), np.uint8)
ROWS = [{"question": "Q1\nOptions: A: x, B: y", "answer": "A", "category": "math", "image": IMG},
        {"question": "Q2\nOptions: A: x, B: y", "answer": "B", "category": "math", "image": IMG},
        {"question": "Q3\nOptions: A: x, B: y", "answer": "C", "category": "science", "image": IMG}]


@pytest.mark.parametrize("text,expected", [("Answer: B", "B"), ("Answer: (c) because", "C"), ("Answer: D.", "D"),
                                           ("Answer: the cat", None), ("Answer: A and B", "A")])
def test_parse_choice_takes_the_first_option_letter(text, expected):
    assert parse_choice(text) == expected


def test_accuracy_and_per_category_breakdown():
    r = run_mmstar(ScriptedPolicy(["Answer: A", "Answer: A", "Answer: C"]), ROWS)
    assert r.n == 3 and r.accuracy == pytest.approx(2 / 3)
    assert r.per_category == {"math": 0.5, "science": 1.0}


def test_a_subset_run_is_labelled_so_it_cannot_pass_as_reported():
    assert run_mmstar(ScriptedPolicy(["Answer: A"] * 3), ROWS).subset is True   # 3 < 1500
    assert MMSTAR_SIZE == 1500


def test_benchmark_decodes_greedily_with_an_answer_prefix():
    calls = []

    class Spy(ScriptedPolicy):
        def act(self, prompt, images, **kw):
            calls.append(kw)
            return super().act(prompt, images)

    run_mmstar(Spy(["Answer: A"] * 3), ROWS)
    assert all(kw == {"prefix": "Answer:", "greedy": True} for kw in calls)


def test_tiny_policy_supports_greedy_and_prefix_override():
    from policy.vlm import Policy
    p = Policy(PolicyConfig(tiny=True, device="cpu", dtype="float32", max_new_tokens=3))
    a = p.act("Which?", [IMG], prefix="Answer:", greedy=True)
    b = p.act("Which?", [IMG], prefix="Answer:", greedy=True)
    assert a.text == b.text and a.text.startswith("Answer:")


@pytest.mark.skipif(not os.environ.get("NANORSI_SLOW"), reason="downloads MMStar; set NANORSI_SLOW=1")
def test_real_mmstar_loads_all_1500():
    from evals.benchmarks import load_mmstar
    rows = load_mmstar()
    assert len(rows) == 1500 and {r["answer"] for r in rows} == {"A", "B", "C", "D"}
