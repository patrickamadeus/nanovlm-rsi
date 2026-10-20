import ast
import subprocess
import sys


def test_walkthrough_is_valid_python_and_up_to_date():
    fresh = subprocess.run([sys.executable, "tools/make_walkthrough.py", "--stdout"],
                           capture_output=True, text=True, check=True).stdout
    ast.parse(fresh)
    with open("walkthrough.py") as f:
        assert f.read() == fresh, "walkthrough.py is stale: run python tools/make_walkthrough.py"


def test_walkthrough_covers_every_organ_in_loop_order():
    text = open("walkthrough.py").read()
    order = ["def rsi(", "def rollout(", "class OutcomeVerifier", "def group_advantages(",
             "def grpo_token_loss(", "class LoRALinear", "def gate("]
    positions = [text.index(marker) for marker in order]
    assert positions == sorted(positions)
