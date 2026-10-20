import pytest

from policy.prompt import ScaffoldContractError, load_scaffold, parse_action, render, validate_scaffold


def test_parse_action_extracts_legal_action():
    assert parse_action("tree ahead\nACT: pick", ["pick", "up"]) == "pick"


def test_parse_action_rejects_missing_act_line():
    assert parse_action("I think I should pick it up", ["pick"]) is None


def test_parse_action_rejects_illegal_but_parseable_action():
    assert parse_action("ACT: teleport", ["pick", "up"]) is None


def test_parse_action_takes_last_act_line():
    assert parse_action("ACT: up\nwait\nACT: pick", ["pick", "up"]) == "pick"


def test_validate_scaffold_requires_all_slots():
    with pytest.raises(ScaffoldContractError):
        validate_scaffold("{system} {actions} {task} ACT:")          # missing {memory}
    with pytest.raises(ScaffoldContractError):
        validate_scaffold("{system} {actions} {memory} {task}")      # missing ACT:


def test_shipped_scaffold_is_valid():
    validate_scaffold(load_scaffold("scaffolds/v0.txt"))


def test_render_fills_slots_and_empty_memory_is_clean():
    out = render(load_scaffold("scaffolds/v0.txt"),
                 system="S", actions=["a", "b"], memory="", task="T")
    assert "S" in out and "a, b" in out and "T" in out
    assert "{" not in out and "\n\n\n" not in out
