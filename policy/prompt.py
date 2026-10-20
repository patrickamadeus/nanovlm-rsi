"""The scaffold is data, not code: a template file with named slots.

Keeping it a file is what lets update/scaffold.py rewrite it as a diffable,
versioned, revertible artifact instead of editing Python.
"""
import re

REQUIRED_SLOTS = ("{system}", "{actions}", "{memory}", "{task}")
ACT_RE = re.compile(r"^\s*ACT:\s*(\S+)", re.MULTILINE)


class ScaffoldContractError(ValueError):
    pass


def load_scaffold(path: str) -> str:
    with open(path) as f:
        return f.read()


def validate_scaffold(template: str) -> None:
    missing = [slot for slot in REQUIRED_SLOTS if slot not in template]
    if "ACT:" not in template:
        missing.append("ACT:")
    if missing:
        raise ScaffoldContractError(f"scaffold is missing {missing}")


def render(template: str, *, system: str, actions: list[str], memory: str, task: str) -> str:
    out = (template.replace("{system}", system).replace("{actions}", ", ".join(actions))
           .replace("{memory}", memory).replace("{task}", task))
    return re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"


def parse_action(text: str, legal: list[str]) -> str | None:
    """The last `ACT: x` line wins; an action outside `legal` is a parse failure, not an env call."""
    found = ACT_RE.findall(text)
    if not found:
        return None
    action = found[-1].strip().rstrip(".,;")
    return action if action in legal else None
