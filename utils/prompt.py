"""The prompt is data, not code: one template file per environment in prompts/, with named slots.

Two answer formats:
  act     one line, "ACT: <action>" (picoRSI's original; the reply is prefilled with "ACT:")
  answer  VAGEN's free_think: "<think>...</think><answer>a1, a2</answer>", up to max_actions per turn
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_SLOTS = ("{system}", "{actions}", "{memory}", "{task}")
ACT_RE = re.compile(r"^\s*ACT:\s*(\S+)", re.MULTILINE)


class ScaffoldContractError(ValueError):
    pass


def load_scaffold(path: str) -> str:
    """A relative path that doesn't exist from the current directory is resolved against the repo root."""
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(ROOT, path)
    with open(path) as f:
        return f.read()


def validate_scaffold(template: str) -> None:
    missing = [slot for slot in REQUIRED_SLOTS if slot not in template]
    if "ACT:" not in template:
        missing.append("ACT:")
    if missing:
        raise ScaffoldContractError(f"scaffold is missing {missing}")


def render(template: str, **slots) -> str:
    """Fill every {slot} the template has; a list (the legal actions) is joined with commas."""
    out = template
    for k, v in slots.items():
        out = out.replace("{" + k + "}", ", ".join(v) if isinstance(v, list) else str(v))
    return re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"


def parse_action(text: str, legal: list[str]) -> str | None:
    """The last `ACT: x` line wins; an action outside `legal` is a parse failure, not an env call."""
    found = ACT_RE.findall(text)
    if not found:
        return None
    action = found[-1].strip().rstrip(".,;")
    if action in legal:
        return action
    bare = action.strip("<>[]\"'`*").rstrip(".,;").lower()      # "Up", "<up>", "'up'." mean up
    by_lower = {a.lower(): a for a in reversed(legal)}
    return by_lower.get(bare)


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _name(a: str) -> str:
    return a.strip().strip("<>[]\"'`*.").lower().replace("_", " ")


def parse_turn(text: str, legal: list[str], fmt: str, max_actions: int, sep: str) -> tuple[list[str], bool]:
    """(actions to execute, format_ok). The actions are every legal one named, up to max_actions; format_ok
    (what VAGEN's format reward pays) also needs reasoning closed before the answer and nothing invalid."""
    if fmt == "act":
        action = parse_action(text, legal)
        return ([action], True) if action else ([], False)
    found = ANSWER_RE.findall(text)
    if not found:
        return [], False
    by_name = {_name(a): a for a in reversed(legal)}
    named = [_name(x) for x in found[-1].split(sep) if x.strip()]
    actions = [by_name[n] for n in named if n in by_name][:max_actions]
    ok = ("</think>" in text.split("<answer>")[0] and bool(actions) and len(actions) == len(named)
          and len(named) <= max_actions)
    return actions, ok