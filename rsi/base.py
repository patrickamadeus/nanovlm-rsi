"""The two contracts every self-improvement mechanism fits. rsi/template.py is a commented, empty copy of each.

A Method is one way the agent changes itself. The loop calls its hooks in this order every iteration:

    context(task, env) -> Context     before a task's attempts: what to add to its prompt (skills, cards)
    signal(batch, env) -> batch       after the judge: reshape rewards or add examples (hindsight, ViSkill reward)
    propose(batch) -> Update | None   turn the batch into one candidate change ...
    commit(update) / revert(update)   ... which the gate applies durably, then undoes exactly if it hurt
    persisted(update) -> bool         reload from disk: is the change there, and did it change anything?
    observe(batch)                    after the gate: bookkeeping (skill utility, use counts)

Every hook has a do-nothing default, so a method implements only what it uses: GRPO only updates, hindsight
only signals, a skill library does all three. `changes` names the part of the agent it updates
("weights" | "memory" | None), which the logs and the update planner read.

A Planner decides what to practise and which methods may update:

    tasks(n) -> list[Task]            n train-split tasks for this iteration
    updates(names) -> list[str]       which of the updating methods run this iteration (default: all)
    observe(batch, results)           what happened: the scored attempts and each update's gate result
"""
from envs.base import Task, stamp_split
from rsi.gate import Update
from utils.rollout import Context


class Method:
    name = "method"
    changes: str | None = None             # "weights" | "memory" | None (signal-only)

    def context(self, task: Task, env) -> Context:
        return Context()

    def signal(self, batch: list, env) -> list:
        return batch

    def propose(self, batch: list) -> Update | None:
        return None

    def commit(self, update: Update) -> None:
        raise NotImplementedError(f"{self.name} proposed an update but cannot commit one")

    def revert(self, update: Update) -> None:
        raise NotImplementedError(f"{self.name} proposed an update but cannot revert one")

    def persisted(self, update: Update) -> bool:
        raise NotImplementedError(f"{self.name} proposed an update but cannot check persistence")

    def observe(self, batch: list) -> None:
        pass


class Planner:
    name = "planner"

    def tasks(self, n: int) -> list[Task]:
        raise NotImplementedError

    def updates(self, names: list[str]) -> list[str]:
        return names

    def observe(self, batch: list, results: dict) -> None:
        pass


def train_only(tasks: list[Task]) -> list[Task]:
    """Every planner's output passes through here: a planner reaching gate or held-out seeds would
    silently contaminate every reported number, so that is an error, not a warning."""
    stamped = [stamp_split(t) for t in tasks]
    leaked = [t.seed for t in stamped if t.split != "train"]
    if leaked:
        raise RuntimeError(f"planner produced non-train seeds {leaked[:5]}")
    return stamped


def merge(contexts: list[Context]) -> Context:
    """Several methods may add to one prompt: texts are joined, images and ids concatenated in order."""
    return Context("\n".join(c.text for c in contexts if c.text),
                   tuple(im for c in contexts for im in c.images), tuple(i for c in contexts for i in c.ids))
