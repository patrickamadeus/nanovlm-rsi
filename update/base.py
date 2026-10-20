"""The Updater contract shared by every carrier. Exactly one carrier runs per experiment."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from infra.trace import Cost


@dataclass(frozen=True)
class Update:
    # taxonomy labels live on the object, so a trace reader never needs the code
    carrier: str               # A3 external memory | A4 harness/scaffold | A5 weights
    role: str                  # B0 policy | B1 memory manager | B2 evaluator | B3 updater | B4 self-referential
    horizon: str               # C1 transient | C2 episode | C3 cross-episode durable | C4 successor version
    clock: str                 # when the update happens
    signal: str                # where the evidence came from; checked against the gate's allowed set
    payload: dict = field(default_factory=dict)
    cost: Cost = field(default_factory=Cost)
    provenance: tuple[str, ...] = ()   # the trajectories that produced this update


class Updater(ABC):
    @abstractmethod
    def propose(self, batch) -> Update | None:
        """Turn scored trajectories into a candidate update, or None when there is no usable signal."""

    @abstractmethod
    def commit(self, update: Update) -> None: ...

    @abstractmethod
    def revert(self, update: Update) -> None:
        """Undo a committed update exactly; the gate depends on this being honest."""
