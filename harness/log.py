"""The run log: one append-only JSONL file of run, rollout, update and eval records.

A run should be attributable from this file alone — carrier, gate outcomes and
cumulative cost are all here, so reading results never requires reading code.
"""
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Cost:
    env_steps: int = 0
    policy_tokens: int = 0
    judge_tokens: int = 0
    gpu_seconds: float = 0.0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(self.env_steps + other.env_steps, self.policy_tokens + other.policy_tokens,
                    self.judge_tokens + other.judge_tokens, self.gpu_seconds + other.gpu_seconds)


def _rollout_cost(rec: dict) -> Cost:
    return Cost(rec.get("env_steps", 0), rec.get("tokens_in", 0) + rec.get("tokens_out", 0),
                rec.get("judge_tokens", 0), rec.get("gpu_seconds", 0.0))


def heal_tail(path: str) -> None:
    """Drop a partial last line left by a killed run, so the next append starts on a clean line."""
    with open(path, "rb+") as f:
        data = f.read()
        if data and not data.endswith(b"\n"):
            f.truncate(data.rfind(b"\n") + 1)


class Trace:
    def __init__(self, run_dir: str, run_id: str | None = None):
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "trace.jsonl")
        self.run_id = run_id or uuid.uuid4().hex[:8]   # every record says which run wrote it
        open(self.path, "a").close()
        heal_tail(self.path)
        self._cost = Cost()
        for rec in self.records("rollout"):
            self._cost += _rollout_cost(rec)

    def _write(self, kind: str, fields: dict) -> None:
        rec = {"kind": kind, "run_id": self.run_id, "iso_ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=_jsonable) + "\n")

    def log_run(self, **fields) -> None:
        self._write("run", fields)

    def log_rollout(self, **fields) -> None:
        # every env step and token, including eval rollouts, is cost: matched-cost comparison depends on it
        self._cost += _rollout_cost(fields)
        self._write("rollout", fields)

    def log_update(self, **fields) -> None:
        self._write("update", fields)

    def log_eval(self, **fields) -> None:
        self._write("eval", fields)

    @property
    def cumulative_cost(self) -> Cost:
        return self._cost

    def records(self, kind: str | None = None) -> list[dict]:
        out = []
        with open(self.path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:        # a run killed mid-write leaves a partial last line
                    continue
                if kind is None or rec.get("kind") == kind:
                    out.append(rec)
        return out

    def success_table(self) -> dict[tuple, tuple[int, int]]:
        """(family, goal) -> (wins, attempts) over train rollouts, in one pass over the trace."""
        table: dict[tuple, list[int]] = {}
        for r in self.records("rollout"):
            if r.get("split") == "train" and "family" in r:
                cell = table.setdefault((r["family"], tuple(r.get("goal", ()))), [0, 0])
                cell[0] += int(r.get("reward", 0) >= 1.0)
                cell[1] += 1
        return {k: (w, n) for k, (w, n) in table.items()}

    def success_rate(self, family: str, goal: tuple) -> float | None:
        """None means never attempted — distinct from always failed, which the curriculum needs to tell apart."""
        wins, n = self.success_table().get((family, tuple(goal)), (0, 0))
        return wins / n if n else None


def _jsonable(x):
    if isinstance(x, Cost):
        return asdict(x)
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (set, tuple)):
        return list(x)
    raise TypeError(f"not JSON serializable: {type(x).__name__}")
