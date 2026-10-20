"""MMStar (Chen et al., 2024) as a catastrophic-forgetting tripwire — not a headline claim.

LoRA updates on synthetic env tasks can quietly erode general visual competence; this is what
catches it. MMStar is broad and vision-indispensable rather than env-matched: a benchmark that
looked like the env would improve alongside it and detect nothing.

Reported numbers use all 1,500 items. At ~35% accuracy a 500-item subset's 95% interval (about
+-4 points) is wider than typical forgetting drops (2-4 points), so subset runs are for debugging
only and are labelled subset=True in the trace. Decoding is greedy: this measures the policy,
not a sample from it.
"""
import re
from dataclasses import dataclass

import numpy as np

MMSTAR_SIZE = 1500
PROMPT = "{question}\nAnswer with the option's letter from the given choices directly."
CHOICE_RE = re.compile(r"\b([A-D])\b")


@dataclass(frozen=True)
class BenchResult:
    name: str
    n: int
    accuracy: float
    subset: bool
    per_category: dict[str, float]


def load_mmstar(n: int | None = None) -> list[dict]:
    """The val split, in its published order. The HF datasets cache is the frozen local snapshot."""
    from datasets import load_dataset   # the [bench] extra: only needed when the tripwire runs
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return [{"question": r["question"], "answer": r["answer"], "category": r["category"],
             "image": np.asarray(r["image"].convert("RGB"))} for r in ds]


def parse_choice(text: str) -> str | None:
    m = CHOICE_RE.search(text.replace("Answer:", " ").upper())
    return m.group(1) if m else None


def run_mmstar(policy, rows: list[dict]) -> BenchResult:
    hits: dict[str, list[float]] = {}
    for r in rows:
        out = policy.act(PROMPT.format(question=r["question"]), [r["image"]], prefix="Answer:", greedy=True)
        hits.setdefault(r["category"], []).append(float(parse_choice(out.text) == r["answer"]))
    every = [h for hs in hits.values() for h in hs]
    return BenchResult("mmstar", len(every), float(np.mean(every)) if every else 0.0, len(every) < MMSTAR_SIZE,
                       {c: float(np.mean(hs)) for c, hs in sorted(hits.items())})


def log_bench(trace, result: BenchResult, when: str) -> None:
    trace.log_eval(split=result.name, n=result.n, success_rate=result.accuracy, bench_subset=result.subset,
                   per_category=result.per_category, when=when, cost=trace.cumulative_cost)
