"""The Cauldron as verifiable single-answer tasks: real images, real questions, a reference answer.

patrickamadeus/the_cauldron is image + multi-turn Q&A from ~50 VQA sources. The loop needs a reward it
can check, so a row becomes a task only through a turn whose answer exact match can verify: a letter
(multiple choice), yes/no, or a whole number 0-20 ("Three." -> 3). One task per image — its first such
turn, seeded by the row index — so an image never sits in two splits. About 1 in 4 turns qualifies;
captions and free-form answers are left out rather than handed to a weaker judge.

    pip install -e ".[cauldron]"
    python train.py --config cauldron_grpo     # downloads the 1.7 GB sample_1pct split on first use
"""
import glob
import io
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

from envs.base import Env, Obs, Task, split_for_seed

WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()
# "Yes." is a right answer, not a format error: both spellings are legal and scored the same
ACTIONS = {"mc": list("ABCDEFGH"), "yn": ["yes", "no", "Yes", "No"], "num": [str(i) for i in range(21)]}


def answer_key(text: str) -> tuple[str, str] | None:
    """(kind, normalized answer) when exact match can check it, else None."""
    a = text.strip()
    if m := re.fullmatch(r"(?:[Aa]nswer:\s*)?\(?([A-H])\)?\.?", a):
        return "mc", m.group(1)
    if m := re.fullmatch(r"(?i)(yes|no)\.?", a):
        return "yn", m.group(1).lower()
    if (m := re.fullmatch(r"(\d{1,2})\.?", a)) and int(m.group(1)) <= 20:
        return "num", str(int(m.group(1)))
    if m := re.fullmatch(r"(?i)(%s)\.?" % "|".join(WORDS), a):
        return "num", str(WORDS.index(m.group(1).lower()))
    return None


@dataclass(frozen=True)
class Item:
    seed: int              # the row index: fixes the split and finds the image
    source: str
    question: str
    kind: str              # "mc" | "yn" | "num"
    answer: str


def items_from_rows(rows, sources: tuple[str, ...] = ()) -> list[Item]:
    items = []
    for i, row in enumerate(rows):
        if row["num_imgs"] != 1 or (sources and row["source"] not in sources):
            continue
        for turn in row["texts"]:
            if key := answer_key(turn["assistant"]):
                items.append(Item(i, row["source"], turn["user"].strip(), *key))
                break
    return items


def load(repo: str, subset: str, revision: str, sources: tuple[str, ...] = ()):
    """(items, image_of) from the subset's train parquet. Images stay memory-mapped until a task needs one."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo, repo_type="dataset", revision=revision, allow_patterns=f"{subset}/train-*.parquet")
    files = sorted(glob.glob(f"{root}/{subset}/train-*.parquet"))
    rows = pa.concat_tables([pq.read_table(f, columns=["texts", "source", "num_imgs"]) for f in files]).to_pylist()
    images = pa.concat_tables([pq.read_table(f, columns=["images"], memory_map=True) for f in files])["images"]

    def image_of(seed: int) -> np.ndarray:
        return np.asarray(Image.open(io.BytesIO(images[seed].as_py()[0]["bytes"])).convert("RGB"))
    return items_from_rows(rows, sources), image_of


class CauldronEnv(Env):
    def __init__(self, items: list[Item], image_of, env_seed: int = 0, heldout_n: int = 200):
        self.items, self.image_of, self.heldout_n = items, image_of, heldout_n
        self._by_seed = {i.seed: i for i in items}
        self._pools = {s: [i for i in items if split_for_seed(i.seed) == s] for s in ("train", "gate", "heldout")}
        held = self._pools["heldout"]                     # frozen: one shuffle per env_seed, ignores the caller's rng
        self._heldout = [held[j] for j in np.random.default_rng(env_seed).permutation(len(held))[:heldout_n]]
        self._item, self._obs = None, None

    def task(self, item: Item) -> Task:
        return Task(item.source, item.seed, (item.answer,), 1, split_for_seed(item.seed))

    def tasks(self, split: str, n: int, rng: np.random.Generator) -> list[Task]:
        if split == "heldout":
            if n > len(self._heldout):
                raise ValueError(f"asked for {n} held-out tasks, only {len(self._heldout)} are frozen")
            return [self.task(i) for i in self._heldout[:n]]
        pool = self._pools[split]
        return [self.task(pool[j]) for j in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]

    @property
    def actions(self) -> list[str]:
        return ACTIONS[self._item.kind] if self._item else sorted({a for v in ACTIONS.values() for a in v})

    def reset(self, task: Task) -> Obs:
        self._item = self._by_seed[task.seed]
        self._obs = Obs(self.image_of(task.seed), self._item.question)
        return self._obs

    def step(self, action: str) -> tuple[Obs, float, bool, dict]:
        return self._obs, float(action.lower() == self._item.answer.lower()), True, {"steps": 1}
