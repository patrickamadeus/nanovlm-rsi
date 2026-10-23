# How picoRSI fits together

`train.py` is the whole loop. Every other folder is one part of it, and each part has one small
contract. Only `train.py` and `evaluate.py` wire parts together; `generate.py` is a debug view.

```
rsi/curriculum.py ──train-split tasks──► harness/rollout.py ──Trajectory──► judges/
        ▲                                   │ act() each step                │ ScoredTrajectory
        │ success table                     ▼                                ▼
   harness/log.py ◄────── every record ── models/smolvlm.py ◄── commit ── rsi/grpo.py | rsi/memory.py
                                                         ▲                  │ Update
                                                         └── revert ── rsi/gate.py (gate split only)
```

## Contracts

| Part | Methods | Files |
|---|---|---|
| model | `act(prompt, images) -> RolloutStep`, `logprobs(prompt, images, token_ids)`, `embed(text, images)`, `snapshot()` / `restore()` | `models/smolvlm.py`, `models/lora.py` |
| env | `reset(task) -> Obs`, `step(action) -> (Obs, reward, done, info)`, `tasks(split, n, rng)`, `actions` | `envs/static.py`, `envs/craftax.py` |
| judge | `score(traj) -> ScoredTrajectory` | `judges/outcome.py`, `judges/vlm.py` |
| update | `propose(batch) -> Update \| None`, `commit`, `revert`, `persisted` | `rsi/grpo.py`, `rsi/memory.py` |
| curriculum | `(policy, log) -> list[Task]` | `rsi/curriculum.py` |

## Splits

`envs/base.py:split_for_seed` maps `crc32(seed) % 10` to train (0–6), gate (7) and held-out
(8–9). `crc32`, not `hash()`, because Python salts `hash()` per process. The curriculum's tasks
are re-stamped and rejected if not train. Held-out is drawn once per `env_seed` and frozen.

## The gate

For each proposed update, on the same 32 gate tasks: evaluate, commit, evaluate again, and revert
unless all five hold — `proposed` (it has provenance), `survives_reset` (the change reloads from
disk and actually changed something), `gate_gain` (success did not drop), `deployment_feedback`
(the signal is in `gate_allowed_signals`), `no_regression` (parse failures rose by at most
`gate_parse_fail_delta`). `gate_enabled: false` exists for ablations and is logged on every update.

Known weakness: both gate evals sample at temperature 1 on 32 tasks, so small gains are
indistinguishable from noise. Greedy gate evals or a margin would make it stricter.

## Reading a log

`runs/<name>/trace.jsonl`, one JSON object per line:

| `kind` | One per | Key fields |
|---|---|---|
| `run` | run | `config` (all of it), `cfg_hash`, `git_sha`, `update`, `curriculum`, `judge`, `env` |
| `rollout` | episode | `family`, `goal`, `split`, `reward`, `parse_fails`, `tokens_in/out`, `gpu_seconds`, `judge_verdict` |
| `update` | iteration | `mode`, `committed`, `conditions`, `gate_before/after`, `loss`, `curriculum_stats` |
| `eval` | eval point | `split`, `success_rate`, `unseen_seed_rate`, `unseen_goal_rate`, `cost` |

```python
from harness.log import Trace
[(e["cost"]["gpu_seconds"], e["success_rate"]) for e in Trace("runs/count_grpo").records("eval")]
```

## Adding an environment

Subclass `Env` and the `SeededTasks` mixin, make each task a pure function of its seed, then add one
branch to `envs/__init__.py:make_env`:

```python
from envs.base import Env, Obs, SeededTasks, Task, split_for_seed

class MyEnv(SeededTasks, Env):
    families = {"my_family"}

    def __init__(self, env_seed=0, heldout_n=200):
        self.env_seed, self.heldout_n = env_seed, heldout_n

    def make_task(self, seed):            # everything about the task comes from the seed
        return Task("my_family", seed, goal=("...",), budget=8, split=split_for_seed(seed))

    @property
    def actions(self): return ["a", "b"]  # strings the policy emits after ACT:

    def reset(self, task): ...            # -> Obs(rgb=uint8 [H,W,3], text="what the agent is told")
    def step(self, action): ...           # -> (Obs, reward, done, {"steps": t})
```

Keep actions to things a small VLM would naturally say: the count task uses bare digits because
SmolVLM answers "3", not "say_3".

## Adding an update mode

A class with the four methods in the table, registered in `rsi/__init__.py:make_update`. `commit`
must write to disk, `revert` must undo exactly, and `persisted` must return `False` for a commit
that changed nothing — the gate relies on all three.
