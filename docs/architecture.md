# How nanoVLM-RSI fits together

Five organs, each behind a small interface, wired by one explicit loop in `rsi.py`.
If you read only one file, read `walkthrough.py`: the whole method, generated from the real
source, in the order the loop runs.

```
            ┌──────────── propose.py ────────────┐
            │  fixed  |  self (competence frontier) │
            └──────────────────┬──────────────────┘
                               │ train-split Tasks only
                               ▼
  policy/vlm.py ──act()──► rollout.py ──► envs/ (static | craftax)
        ▲                      │ Trajectory
        │                      ▼
        │                  verify.py  (outcome | judge | composite)
        │                      │ ScoredTrajectory
        │                      ▼
        │              update/ (weights | memory)  ── propose() ──► Update
        │                      │
        └───── commit/revert ── infra/gate.py  (five conditions, gate split only)
                               │
                        infra/trace.py  (every rollout, update, eval — with cost)
                               │
                        evaluate.py  (held-out, frozen)   evals/benchmarks.py (MMStar)
```

## The organs

| Organ | Interface | Implementations |
|---|---|---|
| **Policy** | `act(prompt, images) -> RolloutStep`, `logprobs(prompt, images, token_ids)`, `embed(text, images)`, `snapshot()` / `restore()` | `policy/vlm.py` wraps any HF image-text-to-text model; `policy/lora.py` adds LoRA |
| **Environment** | `reset(task)`, `step(action)`, `tasks(split, n, rng)`, `actions` | `envs/static.py` (numpy, CPU), `envs/craftax.py` (JAX, optional) |
| **Verifier** | `score(trajectory) -> ScoredTrajectory` | `OutcomeVerifier`, `JudgeVerifier`, `CompositeVerifier` in `verify.py` |
| **Update rule (carrier)** | `propose(batch) -> Update \| None`, `commit`, `revert`, `persisted` | `update/weights.py` (GRPO + LoRA), `update/memory.py` (skill library) |
| **Proposer** | `(policy, trace) -> list[Task]` | `fixed`, `self` in `propose.py` |

Exactly one carrier runs per experiment (`--update_rule`). The proposer is independent of it
(`--proposer`), so carrier × proposer is a clean experimental grid.

## What keeps a result honest

**Three splits, by seed hash.** `envs/base.py:split_for_seed` maps `crc32(seed) % 10` to `train`
(0–6), `gate` (7) and `heldout` (8–9). Proposers only ever see `train`; every task they return is
re-stamped by the harness and rejected if it is not train. The gate only sees `gate`. `heldout`
is only used for reporting. `crc32` rather than `hash()`, because Python salts `hash()` per process.

**Frozen held-out.** `tasks("heldout", n, rng)` ignores `rng` and returns the same tasks for a
given `env_seed`, so runs are comparable.

**The gate** (`infra/gate.py`) evaluates on the gate split, commits, reloads the organ from
disk, evaluates again, and reverts unless all five hold:

| Condition | Checked as |
|---|---|
| `proposed` | the update has provenance (trajectories that produced it) |
| `survives_reset` | `updater.persisted(update)`: the change is on disk and actually changed something |
| `gate_gain` | gate-split success after ≥ before |
| `deployment_feedback` | the update's signal source is in `gate_allowed_verdict_sources` |
| `no_regression` | parse-fail rate did not rise by more than `gate_parse_fail_delta` |

`--gate_enabled false` exists for ablations and is recorded on every update record.

**Matched cost.** Every rollout — training, gate and eval — is logged with env steps, prompt and
completion tokens, judge tokens and model time. Compare carriers with `tools/plot.py`, which plots
success against cumulative cost, never against iteration.

## Reading a trace

`runs/<name>/trace.jsonl`, one JSON object per line, every record carrying `run_id`:

| `kind` | One per | Key fields |
|---|---|---|
| `run` | run | `cfg_hash`, `git_sha`, `carrier`, `proposer`, `gate_enabled` |
| `rollout` | episode | `family`, `goal`, `split`, `reward`, `parse_fails`, `tokens_in/out`, `gpu_seconds`, `retrieved_skill_ids`, `judge_verdict` |
| `update` | iteration | `carrier` `role` `horizon` labels, `committed`, `conditions`, `gate_before/after`, `loss`, `discarded_groups`, `proposer_stats` |
| `eval` | eval point | `split` (`heldout` or `mmstar`), `success_rate`, `unseen_seed_rate`, `unseen_goal_rate`, `cost` |

```python
from infra.trace import Trace
tr = Trace("runs/headline")
[(e["cost"]["gpu_seconds"], e["success_rate"]) for e in tr.records("eval") if e["split"] == "heldout"]
```

## Adding an environment

Subclass `Env` and the `SeededTasks` mixin, make each task a pure function of its seed, and
declare which families have a programmatic checker:

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

Then add a branch to `make_env` in `rsi.py`. Keep actions to things a small VLM would naturally
say: the count task uses bare digits because SmolVLM answers "3", not "say_3".

## Adding a carrier

Subclass `update.base.Updater` and register it in `update/__init__.py:make_updater`:

- `propose(batch)` returns an `Update` labelled with its carrier, role and horizon, or `None` when
  the batch carries no signal.
- `commit(update)` applies it **durably** (to disk).
- `revert(update)` undoes it **exactly**. The gate depends on this.
- `persisted(update)` rebuilds the organ from disk and reports whether the change is there and
  actually changed something. A commit that changed nothing must return `False`.

`update/scaffold.py` (prompt self-edit) and `update/verifier.py` (judge self-improvement) are
stubs with their designs in the docstrings — good first carriers to implement.
