# nanoVLM-RSI — Design Spec

Date: 2026-10-20
Status: approved design, pending implementation plan

## 1. Purpose

nanoVLM-RSI is a minimal, readable harness for **multimodal recursive
self-improvement (RSI) research**. It is a substrate: it carries whichever
self-improvement loop a researcher wants to study, rather than implementing one
loop as a product.

It makes the same contract nanoVLM made for vision-language models — *here are
the essential parts, each in one file sized to its idea, implemented from
scratch so you can see the mechanism* — but the object of study is a **loop**,
not a model.

Two audiences, in priority order:

1. The author, who learns RSI by implementing its methods from scratch.
2. Other researchers, who get a compute-efficient playground where a
   lightweight policy loads immediately and a new experiment is a config plus
   one file.

### Non-goals

- Not a SOTA result. Like nanoVLM, credibility comes from one checkable claim,
  not a leaderboard.
- Not a framework. No plugin registry, no dependency injection, no
  config-as-DSL. Generality comes from interfaces being *small*.
- Not domain-specific. No feature exists only to make one benchmark number go
  up.
- Not a general agent platform. Tool calling and web access exist only insofar
  as the loop needs them.

## 2. The claim the stack is built to support

**In-domain held-out improvement.** A policy self-improves on tasks sampled
from a family, and measurably improves on *held-out tasks from that same
family*, at matched cost.

External multimodal benchmarks are a **regression tripwire**, not a headline
claim. Transfer from sandbox RSI to external benchmarks is explicitly out of
scope as a claim, though the stack does not prevent studying it.

Two held-out axes are measured and reported **separately**, because conflating
them is the standard way agentic RSI results get oversold:

- **Unseen seed, trained goal** — new world, known skill.
- **Unseen goal, same family** — compositional generalization.

## 3. Settled decisions

| Decision | Choice |
|---|---|
| Success claim | In-domain held-out improvement at matched cost |
| Environments | Craftax (JAX, quarantined) + from-scratch static verifiable tasks |
| Action interface | Short reasoning + `ACT: <action>` string; sequence-level GRPO |
| Carriers | `weights \| memory \| scaffold`, **exactly one per run** |
| Verifier self-improvement | Stub: interface present, training data collected, update not implemented |
| Proposer | Orthogonal knob: `fixed \| self` |
| Policy | HF-wrapped small VLM; every RSI method from scratch in PyTorch + numpy |
| Weight update | LoRA by default (written from scratch), `full` via config |
| Judge | Separate small frozen VLM, not the policy's own weights |
| Held-out set | Frozen: 200 tasks drawn once per `env_seed`, reused across runs |
| Splits | Three-way: `train` / `gate` / `heldout` |
| Architecture | Flat organs, explicit loop, dataclass config |

### Rationale for the load-bearing ones

**One carrier per run.** Simultaneous commits would need arbitration logic and
would make attribution ambiguous. One carrier per run means carrier comparison
falls out of running each mode under the same budget, and no organ needs to
know about another.

**Three-way split.** If `gate()` consults held-out data to accept or reject an
update, held-out has silently become a validation set and the headline number is
contaminated. `gate` is a separate split touched by accept/reject decisions;
`heldout` is touched only by reporting. This costs ~10% of task space and is
impossible to retrofit into already-published results.

**Scaffold as data, not code.** Because `update/scaffold.py` must rewrite the
scaffold, the prompt cannot be an f-string in Python. It is a template file with
named slots, making self-edit diffable, versionable and revertible — and
avoiding the sandbox requirement that would otherwise make this the slice that
never lands.

**Separate judge VLM.** A judge sharing the policy's weights shares its blind
spots, which is precisely the failure the judge exists to catch.

**LoRA default.** The whole loop runs on one 24GB card, which is what makes the
playground playable. Accepted cost: LoRA caps how far the weights carrier can
move, tilting a weights-vs-memory comparison slightly toward memory. Runs
intended for carrier comparison should state which mode was used.

## 4. Architecture

```
rsi.py          # THE loop, ~200 lines — read this first
evaluate.py     # held-out eval + benchmark regression
play.py         # watch a policy act, no learning (debug/demo)
config.py       # PolicyConfig + LoopConfig dataclasses

policy/vlm.py      # ~100 lines: HF wrapper -> act(), logprobs()
policy/prompt.py   # scaffold rendering, ACT: parsing, memory injection

envs/base.py       # Env ABC + Task + Obs
envs/static.py     # from-scratch numpy procedural verifiable tasks
envs/craftax.py    # JAX quarantine, optional extra

verify.py          # OutcomeVerifier, JudgeVerifier, CompositeVerifier
propose.py         # fixed sampler + self-curriculum

update/base.py     # Updater ABC + Update
update/weights.py  # GRPO + LoRA, from scratch
update/memory.py   # skill/episodic library write, retrieve, expire
update/scaffold.py # prompt self-edit + slot-contract validation
update/verifier.py # stub

infra/memory.py    # the store (JSONL + retrieval)
infra/tools.py     # tool calling
infra/trace.py     # attribution/logging schema
infra/gate.py      # the acceptance bar

scaffolds/v0.txt   # versioned prompt templates
evals/benchmarks.py
tests/
```

### The loop

```python
# rsi.py
def rsi(cfg):
    policy   = Policy(cfg.policy)
    env      = make_env(cfg.env)
    verifier = make_verifier(cfg.verify)
    updater  = make_updater(cfg.update_rule, policy)   # weights | memory | scaffold
    propose  = make_proposer(cfg.proposer, env)
    trace    = Trace(cfg.run_dir)

    for it in range(cfg.iters):
        tasks = propose(policy, trace)                 # train split only
        batch = []
        for task in tasks:
            for _ in range(cfg.group_size):            # GRPO needs a group
                traj = rollout(policy, env, task, cfg)
                batch.append(verifier.score(traj))

        update = updater.propose(batch)
        if update is not None:
            if gate(update, policy, env, trace):
                updater.commit(update)
            else:
                updater.revert(update)
        trace.log(it, batch, update)

        if it % cfg.eval_every == 0:
            evaluate(policy, env.tasks("heldout", cfg.eval_n, rng), trace)

    evaluate(policy, env.tasks("heldout", cfg.eval_n, rng), trace, final=True)
```

Swapping carrier changes one config line. `gate()` is the single place the
acceptance bar is enforced.

### Rejected alternatives

- **Protocol + registry plugin system.** More extensible, but opening `rsi.py`
  would reveal a wiring step instead of a loop. Readability is the product.
  Named constructors (`make_*`) are the seam if this is ever needed.
- **Single-file `nano_rsi.py`.** Maximally readable top-to-bottom, but three
  carriers and two envs in one file means every experiment edits the same file,
  defeating fork-per-paper. Retained instead as a *derived* teaching artifact
  (`walkthrough.py`) generated from the real modules.

## 5. Environment and task interface

```python
@dataclass(frozen=True)
class Task:
    family: str          # "fetch_craft" | "count_objects" | ...
    seed: int            # drives procedural generation, fully reproducible
    goal: tuple[str,...] # goal spec the verifier reads
    budget: int          # max env steps — part of matched-cost accounting
    split: str           # "train" | "gate" | "heldout" — set by harness only

@dataclass(frozen=True)
class Obs:
    rgb: np.ndarray      # [H,W,3] — mandatory, policy is genuinely multimodal
    text: str | None

class Env(ABC):
    def reset(self, task: Task) -> Obs: ...
    def step(self, action: str) -> tuple[Obs, float, bool, dict]: ...
    def tasks(self, split: str, n: int, rng) -> list[Task]: ...
    @property
    def actions(self) -> list[str]: ...   # legal action strings, for the prompt
```

**Split enforcement is structural.** `tasks()` derives split from
`hash(seed) % 10`: buckets 0–6 `train`, bucket 7 `gate`, buckets 8–9 `heldout`.
A proposer is only ever handed `split="train"`; any Task it constructs is
re-stamped by the harness, and anything hashing into gate/heldout is rejected
and resampled. This is a code property with a dedicated test, because a
self-improving system has gradient toward proposing its own eval set.

**Held-out is frozen:** 200 tasks drawn once per `env_seed` and reused, chosen
for cross-run comparability. Fresh-sample numbers may be reported as a
secondary check.

**Craftax mapping.** `Task = (world seed, target achievement subset, step
budget)`. `step()` maps the action string to a discrete index, calls the jitted
step, and converts pixel obs `jax -> numpy` at the boundary. JAX is imported
nowhere else, and `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set in this module so
JAX and PyTorch can share a GPU.

**Static env mapping.** Seeds drive a procedural scene (objects, counts, spatial
relations) rendered to RGB by ~80 lines of numpy. Goals are exactly checkable.
Runs on CPU in seconds; serves as CI and as proof the Env interface is general.

## 6. Policy and scaffold

```python
class Policy:
    def act(self, prompt, images) -> RolloutStep:
        """Sample a generation; returns text, token_ids, per-token logprobs."""
    def logprobs(self, prompt, images, token_ids) -> torch.Tensor:
        """Re-score an existing generation under current params — needed for the
           GRPO ratio and the KL term."""
```

~100 lines over an HF VLM. `transformers` owns weight loading and the forward
pass; it owns nothing in the control flow.

**Scaffold template** (`scaffolds/v0.txt`), slots in braces:

```
{system}
Actions available: {actions}
{memory}
Observation: <image>
{task}
Think briefly, then emit one line: ACT: <action>
```

`policy/prompt.py` renders and parses. `{memory}` is empty when the carrier is
not `memory`.

**Parse failures are first-class.** A generation with no valid `ACT:` line
becomes a no-op step with a small negative shaping term, counted as
`parse_fail_rate`. This is a load-bearing diagnostic: a run that "improves"
mainly by learning to emit parseable actions is a result about formatting, and
must be visible in logs rather than discovered in a plot.

**Two frozen artifacts:** a `pi_old` snapshot per iteration (importance ratio)
and a `pi_init` frozen copy (KL-to-init, coefficient configurable, 0 allowed).

## 7. Verifier

```python
class Verifier(ABC):
    def score(self, traj) -> ScoredTrajectory:
        """reward, per_step_credit|None, verdict_source, confidence, flags"""
```

- **`OutcomeVerifier`** — env ground truth: achievements unlocked, exact answer
  match. Binary success plus optional partial credit from achievement subsets.
- **`JudgeVerifier`** — separate small frozen VLM, prompted with (goal, final
  obs, trajectory summary). Required because a self-curriculum proposes goals
  the env has no programmatic checker for.
- **`CompositeVerifier`** — routes env-native goals to outcome, proposed goals
  to judge.

**Agreement logging.** On env-native tasks, where truth is available, the judge
also runs at sample rate `rho` (default 0.1) and agreement is logged. This
accumulates the disagreement dataset a verifier-updater would train on, so
`update/verifier.py` is a stub with its data already waiting.

**Sparse reward is a known constraint.** Binary success over a group frequently
yields an all-zero group, zero advantage, no gradient. Mitigations: partial
credit from achievement subsets; discard all-equal groups and log the discard
rate; and the proposer selecting tasks at the competence frontier. **The
self-curriculum is therefore not optional for sample-efficient GRPO** — a
`fixed` proposer with sparse rewards will under-perform for reasons unrelated to
the update rule, and this must be stated in any comparison.

Hack flags per trajectory: success in 0 steps, degenerate action loops, no-op
spam.

## 8. Update rules

```python
class Updater(ABC):
    def propose(self, batch) -> Update | None: ...
    def commit(self, update) -> None: ...
    def revert(self, update) -> None: ...
```

`Update` carries its own taxonomy labels — carrier (A3/A4/A5), role (B0–B4),
horizon (C), clock (D), signal (E) — plus payload, cost, and provenance (the
trajectories that produced it). Labels live on the object, not in prose.

### weights.py — GRPO from scratch (~100 lines)

- Group-relative advantage `A = (r - mean) / (std + eps)` within a task group.
- Clipped per-token objective, masked to completion tokens.
- k3 KL estimator (`ratio - log ratio - 1`) against `pi_init`, coefficient
  `beta`.
- Per-token mean normalization, not per-sequence, to avoid length bias.
- One inner epoch by default, keeping the loop on-policy.
- LoRA from scratch: `LoRALinear` wrapping the LM's q/v projections; vision
  tower frozen. LoRA state is ~10MB, so a pre-update snapshot is free, making
  `revert()` exact and cheap — this is what lets the gate be real rather than
  advisory.

### memory.py — skill library (~120 lines)

- Append-only JSONL store.
- Write on success above threshold; dedupe by similarity; consolidate
  near-duplicates keeping best reward.
- Retrieve top-k by cosine over embeddings taken from **the policy VLM's own
  pooled features**, so no extra model joins the dependency floor. Brute-force
  numpy; the library is small.
- Expire by usefulness, tracking retrieval count and whether retrieval preceded
  success.
- `revert()` is truncation to a byte offset.
- **Use-attribution:** every rollout logs which skill IDs were in context.
  Without this one can show a skill was *retrieved* but not that any decision
  *depended* on it, and present-vs-used is what separates a real memory result
  from a decorative one.

### scaffold.py — prompt self-edit (~120 lines)

- A meta-prompt gives the policy its current template plus sampled failures and
  asks for a rewrite.
- A validator enforces the slot contract (`{actions}`, `{memory}`, `{task}`, the
  `ACT:` line) and rejects malformed proposals before gating.
- Versioned `scaffolds/v0.txt, v1.txt, …`; commit writes a new file and repoints
  config; revert repoints back.

### verifier.py — stub

Interface plus `NotImplementedError`, with a docstring naming the trace fields
that feed it.

## 9. gate() — the acceptance bar

`infra/gate.py`, ~60 lines. Each condition logged pass/fail:

| Condition | Mechanized as |
|---|---|
| System materially proposed it | `update.provenance` non-empty, recorded |
| Survives reset | **Reload probe**: re-instantiate the organ from disk, confirm the update is present |
| Held-out gain at matched cost | Eval on the **gate split** before/after; require delta >= 0 |
| Feedback would exist at deployment | `verdict_source` in the config-declared allowed set |
| Regression and safety | `parse_fail_rate` not worse by more than `delta`; no crash; budgets respected; slot contract valid |

Failure triggers `revert()` and logs the reason. `gate.enabled=false` is
permitted for ablations **and recorded in the trace**, so ungated results cannot
later be reported as gated.

## 10. Proposer

- **`fixed`** — uniform sample from train split.
- **`self`** — competence-frontier curriculum. Maintain running success rate per
  (family, goal) from the trace; sample where estimated success is in
  [0.2, 0.8]; epsilon-explore unseen families. Optionally the policy generates
  goal specs in text, validated against the env's goal grammar, invalid
  rejected, novel goals routed to the judge.

Logged: proposer entropy and family coverage, so curriculum collapse is visible
rather than inferred.

## 11. Trace and evaluation

Append-only JSONL, four record types:

- **run** — config hash, git sha, seeds, carrier, proposer, gate settings.
- **rollout** — task id/family/seed/split, tokens in/out, env steps, reward,
  verdict source, judge verdict if sampled, parse_fail, retrieved skill IDs,
  wall ms.
- **update** — all five taxonomy labels, cost, per-condition gate results,
  committed bool, revert reason.
- **eval** — split, n, success rate, both held-out axes separately, cumulative
  cost at that point.

**Matched cost is the spine.** Every eval point records cumulative env steps,
policy tokens, judge tokens, and GPU-seconds. Plots are **success versus cost,
not success versus iteration**: a memory update costs almost no GPU time while
GRPO costs a great deal, so iteration-indexed curves would make memory look slow
and weights look efficient purely as an artifact of the x-axis.

`evaluate.py` runs a split with learning disabled and prints the table.
`--benchmarks` runs `evals/benchmarks.py`, a small exact-match multimodal subset
that is explicitly a regression tripwire, not a headline claim.

## 12. Tests

| Test | Protects |
|---|---|
| Split leak | Proposer can never emit a gate/heldout seed |
| GRPO gradient sign | Analytic toy case with known advantages |
| Advantage normalization | Group-relative math |
| Memory revert | Byte-identical state after truncation |
| Scaffold slot contract | Malformed self-edits rejected |
| Reload probe | Updates genuinely persist to disk |
| End-to-end smoke | Static env, tiny policy, CPU, under 60s, CI-able |

## 13. Dependencies

Floor: `torch`, `numpy`, `transformers`, `pillow`, `safetensors`.
Extras: `[craftax]` -> `jax`, `craftax`; `[wandb]`.

**Enforced socially, as nanoVLM does:** no framework may own the control flow.
No `trl`, no `peft`, no `accelerate`, no RL library. GRPO, LoRA, advantage
estimation, memory operations, the proposer and the scaffold editor are written
from scratch — that is the point of the project, not an incidental constraint.

## 14. Implementation staging

The spine is defined completely; carriers land as slices against stable
interfaces.

1. **Spine + static env + memory carrier.** config, Task/Env/Obs, static env,
   policy wrapper, prompt/scaffold, OutcomeVerifier, trace, gate, evaluate,
   `rsi.py`. End-to-end on CPU. Proves the plumbing before any GPU cost.
2. **Weights carrier.** LoRA + GRPO from scratch, static env, one GPU.
3. **Craftax adapter.** JAX quarantine, achievement-based tasks.
4. **Judge verifier** + agreement logging.
5. **Self-curriculum proposer.**
6. **Scaffold carrier.**
7. **Benchmark regression** + `walkthrough.py` teaching artifact.

Each slice ends with the loop still runnable and its tests passing.

## 15. Success criteria for the project

1. `rsi.py` is readable top-to-bottom by someone who has not seen the repo.
2. Every RSI method is from-scratch PyTorch/numpy with no framework in the
   control flow.
3. One config line switches carrier; one switches proposer.
4. A run produces a trace from which carrier, role, horizon, clock, signal,
   cost and gate outcomes can all be read without consulting the code.
5. Held-out improvement can be demonstrated on at least one env/carrier pair,
   reported against cost, with both held-out axes separated.
6. A researcher can clone, install, and close a loop on CPU in under ten
   minutes.
