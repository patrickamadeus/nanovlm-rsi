# nanoVLM-RSI

The simplest harness for **multimodal recursive self-improvement** research, in plain PyTorch + numpy.

nanoVLM showed that the essential parts of a VLM fit in a handful of readable files. nanoVLM-RSI does the
same for the *loop*: a small VLM policy acts in a verifiable environment, a verifier scores what it did,
an update rule turns that into a durable change, a gate decides whether the change is kept, and a trace
records enough to attribute the result. Every RSI method here is written from scratch so you can read how
it works. This is not a SOTA system. It's a playground where a loop closes on CPU in under a minute and a
new experiment means one config line plus one file.

## The loop

```
propose tasks -> roll out a group per task -> verify -> carrier proposes an update
              -> gate commits or reverts it -> trace everything -> evaluate on held-out
```

`rsi.py` *is* this loop, top to bottom. Read it first.

## Files

| File | Lines | What it is |
|---|---:|---|
| `rsi.py` | 107 | the loop and its CLI |
| `rollout.py` | 48 | one episode: render scaffold, act, step, until done or out of budget |
| `evaluate.py` | 96 | held-out eval with learning disabled; both held-out axes, against cost |
| `config.py` | 43 | every knob, in two dataclasses |
| `envs/base.py` | 55 | `Task`, `Obs`, the 4-method `Env`, process-stable train/gate/heldout split |
| `envs/static.py` | 115 | from-scratch numpy env: `count_objects` (1 step), `gather_order` (multi-step) |
| `policy/vlm.py` | 102 | HF VLM wrapper: `act()`, `logprobs()`, `embed()`, `snapshot()` |
| `policy/prompt.py` | 41 | scaffold as data: slot contract, rendering, strict `ACT:` parsing |
| `verify.py` | 66 | outcome verifier + reward-hacking flags |
| `propose.py` | 32 | task proposer, structurally unable to emit gate/held-out seeds |
| `update/base.py` | 35 | `Update` (carries its own taxonomy labels) and the `Updater` contract |
| `update/memory.py` | 41 | memory carrier: successful trajectories become retrievable skills |
| `infra/memory.py` | 128 | skill store: append-only event log with byte-exact revert |
| `infra/gate.py` | 50 | the five-condition acceptance bar, as code |
| `infra/trace.py` | 88 | append-only JSONL: run, rollout, update and eval records with cumulative cost |
| `play.py` | 42 | watch a policy act, step by step |

## Quickstart

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"            # or: pip install torch numpy transformers pillow safetensors pytest

pytest -q                              # full suite, CPU
python rsi.py --tiny --iters 2         # close a loop on CPU with a tiny random policy (~30s)
python rsi.py --iters 20               # real run: SmolVLM-256M policy, GPU if available
python play.py --tiny                  # watch one episode
python evaluate.py --tiny --n 50       # held-out eval only
```

Every `LoopConfig` field is also a CLI flag (`--update_rule`, `--group_size`, `--gate_enabled false`, ...).

## The knobs that define an experiment

- `update_rule`: **the carrier**, exactly one per run. `memory` works now; `weights` (GRPO + LoRA) and
  `scaffold` (prompt self-edit) raise `NotImplementedError` naming the slice they land in.
- `proposer`: `fixed` now; `self` (competence-frontier curriculum) later. It's independent of the carrier.
- `gate_enabled`: you can turn it off for ablations, and the trace records that you did, so an ungated
  run can't be passed off as gated.

## What makes a result trustworthy here

- **Three-way split by `crc32(seed)`.** `train` 70%, `gate` 10%, `heldout` 20%. Accept/reject decisions
  only ever see `gate`, and `heldout` is only used for reporting. Builtin `hash()` isn't used because
  it's salted per process.
- **Frozen held-out.** 200 tasks drawn once per `env_seed`, so runs are comparable.
- **Two held-out axes reported separately.** *Unseen seed* (new world, known goal) and *unseen goal*
  (compositional generalization).
- **Matched cost.** Every rollout counts, including gate and eval rollouts. Plot success against cost,
  not against iteration.
- **Parse failures are first-class.** If a run "improves" only by learning to emit `ACT:` lines, that
  shows up as `parse_fail_rate`.

## Status

| Slice | | |
|---|---|---|
| 1 | spine + static env + memory carrier, CPU end-to-end | done |
| 2 | weights carrier: GRPO + LoRA from scratch | next |
| 3 | Craftax adapter (JAX quarantined to `envs/craftax.py`) | |
| 4 | judge verifier (separate small VLM) + judge/truth agreement logging | |
| 5 | self-curriculum proposer | |
| 6 | scaffold carrier | |
| 7 | MMStar (full 1,500) forgetting tripwire + `walkthrough.py` | |

## Dependencies

`torch`, `numpy`, `transformers`, `pillow`, `safetensors`. Optional extras: `[craftax]`, `[bench]`, `[wandb]`.

**No framework may own the control flow.** No `trl`, `peft`, `accelerate`, or RL library. `transformers`
loads weights and runs the forward pass, and that's all it does. GRPO, LoRA, advantage estimation, memory
operations, the proposer and the gate are all written here.

## Provenance

- Environment contract follows the reset/step interface of Gym, reduced to four methods.
- Skill memory follows the skill-library idea from Voyager (Wang et al., 2023), reduced to retrieval
  over the policy's own embeddings.
- The gate turns a five-condition operational bar for durable self-improvement into code: the system
  proposed the update, it survives a reset, it gains on held-out data at matched cost, its feedback would
  exist at deployment, and it doesn't regress.
- Design and rationale: `docs/design/spec.md`.
