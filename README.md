# nanoVLM-RSI

The simplest repository for **multimodal recursive self-improvement** research, in plain PyTorch + numpy.

nanoVLM showed that the essential parts of a vision-language model fit in a handful of readable
files. nanoVLM-RSI does the same for the *self-improvement loop*. A small VLM acts in a verifiable
environment. A verifier scores what it did. An update rule (GRPO on LoRA, or a skill library)
turns that into a durable change. A gate keeps the change only if it helps on data the loop never
trains on. A trace records enough to attribute the result.

Every method is written from scratch, so you can read how it works: GRPO, LoRA, advantage
estimation, the competence-frontier curriculum, the skill store, the gate. `transformers` loads
weights and runs the forward pass, and nothing else. This is not a SOTA system. It's a playground
where a loop closes on CPU in under a minute, and a new experiment is one config flag plus one file.

## Result

GRPO, written from scratch, teaches SmolVLM-256M to count objects in images it has never seen.
The run took 77 minutes on one A40.

![held-out success vs GPU seconds](assets/headline.png)

| | before | after 100 iterations |
|---|---:|---:|
| **held-out success** (200 frozen tasks, new images) | **3.0%** | **51.5%** |
| parse-failure rate | 38.0% | 0.5% |
| success when the answer parsed | 4.8% | 51.8% |
| **MMStar** (all 1,500; forgetting tripwire) | 25.9% | 28.5% |

Setup: LoRA r=8, self-curriculum, `count_objects` only, 8 tasks × 8 rollouts per iteration. 59 of
the 100 proposed updates passed the gate; the other 41 were reverted. Total cost: 13,772 env steps,
2.6M policy tokens, 2,936 GPU-seconds of model time. The full trace is in
`results/headline_trace.jsonl.gz`.

Read it honestly:
- **Part of the early gain is format.** The parse-failure rate fell from 38% to 0.5%. Success
  *among parsed answers* rose from 4.8% to 51.8%, so the counting itself improved too. (Chance on
  1–9 is about 11%.)
- **No forgetting detected.** MMStar moved +2.5 points, which is within the noise of a 1,500-item
  benchmark. It's measured with one 512px tile and a plain prompt, so it sits below SmolVLM's
  published number; baseline and final use identical settings.
- **One seed, one task family, no ablations yet.** All nine count goals are seen in training, so
  this is the unseen-*image* axis, not unseen-goal generalization.

Reproduce:

```bash
python rsi.py --update_rule weights --proposer self --static_families count_objects \
    --iters 100 --tasks_per_iter 8 --group_size 8 --eval_every 10 --eval_n 200 --gate_n 32 \
    --bench true --run_dir runs/headline
```

## The loop

```
propose tasks ─► roll out a group per task ─► verify ─► carrier proposes an update
              ─► gate commits or reverts it ─► trace everything ─► evaluate on held-out
```

`rsi.py` *is* this loop. `walkthrough.py` is the whole method in one file, in the order the loop
runs, generated from the real source.

## What's in the box

| File | Lines | What it is |
|---|---:|---|
| `rsi.py` | 138 | the loop and its CLI: every config field is a flag |
| `rollout.py` | 74 | one episode: render the prompt, act, step, until done or out of budget |
| `policy/vlm.py` | 169 | any HF VLM behind `act()` / `logprobs()` / `embed()`; chat template, prefilled `ACT:` |
| `policy/lora.py` | 53 | LoRA from scratch; switching adapters off recovers π_init exactly |
| `update/weights.py` | 116 | GRPO from scratch: group advantages, clipped token-level loss, k3 KL |
| `update/memory.py` | 45 | skill-library carrier: successful episodes become retrievable skills |
| `infra/memory.py` | 134 | skill store: an append-only event log with byte-exact revert |
| `infra/gate.py` | 50 | the acceptance bar: five conditions, gate split only |
| `infra/trace.py` | 108 | JSONL trace: every rollout, update and eval, with cumulative cost |
| `verify.py` | 134 | ground truth, VLM-as-judge, and routing with judge-vs-truth agreement logging |
| `propose.py` | 64 | fixed sampling, or a curriculum at the competence frontier |
| `envs/static.py` | 102 | from-scratch numpy env: count objects, gather in order. CPU, seconds |
| `envs/craftax.py` | 73 | Craftax-Classic adapter, the only file that imports JAX (optional) |
| `evaluate.py` | 103 | held-out eval with learning off: unseen-seed and unseen-goal axes, separately |
| `evals/benchmarks.py` | 58 | MMStar (all 1,500) as a catastrophic-forgetting tripwire |

## Quickstart

```bash
git clone https://github.com/patrickai0309/nanovlm-rsi && cd nanovlm-rsi
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"                  # or: pip install -e ".[dev]"

pytest -q                                   # full suite, CPU
python rsi.py --tiny --iters 2              # close a loop on CPU with a tiny random policy (~30 s)
python play.py --tiny                       # watch one episode, step by step
```

Or open `quickstart.ipynb` in Colab.

**The real run** (SmolVLM-256M, one GPU):

```bash
python rsi.py --update_rule weights --proposer self --static_families count_objects \
    --iters 100 --tasks_per_iter 8 --group_size 8 --eval_every 10 --gate_n 32 \
    --bench true --run_dir runs/headline     # --bench needs: pip install -e ".[bench]"

python tools/plot.py runs/headline --x gpu_seconds --out headline.png     # needs ".[plot]"
python evaluate.py --split heldout          # held-out only, learning off
python evaluate.py --benchmarks             # MMStar only
```

Open-world: `pip install -e ".[craftax]"`, then `--env craftax`.

## The knobs that define an experiment

| Flag | Values | What it changes |
|---|---|---|
| `--update_rule` | `weights` · `memory` | **the carrier**: exactly one per run. `weights` = GRPO on LoRA (`--grpo_mode full` for all LM weights), `memory` = skill library |
| `--proposer` | `fixed` · `self` | uniform sampling, or the competence-frontier curriculum |
| `--verifier` | `outcome` · `judge` · `composite` | ground truth, a separate frozen VLM judge, or ground truth shadowed by the judge (agreement logging) |
| `--env` | `static` · `craftax` | procedural verifiable tasks, or open-world Craftax |
| `--gate_enabled` | `true` · `false` | turn the gate off for ablations; the trace records that you did |

Carrier × proposer × verifier is a clean grid: each axis changes one component.

## What makes a result here trustworthy

- **Three splits by seed hash**: train 70%, gate 10%, held-out 20%. Proposers only ever see train,
  and this is enforced in code. Accept/reject decisions see only gate. Held-out is only used for
  reporting, and it's frozen per `env_seed`.
- **The gate** keeps an update only if it is on disk, actually changed something, didn't hurt the
  gate split, came from a signal that would exist at deployment, and didn't raise the parse-failure
  rate.
- **Matched cost**: every rollout counts, including gate and eval ones. Plot success against
  cost, never against iteration.
- **Two held-out axes** reported separately: unseen seed (new image, known goal) and unseen goal.
- **A forgetting tripwire**: full MMStar before and after, policy only.

See [`docs/architecture.md`](docs/architecture.md) for every interface and for how to add your own
environment or carrier.

## Hardware

The default policy is `HuggingFaceTB/SmolVLM-256M-Instruct` in bf16 with LoRA r=8 on the language
model's q/v projections. The vision tower is frozen. The adapters train in fp32. One action takes
about 0.16 s on an A40. Other HF image-text-to-text models load via `PolicyConfig.model_id`, but LoRA currently targets SmolVLM/Idefics3 module names (`text_model`), so other families need the LoRA scope adjusted (`policy/lora.py:apply_lora`).

## Status

| | |
|---|---|
| GRPO + LoRA carrier, skill-library carrier | ✅ |
| static env, Craftax adapter | ✅ |
| outcome / judge / composite verifiers, agreement logging | ✅ |
| fixed and self-curriculum proposers | ✅ |
| gate, trace, held-out eval, MMStar tripwire | ✅ |
| scaffold self-edit carrier (`update/scaffold.py`) | stub, design in its docstring |
| verifier self-improvement (`update/verifier.py`) | stub; its training data is already logged |

Known limitations: rollouts in a GRPO group are generated one at a time (batching would be a
large speedup), and memory retrieval is keyed on text only, so it can't tell apart tasks that
share a prompt.

## Contributing

Same rule as nanoVLM: **no framework may own the control flow**. No `trl`, `peft`, `accelerate`
or RL libraries. New carriers, environments and verifiers are very welcome. Each one should be one
file behind the existing interface, with tests.

## Citation

```bibtex
@misc{irawan2026nanovlmrsi,
  author       = {Patrick Amadeus Irawan},
  title        = {nanoVLM-RSI: The simplest harness for multimodal recursive self-improvement},
  year         = {2026},
  howpublished = {\url{https://github.com/patrickai0309/nanovlm-rsi}}
}
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
