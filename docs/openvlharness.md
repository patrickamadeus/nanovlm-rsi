# Running picoRSI through OpenVLHarness

The `complex` branch can swap our 80-line rollout loop (`harness/rollout.py`) for
[OpenVLHarness](https://github.com/uclanlp/OpenVLHarness): a served VLM with tools (zoom, Python,
and optionally grounding, OCR, depth, search) and a multimodal memory. Everything else — envs,
judges, the memory update, the curriculum, the gate and the log — is unchanged.

**What works:** single-answer tasks (`count_objects`) with prompt-level updates (`update: memory`).
**Not yet:** GRPO (a served 2B model's trajectories carry no token ids) and multi-step envs
(the agent answers once; env actions would have to become tools).

## Setup (one GPU, ~20 GB)

```bash
pip install -e ".[dev]"
pip install vllm "openvlharness[qwen,server] @ git+https://github.com/uclanlp/OpenVLHarness.git"

# 1. serve the model
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-VL-2B-Instruct \
  --served-model-name Qwen/Qwen3-VL-2B-Instruct --host 127.0.0.1 --port 8222 --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.40 --enable-auto-tool-choice --tool-call-parser hermes \
  --enforce-eager --limit-mm-per-prompt '{"video":0}'

# 2. start the tool router (zoom only: no GPU workers, no API keys, no code execution)
export MODEL_BASE_URL=http://127.0.0.1:8222/v1 MODEL_API_KEY=empty TOOL_SERVER_URL=http://127.0.0.1:8101
python -m openvlharness.services.server --config configs/openvlharness/qwen3_2b_zoom.yaml

# 3. run the loop (same env vars)
python train.py --config ovh_count_memory
```

`configs/openvlharness/qwen3_2b_zoom_code.yaml` adds the coding tool. It runs model-written Python
on your machine without a sandbox (OpenVLHarness's own warning), so it is opt-in: use it only inside
a container or VM you're comfortable with, and point both the router and `ovh_config` at it.

## First numbers

| | held-out success |
|---|---:|
| SmolVLM-256M, pico harness, before GRPO | 3.0% |
| SmolVLM-256M, pico harness, after GRPO | 51.5% |
| Qwen3-VL-2B, OpenVLHarness (zoom + coding), no learning, 20 tasks | 35% (7/20) |
| Qwen3-VL-2B, OpenVLHarness (zoom only), no learning, 40 tasks | 52.5% |

Every one of the 20 episodes used tools. 4 of 20 hit the 8-turn limit without answering, and the 2B
model often calls tools that aren't configured (e.g. `Visual_Grounding_Tool`) — tool-use errors a
scaffold or memory update could learn to avoid.

## A memory loop through OpenVLHarness (`--config ovh_count_memory`)

6 iterations, 4 tasks × 2 rollouts each, 40 held-out tasks, 43 minutes, 2.7M tokens:

| eval at iteration | 0 | 3 | final |
|---|---:|---:|---:|
| held-out success | 52.5% | 65.0% | 40.0% |

The gate rejected 5 of the 6 memory updates: putting recalled skills ("- How many blue squares…
-> 3") into the question lowered gate-split success. So this run shows the plumbing and the gate
working, **not** improvement: for a 2B model with tools, copying past answers into the prompt is not
a useful skill format, and 40-task evals swing by ±15 points. Better next steps here are skills that
describe *procedure* (e.g. "zoom into each quadrant, then add") rather than answers, larger evals,
and scaffold edits targeting the tool-call errors above.
