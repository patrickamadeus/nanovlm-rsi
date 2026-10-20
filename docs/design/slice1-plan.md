# nanoVLM-RSI Slice 1 (Spine + Static Env + Memory Carrier) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A complete RSI loop that closes end-to-end on CPU: a tiny VLM policy acts in a from-scratch verifiable environment, an outcome verifier scores trajectories, a skill-memory updater proposes a durable update, a gate accepts or reverts it, and a trace records enough to attribute the result.

**Architecture:** Five organs behind small ABCs (Env, Policy, Verifier, Updater, proposer function), wired by an explicit loop in `rsi.py`. Exactly one carrier per run; this slice implements the `memory` carrier and stubs the others. Generality comes from interfaces being small, not from a registry.

**Tech Stack:** Python 3.12, PyTorch, numpy, transformers, pillow, safetensors, pytest.

**Spec:** `docs/design/spec.md`

## Global Constraints

- Python 3.12. Dependency floor: `torch`, `numpy`, `transformers`, `pillow`, `safetensors`. Dev: `pytest`.
- **No framework may own the control flow.** No `trl`, `peft`, `accelerate`, or any RL library — not in this slice, not later. GRPO, LoRA, advantage estimation, memory operations, proposer and scaffold editor are written from scratch.
- No JAX in this slice. When `envs/craftax.py` lands (Slice 3) it is the only file permitted to import it.
- Splits are three-way: `train` = hash buckets 0-6, `gate` = bucket 7, `heldout` = buckets 8-9.
- Held-out is frozen: 200 tasks drawn once per `env_seed`, reused across runs.
- Split assignment uses a **process-stable** hash (`zlib.crc32`), never builtin `hash()`.
- A proposer is only ever handed `split="train"`; every Task it constructs is re-stamped by the harness.
- `Obs.rgb` is mandatory on every observation; the policy is genuinely multimodal.
- Every file stays sized to its idea. If a file exceeds ~250 lines, that is a signal it holds two ideas.

## Review Focus

Five failure modes the spec implies but which no task's happy-path tests would exercise. Each has a test assigned to the task that owns the code.

1. **Split determinism across processes.** `hash()` on Python strings is salted per process, so a seed could land in `train` in one run and `heldout` in the next, silently contaminating results. Splits must be identical across interpreter invocations. — Task 2
2. **Parseable but illegal action.** The policy emits `ACT: teleport` — syntactically fine, not in `env.actions`. This is a different path from "no `ACT:` line at all" and must be treated as a parse failure, not passed to `step()`. — Task 4
3. **Degenerate batch.** Every rollout in an iteration parse-fails or scores identically, so there is no signal. `propose()` must return `None` and the loop must continue rather than crash or commit an empty update. — Task 11
4. **Truncated trace or store.** A run killed mid-write leaves a partial final JSONL line. Reading the store or trace must skip the partial line, not raise. — Tasks 6, 10
5. **Frozen held-out identity.** Two runs with the same `env_seed` must produce byte-identical held-out task lists, or cross-run comparison is meaningless. — Task 3

---

### Task 1: Project scaffolding and config

**Files:**
- Create: `pyproject.toml`, `config.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PolicyConfig`, `LoopConfig` dataclasses; `LoopConfig.hash() -> str`.

- [ ] **Step 1: Write the failing test**

```python
def test_config_hash_is_stable_and_sensitive():
    a, b = LoopConfig(), LoopConfig()
    assert a.hash() == b.hash()
    assert LoopConfig(iters=5).hash() != LoopConfig(iters=6).hash()

def test_defaults_match_spec():
    cfg = LoopConfig()
    assert cfg.update_rule == "memory"
    assert cfg.proposer == "fixed"
    assert cfg.group_size == 8
    assert cfg.gate_enabled is True
    assert cfg.heldout_n == 200
    assert isinstance(cfg.policy, PolicyConfig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Write `pyproject.toml`**

Package name `nanovlm-rsi`, `requires-python = ">=3.12"`, the dependency floor from Global Constraints, optional extras `craftax`, `bench` and `wandb` declared but empty-bodied for now, and `[tool.pytest.ini_options] testpaths = ["tests"]`.

- [ ] **Step 4: Implement `PolicyConfig` and `LoopConfig` in `config.py`**

Two flat dataclasses with prefixed field names (nanoVLM idiom: `env_*`, `gate_*`, `mem_*`, `grpo_*`), each knob commented inline. `PolicyConfig`: `model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"`, `tiny: bool = False`, `max_new_tokens: int = 64`, `temperature: float = 1.0`, `dtype: str = "bfloat16"`, `device: str = "auto"`, `scaffold_path: str = "scaffolds/v0.txt"`. `LoopConfig`: `iters`, `group_size: int = 8`, `tasks_per_iter: int = 4`, `update_rule: str = "memory"`, `proposer: str = "fixed"`, `env: str = "static"`, `env_seed: int = 0`, `heldout_n: int = 200`, `gate_n: int = 32`, `eval_every`, `eval_n`, `gate_enabled: bool = True`, `gate_parse_fail_delta: float = 0.05`, `gate_allowed_verdict_sources: tuple[str, ...] = ("outcome",)`, `mem_write_threshold: float = 1.0`, `mem_top_k: int = 3`, `mem_max_size: int = 500`, `mem_dedupe_sim: float = 0.95`, `judge_sample_rate: float = 0.1`, `run_dir: str = "runs/dev"`, `seed: int = 0`, and `policy: PolicyConfig = field(default_factory=PolicyConfig)` so `rsi.py` can call `Policy(cfg.policy)`. `hash()` returns the first 12 chars of a sha256 over the sorted `asdict()` repr.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml config.py tests/test_config.py
git commit -m "feat: project scaffolding and loop/policy config"
```

---

### Task 2: Task, Obs, Env ABC and split enforcement

**Files:**
- Create: `envs/__init__.py`, `envs/base.py`, `tests/test_splits.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Task(family: str, seed: int, goal: tuple[str,...], budget: int, split: str)` frozen dataclass; `Obs(rgb: np.ndarray, text: str | None)` frozen dataclass; `Env` ABC with `reset(task) -> Obs`, `step(action: str) -> tuple[Obs, float, bool, dict]`, `tasks(split: str, n: int, rng) -> list[Task]`, property `actions -> list[str]`; `split_for_seed(seed: int) -> str`; `stamp_split(task: Task) -> Task`.

- [ ] **Step 1: Write the failing test**

```python
def test_split_buckets_match_spec():
    # buckets 0-6 train, 7 gate, 8-9 heldout
    seen = {split_for_seed(s) for s in range(2000)}
    assert seen == {"train", "gate", "heldout"}
    counts = Counter(split_for_seed(s) for s in range(10000))
    assert 0.6 < counts["train"] / 10000 < 0.8
    assert 0.05 < counts["gate"] / 10000 < 0.15
    assert 0.15 < counts["heldout"] / 10000 < 0.25

def test_split_is_stable_across_processes():
    # Review Focus 1: builtin hash() is salted per process; splits must not be
    out = subprocess.run(
        [sys.executable, "-c",
         "from envs.base import split_for_seed;"
         "print(','.join(split_for_seed(s) for s in range(50)))"],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONHASHSEED": "random"},
    )
    local = ",".join(split_for_seed(s) for s in range(50))
    assert out.stdout.strip() == local

def test_stamp_split_overrides_claimed_split():
    t = Task(family="f", seed=1, goal=("g",), budget=1, split="train")
    assert stamp_split(t).split == split_for_seed(1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_splits.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'envs'`

- [ ] **Step 3: Implement `envs/base.py`**

`split_for_seed` uses `zlib.crc32(str(seed).encode()) % 10` mapped to the bucket ranges in Global Constraints — explicitly not builtin `hash()`. `stamp_split` returns `dataclasses.replace(task, split=split_for_seed(task.seed))`. The `Env` ABC carries docstrings naming each method's contract; no implementations.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_splits.py -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Commit**

```bash
git add envs/ tests/test_splits.py
git commit -m "feat: Task/Obs/Env interfaces with process-stable split enforcement"
```

---

### Task 3: Static environment

**Files:**
- Create: `envs/static.py`, `tests/test_static_env.py`

**Interfaces:**
- Consumes: `Task`, `Obs`, `Env`, `split_for_seed`, `stamp_split` from Task 2.
- Produces: `StaticEnv(Env)` with `__init__(self, env_seed: int = 0, heldout_n: int = 200)`; families `"count_objects"` and `"gather_order"`; `actions` returns `["say_0".."say_9", "up", "down", "left", "right", "pick"]`; `render(state) -> np.ndarray` as a module function.

- [ ] **Step 1: Write the failing test**

```python
def test_reset_is_deterministic_by_seed():
    env = StaticEnv()
    t = Task("count_objects", seed=7, goal=("3",), budget=1, split="train")
    assert np.array_equal(env.reset(t).rgb, env.reset(t).rgb)

def test_obs_is_rgb_image():
    env = StaticEnv()
    obs = env.reset(env.tasks("train", 1, np.random.default_rng(0))[0])
    assert obs.rgb.ndim == 3 and obs.rgb.shape[2] == 3
    assert obs.rgb.dtype == np.uint8

def test_count_objects_rewards_correct_answer_only():
    env = StaticEnv()
    t = [x for x in env.tasks("train", 50, np.random.default_rng(0))
         if x.family == "count_objects"][0]
    env.reset(t)
    _, r_bad, done, _ = env.step("say_9" if t.goal[0] != "9" else "say_1")
    assert r_bad == 0.0 and done is True
    env.reset(t)
    _, r_good, done, _ = env.step(f"say_{t.goal[0]}")
    assert r_good == 1.0 and done is True

def test_gather_order_requires_correct_order_and_respects_budget():
    env = StaticEnv()
    t = [x for x in env.tasks("train", 200, np.random.default_rng(0))
         if x.family == "gather_order"][0]
    assert t.budget > 1
    env.reset(t)
    done = False
    for _ in range(t.budget):
        _, r, done, info = env.step("pick")
        if done:
            break
    assert done is True            # budget exhaustion always terminates
    assert "steps" in info

def test_tasks_only_returns_requested_split():
    env = StaticEnv()
    for split in ("train", "gate", "heldout"):
        ts = env.tasks(split, 30, np.random.default_rng(1))
        assert len(ts) == 30
        assert {x.split for x in ts} == {split}

def test_heldout_is_frozen_per_env_seed():
    # Review Focus 5
    a = StaticEnv(env_seed=3).tasks("heldout", 200, np.random.default_rng(99))
    b = StaticEnv(env_seed=3).tasks("heldout", 200, np.random.default_rng(1234))
    assert a == b
    c = StaticEnv(env_seed=4).tasks("heldout", 200, np.random.default_rng(99))
    assert a != c
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_static_env.py -v`
Expected: FAIL with `ImportError: cannot import name 'StaticEnv'`

- [ ] **Step 3: Implement `envs/static.py`**

Procedural generation is driven entirely by `task.seed` via `np.random.default_rng(seed)`, so nothing depends on call order. `count_objects`: scatter N shapes (N in 1..9), goal is `(str(N),)`, budget 1, correct `say_N` gives reward 1.0 and terminates. `gather_order`: an 8x8 grid holding 3 distinct objects, goal is the required pick order as a tuple, budget 24; `pick` on the correct next object advances, a wrong pick terminates with 0.0, collecting all gives 1.0. `render` draws the grid or scene into a `[64,64,3] uint8` array with numpy slice assignment — distinct solid colors per object type, no external drawing library. Held-out freezing: `tasks("heldout", n, rng)` ignores `rng` and returns a list built once from `np.random.default_rng(self.env_seed)` by drawing candidate seeds and keeping those whose `split_for_seed` is `"heldout"` until `heldout_n` are collected; cache it on the instance. `train` and `gate` draw from `rng`, rejecting seeds whose stamped split does not match, with a bounded retry count that raises a clear error rather than looping forever.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_static_env.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add envs/static.py tests/test_static_env.py
git commit -m "feat: from-scratch static verifiable env with frozen held-out set"
```

---

### Task 4: Scaffold rendering and action parsing

**Files:**
- Create: `policy/__init__.py`, `policy/prompt.py`, `scaffolds/v0.txt`, `tests/test_prompt.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `REQUIRED_SLOTS: tuple[str,...]`; `load_scaffold(path) -> str`; `validate_scaffold(template: str) -> None` raising `ScaffoldContractError`; `render(template, *, system, actions, memory, task) -> str`; `parse_action(text: str, legal: list[str]) -> str | None`.

- [ ] **Step 1: Write the failing test**

```python
def test_parse_action_extracts_legal_action():
    assert parse_action("tree ahead\nACT: pick", ["pick", "up"]) == "pick"

def test_parse_action_rejects_missing_act_line():
    assert parse_action("I think I should pick it up", ["pick"]) is None

def test_parse_action_rejects_illegal_but_parseable_action():
    # Review Focus 2
    assert parse_action("ACT: teleport", ["pick", "up"]) is None

def test_parse_action_takes_last_act_line():
    assert parse_action("ACT: up\nwait\nACT: pick", ["pick", "up"]) == "pick"

def test_validate_scaffold_requires_all_slots():
    with pytest.raises(ScaffoldContractError):
        validate_scaffold("{system} {actions} {task}")   # missing {memory}
    with pytest.raises(ScaffoldContractError):
        validate_scaffold("{system} {actions} {memory} {task}")  # missing ACT:

def test_shipped_scaffold_is_valid():
    validate_scaffold(load_scaffold("scaffolds/v0.txt"))

def test_render_fills_slots_and_empty_memory_is_clean():
    out = render(load_scaffold("scaffolds/v0.txt"),
                 system="S", actions=["a", "b"], memory="", task="T")
    assert "S" in out and "a, b" in out and "T" in out
    assert "{" not in out and "\n\n\n" not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'policy'`

- [ ] **Step 3: Write `scaffolds/v0.txt` and implement `policy/prompt.py`**

`scaffolds/v0.txt` is the template from spec §6 verbatim, slots `{system} {actions} {memory} {task}` plus the literal `ACT:` contract line. `REQUIRED_SLOTS = ("{system}", "{actions}", "{memory}", "{task}")`; `validate_scaffold` checks every required slot plus the substring `ACT:` is present. `parse_action` scans lines for the last one matching `^\s*ACT:\s*(\S+)` and returns the capture only if it is in `legal`, else `None`. `render` joins `actions` with `", "` and collapses runs of 3+ newlines to 2 so an empty `{memory}` leaves no gap.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add policy/ scaffolds/ tests/test_prompt.py
git commit -m "feat: scaffold template with slot contract and strict action parsing"
```

---

### Task 5: Policy wrapper

**Files:**
- Create: `policy/vlm.py`, `tests/test_policy.py`

**Interfaces:**
- Consumes: `PolicyConfig` from Task 1.
- Produces: `RolloutStep(text: str, token_ids: list[int], logprobs: np.ndarray, prompt_tokens: int)` frozen dataclass; `Policy` with `__init__(cfg: PolicyConfig)`, `act(prompt: str, images: list[np.ndarray]) -> RolloutStep`, `logprobs(prompt, images, token_ids) -> torch.Tensor`, `embed(text: str, images: list[np.ndarray]) -> np.ndarray`, `snapshot() -> dict`, `restore(state: dict) -> None`.

- [ ] **Step 1: Write the failing test**

```python
@pytest.fixture(scope="module")
def tiny():
    return Policy(PolicyConfig(tiny=True, max_new_tokens=8, device="cpu"))

def test_act_returns_aligned_tokens_and_logprobs(tiny):
    step = tiny.act("hello", [np.zeros((64, 64, 3), np.uint8)])
    assert isinstance(step.text, str)
    assert len(step.token_ids) == len(step.logprobs) > 0
    assert step.prompt_tokens > 0

def test_logprobs_rescores_consistently(tiny):
    img = [np.zeros((64, 64, 3), np.uint8)]
    step = tiny.act("hello", img)
    again = tiny.logprobs("hello", img, step.token_ids)
    assert again.shape[0] == len(step.token_ids)
    np.testing.assert_allclose(
        again.detach().cpu().numpy(), step.logprobs, rtol=1e-3, atol=1e-3)

def test_embed_returns_fixed_width_unit_vector(tiny):
    v = tiny.embed("a situation", [np.zeros((64, 64, 3), np.uint8)])
    assert v.ndim == 1 and np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)
    assert v.shape == tiny.embed("other", [np.ones((64, 64, 3), np.uint8)]).shape

def test_snapshot_restore_roundtrip(tiny):
    s = tiny.snapshot()
    tiny.restore(s)
    assert tiny.logprobs is not None   # still usable after restore
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_policy.py -v`
Expected: FAIL with `ImportError: cannot import name 'Policy'`

- [ ] **Step 3: Implement `policy/vlm.py`**

A thin wrapper: `transformers` loads the processor and model and owns the forward pass; nothing in this file touches control flow. When `cfg.tiny` is set, build a randomly-initialized model from a deliberately tiny config (a handful of layers, small hidden size) so CI runs on CPU in seconds — the tiny path must go through the same code as the real path, not a separate branch of `act`. `act` samples with `temperature` and `max_new_tokens`, then gathers the per-token logprob of each sampled token from the scores; only completion tokens are returned, never prompt tokens. `logprobs` runs one forward pass over prompt+completion and gathers the same positions, which is why its result must match `act` to tolerance. `embed` mean-pools last hidden states over non-padding positions and L2-normalizes — this is the memory store's embedding function, so no second model is ever needed. `snapshot`/`restore` deep-copy trainable state on CPU; in this slice nothing trains, but `gate()` and Slice 2 both need the hooks to exist.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_policy.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add policy/vlm.py tests/test_policy.py
git commit -m "feat: HF-wrapped VLM policy with act/logprobs/embed and tiny CPU path"
```

---

### Task 6: Trace

**Files:**
- Create: `infra/__init__.py`, `infra/trace.py`, `tests/test_trace.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Cost(env_steps: int, policy_tokens: int, judge_tokens: int, gpu_seconds: float)` with `__add__`; `Trace` with `__init__(run_dir: str)`, `log_run(cfg_hash, git_sha, seeds, carrier, proposer, gate_enabled)`, `log_rollout(**fields)`, `log_update(**fields)`, `log_eval(**fields)`, property `cumulative_cost -> Cost`, `records(kind: str | None = None) -> list[dict]`, `success_rate(family: str, goal: tuple) -> float | None`.

- [ ] **Step 1: Write the failing test**

```python
def test_records_roundtrip_by_kind(tmp_path):
    tr = Trace(str(tmp_path))
    tr.log_run(cfg_hash="abc", git_sha="def", seeds={"seed": 0},
               carrier="memory", proposer="fixed", gate_enabled=True)
    tr.log_rollout(task_id="t1", family="count_objects", seed=1, split="train",
                   reward=1.0, verdict_source="outcome", parse_fail=False,
                   retrieved_skill_ids=[], env_steps=1, tokens_out=5)
    assert len(tr.records("run")) == 1
    assert tr.records("rollout")[0]["family"] == "count_objects"
    assert tr.records()[0]["kind"] == "run"

def test_cumulative_cost_accumulates():
    ...  # two rollouts of 3 and 4 env_steps
    assert tr.cumulative_cost.env_steps == 7

def test_eval_record_separates_heldout_axes(tmp_path):
    tr = Trace(str(tmp_path))
    tr.log_eval(split="heldout", n=200, success_rate=0.4,
                unseen_seed_rate=0.5, unseen_goal_rate=0.3,
                cost=Cost(10, 20, 0, 1.0))
    rec = tr.records("eval")[0]
    assert rec["unseen_seed_rate"] == 0.5 and rec["unseen_goal_rate"] == 0.3
    assert rec["cost"]["env_steps"] == 10

def test_reader_skips_truncated_final_line(tmp_path):
    # Review Focus 4
    tr = Trace(str(tmp_path))
    tr.log_rollout(task_id="t1", reward=1.0)
    (tmp_path / "trace.jsonl").open("a").write('{"kind": "rollout", "rew')
    assert len(Trace(str(tmp_path)).records("rollout")) == 1

def test_success_rate_tracks_family_goal(tmp_path):
    ...  # three rollouts on ("count_objects", ("3",)) with rewards 1,0,1
    assert tr.success_rate("count_objects", ("3",)) == pytest.approx(2 / 3)
    assert tr.success_rate("count_objects", ("9",)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trace.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'infra'`

- [ ] **Step 3: Implement `infra/trace.py`**

One append-only `trace.jsonl` under `run_dir`, created on init. Each `log_*` writes one object with a `kind` field (`run`/`rollout`/`update`/`eval`), an `iso_ts`, and flushes. `records()` reads the file, parsing line by line and silently skipping any line that fails `json.loads` — that is the truncation defense, and it must skip rather than raise. `cumulative_cost` folds `Cost` over rollout and eval records. `success_rate` returns `None` for an unseen `(family, goal)` key rather than 0.0, since "never attempted" and "always failed" must be distinguishable by the proposer.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trace.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add infra/ tests/test_trace.py
git commit -m "feat: append-only trace with cost accounting and truncation-safe reads"
```

---

### Task 7: Outcome verifier

**Files:**
- Create: `verify.py`, `tests/test_verify.py`

**Interfaces:**
- Consumes: `Task` from Task 2.
- Produces: `Trajectory(task: Task, steps: list[dict], total_reward: float, env_steps: int, parse_fails: int, tokens_out: int)`; `ScoredTrajectory(traj: Trajectory, reward: float, per_step_credit: list[float] | None, verdict_source: str, confidence: float, flags: tuple[str,...])`; `Verifier` ABC with `score(traj) -> ScoredTrajectory`; `OutcomeVerifier(Verifier)`; `make_verifier(cfg) -> Verifier`.

- [ ] **Step 1: Write the failing test**

```python
def test_binary_success_from_env_reward():
    s = OutcomeVerifier().score(traj(total_reward=1.0, env_steps=3))
    assert s.reward == 1.0 and s.verdict_source == "outcome" and s.confidence == 1.0

def test_partial_credit_is_passed_through():
    s = OutcomeVerifier().score(traj(total_reward=0.5, env_steps=4))
    assert s.reward == 0.5

def test_flags_zero_step_success():
    s = OutcomeVerifier().score(traj(total_reward=1.0, env_steps=0))
    assert "zero_step_success" in s.flags

def test_flags_degenerate_action_loop():
    s = OutcomeVerifier().score(traj(actions=["up"] * 12, total_reward=0.0))
    assert "action_loop" in s.flags

def test_flags_noop_spam_from_parse_fails():
    s = OutcomeVerifier().score(traj(total_reward=0.0, env_steps=10, parse_fails=9))
    assert "noop_spam" in s.flags
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_verify.py -v`
Expected: FAIL with `ImportError: cannot import name 'OutcomeVerifier'`

- [ ] **Step 3: Implement `verify.py`**

`OutcomeVerifier.score` reads the environment's own reward — it does not re-derive success, since the env is the ground truth. `verdict_source = "outcome"`, `confidence = 1.0`. Flags: `zero_step_success` when `total_reward > 0 and env_steps == 0`; `action_loop` when a single action is at least 80% of at least 8 steps; `noop_spam` when `parse_fails / max(env_steps, 1) > 0.5`. `make_verifier` returns `OutcomeVerifier()` for this slice and raises `NotImplementedError` naming the slice for `judge` and `composite`. Include a module docstring stating that `JudgeVerifier` and agreement logging land in Slice 4 and listing the trace fields reserved for them.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_verify.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add verify.py tests/test_verify.py
git commit -m "feat: outcome verifier with reward-hacking flags"
```

---

### Task 8: Rollout

**Files:**
- Create: `rollout.py`, `tests/conftest.py`, `tests/test_rollout.py`

**Interfaces:**
- Consumes: `Policy` (Task 5), `Env`/`Task`/`Obs` (Task 2), `Trajectory` (Task 7), `render`/`parse_action` (Task 4), `LoopConfig` (Task 1).
- Produces: `rollout(policy, env, task, cfg, *, memory_text: str = "") -> Trajectory`; and in `tests/conftest.py` the shared stub `ScriptedPolicy(generations: Iterable[str])` satisfying the `Policy` surface (`act` pops the next generation, records the prompt on `.prompts`, returns a `RolloutStep` with dummy token ids and logprobs; `embed` returns a deterministic unit vector from a hash of its input). Tasks 13 and 14 reuse it.

- [ ] **Step 1: Write the failing test**

Tests use a `ScriptedPolicy` stub (returns queued generations) and `StaticEnv`, so no model loads.

```python
def test_rollout_records_one_step_per_action():
    tr = rollout(ScriptedPolicy(["ACT: say_3"]), env, count_task_with_goal_3, cfg)
    assert tr.env_steps == 1 and tr.total_reward == 1.0 and tr.parse_fails == 0

def test_parse_failure_becomes_noop_with_negative_shaping():
    tr = rollout(ScriptedPolicy(["no action here"]), env, gather_task, cfg)
    assert tr.parse_fails == 1
    assert tr.steps[0]["action"] is None
    assert tr.steps[0]["shaping"] < 0

def test_illegal_action_counts_as_parse_failure():
    # Review Focus 2, at the loop level
    tr = rollout(ScriptedPolicy(["ACT: teleport"]), env, gather_task, cfg)
    assert tr.parse_fails == 1 and tr.steps[0]["action"] is None

def test_rollout_never_exceeds_task_budget():
    tr = rollout(ScriptedPolicy(["ACT: up"] * 100), env, gather_task, cfg)
    assert tr.env_steps <= gather_task.budget

def test_memory_text_reaches_the_prompt():
    p = ScriptedPolicy(["ACT: say_3"])
    rollout(p, env, count_task_with_goal_3, cfg, memory_text="RECALL: count shapes")
    assert "RECALL: count shapes" in p.prompts[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rollout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rollout'`

- [ ] **Step 3: Implement `rollout(policy, env, task, cfg, *, memory_text="") -> Trajectory`**

Reset the env, then loop to `task.budget`: render the scaffold with the current obs and `memory_text`, call `policy.act`, `parse_action` against `env.actions`. A `None` parse writes a step with `action=None` and `shaping=-0.01`, consumes a budget step, increments `parse_fails`, and does not call `env.step`. Otherwise call `env.step` and record `action`, `reward`, `tokens_out`. Break on `done`. `total_reward` is the env reward sum; shaping is recorded per-step but kept out of `total_reward` so the verifier sees unmodified environment signal.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rollout.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add rollout.py tests/test_rollout.py
git commit -m "feat: rollout loop with first-class parse failures and budget enforcement"
```

---

### Task 9: Fixed proposer

**Files:**
- Create: `propose.py`, `tests/test_propose.py`

**Interfaces:**
- Consumes: `Env`/`Task`/`stamp_split` (Task 2), `Trace` (Task 6), `LoopConfig` (Task 1).
- Produces: `make_proposer(name: str, env, cfg) -> Callable[[Policy, Trace], list[Task]]`; `fixed_proposer`.

- [ ] **Step 1: Write the failing test**

```python
def test_fixed_proposer_returns_only_train_tasks():
    tasks = make_proposer("fixed", StaticEnv(), LoopConfig(tasks_per_iter=16))(None, None)
    assert len(tasks) == 16
    assert {t.split for t in tasks} == {"train"}

def test_proposer_can_never_emit_gate_or_heldout_seeds():
    # Review Focus 1 consequence; the central leak test
    env, prop = StaticEnv(), make_proposer("fixed", StaticEnv(), LoopConfig(tasks_per_iter=64))
    heldout_seeds = {t.seed for t in env.tasks("heldout", 200, np.random.default_rng(0))}
    for _ in range(20):
        for t in prop(None, None):
            assert split_for_seed(t.seed) == "train"
            assert t.seed not in heldout_seeds

def test_self_proposer_not_yet_available():
    with pytest.raises(NotImplementedError, match="Slice 5"):
        make_proposer("self", StaticEnv(), LoopConfig())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_propose.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propose'`

- [ ] **Step 3: Implement `propose.py`**

`fixed_proposer` samples `cfg.tasks_per_iter` from `env.tasks("train", ...)` and re-stamps every task through `stamp_split` before returning, asserting the result is `"train"`. The proposer never receives or constructs a gate/heldout seed. `make_proposer` raises `NotImplementedError("self proposer lands in Slice 5")` for `"self"`, with a docstring noting the competence-frontier design and that `Trace.success_rate` already supplies its input.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_propose.py -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Commit**

```bash
git add propose.py tests/test_propose.py
git commit -m "feat: fixed proposer with enforced train-split-only sampling"
```

---

### Task 10: Skill store

**Files:**
- Create: `infra/memory.py`, `tests/test_memory_store.py`
- Modify: `tests/conftest.py` (add `fake_embed(text) -> np.ndarray`, a deterministic bag-of-words unit vector, so store tests need no model)

**Interfaces:**
- Consumes: nothing (embedding function is injected).
- Produces: `Skill(id: str, family: str, situation: str, actions: tuple[str,...], reward: float, embedding: np.ndarray, uses: int, uses_before_success: int)`; `SkillStore` with `__init__(path: str, embed_fn: Callable[[str], np.ndarray], max_size: int = 500, dedupe_sim: float = 0.95)`, `offset() -> int`, `append(skill) -> str | None`, `retrieve(situation: str, k: int) -> list[Skill]`, `truncate(offset: int) -> None`, `note_use(skill_ids, succeeded: bool) -> None`, `expire() -> int`, `as_text(skills) -> str`, `__len__`.

- [ ] **Step 1: Write the failing test**

```python
def test_append_and_retrieve_top_k(tmp_path):
    store = SkillStore(str(tmp_path / "skills.jsonl"), embed_fn=fake_embed)
    store.append(skill("count three shapes"))
    store.append(skill("gather in order"))
    got = store.retrieve("counting shapes", k=1)
    assert len(got) == 1 and "count" in got[0].situation

def test_dedupe_rejects_near_duplicate(tmp_path):
    store = SkillStore(..., dedupe_sim=0.95)
    assert store.append(skill("count three shapes", reward=1.0)) is not None
    assert store.append(skill("count three shapes", reward=0.5)) is None
    assert len(store) == 1

def test_dedupe_keeps_higher_reward_on_collision(tmp_path):
    store.append(skill("count three shapes", reward=0.5))
    store.append(skill("count three shapes", reward=1.0))
    assert len(store) == 1 and store.retrieve("count", 1)[0].reward == 1.0

def test_truncate_restores_byte_identical_state(tmp_path):
    store.append(skill("a"))
    off, before = store.offset(), path.read_bytes()
    store.append(skill("b"))
    store.truncate(off)
    assert path.read_bytes() == before and len(store) == 1

def test_reader_skips_truncated_final_line(tmp_path):
    # Review Focus 4
    store.append(skill("a"))
    path.open("a").write('{"id": "x", "situ')
    assert len(SkillStore(str(path), embed_fn=fake_embed)) == 1

def test_note_use_tracks_use_attribution(tmp_path):
    sid = store.append(skill("a"))
    store.note_use([sid], succeeded=True)
    store.note_use([sid], succeeded=False)
    s = store.retrieve("a", 1)[0]
    assert s.uses == 2 and s.uses_before_success == 1

def test_expire_caps_size_by_usefulness(tmp_path):
    store = SkillStore(..., max_size=2)
    for i in range(5):
        store.append(skill(f"s{i}", reward=1.0))
    assert store.expire() == 3 and len(store) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_store.py -v`
Expected: FAIL with `ImportError: cannot import name 'SkillStore'`

- [ ] **Step 3: Implement `infra/memory.py`**

Append-only JSONL; embeddings stored as lists and rehydrated to numpy on load. `offset()` returns the file size in bytes, so `truncate(offset)` is an exact byte-level revert — that is what makes the gate's revert honest. Loading parses line by line and skips unparseable lines. `retrieve` is brute-force cosine over the in-memory list; the library is capped at `max_size` so this stays trivial. `append` returns `None` when cosine similarity to an existing skill exceeds `dedupe_sim` and the incoming reward is not higher; when it is higher, the loser is dropped and the winner appended. `expire` removes lowest-usefulness skills until `len <= max_size`, ranking by `uses_before_success / max(uses, 1)` then `reward`, and returns the count removed. `as_text` formats skills into the `{memory}` slot.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_store.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add infra/memory.py tests/test_memory_store.py
git commit -m "feat: skill store with byte-exact revert, dedupe and use-attribution"
```

---

### Task 11: Updater interface and memory carrier

**Files:**
- Create: `update/__init__.py`, `update/base.py`, `update/memory.py`, `update/weights.py`, `update/scaffold.py`, `update/verifier.py`, `tests/test_update_memory.py`

**Interfaces:**
- Consumes: `ScoredTrajectory` (Task 7), `SkillStore` (Task 10), `Policy` (Task 5), `Cost` (Task 6), `LoopConfig` (Task 1).
- Produces: `Update(carrier: str, role: str, horizon: str, clock: str, signal: str, payload: dict, cost: Cost, provenance: tuple[str,...])`; `Updater` ABC with `propose(batch) -> Update | None`, `commit(update)`, `revert(update)`; `MemoryUpdater(Updater)`; `make_updater(name, policy, cfg) -> Updater`.

- [ ] **Step 1: Write the failing test**

```python
def test_update_carries_taxonomy_labels():
    u = MemoryUpdater(store, cfg).propose([scored(reward=1.0)])
    assert (u.carrier, u.role, u.horizon) == ("A3", "B0", "C3")
    assert u.signal == "outcome" and u.provenance

def test_proposes_only_successes_above_threshold():
    assert MemoryUpdater(store, cfg).propose([scored(reward=0.4)]) is None
    assert MemoryUpdater(store, cfg).propose([scored(reward=1.0)]) is not None

def test_degenerate_batch_returns_none():
    # Review Focus 3
    up = MemoryUpdater(store, cfg)
    assert up.propose([]) is None
    assert up.propose([scored(reward=0.0, parse_fail=True) for _ in range(8)]) is None

def test_commit_then_revert_restores_store(tmp_path):
    up, before = MemoryUpdater(store, cfg), store_bytes()
    u = up.propose([scored(reward=1.0)])
    up.commit(u)
    assert len(store) == 1
    up.revert(u)
    assert store_bytes() == before and len(store) == 0

def test_weights_and_scaffold_carriers_not_yet_available():
    with pytest.raises(NotImplementedError, match="Slice 2"):
        make_updater("weights", policy, LoopConfig())
    with pytest.raises(NotImplementedError, match="Slice 6"):
        make_updater("scaffold", policy, LoopConfig())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_update_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'update'`

- [ ] **Step 3: Implement the `update/` package**

`update/base.py` holds `Update` and the `Updater` ABC; the taxonomy fields are plain strings on the dataclass so a trace reader never needs the code. `MemoryUpdater.propose` selects trajectories with `reward >= cfg.mem_write_threshold`, summarizes each into a `Skill` candidate (family, a one-line situation string built from the task, the action sequence, reward), records the store's `offset()` in `payload["revert_offset"]`, and returns `None` for an empty or all-failed batch. `commit` appends each candidate then calls `expire()`. `revert` calls `store.truncate(payload["revert_offset"])`. `update/weights.py`, `update/scaffold.py` and `update/verifier.py` are stubs: each defines its class with the `Updater` signature, raises `NotImplementedError` naming its slice, and carries a docstring describing the design from spec §8 — `update/verifier.py` additionally names the trace fields (`verdict_source`, `judge_verdict`) that will feed it.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_update_memory.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add update/ tests/test_update_memory.py
git commit -m "feat: Updater interface, memory carrier, and stubs for later carriers"
```

---

### Task 12: Gate

**Files:**
- Create: `infra/gate.py`, `tests/test_gate.py`

**Interfaces:**
- Consumes: `Update` (Task 11), `Policy` (Task 5), `Env` (Task 2), `Trace` (Task 6), `LoopConfig` (Task 1).
- Produces: `GateResult(passed: bool, conditions: dict[str, bool], reason: str | None)`; `gate(update, policy, env, trace, cfg, *, eval_fn) -> GateResult`.

- [ ] **Step 1: Write the failing test**

```python
def test_all_conditions_pass_for_a_good_update():
    r = gate(good_update, policy, env, trace, cfg, eval_fn=eval_improving)
    assert r.passed and all(r.conditions.values())
    assert set(r.conditions) == {"proposed", "survives_reset", "gate_gain",
                                 "deployment_feedback", "no_regression"}

def test_empty_provenance_fails_proposed():
    r = gate(replace(good_update, provenance=()), ..., eval_fn=eval_improving)
    assert not r.passed and r.conditions["proposed"] is False

def test_reload_probe_fails_when_update_not_on_disk():
    r = gate(update_not_persisted, ..., eval_fn=eval_improving)
    assert not r.passed and r.conditions["survives_reset"] is False

def test_negative_gate_delta_fails():
    r = gate(good_update, ..., eval_fn=eval_worsening)
    assert not r.passed and r.conditions["gate_gain"] is False
    assert "gate_gain" in r.reason

def test_disallowed_verdict_source_fails():
    cfg2 = replace(cfg, gate_allowed_verdict_sources=("outcome",))
    r = gate(replace(good_update, signal="oracle"), ..., cfg=cfg2, eval_fn=eval_improving)
    assert r.conditions["deployment_feedback"] is False

def test_parse_fail_regression_fails():
    r = gate(good_update, ..., eval_fn=eval_improving_but_parse_worse)
    assert r.conditions["no_regression"] is False

def test_disabled_gate_passes_and_is_recorded(tmp_path):
    cfg2 = replace(cfg, gate_enabled=False)
    r = gate(good_update, ..., cfg=cfg2, eval_fn=eval_worsening)
    assert r.passed and trace.records("update")[-1]["gate_enabled"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gate.py -v`
Expected: FAIL with `ImportError: cannot import name 'gate'`

- [ ] **Step 3: Implement `infra/gate.py`**

Five conditions, evaluated in the order listed in spec §9, each recorded in `conditions` even when an earlier one already failed, so the trace shows the full picture. `proposed`: `update.provenance` non-empty. `survives_reset`: the reload probe — re-instantiate the organ from disk (for the memory carrier, construct a fresh `SkillStore` over the same path) and confirm the update's payload is present. `gate_gain`: `eval_fn` is called on `env.tasks("gate", cfg.gate_n, ...)` before and after, requiring `delta >= 0`; `eval_fn` is injected so the gate never imports `evaluate` and the tests need no model. `deployment_feedback`: `update.signal in cfg.gate_allowed_verdict_sources`. `no_regression`: post-update `parse_fail_rate` not worse than pre by more than `cfg.gate_parse_fail_delta`. When `cfg.gate_enabled` is false, return `passed=True` with every condition recorded as `None`-equivalent and ensure the update record carries `gate_enabled: False` — ungated results must be self-identifying.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gate.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add infra/gate.py tests/test_gate.py
git commit -m "feat: acceptance gate mechanizing the five-condition bar"
```

---

### Task 13: Evaluation

**Files:**
- Create: `evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `rollout` (Task 8), `Trace`/`Cost` (Task 6), `Env`/`Task` (Task 2), `SkillStore` (Task 10).
- Produces: `EvalResult(split: str, n: int, success_rate: float, unseen_seed_rate: float, unseen_goal_rate: float, parse_fail_rate: float, cost: Cost)`; `evaluate(policy, tasks, trace, cfg, *, store=None, final=False) -> EvalResult`; `main()` CLI with `--split`, `--n`, `--benchmarks`.

- [ ] **Step 1: Write the failing test**

```python
def test_evaluate_reports_both_heldout_axes_separately():
    r = evaluate(ScriptedPolicy(...), tasks_mixed, trace, cfg)
    assert 0.0 <= r.unseen_seed_rate <= 1.0
    assert r.unseen_seed_rate != r.unseen_goal_rate   # computed independently

def test_evaluate_does_not_write_to_the_store(tmp_path):
    before = len(store)
    evaluate(ScriptedPolicy(["ACT: say_3"]), tasks, trace, cfg, store=store)
    assert len(store) == before     # learning disabled during eval

def test_evaluate_logs_an_eval_record_with_cost():
    evaluate(..., trace, cfg)
    rec = trace.records("eval")[-1]
    assert rec["split"] == "heldout" and rec["cost"]["env_steps"] > 0

def test_evaluate_on_empty_task_list_returns_zeros_not_nan():
    r = evaluate(ScriptedPolicy([]), [], trace, cfg)
    assert r.n == 0 and r.success_rate == 0.0 and not math.isnan(r.success_rate)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate'`

- [ ] **Step 3: Implement `evaluate.py`**

Run `rollout` over each task with memory retrieval allowed but all writes suppressed — pass the store for `retrieve` only, never call `append`. A task counts toward `unseen_goal_rate` when its `goal` was never seen in any training rollout in the trace, and toward `unseen_seed_rate` otherwise; compute the two rates over their own denominators and guard division by zero. Log one `eval` record with cumulative cost. `main()` prints a plain table; `--benchmarks` raises `NotImplementedError("MMStar benchmark regression lands in Slice 7")`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluation with separated held-out axes and learning disabled"
```

---

### Task 14: The loop, CLI, and end-to-end smoke

**Files:**
- Create: `rsi.py`, `play.py`, `tests/test_smoke.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: every prior task.
- Produces: `make_env(cfg) -> Env`; `rsi(cfg) -> EvalResult`; `main()` CLI over `LoopConfig` fields.

- [ ] **Step 1: Write the failing test**

```python
def test_smoke_closes_the_loop_on_cpu(tmp_path):
    cfg = LoopConfig(iters=2, tasks_per_iter=2, group_size=2, eval_every=1,
                     eval_n=4, gate_n=4, heldout_n=8,
                     run_dir=str(tmp_path), env="static", update_rule="memory")
    result = rsi(cfg, policy=ScriptedPolicy(cycle(["ACT: say_3", "ACT: pick"])))
    assert isinstance(result, EvalResult)
    kinds = {r["kind"] for r in Trace(str(tmp_path)).records()}
    assert kinds == {"run", "rollout", "update", "eval"}

def test_smoke_runs_under_60s_on_cpu(tmp_path):
    t0 = time.monotonic()
    rsi(LoopConfig(iters=1, ..., policy=PolicyConfig(tiny=True)), ...)
    assert time.monotonic() - t0 < 60

def test_loop_survives_an_iteration_with_no_update(tmp_path):
    # Review Focus 3, at the loop level
    result = rsi(cfg, policy=ScriptedPolicy(cycle(["garbage"])))
    assert isinstance(result, EvalResult)
    assert all(r["committed"] is False for r in trace.records("update")) or \
           not trace.records("update")

def test_run_record_captures_provenance(tmp_path):
    rsi(cfg, policy=ScriptedPolicy(...))
    rec = Trace(str(tmp_path)).records("run")[0]
    assert rec["carrier"] == "memory" and rec["cfg_hash"] and "git_sha" in rec
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rsi'`

- [ ] **Step 3: Implement `rsi.py` and `play.py`**

`rsi.py` is the loop from spec §4, verbatim in structure, with `policy` an optional injected argument so tests can pass a stub — defaulting to `Policy(cfg.policy)`. `make_env` returns `StaticEnv` for `"static"` and raises `NotImplementedError("craftax lands in Slice 3")` otherwise. The memory carrier's retrieval happens in the loop: before each rollout, `store.retrieve` then `as_text` into `memory_text`, and `note_use` after scoring. `log_run` records `cfg.hash()` and the git sha. Keep this file under ~200 lines; it is the file a newcomer reads first, and anything that does not belong to the loop's shape belongs elsewhere. `play.py` runs one task with learning disabled and prints each step's prompt, generation and reward — the debug path.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_smoke.py -v && pytest -q`
Expected: PASS, 4 smoke tests; full suite green

- [ ] **Step 5: Write `README.md`**

What nanoVLM-RSI is and the one-sentence non-SOTA framing; per-file line counts as nanoVLM does, since file size is the documentation; install (`uv` first, pip fallback); a 60-second CPU quickstart that closes a loop on the static env; the carrier/proposer knobs; which slices exist versus which raise `NotImplementedError`; the dependency floor and the no-framework rule; and a provenance section citing the method each from-scratch file implements.

- [ ] **Step 6: Commit**

```bash
git add rsi.py play.py tests/test_smoke.py README.md
git commit -m "feat: explicit RSI loop, play CLI, end-to-end CPU smoke test"
```

---

## Done when

- `pytest -q` is green and the smoke test closes a loop on CPU in under 60 seconds.
- `python rsi.py --iters 2 --update_rule memory` produces a trace containing all four record types, with taxonomy labels and per-condition gate results readable without opening the code.
- `evaluate.py --split heldout` prints both held-out axes separately.
- Every later-slice entry point raises `NotImplementedError` naming its slice.
