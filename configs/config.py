"""Every knob of a run. A YAML file in configs/yaml/ picks values; CLI flags override the YAML."""
import hashlib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent


@dataclass
class PolicyConfig:
    model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    tiny: bool = False             # randomly-initialized tiny VLM for CPU tests; same code path as the real one
    max_new_tokens: int = 64
    temperature: float = 1.0       # GRPO needs real sampling diversity within a group
    dtype: str = "bfloat16"
    device: str = "auto"           # "auto" picks cuda when available
    scaffold_path: str = "utils/scaffold.txt"
    image_side: int = 512          # one 512px tile; the processor default turns a 64px frame into 17 tiles
    stop_at_newline: bool = True   # the action is one line; generating past it is wasted compute
    response_prefix: str = "ACT:"  # prefilled assistant turn: 39/40 parseable vs 4/40 without, for a 256M model


@dataclass
class LoopConfig:
    # what runs: one choice per folder
    env: str = "static"            # envs/:    "static" | "craftax"
    judge: str = "outcome"         # judges/:  "outcome" | "vlm" | "composite"
    update: str = "grpo"           # rsi/:     "grpo" | "memory"  — exactly one per run
    curriculum: str = "uniform"    # rsi/:     "uniform" | "frontier"
    harness: str = "pico"          # utils/:   "pico" (rollout.py) | "openvlharness" (openvlharness.py: tools, served model)
    ovh_config: str = "configs/openvlharness/qwen3_2b_zoom.yaml"

    iters: int = 20
    tasks_per_iter: int = 4
    group_size: int = 8            # rollouts per task; GRPO advantages are relative within this group
    seed: int = 0
    run_dir: str = "runs/dev"
    resume: bool = False           # reusing a non-empty run_dir is refused unless resuming on purpose

    # envs
    env_seed: int = 0              # fixes the frozen held-out set
    static_families: tuple[str, ...] = ("count_objects", "gather_order")
    craftax_budget: int = 64

    # evaluation and the gate
    eval_every: int = 5
    eval_n: int = 200
    heldout_n: int = 200
    gate_enabled: bool = True      # off is allowed for ablations, and is recorded in the log
    gate_n: int = 32
    gate_parse_fail_delta: float = 0.05
    gate_allowed_signals: tuple[str, ...] = ("outcome",)   # signals that would exist at deployment

    # judges
    judge_model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct"   # a different model from the policy, on purpose
    judge_sample_rate: float = 0.1 # composite: how often the VLM judge shadows ground truth

    # rsi/grpo.py
    grpo_mode: str = "lora"        # "lora" | "full"
    grpo_lr: float = 1e-4
    grpo_clip: float = 0.2
    grpo_beta: float = 0.04        # KL-to-init coefficient; 0 disables the reference pass
    grpo_max_grad_norm: float = 1.0
    lora_r: int = 8
    lora_alpha: float = 16

    # rsi/memory.py
    mem_write_threshold: float = 1.0
    mem_top_k: int = 3
    mem_max_size: int = 500
    mem_dedupe_sim: float = 0.95

    # rsi/curriculum.py (frontier)
    curriculum_pool: int = 8       # candidates drawn per task returned
    curriculum_epsilon: float = 0.1
    frontier_band: tuple[float, float] = (0.2, 0.8)   # where a GRPO group carries signal
    frontier_unseen_weight: float = 1.0
    frontier_off_band_weight: float = 0.05

    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def hash(self) -> str:
        return hashlib.sha256(repr(sorted(asdict(self).items())).encode()).hexdigest()[:12]


def _coerce(cls, values: dict) -> dict:
    """YAML lists become the tuples the dataclass declares."""
    types = {f.name: f.type for f in fields(cls)}
    return {k: tuple(v) if isinstance(v, list) and str(types.get(k, "")).startswith("tuple") else v
            for k, v in values.items()}


def load_config(name: str | None = None, **overrides) -> LoopConfig:
    """configs/<name>.yaml (or a path), then overrides. Unknown keys are an error, not a silent no-op."""
    values = {}
    if name:
        path = Path(name) if name.endswith((".yaml", ".yml")) else CONFIG_DIR / "yaml" / f"{name}.yaml"
        values = yaml.safe_load(path.read_text()) or {}
    policy = {**values.pop("policy", {}), **overrides.pop("policy", {})}
    cfg = LoopConfig(**_coerce(LoopConfig, {**values, **overrides}))
    cfg.policy = PolicyConfig(**policy)
    return cfg
