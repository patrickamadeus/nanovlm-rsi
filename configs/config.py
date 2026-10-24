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
    batch_size: int = 64           # act_batch: prompts per generate call
    response_prefix: str = "ACT:"  # prefilled assistant turn: 39/40 parseable vs 4/40 without, for a 256M model


@dataclass
class LoopConfig:
    # what runs: one choice per folder
    env: str = "static"            # envs/:    "static" | "frozenlake" | "sokoban" | "craftax" | "cauldron"
    judge: str = "outcome"         # rsi/judges.py: "outcome" | "vlm" | "composite"
    methods: tuple[str, ...] = ("grpo",)      # rsi/<name>.py, any combination: "grpo" | "memory" | ...
    planner: str = "uniform"       # which tasks to practise: "uniform" | "frontier" | ...
    update_planner: str = "all"    # which methods update each iteration: "all" | ...

    iters: int = 20
    tasks_per_iter: int = 4
    group_size: int = 8            # rollouts per task; GRPO advantages are relative within this group
    seed: int = 0
    run_dir: str = "runs/dev"
    resume: bool = False           # reusing a non-empty run_dir is refused unless resuming on purpose

    # envs
    env_seed: int = 0              # fixes the frozen held-out set
    static_families: tuple[str, ...] = ("count_objects", "gather_order")
    frozenlake_levels: tuple[tuple[int, int], ...] = ((4, 3),)   # (size, holes) per level; ViSkill/VAGEN use 4x4
    sokoban_levels: tuple[tuple[int, int], ...] = ((6, 1),)      # (size, boxes); ViSkill/VAGEN use 6x6 with one box
    craftax_budget: int = 64
    cauldron_repo: str = "patrickamadeus/the_cauldron"
    cauldron_subset: str = "sample_1pct"      # 17k rows, ~11k checkable tasks; also sample_10pct ... all
    cauldron_revision: str = "3047119200ebf8c4c7a42fedc591303a2e703d34"   # pinned: seeds are row indices
    cauldron_sources: tuple[str, ...] = ()    # empty = every source

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
    micro_batch: int = 8           # turns per training forward pass (every weight method)
    rest_lr: float = 1e-5          # rsi/rest.py: fine-tuning on its own successes
    ppo_value_lr: float = 1e-3     # rsi/ppo.py: the value head (the policy uses grpo_lr and grpo_clip)
    lora_r: int = 8
    lora_alpha: float = 16

    # rsi/memory.py
    mem_write_threshold: float = 1.0
    mem_top_k: int = 3
    mem_max_size: int = 500
    mem_dedupe_sim: float = 0.95
    mem_retrieve_alpha: float = 1.0   # score = alpha * similarity + (1 - alpha) * utility; ViSkill uses 0.9
    mem_retrieve_min: float = -1.0    # skills scoring below this are not shown; ViSkill uses 0.8
    mem_max_excess_steps: int = -1    # admit a success only within this many steps of the solver; -1 = any length
    skill_format: str = "action"      # what a skill shows: action | text | text+action | visual | text+visual
    viskill_lambda: float = 0.8       # rsi/viskill.py: weight of the skill-following reward on failures
    hindsight_per_task: int = 1       # rsi/hindsight.py: relabelled copies kept per task per iteration
    expel_top_k: int = 5              # rsi/expel.py: rules shown in every prompt

    # rsi/curriculum.py (the frontier planner)
    curriculum_pool: int = 8       # candidates drawn per task returned
    curriculum_epsilon: float = 0.1
    frontier_band: tuple[float, float] = (0.2, 0.8)   # where a GRPO group carries signal
    frontier_unseen_weight: float = 1.0
    frontier_off_band_weight: float = 0.05

    # planners: rsi/tscl.py, rsi/accel.py, rsi/exp3.py (rsi/voyager.py falls back to tscl)
    tscl_alpha: float = 0.3        # smoothing of the success-rate slope
    accel_replay: float = 0.5      # share of slots that replay (or edit) a buffered level
    accel_edit: float = 0.5        # share of replays that become an edited new level
    accel_buffer: int = 64
    exp3_gamma: float = 0.2        # exploration of the update bandit
    exp3_k: int = 1                # updating methods run per iteration

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
