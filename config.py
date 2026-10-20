import hashlib
from dataclasses import asdict, dataclass, field


@dataclass
class PolicyConfig:
    model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    tiny: bool = False             # randomly-initialized tiny VLM for CPU tests; same code path as the real one
    max_new_tokens: int = 64       # reasoning budget per step, the "Think briefly" in the scaffold
    temperature: float = 1.0       # GRPO needs real sampling diversity within a group
    dtype: str = "bfloat16"
    device: str = "auto"           # "auto" picks cuda when available
    scaffold_path: str = "scaffolds/v0.txt"


@dataclass
class LoopConfig:
    iters: int = 20
    group_size: int = 8            # rollouts per task; GRPO advantages are relative within this group
    tasks_per_iter: int = 4
    update_rule: str = "memory"    # the carrier: "weights" | "memory" | "scaffold" — exactly one per run
    proposer: str = "fixed"        # "fixed" | "self"
    env: str = "static"            # "static" | "craftax"
    env_seed: int = 0              # fixes the frozen held-out set
    heldout_n: int = 200
    gate_n: int = 32               # tasks drawn from the gate split per accept/reject decision
    eval_every: int = 5
    eval_n: int = 200
    gate_enabled: bool = True      # disabling is allowed for ablations, and is recorded in the trace
    gate_parse_fail_delta: float = 0.05
    gate_allowed_verdict_sources: tuple[str, ...] = ("outcome",)  # signals that would exist at deployment
    mem_write_threshold: float = 1.0
    mem_top_k: int = 3
    mem_max_size: int = 500
    mem_dedupe_sim: float = 0.95
    judge_sample_rate: float = 0.1
    run_dir: str = "runs/dev"
    seed: int = 0
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def hash(self) -> str:
        return hashlib.sha256(repr(sorted(asdict(self).items())).encode()).hexdigest()[:12]
