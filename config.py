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
    response_prefix: str = "ACT:"  # prefilled assistant turn: 39/40 parseable vs 4/40 without, for a 256M model.
                                   # "" restores free reasoning before the ACT: line, for larger policies


@dataclass
class LoopConfig:
    iters: int = 20
    group_size: int = 8            # rollouts per task; GRPO advantages are relative within this group
    tasks_per_iter: int = 4
    update_rule: str = "memory"    # the carrier: "weights" | "memory" | "scaffold" — exactly one per run
    proposer: str = "fixed"        # "fixed" | "self"
    proposer_pool: int = 8         # self proposer: candidates drawn per task it returns
    proposer_band: tuple[float, float] = (0.2, 0.8)   # the competence frontier: where GRPO groups carry signal
    proposer_epsilon: float = 0.1  # uniform mixing, so nothing is ever starved
    proposer_unseen_weight: float = 1.0
    proposer_off_band_weight: float = 0.05
    env: str = "static"            # "static" | "craftax"
    verifier: str = "outcome"      # "outcome" | "judge" | "composite"
    env_seed: int = 0              # fixes the frozen held-out set
    craftax_budget: int = 64       # env steps per Craftax episode
    heldout_n: int = 200
    gate_n: int = 32               # tasks drawn from the gate split per accept/reject decision
    eval_every: int = 5
    eval_n: int = 200
    gate_enabled: bool = True      # disabling is allowed for ablations, and is recorded in the trace
    gate_parse_fail_delta: float = 0.05
    gate_allowed_verdict_sources: tuple[str, ...] = ("outcome",)  # signals that would exist at deployment
    grpo_mode: str = "lora"        # "lora" (one 24GB card) | "full"
    grpo_lr: float = 1e-4
    grpo_clip: float = 0.2
    grpo_beta: float = 0.04        # KL-to-init coefficient; 0 disables the reference pass
    grpo_max_grad_norm: float = 1.0
    lora_r: int = 8
    lora_alpha: float = 16
    mem_write_threshold: float = 1.0
    mem_top_k: int = 3
    mem_max_size: int = 500
    mem_dedupe_sim: float = 0.95
    judge_sample_rate: float = 0.1 # composite verifier: how often the judge shadows ground truth
    judge_model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct"   # a different model from the policy, on purpose
    run_dir: str = "runs/dev"
    resume: bool = False           # reusing a non-empty run_dir is refused unless resuming on purpose
    seed: int = 0
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def hash(self) -> str:
        return hashlib.sha256(repr(sorted(asdict(self).items())).encode()).hexdigest()[:12]
