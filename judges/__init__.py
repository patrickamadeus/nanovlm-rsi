"""Judges turn a trajectory into a reward. Each has one method: score(traj) -> ScoredTrajectory."""
from judges.outcome import OutcomeJudge, ScoredTrajectory


def make_judge(cfg, env=None):
    if cfg.judge == "outcome":
        return OutcomeJudge()
    if cfg.judge in ("vlm", "composite"):
        from configs.config import PolicyConfig
        from judges.vlm import CompositeJudge, VLMJudge, YesNo
        p = cfg.policy
        vlm = VLMJudge(YesNo(PolicyConfig(model_id=cfg.judge_model_id, tiny=p.tiny, device=p.device, dtype=p.dtype)))
        if cfg.judge == "vlm":
            return vlm
        return CompositeJudge(OutcomeJudge(), vlm, getattr(env, "families", set()), cfg.judge_sample_rate, cfg.seed)
    raise ValueError(f"unknown judge {cfg.judge!r}")


__all__ = ["OutcomeJudge", "ScoredTrajectory", "make_judge"]
