"""Policies: act(prompt, images) -> RolloutStep, logprobs(prompt, images, token_ids), embed(text, images)."""


def make_policy(policy_cfg):
    from models.smolvlm import Policy        # any HF image-text-to-text model; LoRA targets SmolVLM's names
    return Policy(policy_cfg)
