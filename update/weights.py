"""Weights carrier (A5), Slice 2: GRPO + LoRA, both from scratch.

Group-relative advantage A = (r - mean) / (std + eps) within a task group; clipped
per-token objective masked to completion tokens; k3 KL (ratio - log ratio - 1)
against pi_init; per-token mean normalization; one inner epoch by default.
LoRALinear wraps the LM q/v projections, vision tower frozen. LoRA state is ~10MB,
so Policy.snapshot() before each update makes revert() exact and free.
"""
from update.base import Updater


class WeightsUpdater(Updater):
    def __init__(self, policy, cfg):
        raise NotImplementedError("weights carrier (GRPO + LoRA) lands in Slice 2")

    def propose(self, batch): ...
    def commit(self, update): ...
    def revert(self, update): ...
    def persisted(self, update): ...
