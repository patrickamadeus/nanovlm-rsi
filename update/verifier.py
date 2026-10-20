"""Verifier carrier (role B2): a stub by design.

Its training data is already being collected: on env-native tasks the judge runs at
judge_sample_rate alongside ground truth, and rollout records carry `verdict_source`
and `judge_verdict`. Judge-vs-truth disagreements are what this updater would learn from.
"""
from update.base import Updater


class VerifierUpdater(Updater):
    def __init__(self, policy, cfg):
        raise NotImplementedError("verifier self-improvement is a stub; its data is collected from Slice 4")

    def propose(self, batch): ...
    def commit(self, update): ...
    def revert(self, update): ...
    def persisted(self, update): ...
