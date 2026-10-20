"""Scaffold carrier (A2 source stored as A4), Slice 6: the policy rewrites its own prompt template.

A meta-prompt shows the current template plus sampled failures and asks for a
rewrite; policy.prompt.validate_scaffold rejects proposals that break the slot
contract before gating. Versions live at scaffolds/v0.txt, v1.txt, ...; commit
repoints PolicyConfig.scaffold_path, revert repoints it back.
"""
from update.base import Updater


class ScaffoldUpdater(Updater):
    def __init__(self, policy, cfg):
        raise NotImplementedError("scaffold carrier lands in Slice 6")

    def propose(self, batch): ...
    def commit(self, update): ...
    def revert(self, update): ...
    def persisted(self, update): ...
