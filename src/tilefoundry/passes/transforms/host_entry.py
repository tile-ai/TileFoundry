"""Target-resolved host entry generation."""

from __future__ import annotations

from tilefoundry.ir import core
from tilefoundry.passes import pass_base


class InsertHostEntryPass(pass_base.ModulePass):
    """Give a Module its Target's host entries; the default has none to add."""

    name = "insert-host-entry"
    requires = ()

    @classmethod
    def create(cls, target, options) -> InsertHostEntryPass:
        """Create the host entry pass selected by the Target."""
        return cls()

    def run(self, module: core.Module) -> core.Module:
        return module


__all__ = ["InsertHostEntryPass"]
