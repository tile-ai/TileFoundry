"""Target-resolved host entry generation."""

from __future__ import annotations

from tilefoundry.ir.core.module import Module
from tilefoundry.passes.pass_base import ModulePass


class InsertHostEntryPass(ModulePass):
    """Give a Module its Target's host entries; the default has none to add."""

    name = "insert-host-entry"
    requires = ()

    @classmethod
    def create(cls, target, options) -> InsertHostEntryPass:
        """Create the host entry pass selected by the Target."""
        return cls()

    def run(self, module: Module) -> Module:
        return module
