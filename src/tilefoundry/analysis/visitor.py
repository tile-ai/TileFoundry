"""Shared structural facts collected from one normalized HIR graph."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core.module import Module
from tilefoundry.target import Target


@dataclass
class AnalyzeContext:
    """Per-call inputs and the current shared lexical scope."""

    module: Module
    target: Target
    level: str | None
    options: object | None
    root: "Scope"
    current: "Scope"
    def enter(self, child: "Scope") -> "AnalyzeContext":
        """Return a context focused on one child lexical scope."""
        return type(self)(self.module, self.target, self.level, self.options, self.root, child)


__all__ = ["AnalyzeContext"]
