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
    @classmethod
    def create(
        cls,
        module: Module,
        graph: Function,
        target: Target,
        level: str | None = None,
        options: object | None = None,
    ) -> "AnalyzeContext":
        """Build a context and collect one scope/access tree for ``graph``."""
        from .scope import build_scopes  # noqa: PLC0415

        root = build_scopes(module, graph)
        return cls(module, target, level, options, root, root)

    def enter(self, child: "Scope") -> "AnalyzeContext":
        """Return a context focused on one child lexical scope."""
        return type(self)(self.module, self.target, self.level, self.options, self.root, child)


__all__ = ["AnalyzeContext"]
