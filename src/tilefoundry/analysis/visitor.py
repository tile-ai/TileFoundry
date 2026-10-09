"""Shared structural facts collected from one normalized HIR graph."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types.mesh import Mesh
from tilefoundry.ir.types.utils import local_type_of
from tilefoundry.target import Target


@dataclass
class AnalyzeContext:
    """Per-call inputs and the current shared lexical scope.

    Analysis reads the program ``check_program`` typed for it, so the type of a
    value is the one stored on that IR. ``root`` and ``current`` are bound once
    the scope tree is built; reading types and relations does not need them.
    ``current_mesh`` is the execution mesh a relation is asked under; analysis
    asks outside any, as type inference does at a Function's top level.
    """

    module: Module
    target: Target
    topology_level: str | None
    options: object | None
    root: "IterationScope | None" = None
    current: "IterationScope | None" = None
    current_mesh: "Mesh | None" = None

    @property
    def topologies(self):
        return self.module.effective_topologies()

    def type_of(self, expr):
        """The type the analyzed IR holds for *expr*."""
        return expr.type

    def local_type_of(self, expr):
        """*expr*'s type as one unit of ``topology_level`` holds it, or whole without one."""
        if self.topology_level is None:
            return expr.type
        return local_type_of(
            expr.type, topology_level=self.topology_level, topologies=self.topologies
        )


__all__ = ["AnalyzeContext"]
