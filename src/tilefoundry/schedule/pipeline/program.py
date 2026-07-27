"""Target-independent construction of one immutable pipeline program view."""

from __future__ import annotations

from dataclasses import dataclass

import isl

from tilefoundry.analysis.poly import TileGraph, TileUnit, extract
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.schedule.kernel_schedule import build_schedule_tree

from .facts import PipelineFactsQuery


@dataclass(frozen=True)
class PipelineProgram:
    """Analysis facts plus a private ISL tree, without mutating TileGraph."""

    graph: TileGraph
    tree: "isl.schedule"
    units: tuple[TileUnit, ...]

    def facts_query(self, topology: str) -> PipelineFactsQuery:
        """Return the explicit query used to project target facts once."""
        return PipelineFactsQuery(
            topology=topology,
            statements=tuple((unit.name, unit.op) for unit in self.units),
        )


def build_pipeline_program(module: Module, function: Function) -> PipelineProgram:
    """Extract and tree one function without attaching schedule state to analysis."""
    if function not in module.functions:
        raise ValueError(
            f"pipeline program: {function.name!r} is not owned by module {module.name!r}"
        )
    graph = extract(function)
    return PipelineProgram(graph=graph, tree=build_schedule_tree(graph), units=graph.units)


__all__ = ["PipelineProgram", "build_pipeline_program"]
