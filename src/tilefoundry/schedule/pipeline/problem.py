"""The closed, target-free constraint input for pipeline scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import isl

from tilefoundry.analysis.poly import access_footprints
from tilefoundry.ir.types.shard import Topology

from .facts import PipelineFacts
from .program import PipelineProgram


class PipelineProblemError(ValueError):
    """The program and projected facts cannot form a finite schedule problem."""


@dataclass(frozen=True)
class PipelineStatementProblem:
    """One statement's explicit legal instruction and resource choices."""

    id: str
    extents: tuple[int, ...]
    candidates: tuple[object, ...]
    resources: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PipelineBufferProblem:
    """One storage object, its users, and measured carried distances."""

    id: str
    producer_ids: tuple[str, ...]
    consumer_ids: tuple[str, ...]
    carried_distances: tuple[int, ...]


@dataclass(frozen=True)
class PipelineProblem:
    """A complete finite problem with no Target object or callback."""

    topology: str
    capacity_bytes: int
    statements: tuple[PipelineStatementProblem, ...]
    buffers: tuple[PipelineBufferProblem, ...]


def build_pipeline_problem(
    program: PipelineProgram, facts: PipelineFacts, topology: Topology
) -> PipelineProblem:
    """Close the problem from immutable analysis and already-projected facts."""
    if facts.stage != topology.name:
        raise PipelineProblemError(
            f"pipeline facts describe {facts.stage!r}, not topology {topology.name!r}"
        )
    if facts.tile_capacity_bytes <= 0:
        raise PipelineProblemError("pipeline facts require a positive tile capacity")
    candidates = {item.statement_id: item.candidates for item in facts.instructions}
    expected = tuple(unit.name for unit in program.units)
    if tuple(candidates) != expected:
        raise PipelineProblemError(
            f"pipeline facts statements {tuple(candidates)!r} do not match program {expected!r}"
        )
    statements: list[PipelineStatementProblem] = []
    time_maps = program.tree.get_map()
    domains: dict[str, object] = {}
    program.graph.domain.foreach_set(
        lambda domain: domains.__setitem__(domain.get_tuple_name(), domain)
    )
    for unit in program.units:
        domain = domains.get(unit.name)
        if domain is None:
            raise PipelineProblemError(f"statement {unit.name!r} has no domain")
        extents = tuple(
            int(domain.dim_max_val(axis).num_si())
            - int(domain.dim_min_val(axis).num_si())
            + 1
            for axis in range(domain.dim(isl.dim_type.SET))
        )
        if not extents:
            raise PipelineProblemError(f"statement {unit.name!r} has no finite extent")
        choices = candidates[unit.name]
        if not choices:
            raise PipelineProblemError(
                f"statement {unit.name!r} has no supported instruction candidates"
            )
        resources = tuple(
            sorted(
                (name, value)
                for fact in choices
                for name, value in fact.resource.items()
                if value > 0
            )
        )
        statements.append(
            PipelineStatementProblem(unit.name, extents, choices, resources)
        )
    writers: dict[str, list[str]] = {}
    readers: dict[str, list[str]] = {}
    for footprint in access_footprints(program.graph, time_maps):
        readers.setdefault(footprint.buffer, []).append(footprint.statement)
    maps: list[object] = []
    program.graph.writes.foreach_map(maps.append)
    for mapping in maps:
        writers.setdefault(
            mapping.get_tuple_name(isl.dim_type.OUT), []
        ).append(mapping.get_tuple_name(isl.dim_type.IN))
    buffers = tuple(
        PipelineBufferProblem(
            id=name,
            producer_ids=tuple(sorted(set(writers.get(name, ()))),),
            consumer_ids=tuple(sorted(set(readers.get(name, ()))),),
            carried_distances=(),
        )
        for name in sorted(set(writers) | set(readers))
    )
    return PipelineProblem(topology.name, facts.tile_capacity_bytes, tuple(statements), buffers)


__all__ = [
    "PipelineBufferProblem",
    "PipelineProblem",
    "PipelineProblemError",
    "PipelineStatementProblem",
    "build_pipeline_problem",
]
