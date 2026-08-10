"""The closed, target-free constraint input for pipeline scheduling.

Everything here is measured off the program's own schedule tree. Two of those
measurements are what make an intra-CTA pipeline a pipeline rather than a
sequence: the dependence distance a buffer carries, which is how many tiles of
it have to be alive at once, and the bytes each statement holds, which is what
the level's tile store has to fit. Neither is a decision, so both are closed
into the problem and neither is left for the solver to guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import isl

from tilefoundry.analysis.poly import (
    AccessFootprint,
    access_footprints,
    carried_distances,
)
from tilefoundry.ir.types.shard import Topology
from tilefoundry.schedule.kernel_schedule import band_statement, schedule_bands

from ..errors import ScheduleError
from .facts import PipelineFacts
from .program import PipelineProgram


class PipelineProblemError(ScheduleError):
    """The program and projected facts cannot form a finite schedule problem.

    A scheduling failure, and reachable as one: a caller asking this
    layer to schedule something catches what the layer raises, and a
    capability that cannot be scheduled is recorded against that. Sitting
    outside `ScheduleError` made a limit of this algorithm unstateable
    except as a bare `ValueError`, which is also what a caller passing
    nonsense gets -- so the two could not be told apart.
    """


@dataclass(frozen=True)
class PipelineStatementProblem:
    """One statement's explicit legal instruction and resource choices.

    `footprint_bytes` is what one instance of this statement holds in each
    buffer it touches, before any ring multiplies it. A dimension is counted at
    the widest extent any of this statement's accesses reaches there, so two
    accesses to one buffer are counted once.
    """

    id: str
    extents: tuple[int, ...]
    candidates: tuple[object, ...]
    resources: tuple[tuple[str, int], ...]
    footprint_bytes: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PipelineBufferProblem:
    """One storage object, its users, and measured carried distances.

    `carried_distances` is per holding statement, because a distance is only
    meaningful against the extents of the statement whose band reported it: the
    same buffer carried two iterations spans a different number of tiles under
    a statement tiled 2 wide than under one tiled 64 wide.
    """

    id: str
    producer_ids: tuple[str, ...]
    consumer_ids: tuple[str, ...]
    carried_distances: tuple[tuple[str, tuple[int, ...]], ...]


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
    if facts.topology != topology.name:
        raise PipelineProblemError(
            f"pipeline facts describe {facts.topology!r}, not topology "
            f"{topology.name!r}"
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
    accesses = access_footprints(program.graph, time_maps)
    bands = _bands_by_statement(program)
    held = _held_by_statement(accesses)
    distances: dict[str, dict[str, tuple[int, ...]]] = {}
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
        distances[unit.name] = _carried_for(program, unit.name, bands, extents)
        statements.append(
            PipelineStatementProblem(
                unit.name,
                extents,
                choices,
                resources,
                footprint_bytes=tuple(
                    (buffer, _occupancy_bytes(occupancy))
                    for buffer, occupancy in sorted(held.get(unit.name, {}).items())
                ),
            )
        )
    writers: dict[str, list[str]] = {}
    readers: dict[str, list[str]] = {}
    for footprint in accesses:
        if footprint.is_read:
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
            carried_distances=tuple(
                (statement, per_buffer[name])
                for statement, per_buffer in sorted(distances.items())
                if name in per_buffer
            ),
        )
        for name in sorted(set(writers) | set(readers))
    )
    return PipelineProblem(topology.name, facts.tile_capacity_bytes, tuple(statements), buffers)


def _bands_by_statement(program: PipelineProgram) -> dict[str, object]:
    """One band per statement, keyed by the statement it schedules."""
    bands = {band_statement(band): band for band in schedule_bands(program.tree)}
    missing = sorted(unit.name for unit in program.units if unit.name not in bands)
    if missing:
        raise PipelineProblemError(
            f"statements {missing!r} have no band in the schedule tree"
        )
    return bands


def _carried_for(
    program: PipelineProgram,
    name: str,
    bands: dict[str, object],
    extents: tuple[int, ...],
) -> dict[str, tuple[int, ...]]:
    """Per buffer, the distances statement *name*'s own band carries.

    Only the buffers that carry something are kept: a buffer with no carried
    dependence needs one slot, and saying so with a tuple of zeros would put
    every buffer in the program into every statement's record.
    """
    time_map = bands[name].get_partial_schedule_union_map()
    return {
        buffer: carried
        for buffer, carried in carried_distances(
            program.graph, time_map, len(extents)
        ).items()
        if any(carried)
    }


def _held_by_statement(
    footprints: tuple[AccessFootprint, ...],
) -> dict[str, dict[str, tuple[tuple[int, ...], int]]]:
    """Held by statement.

    Group *footprints* by statement then buffer, widening each buffer
    dimension to every extent any access in that group needs there.
    """
    dims: dict[tuple[str, str], list[set[int]]] = {}
    elem_bytes: dict[tuple[str, str], int] = {}
    for footprint in footprints:
        group = (footprint.statement, footprint.buffer)
        if group not in dims:
            dims[group] = [set() for _ in footprint.dims]
            elem_bytes[group] = footprint.elem_bytes
        if len(dims[group]) != len(footprint.dims):
            raise PipelineProblemError(
                f"buffer {footprint.buffer!r} is accessed with "
                f"{len(footprint.dims)} and {len(dims[group])} dimension(s)"
            )
        for position, extent in enumerate(footprint.dims):
            dims[group][position].add(extent.extent)
    held: dict[str, dict[str, tuple[tuple[int, ...], int]]] = {}
    for (statement, buffer), per_dim in dims.items():
        widest = tuple(max(options) for options in per_dim)
        held.setdefault(statement, {})[buffer] = (
            widest,
            elem_bytes[(statement, buffer)],
        )
    return held


def _occupancy_bytes(occupancy: tuple[tuple[int, ...], int]) -> int:
    """Bytes one instance of a statement holds in one buffer."""
    widest, elem_bytes = occupancy
    return math.prod(widest) * elem_bytes


__all__ = [
    "PipelineBufferProblem",
    "PipelineProblem",
    "PipelineProblemError",
    "PipelineStatementProblem",
    "build_pipeline_problem",
]
