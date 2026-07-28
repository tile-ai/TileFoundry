"""Deterministic closed-problem scheduling decisions."""

from __future__ import annotations

from dataclasses import dataclass

from .problem import PipelineBufferProblem, PipelineProblem


class PipelineSolveError(ValueError):
    """A closed pipeline problem has no legal finite solution."""


@dataclass(frozen=True)
class PipelineStatementSolution:
    """One chosen instruction and its half-open interval.

    `footprint_bytes` counts the rings this statement's buffers were given, so
    it is what the statement really occupies once the pipeline is deep enough
    to run. `fits_capacity` records that against the level's tile store rather
    than enforcing it: a statement too wide for the store still has a
    schedule, only a worse one, and a solver that silently dropped it would
    report a plan for a program nobody asked for.
    """

    id: str
    instruction: object
    start: int
    end: int
    resources: tuple[tuple[str, int], ...]
    footprint_bytes: int
    fits_capacity: bool


@dataclass(frozen=True)
class PipelineBufferSolution:
    """One solved ring depth."""

    id: str
    ring_depth: int
    producer_ids: tuple[str, ...]
    consumer_ids: tuple[str, ...]


@dataclass(frozen=True)
class PipelineSolution:
    """All typed decisions exported by the pipeline plan."""

    statements: tuple[PipelineStatementSolution, ...]
    buffers: tuple[PipelineBufferSolution, ...]


def solve_pipeline_problem(problem: PipelineProblem) -> PipelineSolution:
    """Select the unique lowest-cost legal choice and sequence resources.

    Choice is a deterministic optimization over the complete candidate set, not
    a catalogue-order fallback. The problem has no target reference at this
    point; all legality and capacity data were frozen in PipelineFacts.
    """
    extents = {statement.id: statement.extents for statement in problem.statements}
    ring = {
        buffer.id: _ring_depth(buffer, extents) for buffer in problem.buffers
    }
    clock = 0
    statements: list[PipelineStatementSolution] = []
    for statement in problem.statements:
        ranked = sorted(
            statement.candidates,
            key=lambda fact: (fact.duration, getattr(getattr(fact, "atom", None), "op", object()).name),
        )
        if not ranked:
            raise PipelineSolveError(f"statement {statement.id!r} has no legal candidate")
        chosen = ranked[0]
        duration = max(1, round(chosen.duration * 1000))
        footprint = sum(
            held * ring.get(buffer, 1) for buffer, held in statement.footprint_bytes
        )
        statements.append(
            PipelineStatementSolution(
                statement.id,
                chosen,
                clock,
                clock + duration,
                statement.resources,
                footprint_bytes=footprint,
                fits_capacity=footprint <= problem.capacity_bytes,
            )
        )
        clock += duration
    buffers = tuple(
        PipelineBufferSolution(
            item.id, ring[item.id], item.producer_ids, item.consumer_ids
        )
        for item in problem.buffers
    )
    return PipelineSolution(tuple(statements), buffers)


def _ring_depth(
    buffer: PipelineBufferProblem, extents: dict[str, tuple[int, ...]]
) -> int:
    """One buffer's ring depth, measured rather than searched for.

    A dependence carried `distance` iterations along a dimension tiled `tile`
    wide spans `ceil(distance / tile)` tiles, and the ring holds one slot more
    than that so the older tile is still alive while the newer one fills. A
    buffer that carries nothing needs the single slot every buffer needs.
    """
    depths = [1]
    for statement, carried in buffer.carried_distances:
        tile = extents.get(statement)
        if tile is None:
            raise PipelineSolveError(
                f"buffer {buffer.id!r} carries a distance for unknown statement "
                f"{statement!r}"
            )
        if len(tile) != len(carried):
            raise PipelineSolveError(
                f"buffer {buffer.id!r} carries {len(carried)} distance(s) for "
                f"statement {statement!r}, which spans {len(tile)} dimension(s)"
            )
        depths.extend(
            -(-distance // width) + 1
            for distance, width in zip(carried, tile)
            if distance
        )
    return max(depths)


__all__ = [
    "PipelineBufferSolution",
    "PipelineSolution",
    "PipelineSolveError",
    "PipelineStatementSolution",
    "solve_pipeline_problem",
]
