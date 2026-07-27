"""Deterministic closed-problem scheduling decisions."""

from __future__ import annotations

from dataclasses import dataclass

from .problem import PipelineProblem


class PipelineSolveError(ValueError):
    """A closed pipeline problem has no legal finite solution."""


@dataclass(frozen=True)
class PipelineStatementSolution:
    """One chosen instruction and its half-open interval."""

    id: str
    instruction: object
    start: int
    end: int
    resources: tuple[tuple[str, int], ...]


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
        statements.append(
            PipelineStatementSolution(
                statement.id, chosen, clock, clock + duration, statement.resources
            )
        )
        clock += duration
    buffers = tuple(
        PipelineBufferSolution(item.id, 1, item.producer_ids, item.consumer_ids)
        for item in problem.buffers
    )
    return PipelineSolution(tuple(statements), buffers)


__all__ = [
    "PipelineBufferSolution",
    "PipelineSolution",
    "PipelineSolveError",
    "PipelineStatementSolution",
    "solve_pipeline_problem",
]
