"""Typed target facts consumed by the private pipeline scheduler."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.schedule.facts import AtomFact


@dataclass(frozen=True)
class PipelineFactsQuery:
    """The target-independent program facts needed for one projection."""

    stage: str
    statements: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class PipelineInstructionFacts:
    """The supported instruction choices for one stable statement ID."""

    statement_id: str
    candidates: tuple[AtomFact, ...]


@dataclass(frozen=True)
class PipelineFacts:
    """All concrete information required to close one pipeline problem."""

    stage: str
    tile_capacity_bytes: int
    max_threads_per_warp: int
    instructions: tuple[PipelineInstructionFacts, ...]


__all__ = ["PipelineFacts", "PipelineFactsQuery", "PipelineInstructionFacts"]
