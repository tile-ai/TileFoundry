"""Typed target facts consumed by the private pipeline scheduler.

The level a pipeline is asked about and the level whose resources bound it are
two different things, and this record keeps them apart. A CUDA pipeline is asked
about `thread`, because what it decides is how the warps of one CTA overlap; the
tile they cooperate on is shared memory, which is a CTA-scoped resource. Stating
one name for both would claim a per-thread capacity that no hardware publishes.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.schedule.facts import AtomFact


@dataclass(frozen=True)
class PipelineFactsQuery:
    """The target-independent program facts needed for one projection."""

    topology: str
    statements: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class PipelineInstructionFacts:
    """The supported instruction choices for one stable statement ID."""

    statement_id: str
    candidates: tuple[AtomFact, ...]


@dataclass(frozen=True)
class PipelineFacts:
    """All concrete information required to close one pipeline problem.

    `topology` is the level that was asked about. `tile_capacity_scope` names the
    level the capacity below it belongs to, which may be a coarser one: the
    cooperating threads of one such unit share that store between them.
    """

    topology: str
    tile_capacity_scope: str
    tile_capacity_bytes: int
    max_threads_per_warp: int
    instructions: tuple[PipelineInstructionFacts, ...]


__all__ = ["PipelineFacts", "PipelineFactsQuery", "PipelineInstructionFacts"]
