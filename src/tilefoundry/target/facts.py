"""Target-owned Facts shared across consumer families and their validation."""

from __future__ import annotations

import dataclasses
from typing import TypeVar

FactsT = TypeVar("FactsT")

TARGET_MEMORY_OWNER = "target"


@dataclasses.dataclass(frozen=True)
class TopologyLimitFacts:
    """The static extent ceiling one topology level admits.

    ``None`` means that the level has no static ceiling and may defer its extent
    to launch. ``from_target`` marks a level whose extent and program ids both
    come from the target instance rather than from a hardware document or a
    register the device can read.
    """

    name: str
    max_static_extent: int | None
    from_target: bool = False


@dataclasses.dataclass(frozen=True)
class TopologyFacts:
    """The topology levels one Target admits, coarsest first.

    The names a program may declare are exactly the names listed here, so the
    level vocabulary of a backend has one place to be stated.
    """

    topologies: tuple[TopologyLimitFacts, ...]


@dataclasses.dataclass(frozen=True)
class ParallelCapacityFacts:
    """How many instances of one topology level run at once.

    This is a compiler policy expressed over a hardware fact, not a hardware
    limit: the number of parallel units the plan assumes it may occupy. A
    tighter policy changes the plan, never the program.
    """

    topology: str
    parallel_units: int


class TargetFactsError(Exception):
    """A Target failed to provide the requested immutable Facts aggregate."""


def facts_result(
    target: object, facts_type: type[FactsT], value: object
) -> FactsT:
    """Validate and return one Facts value provided by *target*."""
    if not isinstance(facts_type, type):
        raise TargetFactsError(
            f"{type(target).__name__}: Facts type must be a class, got "
            f"{type(facts_type).__name__}"
        )
    if not dataclasses.is_dataclass(facts_type) or not facts_type.__dataclass_params__.frozen:
        raise TargetFactsError(
            f"{type(target).__name__}: {facts_type.__name__} must be a frozen "
            "dataclass Facts aggregate"
        )
    if not isinstance(value, facts_type):
        raise TargetFactsError(
            f"{type(target).__name__}: Facts projection for {facts_type.__name__} "
            f"returned {type(value).__name__}"
        )
    return value


__all__ = [
    "FactsT",
    "ParallelCapacityFacts",
    "TARGET_MEMORY_OWNER",
    "TargetFactsError",
    "TopologyFacts",
    "TopologyLimitFacts",
    "facts_result",
]
