"""Target-owned Facts shared across consumer families and their validation."""

from __future__ import annotations

import dataclasses
from typing import TypeVar

FactsT = TypeVar("FactsT")

TARGET_MEMORY_OWNER = "target"


@dataclasses.dataclass(frozen=True)
class TopologyLevelFacts:
    """The program-side and machine-side unit counts for one topology level.

    ``max_logical_units`` is how many units a program may declare; ``None``
    means there is no static ceiling and the extent may be deferred to launch.
    ``max_physical_units`` is how many machine positions the level divides over;
    for CUDA CTAs it counts one CTA per SM and is not the hardware resident-CTA
    limit. ``from_target`` marks a level whose extent and program ids both come
    from the target instance rather than from a hardware document or a register
    the device can read.
    """

    name: str
    max_logical_units: int | None
    max_physical_units: int | None
    from_target: bool = False


@dataclasses.dataclass(frozen=True)
class TopologyFacts:
    """The topology levels one Target admits, coarsest first.

    The names a program may declare are exactly the names listed here, so the
    level vocabulary of a backend has one place to be stated.
    """

    topologies: tuple[TopologyLevelFacts, ...]
    parallel_level: str | None = None

    def __post_init__(self) -> None:
        if self.parallel_level is not None and self.level(self.parallel_level) is None:
            raise ValueError(
                f"parallel topology level {self.parallel_level!r} is not among "
                f"{tuple(level.name for level in self.topologies)}"
            )

    def level(self, name: str | None) -> TopologyLevelFacts | None:
        """Return the facts for *name*, or ``None`` when it is not stated."""
        return next((level for level in self.topologies if level.name == name), None)

    def parallel(self) -> TopologyLevelFacts | None:
        """Return the default parallel level, or ``None`` for an empty aggregate."""
        return self.level(self.parallel_level)


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
    "TARGET_MEMORY_OWNER",
    "TargetFactsError",
    "TopologyFacts",
    "TopologyLevelFacts",
    "facts_result",
]
