"""The hardware each analysis family asks for, and nothing more.

A family declares the narrow aggregate it needs; the target package registers
the conversion that builds it. What a family cannot see it cannot depend on, so
these types are the record of how much hardware each measurement actually rests
on -- and none of them names a backend. A fact used by more than one consumer
family belongs in ``tilefoundry.target.facts`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tilefoundry.ir.types import DType


class MemoryRelationKind(Enum):
    """How two memory levels are related.

    The vocabulary is deliberately small and non-hierarchical. A hierarchy that
    is only ever a tree cannot say that a cache and an addressable level divide
    one physical block, which is exactly the relationship that decides how much
    of that block either one gets.
    """

    CACHES = "caches"
    SHARES_CAPACITY_WITH = "shares_capacity_with"


@dataclass(frozen=True)
class ExplicitMemoryLevelFacts:
    """A level a program places values in by name.

    ``scope`` is the topology level ``capacity_bytes`` is stated per, so a
    per-CTA and a per-device capacity are not accidentally compared.
    """

    name: str
    capacity_bytes: int | None
    scope: str


@dataclass(frozen=True)
class ImplicitMemoryLevelFacts:
    """A level traffic passes through without being placed there.

    A program never allocates in a cache, so this level has no footprint of its
    own. Its capacity is still worth knowing: it decides whether the working set
    of the level it caches will stay resident.
    """

    name: str
    capacity_bytes: int | None
    scope: str


@dataclass(frozen=True)
class MemoryLevelRelation:
    """One edge between two memory levels.

    ``shared_capacity_bytes`` is set only on a capacity-sharing edge, where it
    is the size of the block both levels are taken from.
    """

    kind: MemoryRelationKind
    near: str
    far: str
    shared_capacity_bytes: int | None = None


@dataclass(frozen=True)
class MemoryHierarchyFacts:
    """Every memory level of one target, as a flat graph.

    The levels are two flat tuples and the structure is a separate edge list,
    rather than a nesting. A target whose cache is shared with an addressable
    level, or whose two caches have no containment relationship at all, is then
    describable without picking a parent for anything.
    """

    explicit_levels: tuple[ExplicitMemoryLevelFacts, ...]
    implicit_levels: tuple[ImplicitMemoryLevelFacts, ...]
    relations: tuple[MemoryLevelRelation, ...]

    def explicit(self, name: str) -> ExplicitMemoryLevelFacts | None:
        """The explicit level called *name*, if the target has one."""
        return next(
            (level for level in self.explicit_levels if level.name == name), None
        )

    def implicit(self, name: str) -> ImplicitMemoryLevelFacts | None:
        """The implicit level called *name*, if the target has one."""
        return next(
            (level for level in self.implicit_levels if level.name == name), None
        )

    def cached_level(self, name: str) -> str | None:
        """The level *name* caches, if it caches one."""
        return next(
            (
                relation.far
                for relation in self.relations
                if relation.kind is MemoryRelationKind.CACHES
                and relation.near == name
            ),
            None,
        )

    def backing_level(self, name: str) -> str:
        """The addressable level *name* ultimately caches.

        A cache in front of a cache is followed to the end of the chain, because
        the working set that decides whether it holds is the one the program
        actually placed somewhere.
        """
        seen = {name}
        current = name
        while (nearer := self.cached_level(current)) is not None:
            if nearer in seen:
                raise ValueError(
                    f"memory hierarchy: caching cycle through {current!r}"
                )
            seen.add(nearer)
            current = nearer
        return current

    def capacity_sharers(self, name: str) -> tuple[tuple[str, int | None], ...]:
        """Every level *name* divides a physical block with, and that block's size."""
        return tuple(
            (
                relation.far if relation.near == name else relation.near,
                relation.shared_capacity_bytes,
            )
            for relation in self.relations
            if relation.kind is MemoryRelationKind.SHARES_CAPACITY_WITH
            and name in (relation.near, relation.far)
        )


@dataclass(frozen=True)
class ThroughputFacts:
    """The rates a roofline divides work by.

    ``bandwidth_level`` names the memory level ``memory_bandwidth`` describes,
    so the bound is computed from the traffic at that level rather than from
    every level summed together.
    """

    peak_flops_per_second: tuple[tuple[DType, int], ...]
    memory_bandwidth_bytes_per_second: int | None
    bandwidth_level: str

    def peak_for(self, dtype: DType) -> int | None:
        """The published rate for *dtype*, or ``None`` when there is none."""
        return next(
            (rate for entry, rate in self.peak_flops_per_second if entry == dtype),
            None,
        )


@dataclass(frozen=True)
class ParallelCapacityFacts:
    """How many instances of one topology level run at once.

    This is a compiler policy expressed over a hardware fact, not a hardware
    limit: the number of parallel units the plan assumes it may occupy. A
    tighter policy changes the plan, never the program.
    """

    topology: str
    parallel_units: int


__all__ = [
    "ExplicitMemoryLevelFacts",
    "ImplicitMemoryLevelFacts",
    "MemoryHierarchyFacts",
    "MemoryLevelRelation",
    "MemoryRelationKind",
    "ParallelCapacityFacts",
    "ThroughputFacts",
]
