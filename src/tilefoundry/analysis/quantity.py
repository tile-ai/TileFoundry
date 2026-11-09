"""Quantities counted in logical, total and per-unit domains, grouped by resource kind."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tilefoundry.interfaces import cost


@dataclass(frozen=True)
class Spread[V]:
    """One typed value in logical, expanded, and topology-unit domains.

    ``per_unit`` follows ``ComputeCostMetadata.topologies``. The value retains
    its semantic structure while ``Spread`` states only its counting domain.
    """

    total: V
    per_unit: tuple[V, ...] = ()
    logical_groups: tuple[tuple[tuple[Var, ...], V], ...] = ()

    @property
    def logical(self) -> V:
        logical = scaled(self.total, 0)
        for _, value in self.logical_groups:
            logical = added(logical, value)
        return logical

    def at(self, index: int) -> V:
        """One topology level's value by position, or ``total`` when absent."""
        return self.per_unit[index] if index < len(self.per_unit) else self.total


@dataclass(frozen=True)
class Breakdown[V]:
    """Typed quantities grouped by resource kind for traffic records."""

    kinds: tuple[tuple[str, Spread[V]], ...] = ()

    def of(self, kind: str) -> Spread[V] | None:
        return next((value for name, value in self.kinds if name == kind), None)

    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.kinds)


def breakdown[V](
    total: "Mapping[str, V]",
    per_unit: "Sequence[Mapping[str, V]]",
    zero: V,
    *,
    logical: "Mapping[str, V] | None" = None,
) -> Breakdown[V]:
    """Gather keyed spreads, filling missing values in every counting domain."""
    logical = total if logical is None else logical
    kinds = sorted({*logical, *total, *(kind for share in per_unit for kind in share)})
    return Breakdown(
        tuple(
            (
                kind,
                Spread(
                    logical_groups=(((), logical.get(kind, zero)),),
                    total=total.get(kind, zero),
                    per_unit=tuple(share.get(kind, zero) for share in per_unit),
                ),
            )
            for kind in kinds
        )
    )


def shares[V](
    held: Breakdown[V],
    topologies: tuple[str, ...],
    topology_level: "str | None" = None,
) -> dict[str, V]:
    """Each kind's value at one topology level, or its total."""
    index = topologies.index(topology_level) if topology_level in topologies else None
    return {
        kind: spread.total if index is None else spread.at(index) for kind, spread in held.kinds
    }


def scaled[V](value: V, factor: int) -> V:
    """Multiply a work count or a directional byte count."""
    if isinstance(value, cost.TrafficBytes):
        return cost.TrafficBytes(value.read * factor, value.write * factor)
    return value * factor


def added[V](left: V, right: V) -> V:
    """Add work counts or directional byte counts."""
    if isinstance(left, cost.TrafficBytes):
        return cost.TrafficBytes(left.read + right.read, left.write + right.write)
    return left + right


__all__ = ["Breakdown", "Spread", "added", "breakdown", "scaled", "shares"]
