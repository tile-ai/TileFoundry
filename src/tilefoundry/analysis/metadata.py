"""Define the records each analysis family attaches to IR.

Compute cost depends only on authored IR; memory and roofline records depend on
a target. Attachment identifies granularity without changing a record's
meaning. Function records describe one analysis call and are never cached
across calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import ClassVar

from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.core.metadata import IRMetadata
from tilefoundry.visitor_registry.contexts import TrafficBytes

from .precision import AnalysisPrecision


@dataclass(frozen=True)
class ExecutionMeshMetadata(IRMetadata):
    """The unique enclosing mesh established while checking one definition."""

    mesh: Mesh | None


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
        for _dependencies, value in self.logical_groups:
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
    if isinstance(value, TrafficBytes):
        return TrafficBytes(value.read * factor, value.write * factor)
    return value * factor


def added[V](left: V, right: V) -> V:
    """Add work counts or directional byte counts."""
    if isinstance(left, TrafficBytes):
        return TrafficBytes(left.read + right.read, left.write + right.write)
    return left + right


@dataclass(frozen=True)
class ComputeCostMetadata(IRMetadata):
    """Floating-point and other operation kinds, each with one counting spread."""

    topologies: tuple[str, ...] = ()
    flops: Breakdown[int] = Breakdown()
    other_ops: Breakdown[int] = Breakdown()
    precision: AnalysisPrecision = AnalysisPrecision.EXACT


@dataclass(frozen=True)
class Traffic:
    """Movement by storage level and crossed communication boundary.

    One movement is counted in both coordinates because it spends both.
    A Call's ``logical`` is authored movement, ``per_unit`` is each unit's share,
    and ``total`` multiplies the counted share by executing mesh positions.
    Storage counts the selected unit; communication counts the boundary's unit.
    Function accumulation multiplies logical movement by varying-loop trips
    and the other domains by all enclosing trips.
    """

    storage: Breakdown[TrafficBytes] = Breakdown()
    communication: Breakdown[TrafficBytes] = Breakdown()


@dataclass(frozen=True)
class Footprint:
    """Unique bytes one wave touches, by source buffer and memory level."""

    groups: tuple[tuple[int, tuple[ReachedAddresses, ...]], ...] = ()

    @cached_property
    def _grouped(self):
        from .footprint import merged  # noqa: PLC0415

        return tuple((windows, merged(items)) for windows, items in self.groups)

    @cached_property
    def reached(self) -> tuple[ReachedAddresses, ...]:
        from .footprint import merged  # noqa: PLC0415

        return merged(item for _windows, items in self._grouped for item in items)

    @cached_property
    def buffers(self) -> tuple[tuple[SSAValue, tuple[str, ...], Breakdown[int]], ...]:
        from .footprint import counted_buffers  # noqa: PLC0415

        return counted_buffers(self)

    @cached_property
    def precision(self) -> AnalysisPrecision:
        from .footprint import counted_precision  # noqa: PLC0415

        return counted_precision(item for _windows, items in self._grouped for item in items)

    def bytes_at(self, level: str) -> tuple[int, int] | None:
        """One window's and every executed window's bytes, when all sit at *level*.

        Reads and writes are one count: they share one bandwidth.
        """
        kinds = tuple(item for *_held, held in self.buffers for item in held.kinds)
        if any(name != level for name, _spread in kinds):
            return None
        return sum(s.logical for _n, s in kinds), sum(s.total for _n, s in kinds)


@dataclass(frozen=True)
class MemoryMetadata(IRMetadata):
    """One Call's memory behavior for one occurrence.

    ``operands`` is positional against ``(*call.args, call)``. ``footprint`` is
    absent until a read-footprint analysis has actually produced a conclusion.
    Address-placed results state one copy's size and the offset of every copy;
    register-resident results retain the defaults because they are not placed.
    """

    topologies: tuple[str, ...] = ()
    traffic: Traffic = Traffic()
    operands: tuple[TrafficBytes, ...] = ()
    footprint: Footprint | None = None
    buffer_bytes: int | None = None
    offsets: tuple[int, ...] = ()


@dataclass(frozen=True)
class MemoryLevelPeak:
    """One level's solved high-water mark or largest logical value.

    ``persistent_bytes`` is the part that cannot be reclaimed within the
    function, so it is the floor the peak can never fall below.
    """

    memory_level: str
    peak_bytes: int
    persistent_bytes: int
    capacity_bytes: int | None = None

    @property
    def exceeds_capacity(self) -> bool:
        """Whether the peak does not fit the stated capacity."""
        return self.capacity_bytes is not None and self.peak_bytes > self.capacity_bytes


@dataclass(frozen=True)
class ValueLifetime:
    """One value's residency on the function's structured SSA event timeline.

    ``persistent`` marks a value that is resident for the whole function rather
    than until its last use. Every parameter is persistent because a function
    cannot reclaim caller-owned storage.

    Display bindings are produced from ``value`` by the expression printer.
    """

    events: ClassVar = ("defined_at", "last_used_at")

    value: SSAValue
    defined_at: int
    last_used_at: int
    memory_level: str = ""
    bytes: int = 0
    persistent: bool = False


@dataclass(frozen=True)
class ReuseWindow:
    """One buffer's re-reads, and what keeping it costs the cache.

    ``time`` is the loop whose next iteration reads it again and ``space`` the
    mesh axis whose units read it at once; either may be absent. ``holds`` is
    everything the wave touches while it must stay resident -- other buffers
    included, because they are what evicts it -- and ``reuse`` the bytes that
    are not fetched again once it does.
    """

    buffer: SSAValue
    ownership: tuple[str, ...] = ()
    time: str = ""
    space: str = ""
    holds_bytes: int = 0
    reuse_bytes: int | None = 0
    fits: bool = True
    precision: AnalysisPrecision = AnalysisPrecision.EXACT
    reason: str = ""


@dataclass(frozen=True)
class RegionMemoryMetadata(IRMetadata):
    """Record one region's memory behavior against a target hierarchy.

    A Function carries whole-function traffic, placement, and lifetimes. A
    MeshRegion carries only the peaks reached while that region executes, and no
    empty record is attached merely to reserve the type.
    """

    solver_status: str
    topologies: tuple[str, ...] = ()
    traffic: Traffic = Traffic()
    footprint: Footprint | None = None
    reuse_windows: tuple[ReuseWindow, ...] = ()
    lifetimes: tuple[ValueLifetime, ...] = ()
    peaks: tuple[MemoryLevelPeak, ...] = ()
    errors: tuple[str, ...] = ()
    advisories: tuple[str, ...] = ()

    def peak_for(self, name: str) -> MemoryLevelPeak | None:
        """The peak recorded for *name*, if this region touches it."""
        return next((item for item in self.peaks if item.memory_level == name), None)


@dataclass(frozen=True)
class RooflineMetadata(IRMetadata):
    """A lower bound on time, and which side of the machine sets it.

    ``bound_by`` names the resource the bound came from, so a caller reads a
    conclusion rather than re-deriving which of two numbers was larger.

    On a Call this is that call's bound. On a Function it is the whole
    function's, which is not the sum of the calls' bounds: the compute and
    memory times are summed across the function first and only then compared,
    because a call bound by memory and a call bound by compute overlap rather
    than each stalling the machine for its own bound.
    """

    compute_ns: int = 0
    memory_ns: int = 0
    ideal_ns: int = 0
    bound_by: str = "none"


@dataclass(frozen=True)
class TimelineMetadata:
    """One interval on the nominal timeline.

    A repeated loop-body occurrence states its first interval plus the trip
    count and stride needed to derive every later interval. This is a value a
    performance record carries rather than a record of its own: what the
    interval spans is decided by the record it sits in.
    """

    events: ClassVar = ("start_ns", "end_ns")

    start_ns: int = 0
    end_ns: int = 0
    trips: int = 1
    stride_ns: int = 0


@dataclass(frozen=True)
class PerformanceMetadata(IRMetadata):
    """One occurrence's interval within one local wave of its Function.

    Only an occurrence with a modeled duration carries one. A structural
    occurrence takes no modeled time, and an empty interval on it would read as
    a measurement rather than as the absence of one.
    """

    timeline: TimelineMetadata


@dataclass(frozen=True)
class PerformanceSummaryMetadata(IRMetadata):
    """One Function's predicted time, and what reaching it took.

    ``timeline`` is the whole-Function envelope from zero, every wave the
    Function's executing positions need, so its duration is the prediction.
    ``wave`` is those positions against the physical units of the selected
    level. The prediction is exact for the model it states, so there is nothing
    here about how it was reached.
    """

    timeline: TimelineMetadata
    wave: tuple[int, int]


__all__ = [
    "Breakdown",
    "MemoryMetadata",
    "ComputeCostMetadata",
    "Footprint",
    "MemoryLevelPeak",
    "PerformanceMetadata",
    "PerformanceSummaryMetadata",
    "RegionMemoryMetadata",
    "ReuseWindow",
    "RooflineMetadata",
    "Spread",
    "TimelineMetadata",
    "Traffic",
    "TrafficBytes",
    "ValueLifetime",
    "breakdown",
    "shares",
]


def shifted(record, origin):
    """Translate only the event coordinates declared by a record."""
    return replace(record, **{name: getattr(record, name) + origin for name in record.events})
