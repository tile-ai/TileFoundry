"""Define the records each analysis family attaches to IR.

Compute cost depends only on authored IR; memory and roofline records depend on
a target. Attachment identifies granularity without changing a record's
meaning. Function records describe one analysis call and are never cached
across calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import ClassVar, Protocol, runtime_checkable

from tilefoundry.ir.core.metadata import IRMetadata
from tilefoundry.visitor_registry.contexts import TrafficBytes

from .access import BufferAlias
from .precision import AnalysisPrecision


@runtime_checkable
class TimingValuation(Protocol):
    """Translate a definition's event and time coordinates to its caller."""

    def event(self, local: int) -> int: ...

    def timeline(self, local: TimelineMetadata) -> TimelineMetadata: ...


@runtime_checkable
class LoopNestValuation(Protocol):
    """Describe the loops surrounding a call in its caller's coordinates."""

    def repeats(self) -> Repeats: ...

    def domain(self) -> isl.set: ...


@runtime_checkable
class VarianceValuation(Protocol):
    """Resolve a formal parameter's dependency on enclosing loops."""

    def varies(self, formal: Var) -> frozenset[int]: ...

    def trips(self, formals: tuple[Var, ...]) -> tuple[int, AnalysisPrecision]: ...


@runtime_checkable
class BindingValuation(Protocol):
    """Resolve parameter and return storage at one call boundary."""

    def actual(self, formal: Var, index: int = 0) -> BufferAlias: ...

    def result(self) -> BufferAlias: ...

    def project(self, source: BufferAlias) -> tuple[BufferAlias, AnalysisPrecision]: ...


@runtime_checkable
class InstanceValuation(Protocol):
    """Name the owning module instance on the current call path."""

    def path(self) -> tuple[str, ...]: ...


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

    logical: V
    total: V
    per_unit: tuple[V, ...] = ()
    logical_groups: tuple[tuple[tuple[Var, ...], V], ...] = ()

    def __post_init__(self) -> None:
        if self.logical_groups:
            logical = scaled(self.logical, 0)
            for _formals, value in self.logical_groups:
                logical = added(logical, value)
            object.__setattr__(self, "logical", logical)

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
                    logical=logical.get(kind, zero),
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


def resolve_counts[V](
    held: Breakdown[V], ctx: ValuationProvider
) -> tuple[Breakdown[V], AnalysisPrecision]:
    """Evaluate logical groups using only the declared variance interface."""
    valuation = ctx.request_valuation(VarianceValuation)
    precision = AnalysisPrecision.EXACT
    entries = []
    for kind, spread in held.kinds:
        logical = scaled(spread.logical, 0)
        for formals, value in spread.logical_groups or (((), spread.logical),):
            factor, bounded = valuation.trips(formals)
            logical = added(logical, scaled(value, factor))
            precision = precision.join(bounded)
        entries.append((kind, replace(spread, logical=logical, logical_groups=())))
    return Breakdown(tuple(entries)), precision


@dataclass(frozen=True)
class ComputeCostMetadata(IRMetadata):
    """Floating-point and other operation kinds, each with one counting spread."""

    topologies: tuple[str, ...] = ()
    flops: Breakdown[int] = Breakdown()
    other_ops: Breakdown[int] = Breakdown()
    precision: AnalysisPrecision = AnalysisPrecision.EXACT
    requires: ClassVar[tuple[type, ...]] = (VarianceValuation,)

    def resolve(self, ctx: ValuationProvider) -> ComputeCostMetadata:
        flops, first = resolve_counts(self.flops, ctx)
        other, second = resolve_counts(self.other_ops, ctx)
        return replace(
            self, flops=flops, other_ops=other, precision=self.precision.join(first).join(second)
        )


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

    buffers: tuple[tuple[str, Breakdown[int]], ...] = ()
    precision: AnalysisPrecision = AnalysisPrecision.EXACT
    reached: tuple[ReachedAddresses, ...] = ()
    memory_level: str = ""
    groups: tuple[tuple[int, tuple[ReachedAddresses, ...]], ...] = ()


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
    alias: BufferAlias | None = None
    requires: ClassVar[tuple[type, ...]] = (VarianceValuation, BindingValuation, LoopNestValuation)

    def resolve(self, ctx: ValuationProvider) -> MemoryMetadata:
        from .footprint import resolve_footprint  # noqa: PLC0415

        storage, _first = resolve_counts(self.traffic.storage, ctx)
        communication, _second = resolve_counts(self.traffic.communication, ctx)
        return replace(
            self,
            traffic=Traffic(storage, communication),
            footprint=resolve_footprint(self.footprint, ctx),
        )


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

    ``binding`` names one value: where an authored name covers several, the later
    ones carry the numeric suffix the printed form of the same program uses.
    """

    binding: str
    memory_level: str
    bytes: int
    defined_at: int
    last_used_at: int
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

    buffer: str
    time: str = ""
    space: str = ""
    holds_bytes: int = 0
    reuse_bytes: int = 0
    fits: bool = True
    precision: AnalysisPrecision = AnalysisPrecision.EXACT


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
    requires: ClassVar[tuple[type, ...]] = ()

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
    requires: ClassVar[tuple[type, ...]] = (VarianceValuation, BindingValuation, LoopNestValuation)


@dataclass(frozen=True)
class TimelineMetadata:
    """One interval on the nominal timeline.

    A repeated loop-body occurrence states its first interval plus the trip
    count and stride needed to derive every later interval. This is a value a
    performance record carries rather than a record of its own: what the
    interval spans is decided by the record it sits in.
    """

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
    requires: ClassVar[tuple[type, ...]] = (TimingValuation, LoopNestValuation)

    def resolve(self, ctx: ValuationProvider) -> PerformanceMetadata:
        timing = ctx.request_valuation(TimingValuation)
        ctx.request_valuation(LoopNestValuation)
        return replace(self, timeline=timing.timeline(self.timeline))


@dataclass(frozen=True)
class PerformanceSummaryMetadata(IRMetadata):
    """One Function's predicted time, and what reaching it took.

    ``timeline`` is the whole-Function envelope from zero, so its duration is
    the prediction; ``waves`` is the uniform scaling between one local wave and
    that envelope. The prediction is exact for the model it states, so there is
    nothing here about how it was reached.
    """

    timeline: TimelineMetadata
    waves: int
    requires: ClassVar[tuple[type, ...]] = (TimingValuation, LoopNestValuation)

    def resolve(self, ctx: ValuationProvider) -> PerformanceSummaryMetadata:
        timing = ctx.request_valuation(TimingValuation)
        ctx.request_valuation(LoopNestValuation)
        return replace(self, timeline=timing.timeline(self.timeline))


__all__ = [
    "BindingValuation",
    "InstanceValuation",
    "LoopNestValuation",
    "TimingValuation",
    "VarianceValuation",
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
