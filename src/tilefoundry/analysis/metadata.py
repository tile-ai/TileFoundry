"""Define the records each analysis family attaches to IR.

Compute cost depends only on authored IR; memory and roofline records depend on
a target. Attachment identifies granularity without changing a record's
meaning. Function records describe one analysis call and are never cached
across calls.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

from tilefoundry.interfaces import cost
from tilefoundry.ir import core, expr

from .precision import Precision
from .quantity import Breakdown
from .reach import Footprint


@dataclass(frozen=True)
class ExecutionMeshMetadata(expr.IRMetadata):
    """The unique enclosing mesh established while checking one definition."""

    mesh: Mesh | None


@dataclass(frozen=True)
class ComputeCostMetadata(expr.IRMetadata):
    """Floating-point and other operation kinds, each with one counting spread."""

    family: ClassVar[str] = "compute-cost"

    topologies: tuple[str, ...] = ()
    flops: Breakdown[int] = Breakdown()
    other_ops: Breakdown[int] = Breakdown()
    precision: Precision = Precision.EXACT


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

    storage: Breakdown[cost.TrafficBytes] = Breakdown()
    communication: Breakdown[cost.TrafficBytes] = Breakdown()


@dataclass(frozen=True)
class MemoryMetadata(expr.IRMetadata):
    """One Call's memory behavior for one occurrence.

    ``operands`` is positional against ``(*call.args, call)``. ``footprint`` is
    absent until a read-footprint analysis has actually produced a conclusion.
    Address-placed results state one copy's size and the offset of every copy;
    register-resident results retain the defaults because they are not placed.
    """

    family: ClassVar[str] = "memory"

    topologies: tuple[str, ...] = ()
    traffic: Traffic = Traffic()
    operands: tuple[cost.TrafficBytes, ...] = ()
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

    value: core.SSAValue
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

    buffer: core.SSAValue
    ownership: tuple[str, ...] = ()
    time: str = ""
    space: str = ""
    holds_bytes: int = 0
    reuse_bytes: int | None = 0
    fits: bool = True
    precision: Precision = Precision.EXACT


@dataclass(frozen=True)
class RegionMemoryMetadata(expr.IRMetadata):
    """Record one region's memory behavior against a target hierarchy.

    A Function carries whole-function traffic, placement, and lifetimes. A
    MeshRegion carries only the peaks reached while that region executes, and no
    empty record is attached merely to reserve the type.
    """

    family: ClassVar[str] = "memory"

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
class RooflineMetadata(expr.IRMetadata):
    """A lower bound on time, and which side of the machine sets it.

    ``bound_by`` names the resource the bound came from, so a caller reads a
    conclusion rather than re-deriving which of two numbers was larger.

    On a Call this is that call's bound. On a Function it is the whole
    function's, which is not the sum of the calls' bounds: the compute and
    memory times are summed across the function first and only then compared,
    because a call bound by memory and a call bound by compute overlap rather
    than each stalling the machine for its own bound.
    """

    family: ClassVar[str] = "roofline"

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
class PerformanceMetadata(expr.IRMetadata):
    """One occurrence's interval within one local wave of its Function.

    Only an occurrence with a modeled duration carries one. A structural
    occurrence takes no modeled time, and an empty interval on it would read as
    a measurement rather than as the absence of one.
    """

    family: ClassVar[str] = "performance"

    timeline: TimelineMetadata


@dataclass(frozen=True)
class PerformanceSummaryMetadata(expr.IRMetadata):
    """One Function's predicted time, and what reaching it took.

    ``timeline`` is the whole-Function envelope from zero, every wave the
    Function's executing positions need, so its duration is the prediction.
    ``wave`` is those positions against the physical units of the selected
    level. The prediction is exact for the model it states, so there is nothing
    here about how it was reached.
    """

    family: ClassVar[str] = "performance"

    timeline: TimelineMetadata
    wave: tuple[int, int]


def shifted(record, origin):
    """Translate only the event coordinates declared by a record."""
    return replace(record, **{name: getattr(record, name) + origin for name in record.events})


__all__ = [
    "ComputeCostMetadata",
    "ExecutionMeshMetadata",
    "MemoryLevelPeak",
    "MemoryMetadata",
    "PerformanceMetadata",
    "PerformanceSummaryMetadata",
    "RegionMemoryMetadata",
    "ReuseWindow",
    "RooflineMetadata",
    "TimelineMetadata",
    "Traffic",
    "ValueLifetime",
    "shifted",
]
