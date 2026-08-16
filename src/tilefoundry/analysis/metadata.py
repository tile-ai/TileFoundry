"""Define the records each analysis family attaches to IR.

Compute cost depends only on authored IR; memory and roofline records depend on
a target. Attachment identifies granularity without changing a record's
meaning. Function records describe one analysis call and are never cached
across calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core.metadata import IRMetadata
from tilefoundry.visitor_registry.contexts import TrafficBytes


@dataclass(frozen=True)
class OccurrenceProvenance(IRMetadata):
    """Identify the authored call and Function-call path of one occurrence."""

    source_call: int
    call_path: tuple[str, ...]


@dataclass(frozen=True)
class ComputeCostMetadata(IRMetadata):
    """Record one occurrence's work or one Function's total work.

    ``flops`` and ``traffic`` state global work; their ``*_per_unit`` partners
    apply shard projection at the requested topology level. On a Call, all
    quantities state one occurrence and ``operands`` is positional against
    ``(*call.args, call)``. On a Function, loops contribute their trip count and
    ``operands`` is empty.
    """

    flops: tuple[tuple[str, int], ...] = ()
    flops_per_unit: tuple[tuple[str, int], ...] = ()
    traffic: tuple[tuple[str, TrafficBytes], ...] = ()
    traffic_per_unit: tuple[tuple[str, TrafficBytes], ...] = ()
    operands: tuple[TrafficBytes, ...] = ()

    def traffic_at(self, level: str) -> TrafficBytes:
        """Traffic at *level*, zero when the call does not touch it."""
        return next(
            (value for name, value in self.traffic if name == level), TrafficBytes()
        )

    def traffic_per_unit_at(self, level: str) -> TrafficBytes:
        """One unit's traffic at *level*, zero when it does not touch it."""
        return next(
            (value for name, value in self.traffic_per_unit if name == level),
            TrafficBytes(),
        )


@dataclass(frozen=True)
class LevelFootprint:
    """How much of one memory level a function needs at its peak.

    ``persistent_bytes`` is the part that cannot be reclaimed within the
    function, so it is the floor the peak can never fall below.
    """

    level: str
    peak_bytes: int
    persistent_bytes: int
    capacity_bytes: int | None = None

    @property
    def exceeds_capacity(self) -> bool:
        """Whether the peak does not fit the stated capacity."""
        return self.capacity_bytes is not None and self.peak_bytes > self.capacity_bytes


@dataclass(frozen=True)
class BufferFootprint:
    """Per-position, device-wide, and repeated bytes touched in one buffer."""

    buffer: str
    level: str
    bytes: int
    device_bytes: int
    repeated_bytes: int


@dataclass(frozen=True)
class LoopFootprintMetadata(IRMetadata):
    """Buffer bytes touched by one authored loop, grouped by storage level.

    ``known`` is false when some access has no representable relation; the
    retained footprints are then a lower bound over the accesses that are known.
    """

    footprints: tuple[BufferFootprint, ...]
    known: bool


@dataclass(frozen=True)
class ValueLifetime:
    """One value's residency, as positions in the function's definition order.

    ``persistent`` marks a value that is resident for the whole function rather
    than until its last use. Every parameter is persistent because a function
    cannot reclaim caller-owned storage.

    ``binding`` names one value: where an authored name covers several, the later
    ones carry the numeric suffix the printed form of the same program uses.
    """

    binding: str
    level: str
    bytes: int
    defined_at: int
    last_used_at: int
    persistent: bool = False


@dataclass(frozen=True)
class MemoryMetadata(IRMetadata):
    """Record one function's memory behavior against a target hierarchy.

    Function attachment reflects that peaks span all live ranges. Advisories
    report cache working-set and order-dependent peak overflow; only a single
    value exceeding an addressable level is an error because no schedule can
    place it.
    """

    footprint: tuple[LevelFootprint, ...] = ()
    traffic: tuple[tuple[str, TrafficBytes], ...] = ()
    lifetimes: tuple[ValueLifetime, ...] = ()
    advisories: tuple[str, ...] = ()

    def level(self, name: str) -> LevelFootprint | None:
        """The footprint recorded for *name*, if the function touches it."""
        return next((item for item in self.footprint if item.level == name), None)


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
class TimelineMetadata(IRMetadata):
    """One occurrence's CTA-local interval on the nominal timeline.

    A repeated loop-body occurrence states its first interval plus the trip
    count and stride needed to derive every later interval.
    """

    start_ns: int = 0
    end_ns: int = 0
    trips: int = 1
    stride_ns: int = 0


@dataclass(frozen=True)
class TimelineSummaryMetadata(IRMetadata):
    """One Function's local schedule and physical-wave estimate."""

    local_makespan_ns: int = 0
    waves: int = 1
    estimated_kernel_ns: int = 0


__all__ = [
    "BufferFootprint",
    "ComputeCostMetadata",
    "LevelFootprint",
    "LoopFootprintMetadata",
    "MemoryMetadata",
    "OccurrenceProvenance",
    "RooflineMetadata",
    "TimelineMetadata",
    "TimelineSummaryMetadata",
    "TrafficBytes",
    "ValueLifetime",
]
