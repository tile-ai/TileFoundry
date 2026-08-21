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
class ComputeCostMetadata(IRMetadata):
    """Record one occurrence's work, or one Function's total work.

    ``flops`` and ``service`` state global work; their ``*_per_unit`` partners
    apply shard projection at the requested topology level. ``service`` counts
    what is not floating point -- comparing, selecting, whole-number arithmetic
    -- by the service it asks for. What an occurrence moves is a separate
    record, kept by the family that knows where values live. On a Call these
    state one occurrence; on a Function, loops contribute their trip count.
    """

    flops: tuple[tuple[str, int], ...] = ()
    flops_per_unit: tuple[tuple[str, int], ...] = ()
    service: tuple[tuple[str, int], ...] = ()
    service_per_unit: tuple[tuple[str, int], ...] = ()

    def service_per_unit_of(self, kind: str) -> int:
        """One unit's count of *kind*, zero when the call asks for none."""
        return next((value for name, value in self.service_per_unit if name == kind), 0)


@dataclass(frozen=True)
class TrafficMetadata(IRMetadata):
    """The bytes one occurrence moves, whole and for one participant.

    Which way a boundary moves is its Op's evaluator's answer and how much is
    its relation's; the family that decides where values live attaches the
    record, and where they landed never corrects a crossing. ``operands`` is
    positional against ``(*call.args, call)`` on a Call and empty on a Function,
    whose totals count each occurrence as often as its loops repeat it.
    """

    whole: tuple[tuple[str, TrafficBytes], ...] = ()
    per_unit: tuple[tuple[str, TrafficBytes], ...] = ()
    operands: tuple[TrafficBytes, ...] = ()

    def at(self, level: str) -> TrafficBytes:
        """Bytes moved at *level*, zero when the occurrence does not touch it."""
        return next(
            (value for name, value in self.whole if name == level), TrafficBytes()
        )

    def per_unit_at(self, level: str) -> TrafficBytes:
        """One unit's bytes at *level*, zero when it does not touch it."""
        return next(
            (value for name, value in self.per_unit if name == level), TrafficBytes()
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
class AllocationMetadata:
    """What showing this function's buffers fit took.

    Where any of them would sit is the solver's business and appears nowhere
    here. What a reader can act on is whether the question was settled.
    """

    solver_status: str


@dataclass(frozen=True)
class MemoryMetadata(IRMetadata):
    """Record one function's memory behavior against a target hierarchy.

    Function attachment reflects that peaks span all live ranges. Advisories
    report cache working-set and order-dependent peak overflow; only a single
    value exceeding an addressable level is an error because no schedule can
    place it.

    ``allocation`` is absent when the function has no addressable buffer to
    place at the level being analysed, which is a different answer from having
    placed one: nothing was decided, so nothing is claimed.
    """

    footprint: tuple[LevelFootprint, ...] = ()
    lifetimes: tuple[ValueLifetime, ...] = ()
    advisories: tuple[str, ...] = ()
    allocation: "AllocationMetadata | None" = None

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


__all__ = [
    "AllocationMetadata",
    "BufferFootprint",
    "ComputeCostMetadata",
    "LevelFootprint",
    "LoopFootprintMetadata",
    "MemoryMetadata",
    "PerformanceMetadata",
    "PerformanceSummaryMetadata",
    "RooflineMetadata",
    "TimelineMetadata",
    "TrafficBytes",
    "ValueLifetime",
]
