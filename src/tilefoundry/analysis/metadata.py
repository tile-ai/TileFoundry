"""Define the records each analysis family attaches to IR.

Compute cost depends only on authored IR; memory and roofline records depend on
a target. Attachment identifies granularity without changing a record's
meaning. Function records describe one analysis call and are never cached
across calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tilefoundry.ir.core.metadata import IRMetadata
from tilefoundry.visitor_registry.contexts import TrafficBytes


@dataclass(frozen=True)
class Spread[V]:
    """One quantity, whole and as one unit of each topology level holds it.

    ``per_unit`` runs in the order the record's ``topologies`` states, one
    entry per level, so a level's name is written once for the whole record
    rather than once per quantity. A finer level's unit sits inside a coarser
    one, so its share is the coarser share divided again by whatever the mesh
    splits between them -- which is why every level is stated rather than only
    the one an analysis was asked about.
    """

    total: V
    per_unit: tuple[V, ...] = ()

    def at(self, index: int) -> V:
        """One level's share by position, or the total when there is none."""
        return self.per_unit[index] if index < len(self.per_unit) else self.total


@dataclass(frozen=True)
class Breakdown[V]:
    """One category's quantities, split by the kind of thing each one is.

    The kind is what the rate pricing it is stated for: a dtype prices flops,
    a service kind prices what is not floating point, a memory level prices
    bytes. Two kinds are never summed, because two rates cannot be.
    """

    kinds: tuple[tuple[str, Spread[V]], ...] = ()

    def of(self, kind: str) -> Spread[V] | None:
        """This kind's quantity, or ``None`` when the record states none."""
        return next((value for name, value in self.kinds if name == kind), None)

    def names(self) -> tuple[str, ...]:
        """Every kind this record states, in the order it states them."""
        return tuple(name for name, _ in self.kinds)


def breakdown[V](
    total: "Mapping[str, V]", per_unit: "Sequence[Mapping[str, V]]", zero: V
) -> Breakdown[V]:
    """Gather one category's kinds, each with its total and every level's share.

    A kind any of them states appears in all of them, at *zero* where it was
    not stated, so one kind's total and its shares stay one row.
    """
    kinds = sorted({*total, *(kind for share in per_unit for kind in share)})
    return Breakdown(
        tuple(
            (
                kind,
                Spread(
                    total.get(kind, zero),
                    tuple(share.get(kind, zero) for share in per_unit),
                ),
            )
            for kind in kinds
        )
    )


def shares[V](
    held: Breakdown[V], topologies: tuple[str, ...], topology_level: "str | None" = None
) -> dict[str, V]:
    """Each kind's value for one unit of *topology_level*, or its total without one.

    The only place a level's name is turned back into a position, because
    ``topologies`` is where the names are written and a ``Spread`` states its
    shares in that order and carries none of its own. A level the record does
    not state reads as the total, which is what a record over one unit says.
    """
    index = topologies.index(topology_level) if topology_level in topologies else None
    return {
        kind: spread.total if index is None else spread.at(index) for kind, spread in held.kinds
    }


@dataclass(frozen=True)
class ComputeCostMetadata(IRMetadata):
    """Record one occurrence's work, or one Function's total work.

    ``service`` counts what is not floating point -- comparing, selecting,
    whole-number arithmetic -- by the service it asks for, because a predicate
    priced as a FLOP is a number about a pipe the work never went down. What an
    occurrence moves is a separate record, kept by the family that knows where
    values live. On a Call these state one occurrence; on a Function, loops
    contribute their trip count.
    """

    topologies: tuple[str, ...] = ()
    flops: Breakdown[int] = Breakdown()
    service: Breakdown[int] = Breakdown()


@dataclass(frozen=True)
class TrafficMetadata(IRMetadata):
    """The bytes one occurrence moves, in the two coordinates a move has.

    ``storage`` says where the bytes are; ``communication`` says whose boundary
    they crossed, which a storage level cannot answer: data handed from one
    card to another is global memory at both ends and has still gone
    somewhere. One movement is counted in both, because it spends both.
    ``operands`` is positional against ``(*call.args, call)`` on a Call and
    empty on a Function, whose totals count each occurrence as often as its
    loops repeat it.
    """

    topologies: tuple[str, ...] = ()
    storage: Breakdown[TrafficBytes] = Breakdown()
    communication: Breakdown[TrafficBytes] = Breakdown()
    operands: tuple[TrafficBytes, ...] = ()


@dataclass(frozen=True)
class MemoryLevelFootprint:
    """How much of one memory level a function needs at its peak.

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
class BufferFootprint:
    """Per-position, device-wide, and repeated bytes touched in one buffer."""

    buffer: str
    memory_level: str
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
    memory_level: str
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

    footprint: tuple[MemoryLevelFootprint, ...] = ()
    lifetimes: tuple[ValueLifetime, ...] = ()
    advisories: tuple[str, ...] = ()
    allocation: "AllocationMetadata | None" = None

    def memory_level(self, name: str) -> MemoryLevelFootprint | None:
        """The footprint recorded for *name*, if the function touches it."""
        return next((item for item in self.footprint if item.memory_level == name), None)


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
    "Breakdown",
    "BufferFootprint",
    "ComputeCostMetadata",
    "LoopFootprintMetadata",
    "MemoryLevelFootprint",
    "MemoryMetadata",
    "PerformanceMetadata",
    "PerformanceSummaryMetadata",
    "RooflineMetadata",
    "Spread",
    "TimelineMetadata",
    "TrafficBytes",
    "ValueLifetime",
    "breakdown",
    "shares",
]
