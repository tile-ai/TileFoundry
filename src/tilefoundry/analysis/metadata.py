"""What each analysis family leaves on the IR.

The record types are split by what the number depends on rather than by
convenience: compute cost is a property of the authored program alone, while the
memory and roofline records only mean anything against a specific target.
Keeping them apart is what lets a caller see which of its numbers would change
on other hardware.

A record's attachment point says what it is about, so the same type serves both
granularities where the quantity is the same quantity: a roofline bound on a
Call is that call's, and one on a Function is that whole function's. What a
record must never do is describe a different quantity depending on where it
hangs.

A Function-attached record is not data the Function inherently has. It is
written only when a call actually analysed that function, and says what this
analysis found; nothing is cached across calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core.metadata import IRMetadata
from tilefoundry.visitor_registry.contexts import TrafficBytes


@dataclass(frozen=True)
class ComputeCostMetadata(IRMetadata):
    """One Call's logical work, as the authored program states it.

    Flops are grouped by compute DType name rather than summed, because an op
    that mixes precisions does not have one flop count -- and which of those
    counts dominates is a question about hardware, asked later.

    ``flops`` is the operation's global arithmetic from the types as written.
    ``flops_per_unit`` is the arithmetic one unit of the analysed topology level
    performs after shard projection.

    ``operands`` breaks ``traffic`` down the other way: per operand rather than
    per level, positional against ``(*call.args, call)``. It is present only for
    a direct call on a primitive op; a call into another Function carries that
    callee's aggregate traffic, which no breakdown of this call's operands
    describes.

    Traffic is global and counted once. A launch-provided topology may make the
    per-unit column target-dependent because the target supplies its extent.
    """

    flops: tuple[tuple[str, int], ...] = ()
    flops_per_unit: tuple[tuple[str, int], ...] = ()
    traffic: tuple[tuple[str, TrafficBytes], ...] = ()
    operands: tuple[TrafficBytes, ...] = ()

    def traffic_at(self, level: str) -> TrafficBytes:
        """Traffic at *level*, zero when the call does not touch it."""
        return next(
            (value for name, value in self.traffic if name == level), TrafficBytes()
        )

    def format_comment(self) -> str:
        flop_text = ",".join(f"{name}:{value}" for name, value in self.flops) or "0"
        local_text = (
            ",".join(f"{name}:{value}" for name, value in self.flops_per_unit)
            or "0"
        )
        traffic_text = (
            ",".join(
                f"{level}:r{value.read}/w{value.write}"
                for level, value in self.traffic
            )
            or "0"
        )
        return f"compute-cost flops={flop_text} per-unit={local_text} bytes={traffic_text}"


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
    """One function's memory behaviour against one target's hierarchy.

    The record is attached to the Function rather than to a value: a peak is a
    property of the whole function's live ranges, and there is no single
    expression it belongs to.

    ``advisories`` carries the capacity findings that are not errors. A cache
    too small for the working set it fronts costs performance, so it is worth
    reporting; an explicit-level peak is likewise order-dependent. Neither
    makes the program invalid. Only one value too large for an addressable
    level fails the call, because no schedule can place it.
    """

    footprint: tuple[LevelFootprint, ...] = ()
    traffic: tuple[tuple[str, TrafficBytes], ...] = ()
    lifetimes: tuple[ValueLifetime, ...] = ()
    advisories: tuple[str, ...] = ()

    def level(self, name: str) -> LevelFootprint | None:
        """The footprint recorded for *name*, if the function touches it."""
        return next((item for item in self.footprint if item.level == name), None)

    def format_comment(self) -> str:
        peaks = (
            ",".join(f"{item.level}:{item.peak_bytes}" for item in self.footprint)
            or "0"
        )
        persistent = sum(item.persistent_bytes for item in self.footprint)
        return (
            f"memory peak={peaks} persistent={persistent} "
            f"advisories={len(self.advisories)}"
        )


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

    def format_comment(self) -> str:
        return (
            f"roofline bound={self.ideal_ns}ns by={self.bound_by} "
            f"compute={self.compute_ns}ns memory={self.memory_ns}ns"
        )


@dataclass(frozen=True)
class TimelineMetadata(IRMetadata):
    """A modeled placement on the nominal timeline.

    On a Call this is the placement of the execution unit that call belongs to.
    Every call fused into one unit carries the same record: the placement was
    decided for the unit, and a distinct one per call would suggest a resolution
    the model does not have.

    On a Function it is the whole function's span, so it starts at the origin
    and ends at the makespan the scheduling model solved for. ``grid_units`` is
    then the widest unit extent and ``waves`` the waves issued in total.
    """

    grid_units: int = 1
    waves: int = 1
    start_ns: int = 0
    end_ns: int = 0

    def format_comment(self) -> str:
        return (
            f"timeline units={self.grid_units} waves={self.waves} "
            f"start={self.start_ns}ns end={self.end_ns}ns"
        )


__all__ = [
    "ComputeCostMetadata",
    "LevelFootprint",
    "MemoryMetadata",
    "RooflineMetadata",
    "TimelineMetadata",
    "TrafficBytes",
    "ValueLifetime",
]
