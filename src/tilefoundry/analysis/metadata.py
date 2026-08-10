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
    """Record one call's logical work as authored.

    ``flops`` groups global work by dtype; ``flops_per_unit`` applies shard
    projection at the requested topology level. Global traffic is counted once.
    ``operands`` is positional against ``(*call.args, call)`` and exists only
    for direct primitive calls, not aggregate calls into another function.
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
