"""Check physical storage reuse on an explicitly supplied execution timeline."""

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from numbers import Real

__all__ = ["BufferAccess", "BufferHazard", "BufferLifetime", "check_buffer_lifetimes"]


@dataclass(frozen=True)
class BufferAccess:
    """An operation's half-open access interval in a caller-selected time unit."""

    operation: str
    start: float
    end: float


@dataclass(frozen=True)
class BufferLifetime:
    """One produced value and all its consumers in a physical byte range.

    Allocation names identify disjoint storage, including its owning device.
    Distinct writes use distinct value names, even when reusing an allocation.
    An initially available input can have a zero-duration producer.
    """

    value: str
    allocation: str
    offset: int
    nbytes: int
    producer: BufferAccess
    consumers: tuple[BufferAccess, ...] = ()

    @property
    def end(self) -> float:
        """Return the end of the last access, including the producer."""
        return max((self.producer.end, *(use.end for use in self.consumers)))


@dataclass(frozen=True)
class BufferHazard:
    """A readiness or overlapping-storage violation with its responsible operations."""

    kind: str
    allocation: str
    values: tuple[str, ...]
    operations: tuple[str, ...]


def check_buffer_lifetimes(
    lifetimes: tuple[BufferLifetime, ...], *, predecessors: dict[str, frozenset[str]] | None = None
) -> tuple[BufferHazard, ...]:
    """Check declared producer readiness and live physical byte-range collisions.

    With predecessors, readiness requires a transitive happens-before edge;
    numeric intervals then only order value versions. Otherwise they share an
    ordered clock domain. Consumers include asynchronous reads through completion.
    """
    names = set()
    hazards = []
    if predecessors is not None:
        for operation, prior in predecessors.items():
            if operation in prior or any(parent not in predecessors or not predecessors[parent] <= prior for parent in prior):
                raise ValueError("predecessors must define a strict transitive happens-before relation")
    def before(earlier, later):
        return (earlier.end <= later.start if predecessors is None
                else earlier.operation in predecessors.get(later.operation, frozenset()))
    for value in lifetimes:
        if not value.value or value.value in names or not value.allocation:
            raise ValueError("values must be uniquely named and identify an allocation")
        names.add(value.value)
        if (
            not isinstance(value.offset, int)
            or isinstance(value.offset, bool)
            or not isinstance(value.nbytes, int)
            or isinstance(value.nbytes, bool)
            or value.offset < 0
            or value.nbytes <= 0
        ):
            raise ValueError("buffer ranges require a nonnegative offset and positive byte count")
        for use in (value.producer, *value.consumers):
            if predecessors is not None and use.operation not in predecessors:
                raise ValueError("every buffer access must appear in the happens-before relation")
            if (
                not use.operation
                or not isinstance(use.start, Real)
                or not isinstance(use.end, Real)
                or not isfinite(use.start)
                or not isfinite(use.end)
                or not 0 <= use.start <= use.end
            ):
                raise ValueError("access intervals must be named, finite and nonnegative")
        for use in value.consumers:
            if not before(value.producer, use):
                hazards.append(
                    BufferHazard(
                        "read_before_ready",
                        value.allocation,
                        (value.value,),
                        (value.producer.operation, use.operation),
                    )
                )
    ordered = sorted(lifetimes, key=lambda value: (value.producer.start, value.value))
    for old, new in combinations(ordered, 2):
        if old.allocation != new.allocation:
            continue
        byte_overlap = max(old.offset, new.offset) < min(
            old.offset + old.nbytes, new.offset + new.nbytes
        )
        if byte_overlap and any(not before(use, new.producer) for use in (old.producer, *old.consumers)):
            blockers = tuple(
                use.operation
                for use in (old.producer, *old.consumers)
                if not before(use, new.producer)
            )
            hazards.append(
                BufferHazard(
                    "overwrite_live_value",
                    old.allocation,
                    (old.value, new.value),
                    (*blockers, new.producer.operation),
                )
            )
    return tuple(hazards)
