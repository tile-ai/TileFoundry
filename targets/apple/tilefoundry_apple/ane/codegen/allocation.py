"""Place complete guarded L2 buffers by inclusive pass lifetimes."""

from collections.abc import Iterable
from dataclasses import dataclass

from tilefoundry_apple.ane import isa


@dataclass(frozen=True)
class Interval:
    """One buffer's inclusive first/last pass and guarded byte reservation."""

    key: object
    first: int
    last: int
    size: int


def allocate(intervals: Iterable[Interval]) -> dict[object, int]:
    """Place live buffers in disjoint aligned ranges, reusing expired storage."""
    addresses = {}
    placed = []
    for interval in sorted(intervals, key=lambda item: item.first):
        if not 0 <= interval.first <= interval.last or interval.size <= 0 or interval.size % 16:
            raise ValueError("L2 intervals require ordered passes and positive 16-byte sizes")
        if interval.key in addresses:
            raise ValueError("duplicate L2 allocation key")
        occupied = sorted(
            (base, base + item.size) for item, base in placed
            if interval.first <= item.last and item.first <= interval.last
        )
        address = 0
        for low, high in occupied:
            if address + interval.size <= low:
                break
            address = max(address, high)
        if address + interval.size > isa.dma.L2_CAPACITY:
            raise ValueError(f"L2 live buffers exceed capacity {isa.dma.L2_CAPACITY}")
        addresses[interval.key] = address
        placed.append((interval, address))
    return addresses


__all__ = ["Interval", "allocate"]
