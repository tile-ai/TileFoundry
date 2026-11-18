"""Apple GPU device resources.

Every value is built from an installed Metal device document; this module
holds the shape of the device value, never a copy of its numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry import target


@dataclass(frozen=True)
class Device(target.Device):
    """One Apple GPU: threadgroup limits and the unified memory it shares with the CPU."""

    name: str
    max_threads_per_threadgroup: int
    threadgroup_memory_bytes: int
    unified_memory_capacity_bytes: int


__all__ = ["Device"]
