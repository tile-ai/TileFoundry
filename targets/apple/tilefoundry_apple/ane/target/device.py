"""Apple Neural Engine device resources.

Every value is built from an installed ANE device document; this module
holds the shape of the device value, never a copy of its numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry import target


@dataclass(frozen=True)
class Device(target.Device):
    """One ANE: its cores, the unified memory it shares with the CPU and its measured fp16 rate."""

    name: str
    core_count: int
    unified_memory_capacity_bytes: int
    f16_throughput: int


__all__ = ["Device"]
