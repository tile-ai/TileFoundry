"""Apple M2 Pro device resources.

Every value is built from the installed ``apple.m2_pro`` document; this module
holds the shape of the device value, never a copy of its numbers. The AMX
register files are ISA geometry and belong to the architecture, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType
from tilefoundry.target.base import Device


@dataclass(frozen=True)
class AppleM2Pro(Device):
    """One Apple M2 Pro package: its cores, caches, memory, and unit rates."""

    name: str
    # The parallel-unit count a makespan divides work over: the AMX units, not
    # the cores -- eight performance cores share two units.
    sm_count: int
    performance_core_count: int
    efficiency_core_count: int
    l1d_bytes_per_performance_core: int
    l1d_bytes_per_efficiency_core: int
    l2_bytes_per_performance_cluster: int
    l2_bytes_per_efficiency_cluster: int
    cache_line_bytes: int
    unified_memory_capacity_bytes: int
    unified_memory_owner: str
    unified_memory_bandwidth_bytes_per_second: int
    # Measured throughput per execution unit, not a published peak: Apple
    # states neither an AMX nor a NEON instruction rate.
    _unit_flops: tuple[tuple[str, tuple[tuple[DType, int], ...]], ...]

    def _python_import_module(self) -> str:
        if type(self) is AppleM2Pro:
            return "tilefoundry.target.amx"
        return super()._python_import_module()

    @property
    def unit_flops_per_second(self) -> dict[str, dict[DType, int]]:
        """Return the measured compute-throughput map of every execution unit."""
        return {unit: dict(entries) for unit, entries in self._unit_flops}

    def throughput_for(self, unit: str, dtype: DType) -> int:
        """Return measured throughput of execution ``unit`` for a ``dtype``."""
        try:
            return self.unit_flops_per_second[unit][dtype]
        except KeyError:
            raise ValueError(
                f"{self.name}: no measured compute-throughput entry for unit "
                f"{unit!r} at dtype {getattr(dtype, 'name', dtype)!r}"
            ) from None


__all__ = ["AppleM2Pro"]
