"""Fixed Apple M2 Pro device facts.

Every value here is recorded with its provenance in
``target/hardware/apple_m2_pro_amx.toml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.types import DType
from tilefoundry.target.base import Device

# Measured f32 throughput per execution unit, not a published peak: Apple states
# neither an AMX nor a NEON instruction rate. See the TOML facts
# `amx_f32_unit_throughput` and `neon_f32_core_throughput`.
_UNIT_FLOPS = (
    ("amx", ((DType.f32, 504_900_000_000),)),
    ("neon", ((DType.f32, 107_700_000_000),)),
)


@dataclass(frozen=True)
class AppleM2Pro(Device):
    """One Apple M2 Pro package with fixed hardware facts and planner policy."""

    name: str = field(default="apple_m2_pro", init=False)
    # The parallel-unit count a makespan divides work over: the AMX units, not
    # the cores -- eight performance cores share two units.
    sm_count: int = field(default=2, init=False)
    performance_core_count: int = field(default=8, init=False)
    efficiency_core_count: int = field(default=4, init=False)
    l1d_bytes_per_performance_core: int = field(default=128 * 1024, init=False)
    l1d_bytes_per_efficiency_core: int = field(default=64 * 1024, init=False)
    l2_bytes_per_performance_cluster: int = field(default=16 * 1024 * 1024, init=False)
    l2_bytes_per_efficiency_cluster: int = field(default=4 * 1024 * 1024, init=False)
    cache_line_bytes: int = field(default=128, init=False)
    unified_memory_capacity_bytes: int = field(default=32 * 1024**3, init=False)
    unified_memory_bandwidth_bytes_per_second: int = field(
        default=200_000_000_000, init=False
    )
    amx_staging_bytes: int = field(default=512, init=False)
    amx_accumulator_bytes: int = field(default=4096, init=False)
    _unit_flops: tuple[tuple[str, tuple[tuple[DType, int], ...]], ...] = field(
        default=_UNIT_FLOPS, init=False, repr=False
    )

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
