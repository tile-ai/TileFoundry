"""Fixed Apple M2 Pro device facts.

Every value here is recorded with its provenance in
``target/hardware/apple_m2_pro_amx.toml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.types import DType
from tilefoundry.target.base import Device

# Measured single-AMX-unit f32 throughput, not a published peak: Apple states no
# AMX instruction rate. See the TOML fact `amx_f32_unit_throughput`.
_AMX_UNIT_FLOPS = ((DType.f32, 504_900_000_000),)


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
    _amx_flops: tuple[tuple[DType, int], ...] = field(
        default=_AMX_UNIT_FLOPS, init=False, repr=False
    )

    # The resource solve reads one tile-memory capacity and one memory bandwidth
    # off the device by name. On AMX a tile lives in the Z accumulator file and
    # is fed from unified memory.

    @property
    def l1_capacity_bytes(self) -> int:
        """Return the capacity one tile's resident footprint is bounded by."""
        return self.amx_accumulator_bytes

    @property
    def l2_bandwidth_bytes_per_second(self) -> int:
        """Return the bandwidth a tile's traffic is charged against."""
        return self.unified_memory_bandwidth_bytes_per_second

    @property
    def amx_unit_flops_per_second(self) -> dict[DType, int]:
        """Return the measured per-unit compute-throughput map."""
        return dict(self._amx_flops)

    def throughput_for(self, dtype: DType) -> int:
        """Return measured per-unit AMX throughput for a compute ``dtype``."""
        try:
            return self.amx_unit_flops_per_second[dtype]
        except KeyError:
            raise ValueError(
                f"{self.name}: no measured AMX compute-throughput entry for dtype "
                f"{getattr(dtype, 'name', dtype)!r}"
            ) from None


__all__ = ["AppleM2Pro"]
