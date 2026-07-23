"""Fixed H200 SXM device facts."""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.types import DType
from tilefoundry.target.base import Device

_H200_DENSE_FLOPS = (
    (DType.f32, 67_000_000_000_000),
    (DType.f16, 989_500_000_000_000),
    (DType.bf16, 989_500_000_000_000),
    (DType.fp8e4m3, 1_979_000_000_000_000),
)


@dataclass(frozen=True)
class H200SXM(Device):
    """One H200 SXM device with fixed hardware facts and planner policy."""

    name: str = field(default="h200_sxm", init=False)
    sm_count: int = field(default=132, init=False)
    max_resident_ctas_per_sm: int = field(default=32, init=False)
    compiler_policy_max_parallel_ctas: int = field(default=132, init=False)
    hbm_capacity_bytes: int = field(default=141_000_000_000, init=False)
    hbm_bandwidth_bytes_per_second: int = field(default=4_800_000_000_000, init=False)
    shared_memory_per_sm_bytes: int = field(default=228 * 1024, init=False)
    shared_memory_per_cta_bytes: int = field(default=227 * 1024, init=False)
    registers_per_sm_32bit: int = field(default=65_536, init=False)
    _dense_flops: tuple[tuple[DType, int], ...] = field(
        default=_H200_DENSE_FLOPS, init=False, repr=False
    )

    @property
    def dense_flops_per_second(self) -> dict[DType, int]:
        """Return the fixed dense compute-throughput map."""
        return dict(self._dense_flops)

    def peak_for(self, dtype: DType) -> int:
        """Return dense device throughput for a compute ``dtype``."""
        try:
            return self.dense_flops_per_second[dtype]
        except KeyError:
            raise ValueError(
                f"{self.name}: no dense compute-throughput entry for dtype "
                f"{getattr(dtype, 'name', dtype)!r}"
            ) from None


__all__ = ["H200SXM"]
