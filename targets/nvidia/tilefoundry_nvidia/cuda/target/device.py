"""CUDA device resources.

One value type answers for every CUDA device. What separates an H200 SXM from a
B200 SXM is what their installed documents record, so there is no class per
product; the per-SM structural limits belong to the architecture rather than
here.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry import target
from tilefoundry.ir import expr


@dataclass(frozen=True)
class Device(target.Device):
    """One CUDA device: how many SMs, and the memory and compute rates."""

    name: str
    sm_count: int
    sm_clock_hz: int | None
    hbm_capacity_bytes: int
    gmem_owner: str
    hbm_bandwidth_bytes_per_second: int

    l2_capacity_bytes: int | None
    dense_flops: tuple[tuple[expr.DType, int], ...]
    service_ops: tuple[tuple[str, int], ...]
    gmem_latency_ns: int | None = None

    @property
    def dense_flops_per_second(self) -> dict[expr.DType, int]:
        """Return the dense compute-throughput map."""
        return dict(self.dense_flops)

    @property
    def service_ops_per_second(self) -> dict[str, int]:
        """What one CTA gets through of each service this device states."""
        return dict(self.service_ops)

    def peak_for(self, dtype: expr.DType) -> int:
        """Return dense device throughput for a compute ``dtype``."""
        try:
            return self.dense_flops_per_second[dtype]
        except KeyError:
            raise ValueError(
                f"{self.name}: no dense compute-throughput entry for dtype "
                f"{getattr(dtype, 'name', dtype)!r}"
            ) from None


__all__ = ["Device"]
