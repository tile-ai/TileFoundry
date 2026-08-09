"""CUDA device resources.

One value type answers for every CUDA device. What separates an H200 SXM from a
B200 SXM is what their installed documents record, so there is no class per
product; the per-SM structural limits belong to the architecture rather than
here.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType
from tilefoundry.target.base import Device


@dataclass(frozen=True)
class CudaDevice(Device):
    """One CUDA device: how many SMs, and the memory and compute rates."""

    name: str
    sm_count: int
    hbm_capacity_bytes: int
    gmem_owner: str
    hbm_bandwidth_bytes_per_second: int
    # None when the product specification states no L2 capacity: an advisory
    # about a cache whose size is unknown is worth less than no advisory.
    l2_capacity_bytes: int | None
    _dense_flops: tuple[tuple[DType, int], ...]

    def _python_import_module(self) -> str:
        # The package re-exports this type, so a rendered constructor imports it
        # from there. A provider's own subclass lives elsewhere and keeps its own
        # module.
        if type(self) is CudaDevice:
            return "tilefoundry.target.cuda"
        return super()._python_import_module()

    @property
    def dense_flops_per_second(self) -> dict[DType, int]:
        """Return the dense compute-throughput map."""
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


__all__ = ["CudaDevice"]
