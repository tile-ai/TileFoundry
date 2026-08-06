"""Provider-owned V100 Target used by the installed smoke workflow."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry import DType
from tilefoundry.target import Architecture, CudaTarget, Device, register_target


@dataclass(frozen=True)
class Volta70(Architecture):
    """Volta compute capability 7.0 resources used by scheduling and analysis."""

    name: str = "sm_70"
    supported_compute_dtypes: tuple[DType, ...] = (DType.f16, DType.f32)
    instruction_capabilities: tuple[str, ...] = ()
    max_threads_per_cta: int = 1024
    max_threads_per_warp: int = 32
    max_warps_per_cta: int = 32
    max_resident_ctas_per_sm: int = 32
    shared_memory_per_sm_bytes: int = 96 * 1024
    shared_memory_per_cta_bytes: int = 96 * 1024
    unified_l1_shared_per_sm_bytes: int = 128 * 1024
    registers_per_sm_32bit: int = 64 * 1024

    def topology_limit(self, name: str) -> int:
        if name == "thread":
            return self.max_threads_per_cta
        raise ValueError(f"{self.name}: no limit for topology {name!r}")


@dataclass(frozen=True)
class TeslaV100SXM2_32GB(Device):
    """Published V100 SXM2 32 GB device capacity and peak rates."""

    name: str = "tesla_v100_sxm2_32gb"
    sm_count: int = 80
    hbm_capacity_bytes: int = 32_000_000_000
    hbm_bandwidth_bytes_per_second: int = 900_000_000_000
    l2_capacity_bytes: int = 6 * 1024 * 1024

    @property
    def dense_flops_per_second(self) -> dict[DType, int]:
        return {
            DType.f16: 125_000_000_000_000,
            DType.f32: 15_700_000_000_000,
        }


VOLTA_70 = Volta70()
V100_SXM2_32GB = TeslaV100SXM2_32GB()


@register_target
@dataclass(frozen=True, init=False)
class V100Target(CudaTarget):
    """External Target with a provider-owned no-argument constructor."""

    name = "nvidia.v100_sxm2_32gb"

    def __init__(self) -> None:
        super().__init__(V100_SXM2_32GB, VOLTA_70)

    def __repr__(self) -> str:
        return "V100Target()"
