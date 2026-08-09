"""Provider-owned V100 Target used by the installed smoke workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tilefoundry import DType
from tilefoundry.target import (
    Analyzer,
    Architecture,
    CudaTarget,
    Device,
    Scheduler,
    register_target,
)

_SERVICE_CALLS = Path(__file__).with_name("v100_service_calls.txt")


def _record_service(kind: str, selector: str) -> None:
    with _SERVICE_CALLS.open("a", encoding="utf-8") as calls:
        calls.write(f"{kind}:{selector}\n")


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
    smem_owner: str = "cta"
    unified_l1_shared_per_sm_bytes: int = 128 * 1024
    registers_per_sm_32bit: int = 64 * 1024
    rmem_owner: str = "thread"
    tensor_memory_per_cta_bytes: int | None = None
    tmem_owner: str = "cta"

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
    gmem_owner: str = "target"
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

    def __init__(
        self,
        architecture: Architecture = VOLTA_70,
        device: Device = V100_SXM2_32GB,
    ) -> None:
        super().__init__(device, architecture)

    def get_analyzer(self, selector: str) -> Analyzer:
        inherited = super().get_analyzer(selector)
        _record_service("analyzer", selector)
        return Analyzer(
            inherited.selector,
            inherited.run,
            inherited.requires,
            inherited.produces,
        )

    def get_scheduler(self, topology: str) -> Scheduler:
        inherited = super().get_scheduler(topology)
        _record_service("scheduler", topology)
        return Scheduler(inherited.topology, inherited.solve)
