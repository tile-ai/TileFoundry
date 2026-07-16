from __future__ import annotations

from tilefoundry.ir.target import CudaTarget

from ..registry import register_provider
from ..services import ServiceCollection, TargetScheduleProfile
from .cost_model import CudaFormulaCostModel
from .profiles import h200_sxm_profiles


class CudaProvider:
    """IoC factory for concrete CUDA architecture/device services."""

    @staticmethod
    def services(target: CudaTarget, level: str) -> ServiceCollection:
        if level != "cta":
            raise ValueError(f"CUDA AutoDist supports level='cta', got {level!r}")
        if target.device is None:
            raise ValueError(
                "CUDA AutoDist requires CudaTarget(device=...), but no concrete device was supplied"
            )
        if target.device != "h200_sxm":
            raise ValueError(f"unsupported CUDA AutoDist device {target.device!r}")
        architecture, device = h200_sxm_profiles(target.arch)
        profile = TargetScheduleProfile(level="cta", topology="cta", max_ctas=device.sm_count)
        return ServiceCollection.from_values(
            architecture=architecture,
            device=device,
            schedule=profile,
            cost_model=CudaFormulaCostModel(architecture, device),
        )

    @classmethod
    def register(cls) -> None:
        register_provider(CudaTarget, level="cta", factory=cls.services)


__all__ = ["CudaProvider"]
