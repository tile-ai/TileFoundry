from __future__ import annotations

import pytest

from tilefoundry.ir.target import CudaTarget
from tilefoundry.providers import resolve_provider_services
from tilefoundry.providers.cuda import (
    CudaArchitectureProfile,
    CudaDeviceProfile,
    CudaFormulaCostModel,
)
from tilefoundry.providers.services import TargetScheduleProfile


def test_h200_cta_services_are_resolved_through_ioc() -> None:
    services = resolve_provider_services(
        CudaTarget(arch="sm_90", device="h200_sxm"), "cta"
    )
    assert services.get(CudaArchitectureProfile).arch == "sm_90"
    assert services.get(CudaDeviceProfile).sm_count == 132
    assert services.get(TargetScheduleProfile).max_ctas == 132
    assert isinstance(services.get(CudaFormulaCostModel), CudaFormulaCostModel)


def test_cuda_autodist_requires_a_concrete_device() -> None:
    with pytest.raises(ValueError, match=r"requires CudaTarget\(device=\.\.\.\)"):
        resolve_provider_services(CudaTarget(arch="sm_90"), "cta")


def test_unsupported_level_and_device_fail_clearly() -> None:
    with pytest.raises(ValueError, match="level 'thread'"):
        resolve_provider_services(
            CudaTarget(arch="sm_90", device="h200_sxm"), "thread"
        )
    with pytest.raises(ValueError, match="unsupported CUDA AutoDist device"):
        resolve_provider_services(
            CudaTarget(arch="sm_90", device="unknown"), "cta"
        )
