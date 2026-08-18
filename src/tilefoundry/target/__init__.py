"""The sole compilation Target implementation package."""

from __future__ import annotations

from tilefoundry.target.amx import AmxTarget, AppleAmx, AppleM2Pro
from tilefoundry.target.base import (
    Architecture,
    Device,
    Target,
    UnsupportedCapabilityError,
    register_target,
    registered_targets,
)
from tilefoundry.target.cpu import CpuTarget
from tilefoundry.target.cuda import CudaArchitecture, CudaDevice, CudaTarget
from tilefoundry.target.cuda.spec import H200_SXM_ID
from tilefoundry.target.facts import (
    TargetFactsError,
    TopologyLimitFacts,
    facts_result,
)
from tilefoundry.target.services import Analyzer, Scheduler

_ANALYSIS_FACTS = {
    "MemoryHierarchyFacts",
    "ParallelCapacityFacts",
    "PerformanceServiceFacts",
    "ThroughputFacts",
}


def __getattr__(name: str):
    if name in _ANALYSIS_FACTS:
        from tilefoundry.analysis import facts  # noqa: PLC0415

        return getattr(facts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _cuda_fallback() -> CudaTarget:
    """Construct the built-in value used at compiler-owned omitted boundaries."""
    return CudaTarget(H200_SXM_ID)


def default_target() -> Target:
    """Return the normal compile-entry default target."""
    return _cuda_fallback()


def validate_cuda_topology_levels(target: Target, names) -> None:
    """Validate names at the CUDA lowering boundary.

    See docs/spec/target.md § Topology levels.
    """
    for name in names:
        if name not in target.topology_levels:
            raise ValueError(
                f"cuda target supports {{{', '.join(target.topology_levels)}}} "
                f"topology levels; got {name!r}"
            )


__all__ = [
    "AmxTarget",
    "Analyzer",
    "AppleAmx",
    "AppleM2Pro",
    "Architecture",
    "CpuTarget",
    "CudaArchitecture",
    "CudaDevice",
    "CudaTarget",
    "Device",
    "MemoryHierarchyFacts",
    "ParallelCapacityFacts",
    "PerformanceServiceFacts",
    "Scheduler",
    "Target",
    "TargetFactsError",
    "TopologyLimitFacts",
    "ThroughputFacts",
    "UnsupportedCapabilityError",
    "default_target",
    "facts_result",
    "register_target",
    "registered_targets",
    "validate_cuda_topology_levels",
]
