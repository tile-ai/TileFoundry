"""The sole compilation Target implementation package."""

from __future__ import annotations

from tilefoundry.target.amx import AmxTarget, AppleAmx, AppleM2Pro
from tilefoundry.target.base import (
    Architecture,
    Device,
    Target,
    TargetLoadError,
    UnsupportedCapabilityError,
    load_targets,
    register_target,
    registered_targets,
)
from tilefoundry.target.cpu import CpuTarget
from tilefoundry.target.cuda import CudaArchitecture, CudaDevice, CudaTarget
from tilefoundry.target.facts import (
    MemoryExecutionFacts,
    TargetFactsError,
    TopologyFacts,
    TopologyLevelFacts,
    facts_result,
)
from tilefoundry.target.services import Analyzer

_ANALYSIS_FACTS = {
    "MemoryHierarchyFacts",
    "PerformanceServiceFacts",
    "ThroughputFacts",
}


def __getattr__(name: str):
    if name in _ANALYSIS_FACTS:
        from tilefoundry.analysis import facts  # noqa: PLC0415

        return getattr(facts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def default_target() -> Target:
    """Return the normal compile-entry default target."""
    return CpuTarget()


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
    "MemoryExecutionFacts",
    "PerformanceServiceFacts",
    "Target",
    "TargetLoadError",
    "TargetFactsError",
    "TopologyFacts",
    "TopologyLevelFacts",
    "ThroughputFacts",
    "UnsupportedCapabilityError",
    "default_target",
    "facts_result",
    "load_targets",
    "register_target",
    "registered_targets",
]
