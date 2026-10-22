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
    TARGET_MEMORY_OWNER,
    MemoryExecutionFacts,
    TargetFactsError,
    TopologyFacts,
    TopologyLevelFacts,
    facts_result,
    owner_levels,
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


def _cuda_fallback() -> CudaTarget:
    """Construct the built-in value used at compiler-owned omitted boundaries."""
    return CudaTarget(H200_SXM_ID)


def default_target() -> Target:
    """Return the normal compile-entry default target."""
    return _cuda_fallback()


def instance_levels(target, storage) -> frozenset[str]:
    """Topology levels owning separate instances of one explicit storage."""
    if target is None:
        return frozenset()
    from tilefoundry.analysis.facts import MemoryHierarchyFacts  # noqa: PLC0415
    from tilefoundry.ir.types.storage import StorageKind  # noqa: PLC0415

    storage = StorageKind.RMEM if storage is StorageKind.UMAT else storage
    name = str(storage)
    try:
        hierarchy = target.get_facts(MemoryHierarchyFacts)
    except UnsupportedCapabilityError:
        return frozenset()
    level = hierarchy.explicit(name)
    if level is None:
        raise ValueError(f"storage {name!r} has no memory owner on {target!r}")
    names = (
        ()
        if level.owner == TARGET_MEMORY_OWNER
        else tuple(one.name for one in target.get_facts(TopologyFacts).topologies)
    )
    try:
        return owner_levels(level.owner, names)
    except ValueError as error:
        raise ValueError(f"storage {name!r}: {error}") from None


def validate_cuda_topology_levels(target: Target, names) -> None:
    """Validate names at the CUDA lowering boundary.

    See docs/spec/target.md § Topology levels.
    """
    supported = tuple(
        level.name for level in target.get_facts(TopologyFacts).topologies
    )
    for name in names:
        if name not in supported:
            raise ValueError(
                f"cuda target supports {{{', '.join(supported)}}} "
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
    "MemoryExecutionFacts",
    "PerformanceServiceFacts",
    "Target",
    "TargetFactsError",
    "TopologyFacts",
    "TopologyLevelFacts",
    "ThroughputFacts",
    "UnsupportedCapabilityError",
    "default_target",
    "facts_result",
    "instance_levels",
    "register_target",
    "registered_targets",
    "validate_cuda_topology_levels",
]
