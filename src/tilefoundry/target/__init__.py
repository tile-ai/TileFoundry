"""The sole compilation Target implementation package."""

from __future__ import annotations

from tilefoundry.target.amx import AmxTarget, AppleAmx, AppleM2Pro
from tilefoundry.target.amx import spec as _amx_spec  # noqa: F401
from tilefoundry.target.base import (
    Architecture,
    Device,
    Target,
    UnsupportedCapabilityError,
    register_target,
    registered_targets,
)
from tilefoundry.target.cpu import CpuTarget
from tilefoundry.target.cuda import H200SXM, SM90, CudaTarget
from tilefoundry.target.cuda.spec import H200_SXM_ID
from tilefoundry.target.facts import TopologyLimitFacts
from tilefoundry.target.services import Analyzer, Scheduler


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
    "CudaTarget",
    "Device",
    "H200SXM",
    "SM90",
    "Scheduler",
    "Target",
    "TopologyLimitFacts",
    "UnsupportedCapabilityError",
    "default_target",
    "register_target",
    "registered_targets",
    "validate_cuda_topology_levels",
]
