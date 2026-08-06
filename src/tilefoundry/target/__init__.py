"""The sole compilation Target implementation package."""

from __future__ import annotations

from tilefoundry.target.amx import AmxTarget, AppleAmx, AppleM2Pro
from tilefoundry.target.amx import spec as _amx_spec  # noqa: F401
from tilefoundry.target.base import Architecture, CpuTarget, Device, Target
from tilefoundry.target.cuda import H200SXM, SM90, CudaTarget
from tilefoundry.target.cuda.spec import H200_SXM_ID
from tilefoundry.target.registration import (
    is_registered_target_type,
    register_target,
    registered_targets,
)


def _cuda_fallback() -> CudaTarget:
    """Construct the built-in value used at compiler-owned omitted boundaries."""
    return CudaTarget(H200_SXM_ID)


def require_target(target: Target) -> Target:
    """Return an authored Target value or reject every other input."""
    if isinstance(target, Target):
        if not is_registered_target_type(type(target)):
            raise TypeError(
                f"target must be an instance of a @register_target class, got "
                f"{type(target).__module__}.{type(target).__qualname__}"
            )
        return target
    if isinstance(target, str):
        raise TypeError(
            "target must be a Target instance, not a string; import and "
            "construct the Target class explicitly"
        )
    raise TypeError(f"target must be a Target instance, got {type(target).__name__}")


def default_target() -> Target:
    """Return the normal compile-entry default target."""
    return _cuda_fallback()


def validate_cuda_topology_levels(names) -> None:
    """Validate names at the CUDA lowering boundary.

    See docs/spec/target.md § Topology levels.
    """
    target = _cuda_fallback()
    for name in names:
        if name not in target.topology_levels:
            raise ValueError(
                f"cuda target supports {{{', '.join(target.topology_levels)}}} "
                f"topology levels; got {name!r}"
            )


__all__ = [
    "AmxTarget",
    "AppleAmx",
    "AppleM2Pro",
    "Architecture",
    "CpuTarget",
    "CudaTarget",
    "Device",
    "H200SXM",
    "SM90",
    "Target",
    "default_target",
    "register_target",
    "registered_targets",
    "require_target",
    "validate_cuda_topology_levels",
]
