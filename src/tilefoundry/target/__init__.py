"""The sole compilation Target implementation package."""

from __future__ import annotations

from tilefoundry.target.base import Architecture, CpuTarget, Device, Target
from tilefoundry.target.cuda import H200SXM, SM90, CudaTarget

_STRING_TARGETS = {"cuda": CudaTarget, "cpu": CpuTarget}


def resolve_target(target: str | Target) -> Target:
    """Resolve a backend name or pass through an immutable Target value."""
    if isinstance(target, Target):
        return target
    if isinstance(target, str):
        factory = _STRING_TARGETS.get(target)
        if factory is None:
            raise ValueError(
                f"unknown target {target!r}; expected one of "
                f"{sorted(_STRING_TARGETS)} or a Target object"
            )
        return factory()
    raise TypeError(f"target must be a str or Target, got {type(target).__name__}")


def default_target() -> Target:
    """Return the normal compile-entry default target."""
    return CudaTarget()


def validate_cuda_topology_levels(names) -> None:
    """Validate topology names against the default CUDA target."""
    target = CudaTarget()
    for name in names:
        if name not in target.topology_levels:
            raise ValueError(
                f"cuda target supports {{{', '.join(target.topology_levels)}}} "
                f"topology levels; got {name!r}"
            )


__all__ = [
    "Architecture",
    "CpuTarget",
    "CudaTarget",
    "Device",
    "H200SXM",
    "SM90",
    "Target",
    "default_target",
    "resolve_target",
    "validate_cuda_topology_levels",
]
