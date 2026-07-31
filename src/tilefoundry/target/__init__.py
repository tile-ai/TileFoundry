"""The sole compilation Target implementation package."""

from __future__ import annotations

from tilefoundry.target.amx import AmxTarget, AppleAmx, AppleM2Pro
from tilefoundry.target.base import Architecture, CpuTarget, Device, Target
from tilefoundry.target.cuda import H200SXM, SM90, CudaTarget

_STRING_TARGETS = {"amx": AmxTarget, "cuda": CudaTarget, "cpu": CpuTarget}


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
    """Validate names at the CUDA lowering boundary.

    See docs/spec/target.md § Topology levels.
    """
    target = CudaTarget()
    for name in names:
        if name not in target.topology_levels:
            raise ValueError(
                f"cuda target supports {{{', '.join(target.topology_levels)}}} "
                f"topology levels; got {name!r}"
            )


def register_facts_projections() -> None:
    """Load each backend's Facts projections, which register on import.

    This cannot happen while this package is initialising: a projection names
    the analysis Facts type it builds, and the analysis layer is built on the IR
    that is still loading the Target at that point. The compiler entry triggers
    it once both halves exist.
    """
    from tilefoundry.target.amx import facts as amx_facts  # noqa: PLC0415, F401
    from tilefoundry.target.cuda import facts as cuda_facts  # noqa: PLC0415, F401


def register_schedule_algorithms() -> None:
    """Load each backend's schedulers, which register on import.

    Deferred for the reason the Facts projections are: a scheduler names the
    public Schedule boundary, which rests on the IR that is still loading this
    package at the time. Which hardware is schedulable at which level is then
    exactly the set of registrations these modules make.
    """
    from tilefoundry.target.amx import schedule as amx_schedule  # noqa: PLC0415, F401
    from tilefoundry.target.cuda import schedule as cuda_schedule  # noqa: PLC0415, F401


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
    "register_facts_projections",
    "register_schedule_algorithms",
    "resolve_target",
    "validate_cuda_topology_levels",
]
