"""Compilation targets.

A function carries a single ``target`` that selects its target-specific
lowering and codegen. A string is reflected into a default target object
(``"cuda"`` -> ``CudaTarget()`` with a default arch); pass a target object
to override the configuration.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    """Base compilation target; ``name`` identifies the backend."""

    name: str


@dataclass(frozen=True)
class CudaTarget(Target):
    """CUDA device target, parameterised by SM architecture."""

    name: str = "cuda"
    arch: str = "sm_90"
    device: str | None = None

    # Program topology levels this target supports (single-card). A kernel's
    # declared topology levels must be a subset; finer levels such as ``warp``
    # are expressed inside a mesh layout, not as a program topology level.
    topology_levels: tuple[str, ...] = ("cta", "thread")


def validate_cuda_topology_levels(names) -> None:
    """Raise unless every topology level name is one the cuda target supports.

    A finer level (e.g. ``warp``) belongs in a mesh layout, not as a program
    topology level; an unknown level (e.g. ``gpu``) is a contract error.
    """
    allowed = CudaTarget.topology_levels
    for name in names:
        if name not in allowed:
            raise ValueError(
                f"cuda target supports {{{', '.join(allowed)}}} topology "
                f"levels; got {name!r}"
            )


@dataclass(frozen=True)
class CpuTarget(Target):
    """Host / CPU target."""

    name: str = "cpu"


_STRING_TARGETS = {"cuda": CudaTarget, "cpu": CpuTarget}


def resolve_target(target: "str | Target") -> Target:
    """Reflect a target spec into a ``Target``.

    A ``Target`` passes through unchanged; a string maps to that backend's
    default target object.
    """
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
    raise TypeError(
        f"target must be a str or Target, got {type(target).__name__}"
    )


def default_target() -> Target:
    """The default target when none is specified (device CUDA)."""
    return CudaTarget()


__all__ = [
    "Target",
    "CudaTarget",
    "CpuTarget",
    "resolve_target",
    "default_target",
]
