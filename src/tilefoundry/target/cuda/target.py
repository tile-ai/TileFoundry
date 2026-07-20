"""CUDA compilation target composition."""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.types.shard import Topology
from tilefoundry.target.base import Architecture, Device, Target, bind_services

from .architecture import SM90
from .device import H200SXM


@dataclass(frozen=True, init=False)
class CudaTarget(Target):
    """CUDA target composed from one architecture and one device."""

    name: str = field(default="cuda", init=False)
    architecture: Architecture = field(default_factory=SM90)
    device: Device = field(default_factory=H200SXM)

    def __init__(
        self,
        architecture: Architecture | None = None,
        device: Device | None = None,
        *,
        arch: str | None = None,
    ) -> None:
        architecture = SM90() if architecture is None else architecture
        device = H200SXM() if device is None else device
        if not isinstance(architecture, Architecture):
            raise TypeError(
                f"CudaTarget.architecture must be an Architecture, got "
                f"{type(architecture).__name__}"
            )
        if not isinstance(device, Device):
            raise TypeError(
                f"CudaTarget.device must be a Device, got {type(device).__name__}"
            )
        if arch is not None and arch != architecture.name:
            raise ValueError(
                f"CudaTarget: arch {arch!r} conflicts with architecture.name "
                f"{architecture.name!r}"
            )
        object.__setattr__(self, "name", "cuda")
        object.__setattr__(self, "architecture", architecture)
        object.__setattr__(self, "device", device)
        from tilefoundry.schedule import Schedule  # noqa: PLC0415

        from .service import _CudaCtaSchedule  # noqa: PLC0415

        bind_services(self, ((Schedule, "cta", _CudaCtaSchedule(self)),))

    @property
    def arch(self) -> str:
        """Return the architecture name used by compilation."""
        return self.architecture.name

    @property
    def topology_levels(self) -> tuple[str, ...]:
        """Return program topology levels admitted by CUDA compilation."""
        return ("cta", "thread")

    def topology_limit(self, name: str) -> int:
        """Return the concrete resource limit for a program topology."""
        if name == "cta":
            return self.device.sm_count
        if name == "thread":
            return self.architecture.topology_limit("thread")
        raise ValueError(
            f"{self!r}: unsupported topology level {name!r}; "
            f"supported levels are {self.topology_levels}"
        )

    def validate_program_topology(self, topology: Topology) -> None:
        """Validate one declared program topology against CUDA facts."""
        if topology.name not in self.topology_levels:
            raise ValueError(
                f"{self!r}: unsupported topology level {topology.name!r}; "
                f"supported levels are {self.topology_levels}"
            )
        if topology.name == "cta" and topology.size is None:
            return
        if not isinstance(topology.size, int) or isinstance(topology.size, bool):
            raise ValueError(
                f"{self!r}: topology {topology.name!r} requires a positive "
                f"static integer extent, got {topology.size!r}"
            )
        limit = self.topology_limit(topology.name)
        if not 1 <= topology.size <= limit:
            raise ValueError(
                f"{self!r}: topology {topology.name!r} extent {topology.size} "
                f"must satisfy 1 <= extent <= {limit}"
            )


__all__ = ["CudaTarget"]
