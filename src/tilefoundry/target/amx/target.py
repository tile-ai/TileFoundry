"""Apple AMX compilation target composition."""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.types.shard import Topology
from tilefoundry.target.base import Architecture, Device, Target, bind_services

from .architecture import AppleAmx
from .device import AppleM2Pro


@dataclass(frozen=True, init=False)
class AmxTarget(Target):
    """AMX target composed from one architecture and one device."""

    name: str = field(default="amx", init=False)
    architecture: Architecture = field(default_factory=AppleAmx)
    device: Device = field(default_factory=AppleM2Pro)

    def __init__(
        self,
        architecture: Architecture | None = None,
        device: Device | None = None,
    ) -> None:
        architecture = AppleAmx() if architecture is None else architecture
        device = AppleM2Pro() if device is None else device
        if not isinstance(architecture, Architecture):
            raise TypeError(
                f"AmxTarget.architecture must be an Architecture, got "
                f"{type(architecture).__name__}"
            )
        if not isinstance(device, Device):
            raise TypeError(
                f"AmxTarget.device must be a Device, got {type(device).__name__}"
            )
        object.__setattr__(self, "name", "amx")
        object.__setattr__(self, "architecture", architecture)
        object.__setattr__(self, "device", device)
        from tilefoundry.analysis import Analysis  # noqa: PLC0415
        from tilefoundry.schedule import Schedule  # noqa: PLC0415

        from .service import _AmxCoreAnalysis, _AmxCoreSchedule  # noqa: PLC0415

        bind_services(
            self,
            (
                (Analysis, "core", _AmxCoreAnalysis(self)),
                (Schedule, "core", _AmxCoreSchedule(self)),
            ),
        )

    @property
    def arch(self) -> str:
        """Return the architecture name used by compilation."""
        return self.architecture.name

    @property
    def topology_levels(self) -> tuple[str, ...]:
        """Return program topology levels admitted by AMX compilation."""
        return ("core", "amx")

    def topology_limit(self, name: str) -> int:
        """Return a topology limit for one AMX level."""
        if name == "core":
            return self.device.performance_core_count
        if name == "amx":
            return self.architecture.topology_limit("amx")
        raise ValueError(
            f"{self!r}: unsupported topology level {name!r}; "
            f"supported levels are {self.topology_levels}"
        )

    def validate_program_topology(self, topology: Topology) -> None:
        """Validate one declared program topology against AMX facts."""
        if topology.name not in self.topology_levels:
            raise ValueError(
                f"{self!r}: unsupported topology level {topology.name!r}; "
                f"supported levels are {self.topology_levels}"
            )
        if not isinstance(topology.size, int) or isinstance(topology.size, bool):
            raise ValueError(
                f"{self!r}: topology {topology.name!r} requires a positive "
                f"static integer extent, got {topology.size!r}"
            )
        if topology.size < 1:
            raise ValueError(
                f"{self!r}: topology {topology.name!r} extent {topology.size} "
                "must be positive"
            )
        limit = self.topology_limit(topology.name)
        if topology.size > limit:
            raise ValueError(
                f"{self!r}: topology {topology.name!r} extent {topology.size} "
                f"must satisfy 1 <= extent <= {limit}"
            )


__all__ = ["AmxTarget"]
