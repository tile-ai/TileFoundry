"""Apple AMX compilation target composition."""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.types.shard import Topology
from tilefoundry.target.base import Architecture, Device, Target, bind_services
from tilefoundry.target.hardware.registry import check_compatible, select


@dataclass(frozen=True, init=False)
class AmxTarget(Target):
    """AMX target composed from one architecture and one device."""

    name: str = field(default="amx", init=False)
    architecture: Architecture = field(init=False)
    device: Device = field(init=False)
    # Identity and digest record where a value came from, not what it says, so
    # they stay out of equality: two targets carrying identical facts must group
    # together for codegen even when one was selected by ID and one supplied
    # directly.
    architecture_id: str | None = field(default=None, init=False, compare=False)
    device_id: str | None = field(default=None, init=False, compare=False)
    architecture_digest: str | None = field(default=None, init=False, compare=False)
    device_digest: str | None = field(default=None, init=False, compare=False)

    def __init__(
        self,
        architecture: Architecture | str | None = None,
        device: Device | str | None = None,
    ) -> None:
        from .spec import APPLE_AMX_ID, APPLE_M2_PRO_ID  # noqa: PLC0415

        architecture = select(
            APPLE_AMX_ID if architecture is None else architecture,
            Architecture,
            role="AmxTarget.architecture",
        )
        device = select(
            APPLE_M2_PRO_ID if device is None else device,
            Device,
            role="AmxTarget.device",
        )
        if architecture.id is not None and device.id is not None:
            check_compatible(architecture, device)
        object.__setattr__(self, "name", "amx")
        object.__setattr__(self, "architecture", architecture.value)
        object.__setattr__(self, "device", device.value)
        object.__setattr__(self, "architecture_id", architecture.id)
        object.__setattr__(self, "device_id", device.id)
        object.__setattr__(self, "architecture_digest", architecture.digest)
        object.__setattr__(self, "device_digest", device.digest)
        from tilefoundry.schedule import Schedule  # noqa: PLC0415

        from .service import _AmxCoreSchedule  # noqa: PLC0415

        bind_services(self, ((Schedule, "core", _AmxCoreSchedule(self)),))

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
