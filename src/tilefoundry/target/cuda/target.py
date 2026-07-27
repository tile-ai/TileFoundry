"""CUDA compilation target composition."""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.types.shard import Topology
from tilefoundry.target.base import Architecture, Device, Target, bind_services
from tilefoundry.target.hardware.registry import check_compatible, select


@dataclass(frozen=True, init=False)
class CudaTarget(Target):
    """CUDA target composed from one architecture and one device."""

    name: str = field(default="cuda", init=False)
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
        *,
        arch: str | None = None,
    ) -> None:
        from .spec import H200_SXM_ID, SM90_ID  # noqa: PLC0415

        architecture = select(
            SM90_ID if architecture is None else architecture,
            Architecture,
            role="CudaTarget.architecture",
        )
        device = select(
            H200_SXM_ID if device is None else device,
            Device,
            role="CudaTarget.device",
        )
        architecture_id, device_id = architecture.id, device.id
        if arch is not None and arch != architecture.value.name:
            raise ValueError(
                f"CudaTarget: arch {arch!r} conflicts with architecture.name "
                f"{architecture.value.name!r}"
            )
        if architecture_id is not None and device_id is not None:
            check_compatible(architecture, device)
        object.__setattr__(self, "name", "cuda")
        object.__setattr__(self, "architecture", architecture.value)
        object.__setattr__(self, "device", device.value)
        object.__setattr__(self, "architecture_id", architecture_id)
        object.__setattr__(self, "device_id", device_id)
        object.__setattr__(self, "architecture_digest", architecture.digest)
        object.__setattr__(self, "device_digest", device.digest)
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

    def topology_limit(self, name: str) -> int | None:
        """Return a per-CTA topology limit, if one exists.

        The CUDA grid is a launch shape, not an SM allocation.  Its static
        extent is therefore deliberately unbounded here; a fixed-wave parallel
        capacity is a scheduling policy applied separately.
        """
        if name == "cta":
            return None
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
        if topology.size < 1:
            raise ValueError(
                f"{self!r}: topology {topology.name!r} extent {topology.size} "
                "must be positive"
            )
        if limit is not None and topology.size > limit:
            raise ValueError(
                f"{self!r}: topology {topology.name!r} extent {topology.size} "
                f"must satisfy 1 <= extent <= {limit}"
            )


__all__ = ["CudaTarget"]
