"""Apple AMX compilation target composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from tilefoundry.ir.types.shard import Topology
from tilefoundry.target.base import Architecture, Device, Target
from tilefoundry.target.hardware.registry import check_compatible, select
from tilefoundry.target.registration import register_target


@register_target
@dataclass(frozen=True, init=False)
class AmxTarget(Target):
    """AMX target composed from one architecture and one device."""

    name: ClassVar[str] = "amx"
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
        object.__setattr__(self, "architecture", architecture.value)
        object.__setattr__(self, "device", device.value)
        object.__setattr__(self, "architecture_id", architecture.id)
        object.__setattr__(self, "device_id", device.id)
        object.__setattr__(self, "architecture_digest", architecture.digest)
        object.__setattr__(self, "device_digest", device.digest)

    def get_analyzer(self, selector: str) -> "Analyzer":
        """Select an AMX analysis service."""
        from tilefoundry.analysis.registry import (  # noqa: PLC0415
            builtin_analyzers,
        )

        try:
            return builtin_analyzers()[selector]
        except KeyError:
            return super().get_analyzer(selector)

    def get_facts(self, facts_type: type, query: object | None = None):
        """Project AMX hardware through the facts this Target owns."""
        from tilefoundry.analysis.facts import (  # noqa: PLC0415
            MemoryHierarchyFacts,
            ParallelCapacityFacts,
            ThroughputFacts,
        )
        from tilefoundry.schedule.pipeline import PipelineFacts  # noqa: PLC0415
        from tilefoundry.target.amx.facts import (  # noqa: PLC0415
            memory_hierarchy,
            parallel_capacity,
            pipeline_facts,
            throughput,
        )
        from tilefoundry.target.facts import facts_result  # noqa: PLC0415

        projections = {
            MemoryHierarchyFacts: memory_hierarchy,
            ThroughputFacts: throughput,
            ParallelCapacityFacts: parallel_capacity,
            PipelineFacts: pipeline_facts,
        }
        try:
            projection = projections[facts_type]
        except KeyError:
            return super().get_facts(facts_type, query)
        return facts_result(self, facts_type, projection(self, query))

    def get_scheduler(self, topology: str) -> "Scheduler":
        """Select the AMX core scheduler."""
        from tilefoundry.target.amx.schedule import amx_schedulers  # noqa: PLC0415

        try:
            return amx_schedulers()[topology]
        except KeyError:
            return super().get_scheduler(topology)

    def __repr__(self) -> str:
        """Return a constructor expression for this concrete AMX Target."""
        constructor = type(self).__name__
        if self.architecture_id is not None and self.device_id is not None:
            from .spec import APPLE_AMX_ID, APPLE_M2_PRO_ID  # noqa: PLC0415

            if (
                self.architecture_id == APPLE_AMX_ID
                and self.device_id == APPLE_M2_PRO_ID
            ):
                return f"{constructor}()"
        architecture = (
            self.architecture_id
            if self.architecture_id is not None
            else self.architecture
        )
        device = self.device_id if self.device_id is not None else self.device
        return (
            f"{constructor}(architecture={architecture!r}, "
            f"device={device!r})"
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
