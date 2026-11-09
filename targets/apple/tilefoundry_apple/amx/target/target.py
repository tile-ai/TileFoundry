"""Apple AMX compilation target composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from tilefoundry import target, utils
from tilefoundry.target import hardware

from .architecture import Architecture
from .device import AppleM2Pro
from .facts import memory_hierarchy, performance_service, throughput
from .spec import (
    APPLE_M2_PRO_ID,
    ARCHITECTURE_SCHEMA,
    DEVICE_SCHEMA,
    build_apple_amx,
    build_apple_m2_pro,
)


@target.register
@dataclass(frozen=True, init=False)
class Target(target.Target):
    """AMX target composed from one architecture and one device."""

    name: ClassVar[str] = "amx"
    hardware: ClassVar[target.HardwareSpec] = target.HardwareSpec(
        package="tilefoundry_apple.amx.target.hardware",
        schemas={
            ARCHITECTURE_SCHEMA: build_apple_amx,
            DEVICE_SCHEMA: build_apple_m2_pro,
        },
    )
    architecture: target.Architecture = field(init=False)
    device: target.Device = field(init=False)




    architecture_id: str | None = field(default=None, init=False, compare=False)
    device_id: str | None = field(default=None, init=False, compare=False)
    architecture_digest: str | None = field(default=None, init=False, compare=False)
    device_digest: str | None = field(default=None, init=False, compare=False)
    architecture_document: hardware.Document | None = field(
        default=None, init=False, compare=False, repr=False
    )
    device_document: hardware.Document | None = field(
        default=None, init=False, compare=False, repr=False
    )

    @property
    def identity(self) -> str:
        return self.device_id or self.name

    @classmethod
    def from_installed(cls) -> tuple[Target, ...]:
        return tuple(cls(device_id) for device_id in target.available_device_ids(cls.hardware))

    def __init__(
        self,
        device: target.Device | str | Path | None = None,
        architecture: target.Architecture | str | Path | None = None,
    ) -> None:
        device = APPLE_M2_PRO_ID if device is None else device
        if architecture is None:
            architecture = target.architecture_of(
                device,
                device_type=AppleM2Pro,
                role="amx.Target.device",
                hardware=self.hardware,
            )
        architecture = target.select(
            architecture,
            Architecture,
            role="amx.Target.architecture",
            hardware=self.hardware,
        )
        device = target.select(
            device,
            AppleM2Pro,
            role="amx.Target.device",
            hardware=self.hardware,
        )
        if architecture.id is not None and device.id is not None:
            target.check_compatible(architecture, device)
        object.__setattr__(self, "architecture", architecture.value)
        object.__setattr__(self, "device", device.value)
        object.__setattr__(self, "architecture_id", architecture.id)
        object.__setattr__(self, "device_id", device.id)
        object.__setattr__(self, "architecture_digest", architecture.digest)
        object.__setattr__(self, "device_digest", device.digest)
        object.__setattr__(self, "architecture_document", architecture.document)
        object.__setattr__(self, "device_document", device.document)

    def topology_facts(self) -> target.TopologyFacts:
        """CPU execution positions; AMX is an instruction capability."""
        count = self.device.performance_core_count + self.device.efficiency_core_count
        return target.TopologyFacts(
            (target.TopologyLevelFacts("thread", count, count),),
            parallel_level="thread",
        )

    def get_facts(self, facts_type: type, query: object | None = None):
        """Project AMX hardware through the facts this Target owns."""
        if facts_type is target.TopologyFacts and query is None:
            return target.facts_result(self, facts_type, self.topology_facts())
        if facts_type is target.MemoryExecutionFacts and query is None:
            return target.facts_result(
                self, facts_type,
                target.MemoryExecutionFacts(tuple(level.name for level in self.topology_facts().topologies)),
            )
        if facts_type is target.TopologyLevelFacts:
            level = self.topology_facts().level(
                query if isinstance(query, str) else None
            )
            if level is not None:
                return target.facts_result(self, facts_type, level)
            return super().get_facts(facts_type, query)

        if facts_type is target.MemoryHierarchyFacts:
            return target.facts_result(self, facts_type, memory_hierarchy(self, query))
        if facts_type is target.PerformanceServiceFacts:
            return target.facts_result(self, facts_type, performance_service(self, query))
        if facts_type is target.ThroughputFacts:
            return target.facts_result(self, facts_type, throughput(self, query))
        return super().get_facts(facts_type, query)

    def to_python(self) -> utils.PythonExpr:
        if type(self) is Target and self.architecture_id and self.device_id:
            return utils.PythonExpr(
                ("from tilefoundry_apple import amx",),
                f'amx.Target("{self.device_id}")',
            )
        return super().to_python()

    @property
    def arch(self) -> str:
        """Return the architecture name used by compilation."""
        return self.architecture.name


__all__ = ["Target"]
