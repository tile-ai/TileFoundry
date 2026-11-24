"""Apple AMX compilation target composition."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
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
@dataclass(frozen=True)
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
    device_id: str | Path = APPLE_M2_PRO_ID

    def __post_init__(self) -> None:
        target.check_compatible(self.selected_architecture, self.selected_device)

    @cached_property
    def selected_device(self) -> hardware.ResolvedResource:
        """The device document and the Device it builds."""
        return target.select(
            self.device_id, AppleM2Pro, role="amx.Target.device", hardware=self.hardware
        )

    @cached_property
    def selected_architecture(self) -> hardware.ResolvedResource:
        """The architecture document the device names and the Architecture it builds."""
        named = target.architecture_of(
            self.device_id, device_type=AppleM2Pro, role="amx.Target.device", hardware=self.hardware
        )
        return target.select(
            named, Architecture, role="amx.Target.architecture", hardware=self.hardware
        )

    @property
    def device(self) -> AppleM2Pro:
        return self.selected_device.value

    @property
    def architecture(self) -> Architecture:
        return self.selected_architecture.value

    @property
    def identity(self) -> str:
        return self.selected_device.id

    @classmethod
    def from_installed(cls) -> tuple[Target, ...]:
        return tuple(cls(device_id) for device_id in target.available_device_ids(cls.hardware))

    def capabilities(self) -> frozenset[str]:
        return frozenset(self.architecture.capabilities)

    def documents(self) -> tuple[hardware.Document, ...]:
        return self.selected_architecture.document, self.selected_device.document

    def topology_facts(self) -> target.TopologyFacts:
        """CPU execution units; AMX is an instruction capability."""
        count = self.device.performance_core_count + self.device.efficiency_core_count
        return target.TopologyFacts(
            (target.TopologyLimits("thread", count, count),),
            parallel_topology="thread",
        )

    def get_facts(self, facts_type: type, query: object | None = None):
        """Project AMX hardware through the facts this Target owns."""
        if facts_type is target.TopologyFacts and query is None:
            return target.facts_result(self, facts_type, self.topology_facts())
        if facts_type is target.MemoryExecutionFacts and query is None:
            return target.facts_result(
                self, facts_type,
                target.MemoryExecutionFacts(tuple(topology.name for topology in self.topology_facts().topologies)),
            )
        if facts_type is target.TopologyLimits:
            limits = self.topology_facts().topology(
                query if isinstance(query, str) else None
            )
            if limits is not None:
                return target.facts_result(self, facts_type, limits)
            return super().get_facts(facts_type, query)

        if facts_type is target.MemoryHierarchyFacts:
            return target.facts_result(self, facts_type, memory_hierarchy(self, query))
        if facts_type is target.PerformanceServiceFacts:
            return target.facts_result(self, facts_type, performance_service(self, query))
        if facts_type is target.ThroughputFacts:
            return target.facts_result(self, facts_type, throughput(self, query))
        return super().get_facts(facts_type, query)

    def to_python(self) -> utils.PythonExpr:
        if type(self) is Target:
            return utils.PythonExpr(
                ("from tilefoundry_apple import amx",),
                f'amx.Target("{self.identity}")',
            )
        return super().to_python()


__all__ = ["Target"]
