"""Metal compilation target composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from tilefoundry.analysis.facts import (
    MemoryHierarchyFacts,
    PerformanceServiceFacts,
    ThroughputFacts,
)
from tilefoundry.target.base import (
    Architecture,
    Device,
    HardwareSpec,
    Target,
    architecture_of,
    available_device_ids,
    check_compatible,
    register_target,
    select,
)
from tilefoundry.target.facts import (
    MemoryExecutionFacts,
    TopologyFacts,
    TopologyLevelFacts,
    facts_result,
)
from tilefoundry.target.hardware.envelope import HardwareDocument, UnknownDocumentError
from tilefoundry.utils.python_source import PythonExpr
from tilefoundry_apple.metal.target.architecture import MetalArchitecture
from tilefoundry_apple.metal.target.device import MetalDevice
from tilefoundry_apple.metal.target.facts import (
    memory_hierarchy,
    performance_service,
    throughput,
)
from tilefoundry_apple.metal.target.spec import (
    ARCHITECTURE_SCHEMA,
    DEVICE_SCHEMA,
    build_architecture,
    build_device,
)


@register_target
@dataclass(frozen=True, init=False)
class MetalTarget(Target):
    """Metal target composed from one architecture and one device."""

    name: ClassVar[str] = "metal"
    hardware: ClassVar[HardwareSpec] = HardwareSpec(
        package="tilefoundry_apple.metal.target.hardware",
        schemas={
            ARCHITECTURE_SCHEMA: build_architecture,
            DEVICE_SCHEMA: build_device,
        },
    )
    architecture: Architecture = field(init=False)
    device: Device = field(init=False)

    architecture_id: str | None = field(default=None, init=False, compare=False)
    device_id: str | None = field(default=None, init=False, compare=False)
    architecture_digest: str | None = field(default=None, init=False, compare=False)
    device_digest: str | None = field(default=None, init=False, compare=False)
    _architecture_document: HardwareDocument | None = field(
        default=None, init=False, compare=False, repr=False
    )
    _device_document: HardwareDocument | None = field(
        default=None, init=False, compare=False, repr=False
    )

    @property
    def identity(self) -> str:
        return self.device_id or self.name

    @classmethod
    def available(cls) -> tuple[MetalTarget, ...]:
        return tuple(cls(device_id) for device_id in available_device_ids(cls.hardware))

    def __init__(
        self,
        device: Device | str | Path,
        architecture: Architecture | str | Path | None = None,
    ) -> None:
        if architecture is None:
            architecture = architecture_of(
                device,
                device_type=MetalDevice,
                role="MetalTarget.device",
                hardware=self.hardware,
            )
        architecture = select(
            architecture,
            MetalArchitecture,
            role="MetalTarget.architecture",
            hardware=self.hardware,
        )
        device = select(
            device,
            MetalDevice,
            role="MetalTarget.device",
            hardware=self.hardware,
        )
        if architecture.id is not None and device.id is not None:
            check_compatible(architecture, device)
        object.__setattr__(self, "architecture", architecture.value)
        object.__setattr__(self, "device", device.value)
        object.__setattr__(self, "architecture_id", architecture.id)
        object.__setattr__(self, "device_id", device.id)
        object.__setattr__(self, "architecture_digest", architecture.digest)
        object.__setattr__(self, "device_digest", device.digest)
        object.__setattr__(self, "_architecture_document", architecture.document)
        object.__setattr__(self, "_device_document", device.document)

    def _topology_facts(self) -> TopologyFacts:
        return TopologyFacts(
            (TopologyLevelFacts("threadgroup", None, self.device.resident_threadgroups),
             TopologyLevelFacts("thread", self.device.max_threads_per_threadgroup,
                                self.device.max_threads_per_threadgroup)),
            parallel_level="threadgroup",
        )

    def get_facts(self, facts_type: type, query: object | None = None):
        """Project Metal hardware through the facts this Target owns."""
        if facts_type is TopologyFacts and query is None:
            return facts_result(self, facts_type, self._topology_facts())
        if facts_type is MemoryExecutionFacts and query is None:
            return facts_result(
                self, facts_type,
                MemoryExecutionFacts(tuple(level.name for level in self._topology_facts().topologies)),
            )
        if facts_type is TopologyLevelFacts:
            level = self._topology_facts().level(
                query if isinstance(query, str) else None
            )
            if level is not None:
                return facts_result(self, facts_type, level)
            return super().get_facts(facts_type, query)

        if facts_type in (MemoryHierarchyFacts, ThroughputFacts) and query is not None:
            raise ValueError(f"{self.identity}: {facts_type.__name__} is hardware-only and accepts no query")
        if facts_type is MemoryHierarchyFacts:
            return facts_result(self, facts_type, memory_hierarchy(self, query))
        if facts_type is PerformanceServiceFacts:
            return facts_result(self, facts_type, performance_service(self, query))
        if facts_type is ThroughputFacts:
            return facts_result(self, facts_type, throughput(self, query))
        return super().get_facts(facts_type, query)

    def to_python(self) -> PythonExpr:
        if type(self) is MetalTarget and self.architecture_id and self.device_id:
            try:
                architecture = self.hardware.resolve(self.architecture_id)
                device = self.hardware.resolve(self.device_id)
            except UnknownDocumentError:
                return super().to_python()
            if (architecture.digest == self.architecture_digest
                    and device.digest == self.device_digest):
                return PythonExpr(
                    ("from tilefoundry_apple import MetalTarget",),
                    f'MetalTarget("{self.device_id}", architecture="{self.architecture_id}")',
                )
        return super().to_python()

    def __repr__(self) -> str:
        return self.to_python().text

    @property
    def arch(self) -> str:
        """Return the architecture name used by compilation."""
        return self.architecture.name


__all__ = ["MetalTarget"]
