"""Apple AMX compilation target composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

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
from tilefoundry.target.hardware.envelope import HardwareDocument
from tilefoundry.utils.python_source import PythonExpr
from tilefoundry_apple.amx.target.architecture import AppleAmx
from tilefoundry_apple.amx.target.device import AppleM2Pro
from tilefoundry_apple.amx.target.spec import (
    ARCHITECTURE_SCHEMA,
    DEVICE_SCHEMA,
    build_apple_amx,
    build_apple_m2_pro,
)


@register_target
@dataclass(frozen=True, init=False)
class AmxTarget(Target):
    """AMX target composed from one architecture and one device."""

    name: ClassVar[str] = "amx"
    hardware: ClassVar[HardwareSpec] = HardwareSpec(
        package="tilefoundry_apple.amx.target.hardware",
        schemas={
            ARCHITECTURE_SCHEMA: build_apple_amx,
            DEVICE_SCHEMA: build_apple_m2_pro,
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
    def available(cls) -> tuple[AmxTarget, ...]:
        return tuple(cls(device_id) for device_id in available_device_ids(cls.hardware))

    def __init__(
        self,
        device: Device | str | Path | None = None,
        architecture: Architecture | str | Path | None = None,
    ) -> None:
        from .spec import APPLE_M2_PRO_ID  # noqa: PLC0415

        device = APPLE_M2_PRO_ID if device is None else device
        if architecture is None:
            architecture = architecture_of(
                device,
                device_type=AppleM2Pro,
                role="AmxTarget.device",
                hardware=self.hardware,
            )
        architecture = select(
            architecture,
            AppleAmx,
            role="AmxTarget.architecture",
            hardware=self.hardware,
        )
        device = select(
            device,
            AppleM2Pro,
            role="AmxTarget.device",
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
        """The two AMX levels, coarsest first.

        Both extents are hardware totals, so neither level takes anything from
        the target instance.
        """
        return TopologyFacts(
            (
                TopologyLevelFacts(
                    "core",
                    self.device.performance_core_count,
                    self.device.performance_core_count,
                ),
                TopologyLevelFacts(
                    "amx", self.architecture.topology_limit("amx"), None
                ),
            ),
            parallel_level="core",
        )

    def get_facts(self, facts_type: type, query: object | None = None):
        """Project AMX hardware through the facts this Target owns."""
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

        from tilefoundry.analysis.facts import (  # noqa: PLC0415
            MemoryHierarchyFacts,
            PerformanceServiceFacts,
            ThroughputFacts,
        )
        from tilefoundry_apple.amx.target.facts import (  # noqa: PLC0415
            memory_hierarchy,
            performance_service,
            throughput,
        )

        if facts_type is MemoryHierarchyFacts:
            return facts_result(self, facts_type, memory_hierarchy(self, query))
        if facts_type is PerformanceServiceFacts:
            return facts_result(self, facts_type, performance_service(self, query))
        if facts_type is ThroughputFacts:
            return facts_result(self, facts_type, throughput(self, query))
        return super().get_facts(facts_type, query)

    def to_python(self) -> PythonExpr:
        if type(self) is AmxTarget and self.architecture_id and self.device_id:
            return PythonExpr(
                ("from tilefoundry_apple import AmxTarget",),
                f'AmxTarget("{self.device_id}")',
            )
        return super().to_python()

    @property
    def arch(self) -> str:
        """Return the architecture name used by compilation."""
        return self.architecture.name


__all__ = ["AmxTarget"]
