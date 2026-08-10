"""Apple AMX compilation target composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from tilefoundry.target.amx.architecture import AppleAmx
from tilefoundry.target.amx.device import AppleM2Pro
from tilefoundry.target.amx.spec import (
    ARCHITECTURE_SCHEMA,
    DEVICE_SCHEMA,
    build_apple_amx,
    build_apple_m2_pro,
)
from tilefoundry.target.base import (
    Architecture,
    Device,
    HardwareSpec,
    Target,
    _architecture_of,
    _available_device_ids,
    check_compatible,
    register_target,
    select,
)
from tilefoundry.target.facts import TopologyLimitFacts, facts_result
from tilefoundry.target.hardware.envelope import HardwareDocument
from tilefoundry.target.services import Scheduler
from tilefoundry.utils.python_source import PythonExpr


@register_target
@dataclass(frozen=True, init=False)
class AmxTarget(Target):
    """AMX target composed from one architecture and one device."""

    name: ClassVar[str] = "amx"
    hardware: ClassVar[HardwareSpec] = HardwareSpec(
        package="tilefoundry.target.amx.hardware",
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
        return tuple(cls(device_id) for device_id in _available_device_ids(cls.hardware))

    def __init__(
        self,
        device: Device | str | Path | None = None,
        architecture: Architecture | str | Path | None = None,
    ) -> None:
        from .spec import APPLE_M2_PRO_ID  # noqa: PLC0415

        device = APPLE_M2_PRO_ID if device is None else device
        if architecture is None:
            architecture = _architecture_of(
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

    def get_facts(self, facts_type: type, query: object | None = None):
        """Project AMX hardware through the facts this Target owns."""
        if facts_type is TopologyLimitFacts:
            if query == "core":
                return facts_result(
                    self,
                    facts_type,
                    TopologyLimitFacts(
                        "core", self.device.performance_core_count
                    ),
                )
            if query == "amx":
                return facts_result(
                    self,
                    facts_type,
                    TopologyLimitFacts(
                        "amx", self.architecture.topology_limit("amx")
                    ),
                )
            return super().get_facts(facts_type, query)

        from tilefoundry.analysis.facts import (  # noqa: PLC0415
            MemoryHierarchyFacts,
            ParallelCapacityFacts,
            ThroughputFacts,
        )
        from tilefoundry.target.amx.facts import (  # noqa: PLC0415
            memory_hierarchy,
            parallel_capacity,
            throughput,
        )

        if facts_type is MemoryHierarchyFacts:
            return facts_result(self, facts_type, memory_hierarchy(self, query))
        if facts_type is ThroughputFacts:
            return facts_result(self, facts_type, throughput(self, query))
        if facts_type is ParallelCapacityFacts:
            return facts_result(self, facts_type, parallel_capacity(self, query))

        from tilefoundry.schedule.pipeline import PipelineFacts  # noqa: PLC0415

        if facts_type is PipelineFacts:
            from tilefoundry.target.amx.facts import pipeline_facts  # noqa: PLC0415

            return facts_result(self, facts_type, pipeline_facts(self, query))
        return super().get_facts(facts_type, query)

    def get_scheduler(self, topology: str) -> Scheduler:
        """Select the AMX core scheduler."""
        from tilefoundry.target.amx.schedule import amx_scheduler  # noqa: PLC0415

        scheduler = amx_scheduler(topology)
        if scheduler is not None:
            return scheduler
        return super().get_scheduler(topology)

    def _python_import_module(self) -> str:
        if type(self) is AmxTarget:
            return "tilefoundry.target.amx"
        return super()._python_import_module()

    def to_python(self) -> PythonExpr:
        if type(self) is AmxTarget and self.architecture_id and self.device_id:
            return PythonExpr(
                ("from tilefoundry.target import AmxTarget",),
                f'AmxTarget("{self.device_id}")',
            )
        return super().to_python()

    @property
    def arch(self) -> str:
        """Return the architecture name used by compilation."""
        return self.architecture.name

    topology_levels: ClassVar[tuple[str, ...]] = ("core", "amx")


__all__ = ["AmxTarget"]
