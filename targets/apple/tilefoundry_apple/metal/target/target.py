"""Apple Metal compilation target composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from tilefoundry import target, utils
from tilefoundry.target import hardware

from .architecture import Architecture
from .device import Device
from .facts import memory_hierarchy
from .spec import (
    APPLE_M2_PRO_GPU_ID,
    ARCHITECTURE_SCHEMA,
    DEVICE_SCHEMA,
    build_architecture,
    build_device,
)


@target.register
@dataclass(frozen=True, init=False)
class Target(target.Target):
    """Metal target composed from one GPU architecture and one device."""

    name: ClassVar[str] = "metal"
    hardware: ClassVar[target.HardwareSpec] = target.HardwareSpec(
        package="tilefoundry_apple.metal.target.hardware",
        schemas={
            ARCHITECTURE_SCHEMA: build_architecture,
            DEVICE_SCHEMA: build_device,
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
        device = APPLE_M2_PRO_GPU_ID if device is None else device
        if architecture is None:
            architecture = target.architecture_of(
                device,
                device_type=Device,
                role="metal.Target.device",
                hardware=self.hardware,
            )
        architecture = target.select(
            architecture,
            Architecture,
            role="metal.Target.architecture",
            hardware=self.hardware,
        )
        device = target.select(
            device,
            Device,
            role="metal.Target.device",
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
        """A threadgroup of threads; threadgroups run in parallel."""
        threads = self.device.max_threads_per_threadgroup
        return target.TopologyFacts(
            (
                target.TopologyLimits("threadgroup", None, None),
                target.TopologyLimits("thread", threads, None),
            ),
            parallel_topology="threadgroup",
        )

    def get_facts(self, facts_type: type, query: object | None = None):
        """Project Metal hardware through the facts this Target owns."""
        if facts_type is target.TopologyFacts and query is None:
            return target.facts_result(self, facts_type, self.topology_facts())
        if facts_type is target.MemoryExecutionFacts and query is None:
            return target.facts_result(
                self, facts_type,
                target.MemoryExecutionFacts((self.topology_facts().topologies[-1].name,)),
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
        return super().get_facts(facts_type, query)

    def to_python(self) -> utils.PythonExpr:
        if type(self) is Target and self.architecture_id and self.device_id:
            return utils.PythonExpr(
                ("from tilefoundry_apple import metal",),
                f'metal.Target("{self.device_id}")',
            )
        return super().to_python()

    @property
    def arch(self) -> str:
        """Return the architecture name used by compilation."""
        return self.architecture.name


__all__ = ["Target"]
