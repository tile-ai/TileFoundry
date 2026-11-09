"""CUDA compilation target composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from tilefoundry import target, utils
from tilefoundry.target import hardware
from tilefoundry_nvidia.cuda import analysis

from .architecture import Architecture
from .device import Device
from .facts import memory_hierarchy, parallel_units, performance_service, throughput
from .spec import ARCHITECTURE_SCHEMA, DEVICE_SCHEMA, build_cuda_architecture, build_cuda_device


@target.register
@dataclass(frozen=True, init=False)
class Target(target.Target):
    """CUDA target composed from one device and the architecture it runs."""

    name: ClassVar[str] = "cuda"
    hardware: ClassVar[target.HardwareSpec] = target.HardwareSpec(
        package="tilefoundry_nvidia.cuda.target.hardware",
        schemas={
            ARCHITECTURE_SCHEMA: build_cuda_architecture,
            DEVICE_SCHEMA: build_cuda_device,
        },
    )
    architecture: target.Architecture = field(init=False)
    device: target.Device = field(init=False)

    device_count: int | None = field(default=None, init=False)
    """How many cards a program of this target may name at its ``gpu`` level."""




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
        device: target.Device | str | Path,
        architecture: target.Architecture | str | Path | None = None,
        *,
        arch: str | None = None,
        device_count: int | None = None,
    ) -> None:
        if device_count is not None and (
            isinstance(device_count, bool)
            or not isinstance(device_count, int)
            or device_count < 1
        ):
            raise ValueError(
                f"cuda.Target: device_count {device_count!r} must be a positive int "
                f"or None, which admits any extent at the gpu level"
            )
        if architecture is None:
            architecture = target.architecture_of(
                device,
                device_type=Device,
                role="cuda.Target.device",
                hardware=self.hardware,
            )
        architecture = target.select(
            architecture,
            Architecture,
            role="cuda.Target.architecture",
            hardware=self.hardware,
        )
        device = target.select(
            device, Device, role="cuda.Target.device", hardware=self.hardware
        )
        architecture_id, device_id = architecture.id, device.id
        if arch is not None and arch != architecture.value.name:
            raise ValueError(
                f"cuda.Target: arch {arch!r} conflicts with architecture.name "
                f"{architecture.value.name!r}"
            )
        if architecture_id is not None and device_id is not None:
            target.check_compatible(architecture, device)
        object.__setattr__(self, "device_count", device_count)
        object.__setattr__(self, "architecture", architecture.value)
        object.__setattr__(self, "device", device.value)
        object.__setattr__(self, "architecture_id", architecture_id)
        object.__setattr__(self, "device_id", device_id)
        object.__setattr__(self, "architecture_digest", architecture.digest)
        object.__setattr__(self, "device_digest", device.digest)
        object.__setattr__(self, "architecture_document", architecture.document)
        object.__setattr__(self, "device_document", device.document)

    def topology_facts(self) -> target.TopologyFacts:
        """The three CUDA levels, coarsest first.

        Only ``gpu`` comes from the target instance: how many cards a deployment
        has is stated by whoever constructs the target, and no card can read
        which of them it is.
        """
        return target.TopologyFacts(
            (
                target.TopologyLevelFacts(
                    "gpu", self.device_count, parallel_units(self, "gpu"), from_target=True
                ),
                target.TopologyLevelFacts("cta", None, parallel_units(self, "cta")),
                target.TopologyLevelFacts(
                    "thread",
                    self.architecture.topology_limit("thread"),
                    parallel_units(self, "thread"),
                ),
            ),
            parallel_level="cta",
        )

    def get_analyzer(self, selector: str) -> target.Analyzer:
        """Add the ``sync`` family to the core analyses."""
        if selector == "sync":
            return target.Analyzer(
                "sync",
                analysis.analyze_sync,
                requires=("memory",),
                produces=(analysis.SyncPlanMetadata, analysis.SyncMetadata),
            )
        return super().get_analyzer(selector)

    def get_facts(self, facts_type: type, query: object | None = None):
        """Project CUDA hardware through the facts this Target owns."""
        if facts_type is target.TopologyFacts and query is None:
            return target.facts_result(self, facts_type, self.topology_facts())
        if facts_type is target.MemoryExecutionFacts and query is None:
            return target.facts_result(
                self, facts_type, target.MemoryExecutionFacts((self.topology_facts().topologies[-1].name,))
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
        if facts_type is target.ThroughputFacts:
            return target.facts_result(self, facts_type, throughput(self, query))
        if facts_type is target.PerformanceServiceFacts:
            return target.facts_result(self, facts_type, performance_service(self, query))
        return super().get_facts(facts_type, query)

    def torch_device_type(self) -> str:
        """The torch device type holding this program's tensor arguments."""
        return "cuda"

    def to_python(self) -> utils.PythonExpr:
        if type(self) is Target and self.device_id and self.architecture_id:
            count = (
                "" if self.device_count is None else f", device_count={self.device_count}"
            )
            return utils.PythonExpr(
                ("from tilefoundry_nvidia import cuda",),
                f'cuda.Target("{self.device_id}"{count})',
            )
        return super().to_python()

    @property
    def arch(self) -> str:
        """Return the architecture's identity name."""
        return self.architecture.name

    def topology_limit(self, name: str) -> int:
        """Return the physical parallel limit for one CUDA topology level."""
        if name == "gpu":
            return self.device_count or 1
        if name == "cta":
            return self.device.sm_count
        return self.architecture.topology_limit(name)


__all__ = ["Target"]
