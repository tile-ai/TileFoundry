"""CUDA compilation target composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

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
from tilefoundry.target.cuda.architecture import CudaArchitecture
from tilefoundry.target.cuda.device import CudaDevice
from tilefoundry.target.cuda.spec import (
    ARCHITECTURE_SCHEMA,
    DEVICE_SCHEMA,
    build_cuda_architecture,
    build_cuda_device,
)
from tilefoundry.target.facts import TopologyLimitFacts, facts_result
from tilefoundry.target.hardware.envelope import HardwareDocument
from tilefoundry.target.services import CodeGenerator, Scheduler
from tilefoundry.utils.python_source import PythonExpr


@register_target
@dataclass(frozen=True, init=False)
class CudaTarget(Target):
    """CUDA target composed from one device and the architecture it runs."""

    name: ClassVar[str] = "cuda"
    hardware: ClassVar[HardwareSpec] = HardwareSpec(
        package="tilefoundry.target.cuda.hardware",
        schemas={
            ARCHITECTURE_SCHEMA: build_cuda_architecture,
            DEVICE_SCHEMA: build_cuda_device,
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
    def available(cls) -> tuple[CudaTarget, ...]:
        return tuple(cls(device_id) for device_id in _available_device_ids(cls.hardware))

    def __init__(
        self,
        device: Device | str | Path,
        architecture: Architecture | str | Path | None = None,
        *,
        arch: str | None = None,
    ) -> None:
        if architecture is None:
            architecture = _architecture_of(
                device,
                device_type=CudaDevice,
                role="CudaTarget.device",
                hardware=self.hardware,
            )
        architecture = select(
            architecture,
            CudaArchitecture,
            role="CudaTarget.architecture",
            hardware=self.hardware,
        )
        device = select(
            device, CudaDevice, role="CudaTarget.device", hardware=self.hardware
        )
        architecture_id, device_id = architecture.id, device.id
        if arch is not None and arch != architecture.value.name:
            raise ValueError(
                f"CudaTarget: arch {arch!r} conflicts with architecture.name "
                f"{architecture.value.name!r}"
            )
        if architecture_id is not None and device_id is not None:
            check_compatible(architecture, device)
        object.__setattr__(self, "architecture", architecture.value)
        object.__setattr__(self, "device", device.value)
        object.__setattr__(self, "architecture_id", architecture_id)
        object.__setattr__(self, "device_id", device_id)
        object.__setattr__(self, "architecture_digest", architecture.digest)
        object.__setattr__(self, "device_digest", device.digest)
        object.__setattr__(self, "_architecture_document", architecture.document)
        object.__setattr__(self, "_device_document", device.document)

    def get_facts(self, facts_type: type, query: object | None = None):
        """Project CUDA hardware through the facts this Target owns."""
        if facts_type is TopologyLimitFacts:
            if query == "cta":
                return facts_result(self, facts_type, TopologyLimitFacts("cta", None))
            if query == "thread":
                return facts_result(
                    self,
                    facts_type,
                    TopologyLimitFacts(
                        "thread", self.architecture.topology_limit("thread")
                    ),
                )
            return super().get_facts(facts_type, query)

        from tilefoundry.analysis.facts import (  # noqa: PLC0415
            MemoryHierarchyFacts,
            ParallelCapacityFacts,
            ThroughputFacts,
        )
        from tilefoundry.target.cuda.facts import (  # noqa: PLC0415
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
            from tilefoundry.target.cuda.facts import pipeline_facts  # noqa: PLC0415

            return facts_result(self, facts_type, pipeline_facts(self, query))

        from tilefoundry.schedule.partition import PartitionFacts  # noqa: PLC0415

        if facts_type is PartitionFacts:
            from tilefoundry.target.cuda.facts import partition_facts  # noqa: PLC0415

            return facts_result(self, facts_type, partition_facts(self, query))
        return super().get_facts(facts_type, query)

    def get_scheduler(self, topology: str) -> Scheduler:
        """Select a CUDA scheduler; subclasses inherit these solvers."""
        from tilefoundry.target.cuda.schedule import cuda_scheduler  # noqa: PLC0415

        scheduler = cuda_scheduler(topology)
        if scheduler is not None:
            return scheduler
        return super().get_scheduler(topology)

    def get_code_generator(self) -> CodeGenerator:
        from tilefoundry.codegen.cuda.module import (  # noqa: PLC0415
            CUDA_CODE_GENERATOR,
        )

        return CUDA_CODE_GENERATOR

    def _python_import_module(self) -> str:
        if type(self) is CudaTarget:
            return "tilefoundry.target.cuda"
        return super()._python_import_module()

    def to_python(self) -> PythonExpr:
        if type(self) is CudaTarget and self.device_id and self.architecture_id:
            return PythonExpr(
                ("from tilefoundry.target import CudaTarget",),
                f'CudaTarget("{self.device_id}")',
            )
        return super().to_python()

    @property
    def arch(self) -> str:
        """Return the architecture name used by compilation."""
        return self.architecture.name

    topology_levels: ClassVar[tuple[str, ...]] = ("cta", "thread")

    def topology_limit(self, name: str) -> int:
        """Return the physical parallel limit for one CUDA topology level."""
        if name == "cta":
            return self.device.sm_count
        return self.architecture.topology_limit(name)


__all__ = ["CudaTarget"]
