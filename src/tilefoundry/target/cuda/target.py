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
from tilefoundry.target.facts import (
    TopologyFacts,
    TopologyLevelFacts,
    facts_result,
)
from tilefoundry.target.hardware.envelope import HardwareDocument
from tilefoundry.target.services import CodeGenerator
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

    device_count: int | None = field(default=None, init=False)
    """How many cards a program of this target may name at its ``gpu`` level."""




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
        device_count: int | None = None,
    ) -> None:
        if device_count is not None and (
            isinstance(device_count, bool)
            or not isinstance(device_count, int)
            or device_count < 1
        ):
            raise ValueError(
                f"CudaTarget: device_count {device_count!r} must be a positive int "
                f"or None, which admits any extent at the gpu level"
            )
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
        object.__setattr__(self, "device_count", device_count)
        object.__setattr__(self, "architecture", architecture.value)
        object.__setattr__(self, "device", device.value)
        object.__setattr__(self, "architecture_id", architecture_id)
        object.__setattr__(self, "device_id", device_id)
        object.__setattr__(self, "architecture_digest", architecture.digest)
        object.__setattr__(self, "device_digest", device.digest)
        object.__setattr__(self, "_architecture_document", architecture.document)
        object.__setattr__(self, "_device_document", device.document)

    def _topology_facts(self) -> TopologyFacts:
        """The three CUDA levels, coarsest first.

        Only ``gpu`` comes from the target instance: how many cards a deployment
        has is stated by whoever constructs the target, and no card can read
        which of them it is.
        """
        from tilefoundry.target.cuda.facts import parallel_units  # noqa: PLC0415

        return TopologyFacts(
            (
                TopologyLevelFacts(
                    "gpu", self.device_count, parallel_units(self, "gpu"), from_target=True
                ),
                TopologyLevelFacts("cta", None, parallel_units(self, "cta")),
                TopologyLevelFacts(
                    "thread",
                    self.architecture.topology_limit("thread"),
                    parallel_units(self, "thread"),
                ),
            ),
            parallel_level="cta",
        )

    def get_facts(self, facts_type: type, query: object | None = None):
        """Project CUDA hardware through the facts this Target owns."""
        if facts_type is TopologyFacts and query is None:
            return facts_result(self, facts_type, self._topology_facts())
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
        from tilefoundry.target.cuda.facts import (  # noqa: PLC0415
            memory_hierarchy,
            performance_service,
            throughput,
        )

        if facts_type is MemoryHierarchyFacts:
            return facts_result(self, facts_type, memory_hierarchy(self, query))
        if facts_type is ThroughputFacts:
            return facts_result(self, facts_type, throughput(self, query))
        if facts_type is PerformanceServiceFacts:
            return facts_result(self, facts_type, performance_service(self, query))
        return super().get_facts(facts_type, query)

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
            count = (
                "" if self.device_count is None else f", device_count={self.device_count}"
            )
            return PythonExpr(
                ("from tilefoundry.target import CudaTarget",),
                f'CudaTarget("{self.device_id}"{count})',
            )
        return super().to_python()

    @property
    def arch(self) -> str:
        """Return the architecture name used by compilation."""
        return self.architecture.name

    def topology_limit(self, name: str) -> int:
        """Return the physical parallel limit for one CUDA topology level."""
        if name == "gpu":
            return self.device_count or 1
        if name == "cta":
            return self.device.sm_count
        return self.architecture.topology_limit(name)


__all__ = ["CudaTarget"]
