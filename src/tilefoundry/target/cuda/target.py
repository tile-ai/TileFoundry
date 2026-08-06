"""CUDA compilation target composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from tilefoundry.ir.types.shard import Topology
from tilefoundry.target.base import Architecture, Device, Target
from tilefoundry.target.hardware.envelope import IncompatiblePairError
from tilefoundry.target.hardware.registry import check_compatible, select
from tilefoundry.target.registration import register_target


def _architecture_of(device: Device | str) -> str:
    """The architecture *device*'s own document declares, when it declares one."""
    if isinstance(device, Device):
        raise ValueError(
            "CudaTarget: a Device supplied directly carries no document to read a "
            "compatible architecture from; name the architecture as well"
        )
    architectures = select(device, Device, role="CudaTarget.device").document.compatibility
    if len(architectures) != 1:
        raise IncompatiblePairError(
            f"device {device!r} declares {list(architectures)} as compatible "
            f"architectures; name the one to build against"
        )
    return architectures[0]


def _dtype_repr(dtype: object) -> str:
    return f"DType.{getattr(dtype, 'name')}"


def _architecture_repr(architecture: Architecture) -> str:
    dtypes = ", ".join(
        _dtype_repr(dtype) for dtype in architecture.supported_compute_dtypes
    )
    if len(architecture.supported_compute_dtypes) == 1:
        dtypes += ","
    return (
        f"{type(architecture).__name__}("
        f"name={architecture.name!r}, "
        f"supported_compute_dtypes=({dtypes}), "
        f"instruction_capabilities={architecture.instruction_capabilities!r}, "
        f"max_threads_per_cta={architecture.max_threads_per_cta}, "
        f"max_threads_per_warp={architecture.max_threads_per_warp}, "
        f"max_warps_per_cta={architecture.max_warps_per_cta}, "
        f"max_resident_ctas_per_sm={architecture.max_resident_ctas_per_sm}, "
        f"shared_memory_per_sm_bytes={architecture.shared_memory_per_sm_bytes}, "
        f"shared_memory_per_cta_bytes={architecture.shared_memory_per_cta_bytes}, "
        "unified_l1_shared_per_sm_bytes="
        f"{architecture.unified_l1_shared_per_sm_bytes}, "
        f"registers_per_sm_32bit={architecture.registers_per_sm_32bit}"
        ")"
    )


def _device_repr(device: Device) -> str:
    flops = ", ".join(
        f"({_dtype_repr(dtype)}, {value})"
        for dtype, value in device.dense_flops_per_second.items()
    )
    if len(device.dense_flops_per_second) == 1:
        flops += ","
    return (
        f"{type(device).__name__}("
        f"name={device.name!r}, "
        f"sm_count={device.sm_count}, "
        f"hbm_capacity_bytes={device.hbm_capacity_bytes}, "
        "hbm_bandwidth_bytes_per_second="
        f"{device.hbm_bandwidth_bytes_per_second}, "
        f"l2_capacity_bytes={device.l2_capacity_bytes!r}, "
        f"_dense_flops=({flops})"
        ")"
    )


@register_target
@dataclass(frozen=True, init=False)
class CudaTarget(Target):
    """CUDA target composed from one device and the architecture it runs."""

    name: ClassVar[str] = "cuda"
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
        device: Device | str,
        architecture: Architecture | str | None = None,
        *,
        arch: str | None = None,
    ) -> None:
        if architecture is None:
            architecture = _architecture_of(device)
        architecture = select(
            architecture, Architecture, role="CudaTarget.architecture"
        )
        device = select(device, Device, role="CudaTarget.device")
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

    def get_analyzer(self, selector: str) -> "Analyzer":
        """Select a CUDA analysis service; subclasses inherit this mapping."""
        from tilefoundry.analysis.registry import (  # noqa: PLC0415
            builtin_analyzers,
        )

        try:
            return builtin_analyzers()[selector]
        except KeyError:
            return super().get_analyzer(selector)

    def get_facts(self, facts_type: type, query: object | None = None):
        """Project CUDA hardware through the facts this Target owns."""
        from tilefoundry.analysis.facts import (  # noqa: PLC0415
            MemoryHierarchyFacts,
            ParallelCapacityFacts,
            ThroughputFacts,
        )
        from tilefoundry.schedule.partition import PartitionFacts  # noqa: PLC0415
        from tilefoundry.schedule.pipeline import PipelineFacts  # noqa: PLC0415
        from tilefoundry.target.cuda.facts import (  # noqa: PLC0415
            memory_hierarchy,
            parallel_capacity,
            partition_facts,
            pipeline_facts,
            throughput,
        )
        from tilefoundry.target.facts import facts_result  # noqa: PLC0415

        projections = {
            MemoryHierarchyFacts: memory_hierarchy,
            ThroughputFacts: throughput,
            ParallelCapacityFacts: parallel_capacity,
            PipelineFacts: pipeline_facts,
            PartitionFacts: partition_facts,
        }
        try:
            projection = projections[facts_type]
        except KeyError:
            return super().get_facts(facts_type, query)
        return facts_result(self, facts_type, projection(self, query))

    def get_scheduler(self, topology: str) -> "Scheduler":
        """Select a CUDA scheduler; subclasses inherit these solvers."""
        from tilefoundry.target.cuda.schedule import (  # noqa: PLC0415
            cuda_schedulers,
        )

        try:
            return cuda_schedulers()[topology]
        except KeyError:
            return super().get_scheduler(topology)

    def get_code_generator(self) -> "CodeGenerator":
        from tilefoundry.codegen.cuda.module import (  # noqa: PLC0415
            CUDA_CODE_GENERATOR,
        )

        return CUDA_CODE_GENERATOR

    def __repr__(self) -> str:
        """Return a constructor expression for this concrete CUDA Target."""
        constructor = type(self).__name__
        if self.device_id is not None and self.architecture_id is not None:
            try:
                rebuilt = type(self)(self.device_id)
            except (TypeError, ValueError, IncompatiblePairError):
                rebuilt = None
            if (
                rebuilt is not None
                and rebuilt == self
                and rebuilt.architecture_id == self.architecture_id
            ):
                return f"{constructor}({self.device_id!r})"
            return (
                f"{constructor}(device={self.device_id!r}, "
                f"architecture={self.architecture_id!r})"
            )
        device = (
            repr(self.device_id)
            if self.device_id is not None
            else _device_repr(self.device)
        )
        architecture = (
            repr(self.architecture_id)
            if self.architecture_id is not None
            else _architecture_repr(self.architecture)
        )
        return (
            f"{constructor}(device={device}, "
            f"architecture={architecture})"
        )

    @property
    def arch(self) -> str:
        """Return the architecture name used by compilation."""
        return self.architecture.name

    @property
    def topology_levels(self) -> tuple[str, ...]:
        """Return program topology levels admitted by CUDA compilation."""
        return ("cta", "thread")

    def topology_limit(self, name: str) -> int | None:
        """Return a per-CTA topology limit, if one exists.

        The CUDA grid is a launch shape, not an SM allocation.  Its static
        extent is therefore deliberately unbounded here; a fixed-wave parallel
        capacity is a scheduling policy applied separately.
        """
        if name == "cta":
            return None
        if name == "thread":
            return self.architecture.topology_limit("thread")
        raise ValueError(
            f"{self!r}: unsupported topology level {name!r}; "
            f"supported levels are {self.topology_levels}"
        )

    def validate_program_topology(self, topology: Topology) -> None:
        """Validate one declared program topology against CUDA facts."""
        if topology.name not in self.topology_levels:
            raise ValueError(
                f"{self!r}: unsupported topology level {topology.name!r}; "
                f"supported levels are {self.topology_levels}"
            )
        if topology.name == "cta" and topology.size is None:
            return
        if not isinstance(topology.size, int) or isinstance(topology.size, bool):
            raise ValueError(
                f"{self!r}: topology {topology.name!r} requires a positive "
                f"static integer extent, got {topology.size!r}"
            )
        limit = self.topology_limit(topology.name)
        if topology.size < 1:
            raise ValueError(
                f"{self!r}: topology {topology.name!r} extent {topology.size} "
                "must be positive"
            )
        if limit is not None and topology.size > limit:
            raise ValueError(
                f"{self!r}: topology {topology.name!r} extent {topology.size} "
                f"must satisfy 1 <= extent <= {limit}"
            )


__all__ = ["CudaTarget"]
