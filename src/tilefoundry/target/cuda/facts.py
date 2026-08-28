"""What a CUDA target tells the algorithms that ask it things.

Each conversion answers exactly one aggregate's question, so what a consumer can
read is visible here rather than spread through the consumers themselves. Nothing
in this module decides anything: it restates the installed hardware documents, and
the target's own atom catalogue, in the shape the asking algorithm declared.
"""

from __future__ import annotations

from types import SimpleNamespace

from tilefoundry.analysis.facts import (
    ExplicitMemoryLevelFacts,
    ImplicitMemoryLevelFacts,
    MemoryHierarchyFacts,
    MemoryLevelRelation,
    MemoryRelationKind,
    ParallelCapacityFacts,
    PerformanceServiceFacts,
    ThroughputFacts,
)
from tilefoundry.ir.types import DType
from tilefoundry.schedule.facts import AtomFact
from tilefoundry.schedule.plan import TargetSpecRef

from .atoms import candidate_atoms
from .target import CudaTarget

_PIPELINE_TOPOLOGY = "thread"



_TILE_CAPACITY_SCOPE = "cta"


_PARTITION_TOPOLOGY = "cta"


def memory_hierarchy(target: CudaTarget, query: object = None) -> MemoryHierarchyFacts:
    """The CUDA memory levels and how they are related.

    L1 has no capacity of its own: it and the shared-memory carveout divide one
    physical block per SM, so how much L1 a kernel gets depends on how much
    shared memory it asked for. That is stated as a capacity-sharing edge rather
    than as a number here, because only the program completes it.
    """
    architecture = target.architecture
    device = target.device
    return MemoryHierarchyFacts(
        explicit_levels=(
            ExplicitMemoryLevelFacts(
                name="gmem",
                capacity_bytes=device.hbm_capacity_bytes,
                scope="device",
                owner=device.gmem_owner,
            ),
            ExplicitMemoryLevelFacts(
                name="smem",
                capacity_bytes=architecture.shared_memory_per_cta_bytes,
                scope="cta",
                owner=architecture.smem_owner,
            ),
            ExplicitMemoryLevelFacts(
                name="rmem",
                capacity_bytes=architecture.registers_per_sm_32bit * 4,
                scope="sm",
                owner=architecture.rmem_owner,
            ),
        )
        + (
            (
                ExplicitMemoryLevelFacts(
                    name="tmem",
                    capacity_bytes=architecture.tensor_memory_per_cta_bytes,
                    scope="cta",
                    owner=architecture.tmem_owner,
                ),
            )
            if architecture.tensor_memory_per_cta_bytes is not None
            else ()
        ),
        implicit_levels=(
            ImplicitMemoryLevelFacts(name="l1", capacity_bytes=None, scope="sm"),
            ImplicitMemoryLevelFacts(
                name="l2", capacity_bytes=device.l2_capacity_bytes, scope="device"
            ),
        ),
        relations=(
            MemoryLevelRelation(kind=MemoryRelationKind.CACHES, near="l1", far="l2"),
            MemoryLevelRelation(kind=MemoryRelationKind.CACHES, near="l2", far="gmem"),
            MemoryLevelRelation(
                kind=MemoryRelationKind.SHARES_CAPACITY_WITH,
                near="l1",
                far="smem",
                shared_capacity_bytes=architecture.unified_l1_shared_per_sm_bytes,
            ),
        ),
    )


def throughput(target: CudaTarget, query: object = None) -> ThroughputFacts:
    """The device rates a roofline divides work by.

    The bandwidth is HBM's, so the memory side of the bound is computed from
    global traffic alone. Shared memory and the register file publish no static
    bandwidth, and inventing one would put a number on the bound that no
    document supports.
    """
    device = target.device
    peaks = tuple(
        sorted(device.dense_flops_per_second.items(), key=lambda item: item[0].name)
    )
    return ThroughputFacts(
        peak_flops_per_second=peaks,
        memory_bandwidth_bytes_per_second=device.hbm_bandwidth_bytes_per_second,
        bandwidth_level="gmem",
    )


def performance_service(
    target: CudaTarget, query: object = None
) -> PerformanceServiceFacts:
    """What one CTA gets through, for every kind of work a program asks for.

    The float rates are the device peaks divided among its SMs, the same
    division the roofline's one-unit rates use. The services are what the
    device's own document states, and a device that states none prices no work
    of that kind rather than pricing it at nothing.
    """
    device = target.device
    return PerformanceServiceFacts(
        unit_flops=tuple(
            (dtype, peak // device.sm_count)
            for dtype, peak in sorted(
                device.dense_flops_per_second.items(), key=lambda item: item[0].name
            )
        ),
        unit_ops=tuple(sorted(device.service_ops_per_second.items())),
        unit_bandwidth=(
            ("gmem", device.hbm_bandwidth_bytes_per_second // device.sm_count),
        ),
        unit="cta",
    )


def parallel_capacity(
    target: CudaTarget, query: object = None
) -> ParallelCapacityFacts:
    """How many CTAs the plan assumes run at once.

    This is a compiler policy, not CUDA's grid limit and not the hardware
    resident-CTA maximum: one active CTA per SM. A tighter policy is a
    scheduling input, and changing it changes the plan rather than the program.
    """
    return ParallelCapacityFacts(topology="cta", parallel_units=target.device.sm_count)


def pipeline_facts(target: CudaTarget, query: object) -> object:
    """Project every instruction and capacity fact before pipeline solving.

    The capacity is per-CTA shared memory and is reported as CTA-scoped, because
    that is whose store it is; the level being decided about is finer.
    """
    from tilefoundry.schedule.pipeline.facts import (  # noqa: PLC0415
        PipelineFacts,
        PipelineFactsQuery,
        PipelineInstructionFacts,
    )

    if not isinstance(query, PipelineFactsQuery):
        raise TypeError(
            "CudaTarget pipeline facts need a PipelineFactsQuery, got "
            f"{type(query).__name__}"
        )
    if query.topology != _PIPELINE_TOPOLOGY:
        raise ValueError(
            f"CudaTarget states no pipeline facts for {query.topology!r}; it "
            f"pipelines {_PIPELINE_TOPOLOGY!r}"
        )
    instructions: list[PipelineInstructionFacts] = []
    for statement_id, op in query.statements:
        if not isinstance(statement_id, str) or not statement_id:
            raise ValueError(f"pipeline statement id must be a non-empty string, got {statement_id!r}")
        try:
            candidates = tuple(candidate_atoms(op, target))
        except NotImplementedError:
            candidates = ()
        if not candidates:
            candidates = (
                AtomFact(
                    shape=(1, 1, 1),
                    dtype=(DType.f32, DType.f32, DType.f32),
                    duration=1.0,
                    compute_duration=1.0,
                    storage={},
                    resource={"lane": 1},
                    is_async=False,
                    atom=SimpleNamespace(op=SimpleNamespace(name="cuda.scalar")),
                ),
            )
        instructions.append(PipelineInstructionFacts(statement_id, candidates))
    return PipelineFacts(
        topology=query.topology,
        tile_capacity_scope=_TILE_CAPACITY_SCOPE,
        tile_capacity_bytes=target.architecture.shared_memory_per_cta_bytes,
        max_threads_per_warp=target.architecture.max_threads_per_warp,
        instructions=tuple(instructions),
    )


def partition_facts(target: CudaTarget, query: object) -> object:
    """Project every rate, capacity, and position count before partitioning.

    All of it is what the installed documents state: how many SMs the device has,
    how fast and how large its memory is, and what it peaks at per dtype. How much
    of that machine to occupy is the caller's to decide, so no policy is encoded
    here. After this call the partition holds numbers, not a target.
    """
    from tilefoundry.schedule.partition.facts import (  # noqa: PLC0415
        PartitionFacts,
        PartitionFactsQuery,
    )

    if not isinstance(query, PartitionFactsQuery):
        raise TypeError(
            "CudaTarget partition facts need a PartitionFactsQuery, got "
            f"{type(query).__name__}"
        )
    if query.topology != _PARTITION_TOPOLOGY:
        raise ValueError(
            f"CudaTarget states no partition facts for {query.topology!r}; it "
            f"partitions {_PARTITION_TOPOLOGY!r}"
        )
    device = target.device
    return PartitionFacts(
        topology=query.topology,
        spec=TargetSpecRef.of(target),
        parallel_units=device.sm_count,
        memory_bandwidth_bytes_per_second=device.hbm_bandwidth_bytes_per_second,
        memory_capacity_bytes=device.hbm_capacity_bytes,
        peak_flops_per_second=tuple(
            sorted(device.dense_flops_per_second.items(), key=lambda item: item[0].name)
        ),
    )


__all__ = [
    "memory_hierarchy",
    "parallel_capacity",
    "partition_facts",
    "pipeline_facts",
    "throughput",
]
