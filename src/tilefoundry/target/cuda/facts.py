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
    ThroughputFacts,
)
from tilefoundry.ir.types import DType
from tilefoundry.schedule.facts import (
    AtomCandidateFacts,
    AtomCandidateQuery,
    AtomFact,
    TileStoreFacts,
)
from tilefoundry.schedule.pipeline.facts import (
    PipelineFacts,
    PipelineFactsQuery,
    PipelineInstructionFacts,
)
from tilefoundry.target.facts import register_target_facts

from .atoms import candidate_atoms
from .target import CudaTarget

# The one topology level CUDA scheduling decides at.
_SCHEDULED_STAGE = "cta"


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
            ),
            ExplicitMemoryLevelFacts(
                name="smem",
                capacity_bytes=architecture.shared_memory_per_cta_bytes,
                scope="cta",
            ),
            ExplicitMemoryLevelFacts(
                name="rmem",
                # The register file is stated per SM in 32-bit slots, and every
                # slot is four bytes wide.
                capacity_bytes=architecture.registers_per_sm_32bit * 4,
                scope="sm",
            ),
            # Tensor memory arrives with a later architecture; SM90 has none, and
            # a level with no capacity is how that is said.
            ExplicitMemoryLevelFacts(name="tmem", capacity_bytes=None, scope="cta"),
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
    return ThroughputFacts(
        peak_flops_per_second=tuple(sorted(
            device.dense_flops_per_second.items(), key=lambda item: item[0].name
        )),
        memory_bandwidth_bytes_per_second=device.hbm_bandwidth_bytes_per_second,
        bandwidth_level="gmem",
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


def tile_store(target: CudaTarget, query: object = None) -> TileStoreFacts:
    """Where a tile of the queried level lives, and how much of it there is.

    A CTA-level tile's resident working set lives in shared memory, whose
    per-CTA capacity is a limit of the architecture rather than of the device.
    """
    if not isinstance(query, str) or not query:
        raise TypeError(
            f"a tile store must be queried by stage name, got {query!r}"
        )
    if query != _SCHEDULED_STAGE:
        raise ValueError(
            f"CudaTarget states no tile store for stage {query!r}; it schedules "
            f"{_SCHEDULED_STAGE!r}"
        )
    return TileStoreFacts(
        stage=query,
        tile_capacity_bytes=target.architecture.shared_memory_per_cta_bytes,
    )


def atom_candidates(
    target: CudaTarget, query: AtomCandidateQuery
) -> AtomCandidateFacts:
    """The atoms this target admits for one operation, in catalogue order."""
    if not isinstance(query, AtomCandidateQuery):
        raise TypeError(
            "CudaTarget atom candidates need an AtomCandidateQuery, got "
            f"{type(query).__name__}"
        )
    if query.stage != _SCHEDULED_STAGE:
        raise ValueError(
            f"CudaTarget enumerates no atoms for stage {query.stage!r}; it "
            f"schedules {_SCHEDULED_STAGE!r}"
        )
    return AtomCandidateFacts(tuple(candidate_atoms(query.op, target)))


def pipeline_facts(target: CudaTarget, query: PipelineFactsQuery) -> PipelineFacts:
    """Project every instruction and capacity fact before pipeline solving."""
    if not isinstance(query, PipelineFactsQuery):
        raise TypeError(
            "CudaTarget pipeline facts need a PipelineFactsQuery, got "
            f"{type(query).__name__}"
        )
    if query.stage != _SCHEDULED_STAGE:
        raise ValueError(
            f"CudaTarget states no pipeline facts for {query.stage!r}; it schedules "
            f"{_SCHEDULED_STAGE!r}"
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
        stage=query.stage,
        tile_capacity_bytes=target.architecture.shared_memory_per_cta_bytes,
        max_threads_per_warp=target.architecture.max_threads_per_warp,
        instructions=tuple(instructions),
    )


register_target_facts(CudaTarget, MemoryHierarchyFacts, memory_hierarchy)
register_target_facts(CudaTarget, ThroughputFacts, throughput)
register_target_facts(CudaTarget, ParallelCapacityFacts, parallel_capacity)
register_target_facts(CudaTarget, TileStoreFacts, tile_store)
register_target_facts(CudaTarget, AtomCandidateFacts, atom_candidates)
register_target_facts(CudaTarget, PipelineFacts, pipeline_facts)


__all__ = [
    "atom_candidates",
    "memory_hierarchy",
    "parallel_capacity",
    "pipeline_facts",
    "throughput",
    "tile_store",
]
