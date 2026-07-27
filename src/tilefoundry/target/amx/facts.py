"""What an AMX target tells the analysis families.

The registrations here exist so the support matrix is read rather than assumed:
an analysis is available on AMX because this module says so, not because AMX
happens to inherit from something.
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
from .target import AmxTarget

# The execution unit whose measured rate the roofline uses. NEON's rate belongs
# to a different unit and would describe a different atom choice.
_ROOFLINE_UNIT = "amx"

# The one topology level AMX scheduling decides at.
_SCHEDULED_STAGE = "core"


def memory_hierarchy(target: AmxTarget, query: object = None) -> MemoryHierarchyFacts:
    """The AMX memory levels and how they are related.

    Nothing shares capacity here: the caches are separate structures from the
    register files, so the relation list is a plain cache chain. That the same
    flat shape describes both this and a GPU's shared L1 block is the point of
    keeping structure out of the level lists.
    """
    device = target.device
    architecture = target.architecture
    return MemoryHierarchyFacts(
        explicit_levels=(
            ExplicitMemoryLevelFacts(
                name="host",
                capacity_bytes=device.unified_memory_capacity_bytes,
                scope="device",
            ),
            # Unified memory is one pool, so the level a kernel names as global
            # and the level the host allocates from are the same bytes.
            ExplicitMemoryLevelFacts(
                name="gmem",
                capacity_bytes=device.unified_memory_capacity_bytes,
                scope="device",
            ),
            ExplicitMemoryLevelFacts(
                name="rmem",
                capacity_bytes=(
                    architecture.staging_bytes + architecture.accumulator_bytes
                ),
                scope="amx",
            ),
        ),
        implicit_levels=(
            ImplicitMemoryLevelFacts(
                name="l1d",
                capacity_bytes=device.l1d_bytes_per_performance_core,
                scope="core",
            ),
            ImplicitMemoryLevelFacts(
                name="l2",
                capacity_bytes=device.l2_bytes_per_performance_cluster,
                scope="cluster",
            ),
        ),
        relations=(
            MemoryLevelRelation(kind=MemoryRelationKind.CACHES, near="l1d", far="l2"),
            MemoryLevelRelation(kind=MemoryRelationKind.CACHES, near="l2", far="gmem"),
        ),
    )


def throughput(target: AmxTarget, query: object = None) -> ThroughputFacts:
    """The measured AMX rates a roofline divides work by.

    Apple publishes no instruction throughput, so these are measured figures for
    one unit rather than a vendor peak. A bound derived from them is a bound
    against this host's observed behaviour.
    """
    device = target.device
    rates = device.unit_flops_per_second.get(_ROOFLINE_UNIT, {})
    return ThroughputFacts(
        peak_flops_per_second=tuple(
            sorted(
                ((dtype, rate) for dtype, rate in rates.items() if isinstance(dtype, DType)),
                key=lambda item: item[0].name,
            )
        ),
        memory_bandwidth_bytes_per_second=(
            device.unified_memory_bandwidth_bytes_per_second
        ),
        bandwidth_level="gmem",
    )


def parallel_capacity(
    target: AmxTarget, query: object = None
) -> ParallelCapacityFacts:
    """How many cores the plan assumes issue AMX work at once.

    The parallel extent is the performance cores, because that is what the
    program's ``core`` topology divides over. The coprocessor count bounds
    throughput rather than the launch shape, and enters through the measured
    rate instead.
    """
    return ParallelCapacityFacts(
        topology="core", parallel_units=target.device.performance_core_count
    )


def tile_store(target: AmxTarget, query: object = None) -> TileStoreFacts:
    """Where a tile of the queried level lives, and how much of it there is.

    A core-level tile's resident working set lives in that core's L1d. The AMX
    register files bound one atom instance rather than a tile, and they do it by
    filtering that atom out of the catalogue.
    """
    if not isinstance(query, str) or not query:
        raise TypeError(
            f"a tile store must be queried by stage name, got {query!r}"
        )
    if query != _SCHEDULED_STAGE:
        raise ValueError(
            f"AmxTarget states no tile store for stage {query!r}; it schedules "
            f"{_SCHEDULED_STAGE!r}"
        )
    return TileStoreFacts(
        stage=query,
        tile_capacity_bytes=target.device.l1d_bytes_per_performance_core,
    )


def atom_candidates(
    target: AmxTarget, query: AtomCandidateQuery
) -> AtomCandidateFacts:
    """The atoms this target admits for one operation, in catalogue order."""
    if not isinstance(query, AtomCandidateQuery):
        raise TypeError(
            "AmxTarget atom candidates need an AtomCandidateQuery, got "
            f"{type(query).__name__}"
        )
    if query.stage != _SCHEDULED_STAGE:
        raise ValueError(
            f"AmxTarget enumerates no atoms for stage {query.stage!r}; it "
            f"schedules {_SCHEDULED_STAGE!r}"
        )
    return AtomCandidateFacts(tuple(candidate_atoms(query.op, target)))


def pipeline_facts(target: AmxTarget, query: PipelineFactsQuery) -> PipelineFacts:
    """Project the finite AMX instruction catalogue before solving."""
    if not isinstance(query, PipelineFactsQuery):
        raise TypeError("AMX pipeline facts need a PipelineFactsQuery")
    if query.stage != _SCHEDULED_STAGE:
        raise ValueError(f"AMX states no pipeline facts for {query.stage!r}")
    instructions: list[PipelineInstructionFacts] = []
    for statement_id, op in query.statements:
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
                    resource={"core": 1},
                    is_async=False,
                    atom=SimpleNamespace(op=SimpleNamespace(name="amx.scalar")),
                ),
            )
        instructions.append(PipelineInstructionFacts(statement_id, candidates))
    return PipelineFacts(
        stage=query.stage,
        tile_capacity_bytes=target.device.l1d_bytes_per_performance_core,
        max_threads_per_warp=1,
        instructions=tuple(instructions),
    )


register_target_facts(AmxTarget, MemoryHierarchyFacts, memory_hierarchy)
register_target_facts(AmxTarget, ThroughputFacts, throughput)
register_target_facts(AmxTarget, ParallelCapacityFacts, parallel_capacity)
register_target_facts(AmxTarget, TileStoreFacts, tile_store)
register_target_facts(AmxTarget, AtomCandidateFacts, atom_candidates)
register_target_facts(AmxTarget, PipelineFacts, pipeline_facts)


__all__ = [
    "atom_candidates",
    "memory_hierarchy",
    "parallel_capacity",
    "pipeline_facts",
    "throughput",
    "tile_store",
]
