"""What a CUDA target tells the analysis families.

Each conversion answers exactly one family's question, so what an analysis can
read is visible here rather than spread through the analyses themselves. Nothing
in this module measures anything: it restates the installed hardware documents in
the shape a family declared.
"""

from __future__ import annotations

from tilefoundry.analysis.facts import (
    ExplicitMemoryLevelFacts,
    ImplicitMemoryLevelFacts,
    MemoryHierarchyFacts,
    MemoryLevelRelation,
    MemoryRelationKind,
    ParallelCapacityFacts,
    ThroughputFacts,
)
from tilefoundry.target.facts import register_target_facts

from .target import CudaTarget


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


register_target_facts(CudaTarget, MemoryHierarchyFacts, memory_hierarchy)
register_target_facts(CudaTarget, ThroughputFacts, throughput)
register_target_facts(CudaTarget, ParallelCapacityFacts, parallel_capacity)


__all__ = ["memory_hierarchy", "parallel_capacity", "throughput"]
