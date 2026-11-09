"""What an AMX target tells the analysis families.

The registrations here exist so the support matrix is read rather than assumed:
an analysis is available on AMX because this module says so, not because AMX
happens to inherit from something.
"""

from __future__ import annotations

from tilefoundry import target
from tilefoundry.ir import expr

ROOFLINE_UNIT = "amx"


def memory_hierarchy(tgt: target.Target, query: object = None) -> target.MemoryHierarchyFacts:
    """The AMX memory levels and how they are related.

    Nothing shares capacity here: the caches are separate structures from the
    register files, so the relation list is a plain cache chain. That the same
    flat shape describes both this and a GPU's shared L1 block is the point of
    keeping structure out of the level lists.
    """
    device = tgt.device
    architecture = tgt.architecture
    return target.MemoryHierarchyFacts(
        explicit_levels=(
            target.ExplicitMemoryLevelFacts(
                name="host",
                capacity_bytes=device.unified_memory_capacity_bytes,
                scope="device",
                owner=device.unified_memory_owner,
            ),


            target.ExplicitMemoryLevelFacts(
                name="gmem",
                capacity_bytes=device.unified_memory_capacity_bytes,
                scope="device",
                owner=device.unified_memory_owner,
            ),
            target.ExplicitMemoryLevelFacts(
                name="rmem",
                capacity_bytes=(
                    2 * architecture.staging_bytes + architecture.accumulator_bytes
                ),
                scope="thread",
                owner=architecture.rmem_owner,
            ),
        ),
        implicit_levels=(
            target.ImplicitMemoryLevelFacts(
                name="l1d",
                capacity_bytes=device.l1d_bytes_per_performance_core,
                scope="core",
            ),
            target.ImplicitMemoryLevelFacts(
                name="l2",
                capacity_bytes=device.l2_bytes_per_performance_cluster,
                scope="cluster",
            ),
        ),
        relations=(
            target.MemoryLevelRelation(kind=target.MemoryRelationKind.CACHES, near="l1d", far="l2"),
            target.MemoryLevelRelation(kind=target.MemoryRelationKind.CACHES, near="l2", far="gmem"),
        ),
    )


def throughput(tgt: target.Target, query: object = None) -> target.ThroughputFacts:
    """The measured AMX rates a roofline divides work by.

    Apple publishes no instruction throughput, so these are measured figures for
    one unit rather than a vendor peak. A bound derived from them is a bound
    against this host's observed behaviour.
    """
    device = tgt.device
    rates = device.unit_flops_per_second.get(ROOFLINE_UNIT, {})
    return target.ThroughputFacts(
        peak_flops_per_second=tuple(
            sorted(
                ((dtype, rate) for dtype, rate in rates.items() if isinstance(dtype, expr.DType)),
                key=lambda item: item[0].name,
            )
        ),
        memory_bandwidth_bytes_per_second=(
            device.unified_memory_bandwidth_bytes_per_second
        ),
        bandwidth_level="gmem",
    )


def performance_service(tgt: target.Target, query: object = None) -> target.PerformanceServiceFacts:
    """Leave generic CPU-thread rates unknown without a P/E placement policy."""
    unit = "thread" if query is None else str(query[0] if isinstance(query, tuple) else query)
    if unit != "thread":
        raise ValueError(f"AMX performance services have no topology level {unit!r}")
    return target.PerformanceServiceFacts(unit_flops=(), unit_ops=(), unit_bandwidth=(), unit=unit)



__all__ = [
    "memory_hierarchy",
    "throughput",
    "performance_service",
]
