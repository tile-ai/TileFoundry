"""What an AMX target tells the analysis families.

The registrations here exist so the support matrix is read rather than assumed:
an analysis is available on AMX because this module says so, not because AMX
happens to inherit from something.
"""

from __future__ import annotations

from tilefoundry.analysis.facts import (
    ExplicitMemoryLevelFacts,
    ImplicitMemoryLevelFacts,
    MemoryHierarchyFacts,
    MemoryLevelRelation,
    MemoryRelationKind,
    PerformanceServiceFacts,
    ThroughputFacts,
)
from tilefoundry.ir.types import DType

from .target import AmxTarget

_ROOFLINE_UNIT = "amx"


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
                owner=device.unified_memory_owner,
            ),


            ExplicitMemoryLevelFacts(
                name="gmem",
                capacity_bytes=device.unified_memory_capacity_bytes,
                scope="device",
                owner=device.unified_memory_owner,
            ),
            ExplicitMemoryLevelFacts(
                name="rmem",
                capacity_bytes=(
                    architecture.staging_bytes + architecture.accumulator_bytes
                ),
                scope="amx",
                owner=architecture.rmem_owner,
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


def performance_service(target: AmxTarget, query: object = None) -> PerformanceServiceFacts:
    """Project the documented M2 Pro compute and host-service model.

    FP16 work shares AMX clusters; FP32 host work uses the measured NEON rate.
    Non-FLOP services use the document's explicitly estimated lane-event rate.
    """
    unit = "core" if query is None else str(query[0] if isinstance(query, tuple) else query)
    if unit not in ("core", "amx"):
        raise ValueError(f"AMX performance services have no topology level {unit!r}")
    device = target.device
    share = device.performance_core_count
    rates = dict(device.unit_flops_per_second.get("neon", {}))
    if device.performance_amx_f16 is not None:
        rates[DType.f16] = device.performance_amx_f16 * device.sm_count // share
    lane_events = device.performance_host_lane_events
    return PerformanceServiceFacts(
        unit_flops=tuple(sorted(rates.items(), key=lambda item: item[0].name)),
        unit_ops=tuple(
            (kind, lane_events)
            for kind in ("integer", "predicate", "select", "local_move", "special")
            if lane_events is not None
        ),
        unit_bandwidth=(("gmem", device.unified_memory_bandwidth_bytes_per_second // share),),
        unit=unit,
    )


__all__ = [
    "memory_hierarchy",
    "throughput",
    "performance_service",
]
