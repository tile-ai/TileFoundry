"""Metal memory ownership and explicitly unknown compute service rates."""
from tilefoundry.analysis.facts import (
    ExplicitMemoryLevelFacts, MemoryHierarchyFacts, PerformanceServiceFacts, ThroughputFacts,
)

def memory_hierarchy(target, query=None):
    d = target.device
    return MemoryHierarchyFacts(
        explicit_levels=(
            ExplicitMemoryLevelFacts("gmem", d.unified_memory_bytes, "device", "target"),
            ExplicitMemoryLevelFacts("smem", d.threadgroup_memory_bytes, "threadgroup", "threadgroup"),
            ExplicitMemoryLevelFacts("rmem", d.register_bytes_per_thread, "thread", "thread"),
        ), implicit_levels=(), relations=(),
    )

def throughput(target, query=None):
    return ThroughputFacts((), target.device.memory_bandwidth_bytes_per_second, "gmem")

def performance_service(target, query=None):
    unit = "threadgroup" if query is None else str(query[0] if isinstance(query, tuple) else query)
    if target._topology_facts().level(unit) is None:
        raise ValueError(f"unknown Metal topology {unit!r}")
    return PerformanceServiceFacts((), (), (), unit)
