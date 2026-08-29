"""A document-free Target provider used by installed smoke coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from tilefoundry import DType
from tilefoundry.analysis import ExplicitMemoryLevelFacts
from tilefoundry.target import (
    MemoryHierarchyFacts,
    ParallelCapacityFacts,
    PerformanceServiceFacts,
    Target,
    ThroughputFacts,
    TopologyLimitFacts,
    facts_result,
    register_target,
)


@register_target
@dataclass(frozen=True)
class VendorNpuTarget(Target):
    """A small backend whose capabilities are expressed directly as Facts."""

    name: ClassVar[str] = "vendor.npu"
    topology_levels: ClassVar[tuple[str, ...]] = ("core",)

    def get_facts(self, facts_type: type, query: object | None = None):
        if facts_type is TopologyLimitFacts and query == "core":
            value = TopologyLimitFacts("core", 256)
        elif facts_type is MemoryHierarchyFacts:
            value = MemoryHierarchyFacts(
                explicit_levels=(
                    ExplicitMemoryLevelFacts(
                        "gmem", 64_000_000_000, "npu", "target"
                    ),
                    ExplicitMemoryLevelFacts("smem", 4_194_304, "core", "core"),
                    ExplicitMemoryLevelFacts("rmem", 1_048_576, "core", "core"),
                ),
                implicit_levels=(),
                relations=(),
            )
        elif facts_type is ThroughputFacts:
            value = ThroughputFacts(
                peak_flops_per_second=((DType.f32, 2_000_000_000_000_000),),
                memory_bandwidth_bytes_per_second=2_000_000_000_000,
                bandwidth_level="gmem",
            )
        elif facts_type is PerformanceServiceFacts:
            value = PerformanceServiceFacts(
                unit_flops=((DType.f32, 2_000_000_000_000_000 // 16),),
                unit_ops=(
                    ("integer", 1_000_000_000),
                    ("predicate", 1_000_000_000),
                    ("select", 1_000_000_000),
                    ("special", 250_000_000),
                ),
                unit_bandwidth=(("gmem", 2_000_000_000_000 // 16),),
                unit="core",
            )
        elif facts_type is ParallelCapacityFacts:
            value = ParallelCapacityFacts("core", 16)
        else:
            return super().get_facts(facts_type, query)
        return facts_result(self, facts_type, value)
