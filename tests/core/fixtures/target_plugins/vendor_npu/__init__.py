"""A document-free Target provider used by installed smoke coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from tilefoundry import DType, target


@target.register
@dataclass(frozen=True)
class VendorNpuTarget(target.Target):
    """A small backend whose capabilities are expressed directly as Facts."""

    name: ClassVar[str] = "vendor.npu"

    def get_facts(self, facts_type: type, query: object | None = None):
        if facts_type is target.TopologyFacts and query is None:
            value = target.TopologyFacts(
                (target.TopologyLevelFacts("core", 256, 16),), parallel_level="core"
            )
        elif facts_type is target.TopologyLevelFacts and query == "core":
            value = target.TopologyLevelFacts("core", 256, 16)
        elif facts_type is target.MemoryHierarchyFacts:
            value = target.MemoryHierarchyFacts(
                explicit_levels=(
                    target.ExplicitMemoryLevelFacts(
                        "gmem", 64_000_000_000, "npu", "target"
                    ),
                    target.ExplicitMemoryLevelFacts("smem", 4_194_304, "core", "core"),
                    target.ExplicitMemoryLevelFacts("rmem", 1_048_576, "core", "core"),
                ),
                implicit_levels=(),
                relations=(),
            )
        elif facts_type is target.ThroughputFacts:
            value = target.ThroughputFacts(
                peak_flops_per_second=((DType.f32, 2_000_000_000_000_000),),
                memory_bandwidth_bytes_per_second=2_000_000_000_000,
                bandwidth_level="gmem",
            )
        elif facts_type is target.PerformanceServiceFacts:
            value = target.PerformanceServiceFacts(
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
        else:
            return super().get_facts(facts_type, query)
        return target.facts_result(self, facts_type, value)
