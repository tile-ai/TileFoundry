"""A document-free Target provider used by installed smoke coverage."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from tilefoundry import DType
from tilefoundry.analysis import ExplicitMemoryLevelFacts
from tilefoundry.schedule import SchedulePlan
from tilefoundry.target import (
    MemoryHierarchyFacts,
    ParallelCapacityFacts,
    Scheduler,
    Target,
    ThroughputFacts,
    TopologyLimitFacts,
    facts_result,
    register_target,
)


@dataclass(frozen=True)
class VendorNpuPlan(SchedulePlan):
    topology: str
    extent: int

    def verify(self, module, function, topology) -> None:
        if topology.name != self.topology or topology.size != self.extent:
            raise ValueError("vendor NPU plan does not match its topology")

    def to_json(self) -> str:
        return json.dumps({"topology": self.topology, "extent": self.extent})

    def render(self) -> str:
        return f"vendor NPU schedule: {self.extent} {self.topology}"


def _schedule_vendor_npu(module, function, target, topology, options) -> SchedulePlan:
    marker = os.environ.get("TF_VENDOR_NPU_SCHEDULER_CALLS")
    if marker is not None:
        with Path(marker).open("a", encoding="utf-8") as calls:
            calls.write(f"{topology.name}\n")
    return VendorNpuPlan(topology.name, topology.size)


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
                peak_flops_per_second_per_unit=(),
                memory_bandwidth_bytes_per_second_per_unit=None,
                rate_unit="core",
            )
        elif facts_type is ParallelCapacityFacts:
            value = ParallelCapacityFacts("core", 16)
        else:
            return super().get_facts(facts_type, query)
        return facts_result(self, facts_type, value)

    def get_scheduler(self, topology: str) -> Scheduler:
        if topology == "core":
            return Scheduler("core", _schedule_vendor_npu)
        return super().get_scheduler(topology)
