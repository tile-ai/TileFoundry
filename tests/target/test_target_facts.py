"""Facts are selected by the exact Target value, not a global registry."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import ClassVar

import pytest

from tilefoundry.analysis.facts import (
    MemoryHierarchyFacts,
    ParallelCapacityFacts,
    ThroughputFacts,
)
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import (
    AmxTarget,
    CudaTarget,
    Target,
    TopologyLimitFacts,
    UnsupportedCapabilityError,
)
from tilefoundry.target.facts import TargetFactsError, facts_result


def test_builtin_targets_own_their_facts_projections() -> None:
    cuda = CudaTarget("nvidia.h200_sxm")

    throughput = cuda.get_facts(ThroughputFacts)
    memory = cuda.get_facts(MemoryHierarchyFacts)

    assert throughput.memory_bandwidth_bytes_per_second == 4_800_000_000_000
    assert memory.explicit("gmem").capacity_bytes == cuda.device.hbm_capacity_bytes
    assert {level.name: level.owner for level in memory.explicit_levels} == {
        "gmem": "target",
        "smem": "cta",
        "rmem": "thread",
        "tmem": "cta",
    }
    assert AmxTarget().get_facts(ThroughputFacts).bandwidth_level == "gmem"
    assert {
        level.name: level.owner
        for level in AmxTarget().get_facts(MemoryHierarchyFacts).explicit_levels
    } == {"host": "target", "gmem": "target", "rmem": "amx"}


def test_two_cuda_products_project_the_hardware_each_one_is() -> None:
    """One projection serves both products.

    One projection serves both products, so what separates them is what their
    documents record rather than a branch per architecture.

    Tensor memory is the case that matters: a level with no capacity says the
    architecture has no such store, and a capacity says a CTA's accumulators live
    in one. A projection that reported the same for both would price a Blackwell
    kernel against Hopper's registers.
    """
    hopper = CudaTarget("nvidia.h200_sxm")
    blackwell = CudaTarget("nvidia.b200_sxm")

    assert hopper.get_facts(MemoryHierarchyFacts).explicit("tmem").capacity_bytes is None
    tmem = blackwell.get_facts(MemoryHierarchyFacts).explicit("tmem")
    assert (tmem.capacity_bytes, tmem.scope) == (262_144, "cta")

    throughput = blackwell.get_facts(ThroughputFacts)
    peaks = dict(throughput.peak_flops_per_second)
    assert throughput.memory_bandwidth_bytes_per_second == 7_672_320_000_000
    assert peaks[DType.f4e2m1] == 9_000_000_000_000_000
    assert DType.f4e2m1 not in dict(hopper.get_facts(ThroughputFacts).peak_flops_per_second)
    assert blackwell.get_facts(ParallelCapacityFacts).parallel_units == 148


def test_a_target_without_a_requested_projection_fails_closed() -> None:
    @dataclass(frozen=True)
    class _UnknownFacts:
        units: int

    with pytest.raises(
        UnsupportedCapabilityError, match=r"CudaTarget \(cuda\): no Facts projection"
    ):
        CudaTarget("nvidia.h200_sxm").get_facts(_UnknownFacts)


def test_topology_limits_are_target_facts_and_base_validation_is_inherited() -> None:
    cuda = CudaTarget("nvidia.h200_sxm")
    amx = AmxTarget()

    assert cuda.get_facts(TopologyLimitFacts, "cta").max_static_extent is None
    assert cuda.get_facts(TopologyLimitFacts, "thread").max_static_extent == 1024
    assert amx.get_facts(TopologyLimitFacts, "core").max_static_extent == 8
    assert amx.get_facts(TopologyLimitFacts, "amx").max_static_extent == 1
    assert cuda.topology_limit("cta") == cuda.device.sm_count == 132
    assert cuda.topology_limit("thread") == cuda.architecture.max_threads_per_cta

    @dataclass(frozen=True)
    class _DirectTarget(Target):
        name: ClassVar[str] = "test.direct-topology"
        topology_levels: ClassVar[tuple[str, ...]] = ("unit",)

        def get_facts(self, facts_type: type, query: object | None = None):
            if facts_type is TopologyLimitFacts and query == "unit":
                return TopologyLimitFacts("unit", 4)
            return super().get_facts(facts_type, query)

    _DirectTarget().validate_program_topology(Topology("unit", 4))
    with pytest.raises(ValueError, match="1 <= extent <= 4"):
        _DirectTarget().validate_program_topology(Topology("unit", 5))


def test_cuda_throughput_projection_leaves_scheduler_families_unloaded() -> None:
    source = """
import sys
from tilefoundry.analysis.facts import ThroughputFacts
from tilefoundry.target import CudaTarget

CudaTarget(\"nvidia.h200_sxm\").get_facts(ThroughputFacts)
assert \"tilefoundry.schedule.partition\" not in sys.modules
assert \"tilefoundry.schedule.pipeline\" not in sys.modules
"""
    completed = subprocess.run(
        (sys.executable, "-c", source), text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_projection_results_are_still_immutable_aggregates_of_the_requested_type() -> None:
    @dataclass(frozen=True)
    class _Facts:
        units: int

    @dataclass
    class _MutableFacts:
        units: int

    @dataclass(frozen=True)
    class _CustomTarget(Target):
        name: ClassVar[str] = "test.custom"

        def get_facts(self, facts_type: type, query: object | None = None):
            if facts_type is _Facts:
                return facts_result(self, facts_type, _Facts(4))
            return super().get_facts(facts_type, query)

    assert _CustomTarget().get_facts(_Facts) == _Facts(4)
    with pytest.raises(TargetFactsError, match="must be a frozen dataclass"):
        facts_result(_CustomTarget(), _MutableFacts, _MutableFacts(4))
    with pytest.raises(TargetFactsError, match="returned int"):
        facts_result(_CustomTarget(), _Facts, 4)
