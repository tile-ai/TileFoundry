"""Facts are selected by the exact Target value, not a global registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from tilefoundry.analysis.facts import MemoryHierarchyFacts, ThroughputFacts
from tilefoundry.target import AmxTarget, CudaTarget, Target
from tilefoundry.target.facts import TargetFactsError, facts_result


def test_builtin_targets_own_their_facts_projections() -> None:
    cuda = CudaTarget("nvidia.h200_sxm")

    throughput = cuda.get_facts(ThroughputFacts)
    memory = cuda.get_facts(MemoryHierarchyFacts)

    assert throughput.memory_bandwidth_bytes_per_second == 4_800_000_000_000
    assert memory.explicit("gmem").capacity_bytes == cuda.device.hbm_capacity_bytes
    assert AmxTarget().get_facts(ThroughputFacts).bandwidth_level == "gmem"


def test_a_target_without_a_requested_projection_fails_closed() -> None:
    @dataclass(frozen=True)
    class _UnknownFacts:
        units: int

    with pytest.raises(ValueError, match=r"CudaTarget \(cuda\): no Facts projection"):
        CudaTarget("nvidia.h200_sxm").get_facts(_UnknownFacts)


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
