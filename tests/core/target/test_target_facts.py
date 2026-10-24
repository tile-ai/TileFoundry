"""Target-neutral tests and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from tilefoundry.ir.types import Topology
from tilefoundry.target import Target, TopologyFacts, TopologyLevelFacts
from tilefoundry.target.facts import TargetFactsError, facts_result


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


def test_topology_limits_are_target_facts_and_base_validation_is_inherited() -> None:
    @dataclass(frozen=True)
    class _DirectTarget(Target):
        name: ClassVar[str] = "test.direct-topology"

        def get_facts(self, facts_type: type, query: object | None = None):
            if facts_type is TopologyFacts and query is None:
                return TopologyFacts((TopologyLevelFacts("unit", 4, 2),))
            if facts_type is TopologyLevelFacts and query == "unit":
                return TopologyLevelFacts("unit", 4, 2)
            return super().get_facts(facts_type, query)

    _DirectTarget().validate_program_topology(Topology("unit", 4))
    with pytest.raises(ValueError, match="1 <= extent <= 4"):
        _DirectTarget().validate_program_topology(Topology("unit", 5))
