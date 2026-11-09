"""Facts are selected by the exact Target value, not a global registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from tilefoundry import target
from tilefoundry.ir import types


def test_projection_results_are_still_immutable_aggregates_of_the_requested_type() -> None:
    @dataclass(frozen=True)
    class _Facts:
        units: int

    @dataclass
    class _MutableFacts:
        units: int

    @dataclass(frozen=True)
    class _CustomTarget(target.Target):
        name: ClassVar[str] = "test.custom"

        def get_facts(self, facts_type: type, query: object | None = None):
            if facts_type is _Facts:
                return target.facts_result(self, facts_type, _Facts(4))
            return super().get_facts(facts_type, query)

    assert _CustomTarget().get_facts(_Facts) == _Facts(4)
    with pytest.raises(target.FactsError, match="must be a frozen dataclass"):
        target.facts_result(_CustomTarget(), _MutableFacts, _MutableFacts(4))
    with pytest.raises(target.FactsError, match="returned int"):
        target.facts_result(_CustomTarget(), _Facts, 4)


def test_topology_limits_are_target_facts_and_base_validation_is_inherited() -> None:
    @dataclass(frozen=True)
    class _DirectTarget(target.Target):
        name: ClassVar[str] = "test.direct-topology"

        def get_facts(self, facts_type: type, query: object | None = None):
            if facts_type is target.TopologyFacts and query is None:
                return target.TopologyFacts((target.TopologyLevelFacts("unit", 4, 2),))
            if facts_type is target.TopologyLevelFacts and query == "unit":
                return target.TopologyLevelFacts("unit", 4, 2)
            return super().get_facts(facts_type, query)

    _DirectTarget().validate_program_topology(types.Topology("unit", 4))
    with pytest.raises(ValueError, match="1 <= extent <= 4"):
        _DirectTarget().validate_program_topology(types.Topology("unit", 5))
