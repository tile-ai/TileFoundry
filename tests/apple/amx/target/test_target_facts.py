"""Facts are selected by the exact Target value, not a global registry."""

from __future__ import annotations

from tilefoundry.analysis.facts import (
    MemoryHierarchyFacts,
    PerformanceServiceFacts,
    ThroughputFacts,
)
from tilefoundry.target import TopologyFacts, TopologyLevelFacts
from tilefoundry_apple import AmxTarget


def test_builtin_targets_own_their_facts_projections() -> None:
    assert AmxTarget().get_facts(ThroughputFacts).bandwidth_level == "gmem"
    assert {
        level.name: level.owner
        for level in AmxTarget().get_facts(MemoryHierarchyFacts).explicit_levels
    } == {"host": "target", "gmem": "target", "rmem": "thread"}


def test_topology_limits_are_target_facts_and_base_validation_is_inherited() -> None:
    amx = AmxTarget()

    amx_levels = amx.get_facts(TopologyFacts)
    assert amx_levels.topologies == (TopologyLevelFacts("thread", 12, 12),)
    assert amx_levels.parallel() == amx_levels.level("thread")
    services = amx.get_facts(PerformanceServiceFacts, "thread")
    assert services.unit == "thread"
    assert services.unit_flops == services.unit_ops == services.unit_bandwidth == ()
