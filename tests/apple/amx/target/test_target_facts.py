"""Facts are selected by the exact Target value, not a global registry."""

from __future__ import annotations

from tilefoundry import target
from tilefoundry_apple import amx


def test_builtin_targets_own_their_facts_projections() -> None:
    assert amx.Target().get_facts(target.ThroughputFacts).bandwidth_level == "gmem"
    assert {
        level.name: level.owner
        for level in amx.Target().get_facts(target.MemoryHierarchyFacts).explicit_levels
    } == {"host": "target", "gmem": "target", "rmem": "thread"}


def test_topology_limits_are_target_facts_and_base_validation_is_inherited() -> None:
    amx_value = amx.Target()

    amx_levels = amx_value.get_facts(target.TopologyFacts)
    assert amx_levels.topologies == (target.TopologyLevelFacts("thread", 12, 12),)
    assert amx_levels.parallel() == amx_levels.level("thread")
    services = amx_value.get_facts(target.PerformanceServiceFacts, "thread")
    assert services.unit == "thread"
    assert services.unit_flops == services.unit_ops == services.unit_bandwidth == ()
