"""Tests and helpers for the owning target."""

from __future__ import annotations

from tilefoundry.analysis.facts import MemoryHierarchyFacts, ThroughputFacts
from tilefoundry.target import TopologyFacts, TopologyLevelFacts
from tilefoundry_apple import AmxTarget


def test_builtin_targets_own_their_facts_projections() -> None:
    assert AmxTarget().get_facts(ThroughputFacts).bandwidth_level == "gmem"
    assert {
        level.name: level.owner
        for level in AmxTarget().get_facts(MemoryHierarchyFacts).explicit_levels
    } == {"host": "target", "gmem": "target", "rmem": "amx"}


def test_topology_limits_are_target_facts_and_base_validation_is_inherited() -> None:
    amx = AmxTarget()

    amx_levels = amx.get_facts(TopologyFacts)
    assert amx_levels.level("core") == TopologyLevelFacts("core", 8, 8)
    assert amx_levels.level("amx") == TopologyLevelFacts("amx", 1, None)
    assert amx_levels.parallel() == amx_levels.level("core")
