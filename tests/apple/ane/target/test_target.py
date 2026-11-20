"""Pin the ANE target's construction, topology and architecture facts.

These stop the hardware documents and the ``Device`` / ``Architecture`` fields
from drifting apart.
"""

from __future__ import annotations

import pytest

from tilefoundry import target
from tilefoundry.ir import types
from tilefoundry.target import hardware
from tilefoundry_apple import ane


def test_target_constructs_and_reads_back():
    """A Target renders as Python that evaluates back to itself; unknown ids are refused."""
    tgt = ane.Target("apple.m2_pro.ane")
    assert ane.Target() == tgt
    assert (tgt.architecture_id, tgt.device_id) == ("apple.h14", "apple.m2_pro.ane")
    rendered = tgt.to_python()
    assert rendered.text == 'ane.Target("apple.m2_pro.ane")'
    namespace: dict[str, object] = {}
    for line in rendered.imports:
        exec(line, namespace)
    assert eval(rendered.text, namespace) == tgt

    with pytest.raises(hardware.UnknownDocumentError, match="apple.m2_pro.ane"):
        ane.Target("apple.m9.ane")


def test_topology_facts():
    """One engine of size 1 runs a program; its L2 is owned by the engine and has no capacity."""
    tgt = ane.Target()
    facts = tgt.get_facts(target.TopologyFacts)
    assert facts.topologies == (target.TopologyLimits("engine", 1, 1),)
    assert facts.parallel_topology == "engine"
    assert tgt.get_facts(target.TopologyLimits, "engine") == target.TopologyLimits("engine", 1, 1)
    hierarchy = tgt.get_facts(target.MemoryHierarchyFacts)
    assert {level.name: level.owner for level in hierarchy.explicit_levels} == {
        "gmem": target.SHARED_OWNER,
        "smem": "engine",
    }
    assert {level.name: level.capacity_bytes for level in hierarchy.explicit_levels} == {
        "gmem": 34359738368,
        "smem": None,
    }
    assert tgt.storage_owners.order == ("engine",)

    tgt.validate_program_topology(types.Topology("engine", 1))
    with pytest.raises(ValueError, match="extent 2 must satisfy 1 <= extent <= 1"):
        tgt.validate_program_topology(types.Topology("engine", 2))
    with pytest.raises(ValueError, match="unsupported topology 'thread'"):
        tgt.validate_program_topology(types.Topology("thread", 1))

    with pytest.raises(target.UnsupportedCapabilityError, match="ane"):
        tgt.get_facts(target.ThroughputFacts)


def test_architecture_facts():
    """The architecture lists six instruction forms and its extent limits."""
    architecture = ane.Target().architecture
    assert len(architecture.capabilities) == len(set(architecture.capabilities)) == 6
    assert all(name.startswith(("ne.", "pe.")) for name in architecture.capabilities)
    assert architecture.max_channels == 65536
    assert architecture.max_kernel == 32
