"""Pin the Metal target's construction, topology and architecture facts.

These stop the hardware documents and the ``Device`` / ``Architecture`` fields
from drifting apart, and keep every recorded source readable upstream.
"""

from __future__ import annotations

import pytest

from tilefoundry import target
from tilefoundry.target import hardware
from tilefoundry_apple import metal


def test_target_constructs_and_reads_back():
    """A Target renders as Python that evaluates back to itself; unknown ids are refused."""
    tgt = metal.Target("apple.m2_pro.gpu")
    assert metal.Target() == tgt
    assert (tgt.architecture_id, tgt.device_id) == ("apple.gpu8", "apple.m2_pro.gpu")
    rendered = tgt.to_python()
    assert rendered.text == 'metal.Target("apple.m2_pro.gpu")'
    namespace: dict[str, object] = {}
    for line in rendered.imports:
        exec(line, namespace)
    assert eval(rendered.text, namespace) == tgt

    with pytest.raises(hardware.UnknownDocumentError, match="apple.m2_pro.gpu"):
        metal.Target("apple.m9.gpu")


def test_topology_facts():
    """Threadgroups are unbounded, one holds at most 1024 threads, and each memory has its owner."""
    tgt = metal.Target()
    facts = tgt.get_facts(target.TopologyFacts)
    assert facts.topologies == (
        target.TopologyLimits("threadgroup", None, None),
        target.TopologyLimits("thread", 1024, None),
    )
    assert facts.parallel_topology == "threadgroup"
    assert tgt.get_facts(target.TopologyLimits, "thread") == target.TopologyLimits("thread", 1024, None)
    hierarchy = tgt.get_facts(target.MemoryHierarchyFacts)
    assert {level.name: level.owner for level in hierarchy.explicit_levels} == {
        "gmem": "target",
        "smem": "threadgroup",
        "rmem": "thread",
    }
    assert tgt.storage_owners.order == ("threadgroup", "thread")


def test_architecture_facts():
    """The architecture lists 27 MSL native functions at SIMD width 32, from public sources."""
    architecture = metal.Target().architecture
    assert architecture.simd_width == 32
    assert len(architecture.capabilities) == len(set(architecture.capabilities)) == 27
    assert all(name.startswith(("simd_", "simdgroup_")) for name in architecture.capabilities)

    for document in metal.Target.hardware.documents().values():
        for fact in document.facts.values():
            for private in ("/Users", "ANE-Research", "artifacts/"):
                assert private not in (fact.source or "")
