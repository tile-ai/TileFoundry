"""Hardware facts and the actual scalar coordinates of pure matrix layouts."""
import pytest
from tilefoundry.analysis.facts import MemoryHierarchyFacts, ThroughputFacts
from tilefoundry.ir.types import Layout, ShardLayout, Topology
from tilefoundry.target import TopologyFacts
from tilefoundry_apple import MetalTarget


def test_target_resources():
    target = MetalTarget("apple.m2_pro.gpu")
    assert target.identity == "apple.m2_pro.gpu"
    topology = target.get_facts(TopologyFacts)
    assert [level.name for level in topology.topologies] == ["threadgroup", "thread"]
    assert topology.parallel().max_physical_units is None
    assert topology.level("thread").max_logical_units == 1024
    target.validate_program_topology(Topology("threadgroup", 4))
    target.validate_program_topology(Topology("thread", 128))
    with pytest.raises(ValueError): target.validate_program_topology(Topology("thread", 1025))
    with pytest.raises(ValueError): target.validate_program_topology(Topology("simdgroup", 2))
    memory = target.get_facts(MemoryHierarchyFacts)
    assert memory.explicit("rmem").owner == "thread"
    assert memory.explicit("rmem").capacity_bytes is None
    assert memory.explicit("smem").owner == "threadgroup"
    assert memory.explicit("smem").capacity_bytes == 32768
    assert target.get_facts(ThroughputFacts).peak_flops_per_second == ()
    assert target in MetalTarget.available()
    assert tuple(MetalTarget.available()) == (target,)
    rendered = target.to_python()
    namespace = {}
    exec("\n".join(rendered.imports), namespace)
    assert eval(rendered.text, namespace) == target

