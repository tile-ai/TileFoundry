"""Hardware facts and the actual scalar coordinates of pure matrix layouts."""
import pytest
from importlib.resources import files
from tilefoundry.target.hardware.envelope import UnknownDocumentError
from tilefoundry.analysis.facts import MemoryHierarchyFacts, ThroughputFacts
from tilefoundry.ir.types import Topology
from tilefoundry.target import TopologyFacts
from tilefoundry_apple import MetalTarget


def test_target_resources(tmp_path):
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


    with pytest.raises(TypeError):
        MetalTarget()
    with pytest.raises(UnknownDocumentError):
        MetalTarget("mlx.nax.profile")
    with pytest.raises(ValueError):
        MetalTarget(target.device)
    values = MetalTarget(target.device, target.architecture)
    assert values == target
    custom = tmp_path / "custom.toml"
    text = files("tilefoundry_apple.metal.target.hardware").joinpath("apple_m2_pro_gpu.toml").read_text()
    custom.write_text(text.replace('id = "apple.m2_pro.gpu"', 'id = "vendor.custom_gpu"'))
    loaded = MetalTarget(custom)
    assert loaded.device_id == "vendor.custom_gpu" and loaded.device_digest
    assert len(MetalTarget.available()) == 1
    for value in (values, loaded):
        expr = value.to_python()
        namespace = {}
        exec("\n".join(expr.imports), namespace)
        assert eval(expr.text, namespace) == value
