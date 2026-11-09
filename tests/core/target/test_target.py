"""Core target registration, facts projections, and authored boundaries."""

from __future__ import annotations

import typing
from pathlib import Path

import pytest

from tests.core.fixtures.target_plugins.vendor_npu import VendorNpuTarget
from tilefoundry import cli, codegen, target
from tilefoundry.dsl import *
from tilefoundry.ir import core

UNREGISTERED = Path(__file__).parents[1] / "fixtures" / "target" / "unregistered_target.py"


def test_target_service_annotations_resolve() -> None:
    assert typing.get_type_hints(target.register)
    assert typing.get_type_hints(core.PrimFunction)
    for getter in (
        target.Target.get_analyzer,
        target.Target.resolve,
        target.bind,
    ):
        assert typing.get_type_hints(getter)


def test_documentless_target_projects_facts() -> None:
    tgt = VendorNpuTarget()

    assert tgt.get_analyzer("roofline").selector == "roofline"
    assert tgt.get_facts(target.TopologyLevelFacts, "core") == target.TopologyLevelFacts("core", 256, 16)
    memory = tgt.get_facts(target.MemoryHierarchyFacts)
    assert memory.explicit("gmem").capacity_bytes == 64_000_000_000
    assert tgt.get_facts(target.ThroughputFacts).peak_for(DType.f32) == 2_000_000_000_000_000
    assert tgt.get_facts(target.TopologyFacts).parallel() == target.TopologyLevelFacts("core", 256, 16)
    tgt.validate_program_topology(Topology("core", 256))
    with pytest.raises(ValueError, match="1 <= extent <= 256"):
        tgt.validate_program_topology(Topology("core", 257))


def test_documentless_target_enforces_boundaries() -> None:
    class BrokenFactsTarget(target.Target):
        name = "tests.core.target.broken_facts"

        def get_facts(self, facts_type: type, query: object | None = None):
            return target.facts_result(self, facts_type, object())

    tgt = VendorNpuTarget()

    with pytest.raises(
        target.FactsError,
        match="BrokenFactsTarget: Facts projection for ThroughputFacts returned object",
    ):
        BrokenFactsTarget().get_facts(target.ThroughputFacts)
    with pytest.raises(target.UnsupportedCapabilityError) as error:
        tgt.resolve(codegen.LinkableModule, None, (), None)
    assert str(error.value) == "VendorNpuTarget (vendor.npu): no code generator"


def test_unregistered_target_instances_accepted() -> None:
    program = cli.load_namespace(str(UNREGISTERED))[0]
    tgt = program["target"]
    module_value = core.Module("unregistered", (), target=tgt)
    function = core.PrimFunction("unregistered", (), core.Sequential(()), target=tgt)
    Decorated = program["Decorated"]

    assert module_value.target is tgt
    assert function.target is tgt
    assert Decorated.target is tgt


def _provider_target(module_name: str, provider_name: str, registered_name: str):
    return type(
        provider_name,
        (target.Target,),
        {
            "__module__": module_name,
            "__qualname__": provider_name,
            "name": registered_name,
        },
    )


def test_target_reload_registration_idempotent() -> None:
    registered_name = "tests.core.target.reload"
    first = _provider_target("tests.providers.reload", "ReloadTarget", registered_name)
    second = _provider_target("tests.providers.reload", "ReloadTarget", registered_name)

    assert target.register(first) is first
    assert target.register(second) is second
    assert target.registered()[registered_name] is first
    assert isinstance(second(), target.Target)
    assert {"cpu"} <= target.registered().keys()
    with pytest.raises(TypeError):
        target.registered()["tests.core.target.mutation"] = target.Target
