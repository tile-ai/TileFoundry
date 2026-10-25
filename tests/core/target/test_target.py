"""Core target registration, facts projections, and authored boundaries."""

from __future__ import annotations

import typing
from pathlib import Path

import pytest

from tests.core.fixtures.target_plugins.vendor_npu import VendorNpuTarget
from tilefoundry.cli.source import load_namespace
from tilefoundry.codegen.linkable_module import LinkableModule
from tilefoundry.dsl import *
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.target import (
    MemoryHierarchyFacts,
    Target,
    TargetFactsError,
    ThroughputFacts,
    TopologyFacts,
    TopologyLevelFacts,
    UnsupportedCapabilityError,
    facts_result,
    register_target,
    registered_targets,
)
from tilefoundry.target.base import bind

UNREGISTERED = Path(__file__).parents[1] / "fixtures" / "target" / "unregistered_target.py"


def test_target_service_annotations_resolve() -> None:
    assert typing.get_type_hints(register_target)
    assert typing.get_type_hints(PrimFunction)
    for getter in (
        Target.get_analyzer,
        Target.resolve,
        bind,
    ):
        assert typing.get_type_hints(getter)


def test_documentless_target_projects_facts() -> None:
    target = VendorNpuTarget()

    assert target.get_analyzer("roofline").selector == "roofline"
    assert target.get_facts(TopologyLevelFacts, "core") == TopologyLevelFacts("core", 256, 16)
    memory = target.get_facts(MemoryHierarchyFacts)
    assert memory.explicit("gmem").capacity_bytes == 64_000_000_000
    assert target.get_facts(ThroughputFacts).peak_for(DType.f32) == 2_000_000_000_000_000
    assert target.get_facts(TopologyFacts).parallel() == TopologyLevelFacts("core", 256, 16)
    target.validate_program_topology(Topology("core", 256))
    with pytest.raises(ValueError, match="1 <= extent <= 256"):
        target.validate_program_topology(Topology("core", 257))


def test_documentless_target_enforces_boundaries() -> None:
    class BrokenFactsTarget(Target):
        name = "tests.core.target.broken_facts"

        def get_facts(self, facts_type: type, query: object | None = None):
            return facts_result(self, facts_type, object())

    target = VendorNpuTarget()

    with pytest.raises(
        TargetFactsError,
        match="BrokenFactsTarget: Facts projection for ThroughputFacts returned object",
    ):
        BrokenFactsTarget().get_facts(ThroughputFacts)
    with pytest.raises(UnsupportedCapabilityError) as error:
        target.resolve(LinkableModule, None, (), None)
    assert str(error.value) == "VendorNpuTarget (vendor.npu): no code generator"


def test_unregistered_target_instances_accepted() -> None:
    program = load_namespace(str(UNREGISTERED))[0]
    target = program["target"]
    module_value = Module("unregistered", (), target=target)
    function = PrimFunction("unregistered", (), Sequential(()), target=target)
    Decorated = program["Decorated"]

    assert module_value.target is target
    assert function.target is target
    assert Decorated.target is target


def _provider_target(module_name: str, provider_name: str, registered_name: str):
    return type(
        provider_name,
        (Target,),
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

    assert register_target(first) is first
    assert register_target(second) is second
    assert registered_targets()[registered_name] is first
    assert isinstance(second(), Target)
    assert {"cpu"} <= registered_targets().keys()
    with pytest.raises(TypeError):
        registered_targets()["tests.core.target.mutation"] = Target
