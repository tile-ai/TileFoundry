"""Core target registration, facts projections, and authored boundaries."""

from __future__ import annotations

import typing

import pytest

from tests.core.installed.smoke_target.vendor_npu import VendorNpuTarget
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


def test_target_registration_and_service_annotations_resolve() -> None:
    assert typing.get_type_hints(register_target)
    assert typing.get_type_hints(PrimFunction)
    for getter in (
        Target.get_analyzer,
        Target.resolve,
        bind,
    ):
        assert typing.get_type_hints(getter)


def test_document_free_target_projects_facts_and_inherits_standard_analysis() -> None:
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


def test_document_free_target_enforces_projection_and_capability_boundaries() -> None:
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


def test_authored_target_boundaries_accept_unregistered_target_instances() -> None:
    class UnregisteredTarget(Target):
        name = "tests.core.target.unregistered"

    target = UnregisteredTarget()
    module_value = Module("unregistered", (), target=target)
    function = PrimFunction("unregistered", (), Sequential(()), target=target)

    @module(target=target)
    class Decorated:
        def forward(self):
            return None

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


def test_registration_is_one_class_boundary_and_reload_is_idempotent() -> None:
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
