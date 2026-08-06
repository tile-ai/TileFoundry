"""What a Target validates about a program, which scheduler it reaches, and how
codegen groups by one.

The composed hardware facts themselves -- which documents a default target
resolves, and which limits belong to the architecture rather than the device --
are asserted where those facts are installed. What is left here is what a Target
is asked during a compilation.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.fixtures.demo_ir import build_demo
from tilefoundry import lower
from tilefoundry.codegen.registry import group_functions_by_target
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import (
    AmxTarget,
    CpuTarget,
    CudaTarget,
    Target,
    register_target,
    registered_targets,
    require_target,
)
from tilefoundry.target.cuda.spec import installed_architecture as _installed_sm90


@register_target
class ExternalCudaTarget(CudaTarget):
    name = "tests.target.external_cuda"


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
    registered_name = "tests.target.reload"
    first = _provider_target("tests.providers.reload", "ReloadTarget", registered_name)
    second = _provider_target("tests.providers.reload", "ReloadTarget", registered_name)

    assert register_target(first) is first
    assert register_target(second) is second
    assert registered_targets()[registered_name] is first
    assert require_target(second()) is not None
    assert {"cpu", "cuda", "amx"} <= registered_targets().keys()
    with pytest.raises(TypeError):
        registered_targets()["tests.target.mutation"] = Target


def test_registration_rejects_ambiguous_or_invalid_provider_classes() -> None:
    inherited = type(
        "InheritedTarget",
        (CudaTarget,),
        {"__module__": "tests.providers.inherited"},
    )
    empty = _provider_target("tests.providers.empty", "EmptyTarget", "")

    with pytest.raises(ValueError, match="declare a class-level"):
        register_target(inherited)
    with pytest.raises(ValueError, match="non-empty"):
        register_target(empty)
    with pytest.raises(TypeError, match="Target subclass"):
        register_target(object)
    with pytest.raises(TypeError, match="Target subclass"):
        register_target("tests.target.argument")

    owner = _provider_target(
        "tests.providers.owner", "OwnedTarget", "tests.target.conflict"
    )
    claimant = _provider_target(
        "tests.providers.claimant", "ClaimantTarget", "tests.target.conflict"
    )
    register_target(owner)
    with pytest.raises(ValueError, match="already owned.*cannot register"):
        register_target(claimant)


def test_authored_target_boundaries_reject_strings_and_keep_exact_instances() -> None:
    target = CudaTarget("nvidia.h200_sxm")
    module = Module("exact", (), target=target)

    assert module.target is target
    assert module.resolve_target() is target
    with pytest.raises(TypeError, match="Target instance, not a string"):
        Module("invalid", (), target="cuda")
    with pytest.raises(TypeError, match="Target instance, not a string"):
        PrimFunction("invalid", (), Sequential(()), target="cuda")


def test_lowering_and_codegen_keep_the_external_target_instance() -> None:
    function, _, _ = build_demo()
    target = ExternalCudaTarget("nvidia.h200_sxm")
    module = Module("external", (function,), function.name, target=target)

    lowered = lower(module)

    assert lowered.target is target
    assert all(fn.target is target for fn in lowered.functions)
    assert target.get_code_generator() is CudaTarget(
        "nvidia.h200_sxm"
    ).get_code_generator()


def test_a_scheduler_is_reached_by_target_inheritance() -> None:
    """CUDA subclasses inherit their base class's Target-owned schedulers."""
    custom = CudaTarget(
        "nvidia.h200_sxm",
        architecture=replace(_installed_sm90(), name="sm_90_custom"),
    )
    assert custom.get_scheduler("cta").solve is CudaTarget("nvidia.h200_sxm").get_scheduler("cta").solve
    assert CudaTarget("nvidia.h200_sxm").get_scheduler("cta").solve is not CudaTarget("nvidia.h200_sxm").get_scheduler("thread").solve

    with pytest.raises(ValueError, match="Target .*no scheduler for 'cta'"):
        Target().get_scheduler("cta")
    with pytest.raises(ValueError, match="AmxTarget .*no scheduler for 'amx'"):
        AmxTarget().get_scheduler("amx")

    assert CudaTarget("nvidia.h200_sxm") == CudaTarget("nvidia.h200_sxm")
    assert hash(CudaTarget("nvidia.h200_sxm")) == hash(CudaTarget("nvidia.h200_sxm"))
    assert AmxTarget() == AmxTarget()


def test_static_topologies_use_target_resource_facts() -> None:
    """A declared extent is validated against the level's own resource fact: a
    grid may be far wider than the machine's SMs and a block may not exceed the
    threads one supports, and a grid whose size is only known at launch is
    accepted as declared."""
    target = CudaTarget("nvidia.h200_sxm")
    target.validate_program_topology(Topology("cta", 132))
    target.validate_program_topology(Topology("cta", 310_000))
    target.validate_program_topology(Topology("thread", 1024))
    target.validate_program_topology(Topology("cta", None))
    with pytest.raises(ValueError, match="must be positive"):
        target.validate_program_topology(Topology("cta", 0))
    with pytest.raises(ValueError, match="1 <= extent <= 1024"):
        target.validate_program_topology(Topology("thread", 1025))


def test_group_functions_by_target_fact_matching() -> None:
    """CUDA functions must agree on Target facts before grouping; CPU
    functions are exempt from the CUDA fact-matching."""
    body = Sequential(body=())
    first = PrimFunction(name="first", params=(), body=body, target=CudaTarget("nvidia.h200_sxm"))
    second = PrimFunction(
        name="second",
        params=(),
        body=body,
        target=CudaTarget(
            "nvidia.h200_sxm",
            architecture=replace(_installed_sm90(), name="sm_90_alt"),
        ),
    )
    with pytest.raises(ValueError, match="mixes unequal device Targets"):
        group_functions_by_target(
            Module(name="mixed", functions=(first, second), entry="first")
        )

    host = PrimFunction(name="host", params=(), body=body, target=CpuTarget())
    groups = group_functions_by_target(
        Module(name="mixed", functions=(first, host), entry="host")
    )
    assert tuple(fn.name for fn in groups[first.target]) == ("first",)
    assert tuple(fn.name for fn in groups[host.target]) == ("host",)
