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

from tilefoundry.codegen.registry import group_functions_by_target
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.ir.types.shard import Topology
from tilefoundry.registry import UnknownAlgorithmError
from tilefoundry.schedule.registry import SCHEDULES
from tilefoundry.target import (
    AmxTarget,
    CpuTarget,
    CudaTarget,
    Target,
)
from tilefoundry.target.cuda.spec import installed_architecture as _installed_sm90


def test_a_scheduler_is_reached_by_target_type_not_by_target_value() -> None:
    """Which levels a target can be scheduled at is a property of its type.

    A target carrying different hardware facts is still the same kind of machine,
    so it resolves the same scheduler; a target with no registration resolves
    none, and says which levels it does serve rather than only that it failed.
    Nothing about scheduling is stored on the target value, which is why two
    equal targets stay equal and interchangeable -- codegen groups functions by
    comparing targets, so a registration reachable by value would split one
    machine into two.
    """
    assert SCHEDULES.selectors_for(CudaTarget) == ("cta", "thread")
    assert SCHEDULES.selectors_for(AmxTarget) == ("core",)
    assert SCHEDULES.selectors_for(CpuTarget) == ()

    custom = CudaTarget(
        "nvidia.h200_sxm",
        architecture=replace(_installed_sm90(), name="sm_90_custom"),
    )
    assert SCHEDULES.resolve(custom, "cta") is SCHEDULES.resolve(CudaTarget("nvidia.h200_sxm"), "cta")
    assert SCHEDULES.resolve(CudaTarget("nvidia.h200_sxm"), "cta") is not SCHEDULES.resolve(
        CudaTarget("nvidia.h200_sxm"), "thread"
    )

    with pytest.raises(UnknownAlgorithmError, match="no 'cta' registered for Target"):
        SCHEDULES.resolve(Target("test"), "cta")
    with pytest.raises(UnknownAlgorithmError, match=r"available: \['core'\]"):
        SCHEDULES.resolve(AmxTarget(), "amx")

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
    with pytest.raises(ValueError, match="differing Target facts"):
        group_functions_by_target(
            Module(name="mixed", functions=(first, second), entry="first")
        )

    host = PrimFunction(name="host", params=(), body=body, target=CpuTarget())
    groups = group_functions_by_target(
        Module(name="mixed", functions=(first, host), entry="host")
    )
    assert tuple(fn.name for fn in groups["cuda"]) == ("first",)
    assert tuple(fn.name for fn in groups["cpu"]) == ("host",)
