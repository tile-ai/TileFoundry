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
    CpuTarget,
    CudaTarget,
    Target,
)
from tilefoundry.target.cuda.spec import installed_architecture as _installed_sm90


def test_cuda_target_composes_fixed_architecture_and_device_facts() -> None:
    target = CudaTarget()

    assert target.name == "cuda"
    assert target.arch == "sm_90"
    assert target.topology_levels == ("cta", "thread")
    # A default target selects the installed documents by their stable IDs and
    # keeps both the ID and the content digest it resolved.
    assert (target.architecture_id, target.device_id) == (
        "nvidia.sm90",
        "nvidia.h200_sxm",
    )
    assert target.architecture_digest and target.device_digest
    assert target == CudaTarget(
        architecture="nvidia.sm90", device="nvidia.h200_sxm"
    )


def test_a_scheduler_is_reached_by_target_type_not_by_target_value() -> None:
    """Which levels a target can be scheduled at is a property of its type.

    A target carrying different hardware facts is still the same kind of
    machine, so it resolves the same scheduler; a target with no registration
    resolves none. Nothing about scheduling is stored on the target value, which
    is why two equal targets stay equal and interchangeable.
    """
    assert SCHEDULES.selectors_for(CudaTarget) == ("cta",)
    assert SCHEDULES.selectors_for(CpuTarget) == ()

    custom = CudaTarget(architecture=replace(_installed_sm90(), name="sm_90_custom"))
    assert SCHEDULES.resolve(custom, "cta") is SCHEDULES.resolve(CudaTarget(), "cta")

    with pytest.raises(UnknownAlgorithmError, match="no 'cta' registered for Target"):
        SCHEDULES.resolve(Target("test"), "cta")

    assert CudaTarget() == CudaTarget()
    assert hash(CudaTarget()) == hash(CudaTarget())


def test_static_topologies_use_target_resource_facts() -> None:
    target = CudaTarget()
    target.validate_program_topology(Topology("cta", 132))
    target.validate_program_topology(Topology("cta", 310_000))
    target.validate_program_topology(Topology("thread", 1024))
    target.validate_program_topology(Topology("cta", None))
    with pytest.raises(ValueError, match="must be positive"):
        target.validate_program_topology(Topology("cta", 0))
    with pytest.raises(ValueError, match="1 <= extent <= 1024"):
        target.validate_program_topology(Topology("thread", 1025))


def test_per_sm_limits_belong_to_the_architecture_not_the_device() -> None:
    """A per-SM limit is a property of the microarchitecture, so every product
    built on it shares the limit; the device only says how many SMs there are
    and how fast its memory system runs."""
    target = CudaTarget()
    architecture, device = target.architecture, target.device

    assert architecture.max_resident_ctas_per_sm == 32
    assert architecture.shared_memory_per_sm_bytes == 228 * 1024
    assert architecture.shared_memory_per_cta_bytes == 227 * 1024
    assert architecture.registers_per_sm_32bit == 65_536
    for moved in (
        "max_resident_ctas_per_sm",
        "shared_memory_per_sm_bytes",
        "shared_memory_per_cta_bytes",
        "registers_per_sm_32bit",
    ):
        assert not hasattr(device, moved)

    assert device.sm_count == 132
    assert device.hbm_capacity_bytes == 141_000_000_000
    assert device.hbm_bandwidth_bytes_per_second == 4_800_000_000_000


def test_group_functions_by_target_fact_matching() -> None:
    """CUDA functions must agree on Target facts before grouping; CPU
    functions are exempt from the CUDA fact-matching."""
    body = Sequential(body=())
    first = PrimFunction(name="first", params=(), body=body, target=CudaTarget())
    second = PrimFunction(
        name="second",
        params=(),
        body=body,
        target=CudaTarget(architecture=replace(_installed_sm90(), name="sm_90_alt")),
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
