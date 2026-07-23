from __future__ import annotations

import pytest

from tilefoundry.codegen.registry import group_functions_by_target
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.ir.types.shard import Topology
from tilefoundry.schedule import Schedule
from tilefoundry.target import (
    H200SXM,
    SM90,
    CpuTarget,
    CudaTarget,
    Target,
)


def test_cuda_target_composes_fixed_architecture_and_device_facts() -> None:
    target = CudaTarget()

    assert target.name == "cuda"
    assert target.arch == "sm_90"
    assert target.architecture == SM90()
    assert target.topology_levels == ("cta", "thread")
    assert target.device == H200SXM()


def test_service_lookup_contract() -> None:
    """Service lookup needs an exact non-empty identity key; services are
    private state excluded from Target equality."""
    target = Target("test")
    with pytest.raises(ValueError, match="non-empty string"):
        target.service(Schedule, "")
    with pytest.raises(ValueError, match="exactly one service"):
        target.service(Schedule, "cta")

    cuda = CudaTarget()
    assert cuda.service(Schedule, "cta").stage == "cta"
    custom = CudaTarget(architecture=SM90(name="sm_90_custom"))
    assert custom.service(Schedule, "cta").stage == "cta"

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


def test_h200_grid_and_parallelism_facts_are_separate() -> None:
    device = H200SXM()

    assert device.sm_count == 132
    assert device.max_resident_ctas_per_sm == 32
    assert device.compiler_policy_max_parallel_ctas == 132
    assert device.shared_memory_per_sm_bytes == 228 * 1024
    assert device.shared_memory_per_cta_bytes == 227 * 1024
    assert device.registers_per_sm_32bit == 65_536


def test_group_functions_by_target_fact_matching() -> None:
    """CUDA functions must agree on Target facts before grouping; CPU
    functions are exempt from the CUDA fact-matching."""
    body = Sequential(body=())
    first = PrimFunction(name="first", params=(), body=body, target=CudaTarget())
    second = PrimFunction(
        name="second",
        params=(),
        body=body,
        target=CudaTarget(architecture=SM90(name="sm_90_alt")),
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
