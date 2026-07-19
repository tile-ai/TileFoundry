from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tilefoundry.codegen.registry import group_functions_by_target
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard import Topology
from tilefoundry.schedule import Schedule
from tilefoundry.target import H200SXM, SM90, CpuTarget, CudaTarget, Target


def test_cuda_target_composes_fixed_architecture_and_device_facts() -> None:
    target = CudaTarget()
    device = target.device

    assert target.name == "cuda"
    assert target.arch == "sm_90"
    assert target.architecture == SM90()
    assert target.topology_levels == ("cta", "thread")
    assert device == H200SXM()
    assert device.sm_count == 132
    assert device.hbm_capacity_bytes == 141_000_000_000
    assert device.hbm_bandwidth_bytes_per_second == 4_800_000_000_000
    assert device.dense_flops_per_second == {
        DType.f32: 67_000_000_000_000,
        DType.f16: 989_500_000_000_000,
        DType.bf16: 989_500_000_000_000,
        DType.fp8e4m3: 1_979_000_000_000_000,
    }
    assert DType.f4e2m1 not in device.dense_flops_per_second
    assert DType.f8e8m0 not in device.dense_flops_per_second
    assert DType.f4e2m1 not in target.architecture.supported_compute_dtypes
    assert DType.f8e8m0 not in target.architecture.supported_compute_dtypes


def test_target_equality_excludes_private_services() -> None:
    left = CudaTarget()
    right = CudaTarget()

    assert left == right
    assert hash(left) == hash(right)
    assert "service" not in repr(left)
    with pytest.raises(ValueError, match="exactly one service"):
        left.service(Schedule, "cta")
    with pytest.raises(FrozenInstanceError):
        left.device = H200SXM()


def test_service_lookup_requires_exact_nonempty_identity_key() -> None:
    target = Target("test")
    with pytest.raises(ValueError, match="non-empty string"):
        target.service(Schedule, "")
    with pytest.raises(ValueError, match="exactly one service"):
        target.service(Schedule, "cta")


def test_function_target_is_optional_until_compile_boundary() -> None:
    fn = Function.build(name="f", params=(), body=None, return_type=object())
    assert fn.target is None


def test_static_topologies_use_target_resource_facts() -> None:
    target = CudaTarget()
    target.validate_program_topology(Topology("cta", 132))
    target.validate_program_topology(Topology("thread", 1024))
    target.validate_program_topology(Topology("cta", None))
    with pytest.raises(ValueError, match="1 <= extent <= 132"):
        target.validate_program_topology(Topology("cta", 133))
    with pytest.raises(ValueError, match="1 <= extent <= 1024"):
        target.validate_program_topology(Topology("thread", 1025))


def test_cuda_functions_with_different_facts_fail_before_grouping() -> None:
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


def test_cpu_functions_are_exempt_from_cuda_fact_matching() -> None:
    body = Sequential(body=())
    device = PrimFunction(name="device", params=(), body=body, target=CudaTarget())
    host = PrimFunction(name="host", params=(), body=body, target=CpuTarget())
    groups = group_functions_by_target(
        Module(name="mixed", functions=(device, host), entry="host")
    )
    assert tuple(fn.name for fn in groups["cuda"]) == ("device",)
    assert tuple(fn.name for fn in groups["cpu"]) == ("host",)
