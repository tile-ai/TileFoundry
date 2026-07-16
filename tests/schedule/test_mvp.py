from __future__ import annotations

import math

from tests.models.deepseek_v4_flash import (
    DIM,
    MOE_INTER,
    N_ACT,
    N_ROUTED,
    dsv4_moe_module,
)
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Expr, Tuple
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.target import CudaTarget
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.parser import parse_module_source
from tilefoundry.schedule import auto_dist, logical_fingerprint


def _walk(expr: Expr):
    if isinstance(expr, Call):
        yield expr
        for argument in expr.args:
            yield from _walk(argument)
    elif isinstance(expr, Tuple):
        for element in expr.elements:
            yield from _walk(element)


def _ops_for_function(result, function_name: str):
    paths = {
        region.call_path
        for region in result.graph.regions
        if region.function.name == function_name
    }
    return [
        item
        for item in result.report.operations
        if item.call_path in paths
    ]


def _overlap(left, right) -> bool:
    return left.start_ns < right.end_ns and right.start_ns < left.end_ns


def _disjoint(left, right) -> bool:
    return (
        left.submesh_offsets[0] + left.submesh_extents[0] <= right.submesh_offsets[0]
        or right.submesh_offsets[0] + right.submesh_extents[0] <= left.submesh_offsets[0]
    )


def test_real_dsv4_moe_passes_whole_graph_autodist_mvp() -> None:
    assert DIM == 4096
    assert N_ROUTED == 256
    assert N_ACT == 6
    assert MOE_INTER == 2048

    mesh = Mesh(Topology("cta", 8), Layout((8,), (1,)))
    result = auto_dist(
        dsv4_moe_module,
        target=CudaTarget(arch="sm_90", device="h200_sxm"),
        mesh=mesh,
    )

    root_targets = {
        op.ir_expr.target.name
        for op in result.graph.ops
        if op.call_path == () and hasattr(op.ir_expr.target, "name")
    }
    assert {"pre_moe_rms_norm", "moe_topk", "shared_expert", "combine_expert_outputs"} <= root_targets
    assert any(call.ir_call.target.name == "moe_experts_core" for call in result.graph.calls)
    assert len(result.graph.constraints) == 1
    assert len(result.report.constraints) == 1
    assert result.report.constraints[0].satisfied is True

    assert all(math.isfinite(float(cost.duration_ns)) for cost in result.costs.all())
    assert all(
        operation.submesh_offsets[0] + operation.submesh_extents[0] <= 132
        for operation in result.report.operations
    )
    assert all(operation.cta_count <= 132 for operation in result.report.operations)

    routed = _ops_for_function(result, "moe_experts_core")
    shared = _ops_for_function(result, "shared_expert")
    assert routed and shared
    assert any(_overlap(left, right) and _disjoint(left, right) for left in routed for right in shared)

    selected_serial = sum(operation.duration_ns for operation in result.report.operations)
    selected_serial += sum(item.end_ns - item.start_ns for item in result.report.reshards)
    assert result.report.predicted_makespan_ns < selected_serial
    assert result.report.status in {"OPTIMAL", "FEASIBLE_NOT_PROVEN"}
    assert result.report.logical_fingerprint == logical_fingerprint(dsv4_moe_module)
    assert logical_fingerprint(result.solution) == logical_fingerprint(dsv4_moe_module)

    forbidden = {"FP8GEMM", "MoERoute", "MoEExpertCompute", "ExpertRoute", "ExpertCompute"}
    assert not forbidden & {type(call.target).__name__ for fn in result.solution.functions for call in _walk(fn.body)}
    for function in result.solution.functions:
        verify_function(function)
    assert any(
        type(call.target).__name__ == "Reshard"
        for function in result.solution.functions
        for call in _walk(function.body)
    )

    printed = as_script(result.solution)
    reparsed = parse_module_source(printed)
    for function in reparsed.functions:
        verify_function(function)
