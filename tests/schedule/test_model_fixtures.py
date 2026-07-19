from __future__ import annotations

from tests.models.deepseek_v4_flash.moe import (
    DIM,
    MOE_INTER,
    N_ACT,
    N_ROUTED,
    deepseek_v4_flash_module,
    deepseek_v4_flash_moe,
    moe_experts_core,
    moe_topk,
)
from tests.models.qwen3_5_30b_a3b.gqa_online import gqa_online_attend
from tests.models.qwen3_5_30b_a3b.static_online import qwen_static_online
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Tuple
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.tensor.gather import Gather
from tilefoundry.ir.hir.tensor.reduce import Reduce
from tilefoundry.ir.hir.tensor.topk import TopK
from tilefoundry.parser import parse_func_source
from tilefoundry.schedule.constraints import (
    LayoutConstraint,
    LayoutDimKind,
    ScheduleConstraintMetadata,
    constraint_metadata,
)
from tilefoundry.target import CudaTarget


def _walk(expr, seen=None):
    if seen is None:
        seen = set()
    if expr is None:
        return
    if id(expr) in seen:
        return
    seen.add(id(expr))
    yield expr
    if isinstance(expr, Call):
        for arg in expr.args:
            yield from _walk(arg, seen)
    elif isinstance(expr, Tuple):
        for element in expr.elements:
            yield from _walk(element, seen)
    elif isinstance(expr, GridRegionExpr):
        for arg in expr.init_args:
            yield from _walk(arg, seen)
        yield from _walk(expr.body, seen)
        for value in expr.yield_values:
            yield from _walk(value, seen)


def _calls(fn):
    return tuple(expr for expr in _walk(fn.body) if isinstance(expr, Call))


def test_deepseek_root_and_helpers_keep_the_real_contract() -> None:
    assert deepseek_v4_flash_moe.target == CudaTarget()
    assert tuple((topology.name, topology.size) for topology in deepseek_v4_flash_moe.topologies) == (
        ("cta", 132),
    )
    assert all(
        helper.target is None and helper.topologies == ()
        for helper in deepseek_v4_flash_module.functions[:-1]
    )
    assert deepseek_v4_flash_moe.params[4].type.shape == (
        N_ROUTED,
        MOE_INTER,
        DIM,
    )
    assert deepseek_v4_flash_moe.params[8].type.shape == (
        N_ROUTED,
        DIM,
        MOE_INTER,
    )

    routed_call = next(
        call
        for call in _calls(deepseek_v4_flash_moe)
        if isinstance(call.target, Function) and call.target.name == "moe_topk"
    )
    routed_metadata = constraint_metadata(routed_call)
    assert isinstance(routed_metadata, ScheduleConstraintMetadata)
    assert len(routed_metadata.constraints) == 1
    routed_layout = routed_metadata.constraints[0]
    assert isinstance(routed_layout, LayoutConstraint)
    assert routed_layout.dims[1].extent == N_ACT
    assert routed_layout.dims[1].kind is LayoutDimKind.SPLIT
    assert routed_layout.dims[1].topology == "cta"
    assert routed_call.type.shape == (1, N_ACT, DIM)

    combined_call = next(
        call
        for call in _calls(deepseek_v4_flash_moe)
        if isinstance(call.target, Function)
        and call.target.name == "combine_expert_outputs"
    )
    combined_metadata = constraint_metadata(combined_call)
    assert isinstance(combined_metadata, ScheduleConstraintMetadata)
    combined_layout = combined_metadata.constraints[0]
    assert isinstance(combined_layout, LayoutConstraint)
    assert all(dim.kind is LayoutDimKind.BROADCAST for dim in combined_layout.dims)


def test_deepseek_routed_path_is_ordinary_batched_dataflow() -> None:
    op_types = {type(call.target) for call in _calls(moe_experts_core)}
    assert {Gather, MatMul}.issubset(op_types)
    assert any(type(call.target).__name__ == "Cast" for call in _calls(moe_experts_core))
    assert any(type(call.target).__name__ == "Reshape" for call in _calls(moe_experts_core))
    assert moe_topk.return_type.shape == (1, N_ACT, DIM)
    assert moe_experts_core.return_type.shape == (1, N_ACT, DIM)
    assert any(isinstance(call.target, TopK) for call in _calls(moe_topk))
    assert any(type(call.target).__name__ == "TupleGetItem" for call in _calls(moe_topk))
    assert any(
        isinstance(call.target, Reduce) and call.target.axes == (1,)
        for call in _calls(deepseek_v4_flash_moe)
    )


def test_static_qwen_has_one_positive_grid_region_with_online_state() -> None:
    regions = tuple(expr for expr in _walk(qwen_static_online.body) if isinstance(expr, GridRegionExpr))
    assert len(regions) == 1
    region = regions[0]
    assert (region.start, region.extent, region.step) == (0, 4096, 1)
    assert {value.name for value in region.carried_args} == {"m", "l", "o"}
    assert qwen_static_online.target == CudaTarget()
    assert tuple((topology.name, topology.size) for topology in qwen_static_online.topologies) == (
        ("cta", 132),
    )


def test_nested_positive_static_grid_regions_are_representable() -> None:
    fn = parse_func_source(
        '''from __future__ import annotations
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.dsl.tf import *

@func
def nested(x: Tensor[(4, 4), "f32"]) -> Tensor[(4, 4), "f32"]:
    acc = tf.full_like(x, value=0.0)
    for i in tile(4):
        for j in tile(2):
            acc = acc + x
    return acc
'''
    )
    regions = tuple(expr for expr in _walk(fn.body) if isinstance(expr, GridRegionExpr))
    assert len(regions) == 2
    assert {region.extent for region in regions} == {2, 4}


def test_dynamic_qwen_grid_region_remains_representable() -> None:
    variants = tuple(gqa_online_attend.variants)
    assert variants
    assert any(
        isinstance(expr, GridRegionExpr)
        and not isinstance(expr.extent, int)
        for variant in variants
        for expr in _walk(variant.body)
    )


def test_static_root_print_roundtrip_preserves_target_and_grid_region() -> None:
    printed = as_script(qwen_static_online)
    reparsed = parse_func_source(printed)
    assert reparsed.target == qwen_static_online.target
    assert tuple((topology.name, topology.size) for topology in reparsed.topologies) == (
        ("cta", 132),
    )
    regions = tuple(
        expr for expr in _walk(reparsed.body) if isinstance(expr, GridRegionExpr)
    )
    assert len(regions) == 1
    assert regions[0].extent == 4096


def test_deepseek_root_printer_keeps_explicit_input_contracts() -> None:
    printed = as_script(deepseek_v4_flash_moe)
    assert "@func(target=CudaTarget(), topologies=(Topology(" in printed
    assert "routed_experts: where(layout=(_, 6 @ cta, DIM))" in printed
    assert "combined: where(layout=(D, D, D))" in printed
