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
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Tuple
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.tensor.gather import Gather
from tilefoundry.ir.hir.tensor.reduce import Reduce
from tilefoundry.ir.hir.tensor.topk import TopK
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types.shard import Broadcast, Split
from tilefoundry.schedule.constraints import LayoutConstraint, constraint_metadata
from tilefoundry.target import CudaTarget


def _walk(expr, seen=None):
    if seen is None:
        seen = set()
    if expr is None or id(expr) in seen:
        return
    seen.add(id(expr))
    yield expr
    if isinstance(expr, Call):
        for arg in expr.args:
            yield from _walk(arg, seen)
    elif isinstance(expr, Tuple):
        for element in expr.elements:
            yield from _walk(element, seen)


def _calls(fn):
    return tuple(expr for expr in _walk(fn.body) if isinstance(expr, Call))


def test_root_helpers_and_constraints_keep_real_model_contract() -> None:
    assert deepseek_v4_flash_moe.target == CudaTarget()
    assert tuple(
        (topology.name, topology.size) for topology in deepseek_v4_flash_moe.topologies
    ) == (("cta", 132),)
    assert all(
        helper.target is None and helper.topologies == ()
        for helper in deepseek_v4_flash_module.functions[:-1]
    )
    assert deepseek_v4_flash_moe.params[4].type.shape == (N_ROUTED, MOE_INTER, DIM)
    assert deepseek_v4_flash_moe.params[8].type.shape == (N_ROUTED, DIM, MOE_INTER)

    routed_call = next(
        call
        for call in _calls(deepseek_v4_flash_moe)
        if isinstance(call.target, Function) and call.target.name == "moe_topk"
    )
    routed = constraint_metadata(routed_call).constraints[0]
    assert isinstance(routed, LayoutConstraint)
    assert repr(routed.layout.shape[0]) == "_"
    assert routed.layout.shape[1:] == (N_ACT, DIM)
    assert routed.bindings == (("cta", Split(1)),)
    assert routed_call.type.shape == (1, N_ACT, DIM)

    combined_call = next(
        call
        for call in _calls(deepseek_v4_flash_moe)
        if isinstance(call.target, Function)
        and call.target.name == "combine_expert_outputs"
    )
    combined = constraint_metadata(combined_call).constraints[0]
    assert isinstance(combined, LayoutConstraint)
    assert combined.bindings == (("cta", Broadcast()),)


def test_routed_path_is_ordinary_batched_dataflow() -> None:
    op_types = {type(call.target) for call in _calls(moe_experts_core)}
    assert {Gather, MatMul}.issubset(op_types)
    assert any(type(call.target).__name__ == "Cast" for call in _calls(moe_experts_core))
    assert any(type(call.target).__name__ == "Reshape" for call in _calls(moe_experts_core))
    assert moe_topk.return_type.shape == (1, N_ACT, DIM)
    assert moe_experts_core.return_type.shape == (1, N_ACT, DIM)

    topk_call = next(call for call in _calls(moe_topk) if isinstance(call.target, TopK))
    assert tuple(field.shape for field in topk_call.type.fields) == (
        (1, N_ACT),
        (1, N_ACT),
    )
    topk_elements = [
        call
        for call in _calls(moe_topk)
        if isinstance(call.target, TupleGetItem) and call.args[0] is topk_call
    ]
    assert len(topk_elements) == 1
    assert all(element.type.shape == (1, N_ACT) for element in topk_elements)
    assert any(
        isinstance(call.target, Gather) and call.type.shape == (1, N_ACT)
        for call in _calls(moe_topk)
    )
    assert any(
        isinstance(call.target, MatMul) and call.type.shape[:2] == (1, N_ACT)
        for call in _calls(moe_experts_core)
    )
    assert any(
        isinstance(call.target, Reduce) and call.target.axes == (1,)
        for call in _calls(deepseek_v4_flash_moe)
    )


def test_root_printer_keeps_explicit_input_contracts() -> None:
    printed = as_script(deepseek_v4_flash_moe)
    assert "@func(target=CudaTarget(), topologies=(Topology(" in printed
    assert f"routed_experts: where(layout=(_, 6 @ cta, {DIM}))" in printed
    assert f"combined: where(layout=((_, _, {DIM}), {{cta @ B()}}))" in printed
