from __future__ import annotations

from tests.models.deepseek_v4_flash.model import REAL, deepseek_v4_flash_module
from tilefoundry.inspection import as_script
from tilefoundry.ir.constraints import LayoutConstraint, constraint_metadata
from tilefoundry.ir.core import Call, Tuple
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.tensor.gather import Gather
from tilefoundry.ir.hir.tensor.reduce import Reduce
from tilefoundry.ir.hir.tensor.topk import TopK
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types.shard import Broadcast, Split


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


#: The real model's shapes, read off the config the block is authored at.
DIM = REAL.dim
MOE_INTER = REAL.moe_inter
N_ACT = REAL.n_act
N_ROUTED = REAL.n_routed


def test_root_helpers_and_constraints_keep_real_model_contract() -> None:
    deepseek_v4_flash_moe = deepseek_v4_flash_module.lookup("deepseek_v4_flash_moe")

    assert tuple(
        (topology.name, topology.size)
        for topology in deepseek_v4_flash_module.effective_topologies()
    ) == (("cta", 132), ("thread", 512))
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
    deepseek_v4_flash_moe = deepseek_v4_flash_module.lookup("deepseek_v4_flash_moe")
    moe_experts_core = deepseek_v4_flash_module.lookup("moe_experts_core")
    moe_topk = deepseek_v4_flash_module.lookup("moe_topk")

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
    printed = as_script(deepseek_v4_flash_module)
    assert '@module(entry="deepseek_v4_flash_moe")' in printed
    assert 'topologies = (Topology("cta", 132), Topology("thread", 512),)' in printed
    assert f"routed_experts: where(layout=(_, 6 @ cta, {DIM}))" in printed
    assert f"combined: where(layout=((_, _, {DIM}), {{cta @ B()}}))" in printed
