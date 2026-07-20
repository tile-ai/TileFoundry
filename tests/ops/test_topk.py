"""TopK contract: (values, indices) with i64 indices, largest/sorted control,
and rejection of an oversized k or a Split on the selected axis."""
from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    raw_shard_tensor_type,
    run_typeinfer_case,
)
from tilefoundry.evaluator import evaluate
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.tensor.topk import TopK
from tilefoundry.ir.types import (
    DType,
    TupleType,
    make_shard_tensor_type,
    make_tensor_type,
)
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, ShardLayout, Split
from tilefoundry.parser.hir_parser import parse_script
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

_BF = DType.bf16
_F32 = DType.f32
_I64 = DType.i64


# ─── typeinfer (AC-0-1 shapes/dtype, AC-0-3 rejections) ─────────────────────

CASES = [
    TypeInferCase(
        "decode_axis_last",
        TopK(k=512, axis=-1),
        (make_tensor_type((1, 1, 16384), _BF),),
        TupleType(fields=(make_tensor_type((1, 1, 512), _BF), make_tensor_type((1, 1, 512), _I64))),
    ),
    TypeInferCase(
        "small_2d",
        TopK(k=6),
        (make_tensor_type((4, 256), _F32),),
        TupleType(fields=(make_tensor_type((4, 6), _F32), make_tensor_type((4, 6), _I64))),
    ),
    TypeInferCase(
        "explicit_axis",
        TopK(k=2, axis=1),
        (make_tensor_type((4, 8, 16), _BF),),
        TupleType(fields=(make_tensor_type((4, 2, 16), _BF), make_tensor_type((4, 2, 16), _I64))),
    ),
    TypeInferCase(
        "axis_out_of_range",
        TopK(k=2, axis=5),
        (make_tensor_type((4,), _BF),),
        ExpectedError(match="out of range"),
    ),
    TypeInferCase(
        "oversized_k_rejected",
        TopK(k=300, axis=-1),
        (make_tensor_type((4, 256), _F32),),
        ExpectedError(match="exceeds axis"),
    ),
    TypeInferCase(
        "negative_k_rejected",
        TopK(k=-1, axis=-1),
        (make_tensor_type((4, 256), _F32),),
        ExpectedError(match="non-negative"),
    ),
    TypeInferCase(
        "split_on_selected_axis_rejected",
        TopK(k=2, axis=-1),
        (make_shard_tensor_type((4, 256), mesh=make_mesh((4,)), attrs=(Split(1),)),),
        ExpectedError(match="must not be Split-sharded"),
    ),
    TypeInferCase(
        "partial_input_rejected",
        TopK(k=2, axis=-1),
        (make_shard_tensor_type((4, 256), mesh=make_mesh((4,)), attrs=(Partial("max"),)),),
        ExpectedError(match="x carries Partial"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_topk_typeinfer(case):
    run_typeinfer_case(case)


# ─── evaluation (AC-0-2 values/indices vs torch.topk) ───────────────────────

def _run_topk(x: torch.Tensor, **attrs):
    param = Var(type=make_tensor_type(tuple(x.shape), DType.f32), name="x")
    call = Call(type=param.type, target=TopK(**attrs), args=(param,))
    result_type = TypeInferVisitor(TypeInferContext()).visit(call)
    call = replace(call, type=result_type)
    fn = Function.build(name="topk_case", params=(param,), body=call, return_type=result_type)
    return evaluate(fn, x, device="cpu")


def test_topk_largest_sorted_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(4, 256)
    vals, idx = _run_topk(x, k=6, axis=-1, largest=True, sorted=True)
    ref_v, ref_i = torch.topk(x, 6, dim=-1, largest=True, sorted=True)
    torch.testing.assert_close(vals, ref_v)
    torch.testing.assert_close(idx.long(), ref_i)


def test_topk_smallest_sorted_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(4, 256)
    vals, idx = _run_topk(x, k=6, axis=-1, largest=False, sorted=True)
    ref_v, ref_i = torch.topk(x, 6, dim=-1, largest=False, sorted=True)
    torch.testing.assert_close(vals, ref_v)
    torch.testing.assert_close(idx.long(), ref_i)


def test_topk_unsorted_selects_same_set():
    """sorted=False: the selected (value,index) pairs match torch.topk as a
    SET, without requiring a particular internal order."""
    torch.manual_seed(0)
    x = torch.randn(4, 256)
    vals, idx = _run_topk(x, k=6, axis=-1, largest=True, sorted=False)
    ref_v, _ = torch.topk(x, 6, dim=-1, largest=True, sorted=True)
    # Values stay paired with their indices (gather along the axis).
    torch.testing.assert_close(vals, x.gather(-1, idx.long()))
    # Same selected values once both are sorted, and same index set per row.
    got_sorted, _ = torch.sort(vals, dim=-1, descending=True)
    torch.testing.assert_close(got_sorted, ref_v)
    for r in range(x.shape[0]):
        assert set(idx[r].long().tolist()) == set(torch.topk(x[r], 6).indices.tolist())


# ─── DSL print/parse preserves largest & sorted (AC-0-2) ────────────────────

def test_topk_printer_preserves_largest_sorted():
    param = Var(type=make_tensor_type((4, 256), DType.f32), name="x")
    call = Call(type=param.type, target=TopK(k=6, axis=-1, largest=False, sorted=True), args=(param,))
    result_type = TypeInferVisitor(TypeInferContext()).visit(call)
    call = replace(call, type=result_type)
    fn = Function.build(name="topk_rt", params=(param,), body=call, return_type=result_type)

    script = as_script(fn)
    assert "largest=False" in script and "sorted=True" in script
    assert "k=6" in script and "axis=-1" in script


def test_topk_output_layout_shrinks_selected_axis_preserving_split():
    """A Split on a non-selected axis must be preserved, and the output shard
    layout's selected axis must shrink to k so size(layout)==size(shape)."""
    x_ty = make_shard_tensor_type((4, 256), mesh=make_mesh((4,)), attrs=(Split(0),))  # split axis 0; TopK on axis 1
    out = infer_call(TopK(k=6, axis=-1), x_ty)
    values_ty, indices_ty = out.fields
    assert values_ty.shape == (4, 6) and indices_ty.shape == (4, 6)
    for t in (values_ty, indices_ty):
        assert isinstance(t.layout, ShardLayout), "sharded input must stay sharded"
        assert any(isinstance(a, Split) and a.axis == 0 for a in t.layout.attrs), (
            "non-selected Split(0) must survive TopK"
        )
        assert math.prod(t.layout.layout.shape) == math.prod(t.shape), (
            f"size(layout)={t.layout.layout.shape} != size(shape)={t.shape}"
        )


def test_topk_all_broadcast_layout_shrinks_selected_axis():
    """A replicated (all-Broadcast) ShardLayout has no Split/Partial to
    propagate, but its output layout must still shrink the selected axis to k
    (not retain the stale input extent)."""
    x_ty = make_shard_tensor_type((4, 256), mesh=make_mesh((4,)), attrs=(Broadcast(),))
    values_ty, indices_ty = infer_call(TopK(k=6, axis=-1), x_ty).fields
    assert values_ty.shape == (4, 6) and indices_ty.shape == (4, 6)
    for t in (values_ty, indices_ty):
        assert isinstance(t.layout, ShardLayout)
        assert all(isinstance(a, Broadcast) for a in t.layout.attrs), "replication preserved"
        assert math.prod(t.layout.layout.shape) == math.prod(t.shape), (
            f"size(layout)={t.layout.layout.shape} != size(shape)={t.shape}"
        )


def test_topk_all_broadcast_layout_with_dynamic_dim():
    """The canonical replicated fallback must handle a dynamic (DimVar) dim on a
    non-selected axis: no int() on a ShapeDim, and — per the HIR invariant that
    every post-typeinfer ShardLayout has concrete strides — it materializes
    explicit all-ones strides rather than leaving them None."""
    s = DimVar("S", 1, 64)
    x_ty = raw_shard_tensor_type(
        (256, s), (256, s), None, (Broadcast(),), make_mesh((4,)), dtype=_F32,
    )
    values_ty, indices_ty = infer_call(TopK(k=6, axis=0), x_ty).fields  # select the static axis
    assert values_ty.shape == (6, s) and indices_ty.shape == (6, s)
    for t in (values_ty, indices_ty):
        assert isinstance(t.layout, ShardLayout)
        assert t.layout.layout.shape == (6, s), t.layout.layout.shape
        assert t.layout.layout.strides == (1, 1), "concrete strides, never None"


def test_topk_parser_preserves_largest_sorted():
    src = (
        "from tilefoundry import func\n"
        "from tilefoundry.dsl import Tensor\n"
        "from tilefoundry.dsl.tf import *\n\n"
        "@func\n"
        'def f(x: Tensor[(4, 256), "f32"]):\n'
        "    v = topk(x, k=6, axis=-1, largest=False, sorted=True)\n"
        "    return v\n"
    )
    topk = parse_script(src).body.target
    assert isinstance(topk, TopK)
    assert topk.k == 6 and topk.axis == -1
    assert topk.largest is False and topk.sorted is True
