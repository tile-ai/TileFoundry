"""TopK's dynamic ``k``, its output layout, its surface, and its consumer.

- dynamic ``k`` (``ShapeDim = int | DimVar | Expr``): a ``k`` derived from a
  context-length ``DimVar`` (e.g. ``dim_min(512, POS // 4)``) is a first-class
  value -- typeinfer propagates it as a symbolic output axis, and the same
  *built* ``Function`` evaluates at any concrete context length without a
  rebuild -- and a ``k`` whose statically-derivable upper bound cannot fit the
  axis is rejected before it ever runs;
- the output ``ShardLayout``: the selected axis shrinks to ``k`` while a Split on
  a non-selected axis survives, including when a non-selected axis is dynamic;
- ``largest`` / ``sorted`` surviving the parse surface, and ``sorted=False``
  keeping each value paired with its own index;
- the downstream consumer: those indices feed ``gather``, whose output shape
  must stay consistent at every binding of the same symbolic ``k``.
"""
from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from tests._source import import_dsl
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    raw_shard_tensor_type,
    run_typeinfer_case,
)
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.tensor.gather import Gather
from tilefoundry.ir.hir.tensor.topk import TopK
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import (
    DType,
    TupleType,
    make_shard_tensor_type,
    make_tensor_type,
)
from tilefoundry.ir.types.dim import DimVar, dim_min
from tilefoundry.ir.types.shard import Layout, make_mesh
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, ShardLayout, Split
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

_F32 = DType.f32
_I64 = DType.i64


# ─── typeinfer rejections ───────────────────────────────────────────────────

CASES = [
    TypeInferCase(
        "oversized_k_rejected",
        TopK(k=300, axis=-1),
        (make_tensor_type((4, 256), _F32),),
        ExpectedError(match="exceeds axis"),
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


# ─── evaluation of a mode no model selects ──────────────────────────────────

def _run_topk(x: torch.Tensor, **attrs):
    param = Var(type=make_tensor_type(tuple(x.shape), DType.f32), name="x")
    call = Call(type=param.type, target=TopK(**attrs), args=(param,))
    result_type = TypeInferVisitor(TypeInferContext()).visit(call)
    call = replace(call, type=result_type)
    fn = Function.build(name="topk_case", params=(param,), body=call, return_type=result_type)
    return evaluate(fn, x, device="cpu")


def test_topk_unsorted_selects_same_set():
    """sorted=False: the selected (value,index) pairs match torch.topk as a
    SET, without requiring a particular internal order. Pairing is what an
    unsorted implementation can silently lose, and no corpus router asks for
    unsorted output, so it is checked here."""
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


# ─── output shard layout ────────────────────────────────────────────────────

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


def test_plain_topk_layout_describes_the_selected_result():
    source = make_tensor_type(
        (4, 256), _F32, layout=Layout(shape=(4, 256), strides=(256, 1))
    )

    values, indices = infer_call(TopK(k=6, axis=-1), source).fields

    assert values.layout == Layout(shape=(4, 6), strides=(6, 1))
    assert indices.layout == values.layout


def test_topk_all_broadcast_layout_with_dynamic_dim():
    """The canonical replicated fallback (an all-Broadcast layout has no
    Split/Partial to propagate, but must still shrink the selected axis to k
    rather than retain the stale input extent) must handle a dynamic (DimVar)
    dim on a non-selected axis: no int() on a ShapeDim, and — per the HIR
    invariant that every post-typeinfer ShardLayout has concrete strides — it
    materializes explicit all-ones strides rather than leaving them None."""
    s = DimVar("S", 1, 64)
    x_ty = raw_shard_tensor_type(
        (256, s), (256, s), None, (Broadcast(),), make_mesh((4,)), dtype=_F32,
    )
    values_ty, indices_ty = infer_call(TopK(k=6, axis=0), x_ty).fields  # select the static axis
    assert values_ty.shape == (6, s) and indices_ty.shape == (6, s)
    for t in (values_ty, indices_ty):
        assert isinstance(t.layout, ShardLayout)
        assert all(isinstance(a, Broadcast) for a in t.layout.attrs), "replication preserved"
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
    topk = import_dsl(src).body.target
    assert isinstance(topk, TopK)
    assert topk.k == 6 and topk.axis == -1
    assert topk.largest is False and topk.sorted is True


# ─── dynamic k: ShapeDim = int | DimVar | Expr ──────────────────────────────

# Context-length-shaped axis; envelope comfortably covers both eval bindings
# exercised below (100 and 4096).
POS = DimVar("POS", 1, 8193)
# The task's motivating example: a decode-time top-k capped at 512 but never
# exceeding a quarter of the current context length.
K = dim_min(512, POS // 4)          # pos=100 -> 25; pos=4096 -> 512


def _build_topk_fn(x_shape, k, *, axis: int = -1) -> tuple[Function, "TupleType"]:
    """A one-``Call`` Function ``x -> (values, indices)`` with ``TopK(k=k,
    axis=axis)``, typeinfer'd. Mirrors this file's ``_run_topk`` but returns
    the built ``Function`` (not the evaluated result) so a caller can
    ``evaluate`` it at more than one concrete binding."""
    x = Var(type=make_tensor_type(x_shape, _F32), name="x")
    call = Call(type=x.type, target=TopK(k=k, axis=axis), args=(x,))
    result_type = TypeInferVisitor(TypeInferContext()).visit(call)
    call = replace(call, type=result_type)
    fn = Function.build(name="topk_dyn_k", params=(x,), body=call, return_type=result_type)
    return fn, result_type


# A bare DimVar whose envelope alone (hi=2000 -> max reachable 1999) exceeds a
# static axis length of 100.
_BIG_K = DimVar("topk_dyn_big_k", 1, 2000)

DYNAMIC_K_TYPEINFER_CASES = [
    TypeInferCase(
        "dynamic_k_from_ctx_len_propagates_as_symbolic_output_axis",
        TopK(k=K, axis=-1),
        (make_tensor_type((4, POS), _F32),),
        TupleType(fields=(make_tensor_type((4, K), _F32), make_tensor_type((4, K), _I64))),
    ),
    TypeInferCase(
        "symbolic_k_hi_bound_exceeds_static_axis_rejected",
        TopK(k=_BIG_K, axis=-1),
        (make_tensor_type((4, 100), _F32),),
        ExpectedError(match="upper bound"),
    ),
    TypeInferCase(
        "invalid_k_type_rejected",
        TopK(k="oops", axis=-1),
        (make_tensor_type((4, 256), _F32),),
        ExpectedError(match="DimVar, or dim expression"),
    ),
]


@pytest.mark.parametrize("case", DYNAMIC_K_TYPEINFER_CASES, ids=lambda c: c.name)
def test_topk_dynamic_k_typeinfer(case):
    run_typeinfer_case(case)


def test_topk_dynamic_k_evaluates_at_two_ctx_bindings():
    """Same built Function; k = min(512, pos // 4) resolves to a different
    concrete int per invocation, driven purely by x's runtime shape -- no
    rebuild (mirrors the dynamic-ctx pattern the corpus decoders use for their
    DimVar-shaped kv cache)."""
    fn, _ = _build_topk_fn((4, POS), K)

    torch.manual_seed(0)
    for pos, expected_k in ((100, 25), (4096, 512)):
        scores = torch.randn(4, pos)
        vals, idx = evaluate(fn, scores, device="cpu")
        assert vals.shape == (4, expected_k)
        assert idx.shape == (4, expected_k)
        ref_v, ref_i = torch.topk(scores, expected_k, dim=-1, largest=True, sorted=True)
        torch.testing.assert_close(vals, ref_v)
        torch.testing.assert_close(idx.long(), ref_i)


# ─── downstream consumer: topk indices -> gather, shape (1, K, D) ──────────

_D = 8


def test_topk_dynamic_k_downstream_gather_shape_consistent():
    """indices from a dynamic-k TopK feed ``gather``; shape (1, K, D) holds,
    at the type level and at both concrete ctx-length bindings, and the
    gathered rows match a plain-torch reference."""
    scores = Var(type=make_tensor_type((1, POS), _F32), name="scores")
    table = Var(type=make_tensor_type((POS, _D), _F32), name="table")

    topk_call = Call(type=scores.type, target=TopK(k=K, axis=-1), args=(scores,))
    topk_ty = TypeInferVisitor(TypeInferContext()).visit(topk_call)
    topk_call = replace(topk_call, type=topk_ty)

    idx_call = Call(type=topk_ty.fields[1], target=TupleGetItem(index=1), args=(topk_call,))
    idx_ty = TypeInferVisitor(TypeInferContext()).visit(idx_call)
    idx_call = replace(idx_call, type=idx_ty)

    gather_call = Call(type=idx_ty, target=Gather(axis=0, batch_dims=0), args=(table, idx_call))
    gather_ty = TypeInferVisitor(TypeInferContext()).visit(gather_call)
    assert gather_ty.shape == (1, K, _D)
    gather_call = replace(gather_call, type=gather_ty)

    fn = Function.build(
        name="topk_dyn_k_gather",
        params=(scores, table),
        body=gather_call,
        return_type=gather_ty,
    )

    torch.manual_seed(0)
    for pos, expected_k in ((100, 25), (4096, 512)):
        scores_data = torch.randn(1, pos)
        table_data = torch.randn(pos, _D)
        out = evaluate(fn, scores_data, table_data, device="cpu")
        assert out.shape == (1, expected_k, _D)
        _, ref_idx = torch.topk(scores_data, expected_k, dim=-1, largest=True, sorted=True)
        ref_out = table_data.index_select(0, ref_idx.reshape(-1)).reshape(1, expected_k, _D)
        torch.testing.assert_close(out, ref_out)
