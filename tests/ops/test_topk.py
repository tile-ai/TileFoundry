"""TopK contract: (values, indices) with i64 indices, largest/sorted control,
rejection of an oversized k or a Split on the selected axis, and dynamic
``k`` (``ShapeDim = int | DimVar | Expr``): a ``k`` derived from a
context-length ``DimVar`` (e.g. ``dim_min(512, POS // 4)``) is a first-class
value -- typeinfer propagates it as a symbolic output axis, and the same
*built* ``Function`` evaluates at any concrete context length without a
rebuild."""
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
from tilefoundry.inspection.python_printer import shape_entry_str
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
        "explicit_axis",
        TopK(k=2, axis=1),
        (make_tensor_type((4, 8, 16), _BF),),
        TupleType(fields=(make_tensor_type((4, 2, 16), _BF), make_tensor_type((4, 2, 16), _I64))),
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


# ─── typeinfer: symbolic k propagates, validity checks (AC on widened k) ────

# A static axis with a symbolic k whose statically-derivable upper bound (see
# ``_dim_upper_bound`` in topk.py) still fits: dim_min(50, V//2) with V's
# envelope hi=101 -> V's bound 100 -> V//2 bound 50 -> min(50, 50) = 50 <= 100.
_V_SMALL = DimVar("topk_dyn_v_small", 1, 101)
_K_WITHIN_BOUND = dim_min(50, _V_SMALL // 2)

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
        "symbolic_k_within_static_hi_bound_accepted",
        TopK(k=_K_WITHIN_BOUND, axis=-1),
        (make_tensor_type((4, 100), _F32),),
        TupleType(fields=(
            make_tensor_type((4, _K_WITHIN_BOUND), _F32),
            make_tensor_type((4, _K_WITHIN_BOUND), _I64),
        )),
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
    TypeInferCase(
        # dim_min(6, 6) is two static ints -> simplify_dim folds it to a
        # Constant (not a plain int) before it ever reaches TopK; TopK must
        # treat that exactly like a plain static k=6 (TensorType.__post_init__
        # canonicalizes the int-valued Constant back to plain int in the
        # output shape -- this pins that no Constant-wrapper survives).
        "static_dim_min_folds_and_behaves_like_plain_int_k",
        TopK(k=dim_min(6, 6), axis=-1),
        (make_tensor_type((4, 256), _F32),),
        TupleType(fields=(make_tensor_type((4, 6), _F32), make_tensor_type((4, 6), _I64))),
    ),
    TypeInferCase(
        "static_dim_min_oversized_rejected",
        TopK(k=dim_min(300, 300), axis=-1),
        (make_tensor_type((4, 256), _F32),),
        ExpectedError(match="exceeds axis"),
    ),
]


@pytest.mark.parametrize("case", DYNAMIC_K_TYPEINFER_CASES, ids=lambda c: c.name)
def test_topk_dynamic_k_typeinfer(case):
    run_typeinfer_case(case)


# ─── printer: shape_entry_str already covers DimMin/DimFloorDiv; TopK's ─────
# ─── symbolic output axis rides that mechanism with no special-casing ───────


def test_topk_dynamic_k_output_shape_prints_consistently():
    """Rendering the Function's return type as a plain ``TensorType`` (the
    ``values`` field alone, via ``tuple_get_item``) exercises the full
    ``_tensor_annotation`` -> ``_shape_tuple`` -> ``shape_entry_str`` path:
    a ``TupleType`` return has no surface annotation at all (python_printer's
    own documented rule), so this is the shape one *can* observe printed.
    """
    rendered = shape_entry_str(K)
    assert rendered == "min(512, POS // 4)"

    fn, result_type = _build_topk_fn((4, POS), K)
    values_call = Call(
        type=result_type.fields[0], target=TupleGetItem(index=0), args=(fn.body,)
    )
    values_ty = TypeInferVisitor(TypeInferContext()).visit(values_call)
    values_call = replace(values_call, type=values_ty)
    fn = Function.build(
        name="topk_dyn_k_printer",
        params=fn.params,
        body=values_call,
        return_type=values_ty,
    )

    script = as_script(fn)
    assert rendered in script
    assert f"-> Tensor[(4, {rendered})" in script


def test_topk_dynamic_k_printer_does_not_crash_on_tuple_return():
    """The (values, indices) TupleType return shape is printer-invisible (see
    above), but printing must still not raise when ``k`` itself is a ``Call``
    (not a plain int) -- pins that the attribute-rendering fallback in
    ``_format_call`` merely reprs an unrecognised attribute value rather than
    special-casing (and possibly choking on) it. That repr is not meant to be
    re-parsed back into the same symbolic k (see the task report's debt
    notes); this only guards against a crash.
    """
    fn, _ = _build_topk_fn((4, POS), K)
    script = as_script(fn)
    assert "topk(" in script


# ─── evaluation: one built Function, two ctx-length bindings ───────────────


def test_topk_dynamic_k_evaluates_at_two_ctx_bindings():
    """Same built Function; k = min(512, pos // 4) resolves to a different
    concrete int per invocation, driven purely by x's runtime shape -- no
    rebuild (mirrors the P0a nested-module dynamic-ctx pattern used by
    tests/models/*/attention.py's DimVar-shaped kv cache)."""
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


def test_topk_dynamic_k_small_ctx_binding_yields_k_zero():
    """A pos below 4 makes pos // 4 == 0 -> k == 0; torch.topk(k=0) is valid
    (selects nothing), so the same built Function handles this edge binding
    like any other k, not a crash or a special case."""
    fn, _ = _build_topk_fn((4, POS), K)
    vals, idx = evaluate(fn, torch.randn(4, 3), device="cpu")
    assert vals.shape == (4, 0)
    assert idx.shape == (4, 0)


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
