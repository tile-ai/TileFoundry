from __future__ import annotations

from typing import Literal

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._helpers import resolve_anchor_storage
from tilefoundry.ir.hir._shard_checks import check_multilinear_partials
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.ir.types import DType, Layout, TensorType
from tilefoundry.ir.types.shard_layout import (
    ShardLayout,
    canonical_shard_layout,
    shard_layout_of,
    split_target_axes,
)
from tilefoundry.ir.types.stride import try_compact_major
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    broadcast_shapes,
    matmul_relations,
    register_access_relation,
    relations_of,
    shape_from_relation,
)
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout

_ACCUMULATOR_OF = {DType.fp8e4m3: DType.f32}


def matmul_result_dtype(operand: DType) -> DType:
    """The dtype a product of two *operand* tensors is summed and returned in.

    No MMA accumulates in fp8e4m3, so its products are summed and returned in f32
    (docs/spec/hir.md MatMul); every other dtype is its own result dtype.
    """
    return _ACCUMULATOR_OF.get(operand, operand)


@register_op
class MatMul(Op):
    """Batched matrix multiplication with explicit physical matrix-axis order."""

    lhs = ParamDef(kind="input", pattern=is_ranked_tensor())
    rhs = ParamDef(kind="input", pattern=is_ranked_tensor())
    a_layout = ParamDef(kind="attribute", annotation=Literal["MK", "KM"], default="MK")
    b_layout = ParamDef(kind="attribute", annotation=Literal["NK", "KN"], default="KN")


def matmul_axes(op: MatMul) -> tuple[int, int, int, int]:
    """Return physical ``(A.M, A.K, B.N, B.K)`` axes for the layout literals."""
    if op.a_layout == "MK":
        a_m, a_k = -2, -1
    elif op.a_layout == "KM":
        a_m, a_k = -1, -2
    else:
        raise ValueError(f"MatMul: a_layout must be 'MK' or 'KM', got {op.a_layout!r}")
    if op.b_layout == "NK":
        b_n, b_k = -2, -1
    elif op.b_layout == "KN":
        b_n, b_k = -1, -2
    else:
        raise ValueError(f"MatMul: b_layout must be 'NK' or 'KN', got {op.b_layout!r}")
    return a_m, a_k, b_n, b_k


def _k_split_axes(t, k_tensor_axis: int) -> "frozenset[int]":
    """The mesh axes on which *t* splits its contraction (K) tensor axis."""
    layout = shard_layout_of(t.layout)
    if layout is None:
        return frozenset()
    targets = split_target_axes(layout, t.shape)
    return frozenset(p for p, ax in enumerate(targets) if ax == k_tensor_axis)


@register_access_relation(MatMul)
def _matmul_access_relation(call: "Call", ctx) -> tuple[AccessRelation, ...]:
    """Every coordinate of each operand a contraction reaches, read once."""
    lhs = ctx.type_of(call.args[0])
    rhs = ctx.type_of(call.args[1])
    return matmul_relations(lhs.shape, rhs.shape, matmul_axes(call.target))


def _elements(shape: tuple) -> int:
    """How many elements a shape of numbers holds."""
    counted = 1
    for extent in shape:
        counted *= extent if isinstance(extent, int) else 1
    return counted


@register_typeinfer(MatMul)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    lhs = ctx.type_of(call.args[0])
    rhs = ctx.type_of(call.args[1])
    try:
        a_m, a_k, b_n, b_k = matmul_axes(call.target)
    except ValueError as error:
        ctx.error(call, str(error).removeprefix("MatMul: "))
    if lhs.dtype != rhs.dtype:
        ctx.error(call, f"MatMul dtype mismatch: {lhs.dtype} vs {rhs.dtype}")
    if len(lhs.shape) < 2 or len(rhs.shape) < 2:
        ctx.error(call, "MatMul requires rank >= 2 on both operands")
    if broadcast_shapes(lhs.shape[:-2], rhs.shape[:-2], raising=False) is None:
        ctx.error(call, f"MatMul batch-dim mismatch {lhs.shape[:-2]} vs {rhs.shape[:-2]}")
    if lhs.shape[a_k] != rhs.shape[b_k]:
        ctx.error(
            call,
            f"MatMul contraction-dim mismatch: lhs K={lhs.shape[a_k]} vs rhs K={rhs.shape[b_k]}",
        )

    if _k_split_axes(lhs, a_k % len(lhs.shape)) != _k_split_axes(rhs, b_k % len(rhs.shape)):
        ctx.error(
            call,
            "MatMul contraction dim K must be split on the same mesh axes for both operands",
        )

    check_multilinear_partials(ctx, call, (("lhs", lhs), ("rhs", rhs)))

    relation = relations_of(call, ctx)

    out_batch = broadcast_shapes(lhs.shape[:-2], rhs.shape[:-2], raising=False)
    out_shape = shape_from_relation(
        relation[len(call.args)], (*out_batch, lhs.shape[a_m], rhs.shape[b_n], lhs.shape[a_k])
    )
    k_domain_dim = len(out_shape)
    try:
        shard = derive_output_shard_layout(
            (lhs, rhs),
            relation,
            out_shape,
            partial_reduction_dims=frozenset({k_domain_dim}),
        )
    except ValueError as e:
        ctx.error(call, str(e))
    layout = shard
    if layout is None:
        held = shard_layout_of(lhs.layout) or lhs.layout
        if isinstance(held, ShardLayout):
            layout = canonical_shard_layout(out_shape, held.mesh, held.attrs)
        elif held is None or tuple(held.shape) == tuple(out_shape):
            layout = held
        else:
            layout = Layout(shape=out_shape, strides=try_compact_major(out_shape))
    storage = resolve_anchor_storage(ctx, call, lhs.storage, rhs.storage)
    return TensorType(
        shape=out_shape, dtype=matmul_result_dtype(lhs.dtype), layout=layout, storage=storage
    )


@register_eval(MatMul)
def _eval_matmul(ctx):
    lhs = ctx.args[0].data
    rhs = ctx.args[1].data
    if ctx.op.a_layout == "KM":
        lhs = lhs.transpose(-1, -2)
    if ctx.op.b_layout == "NK":
        rhs = rhs.transpose(-1, -2)
    if ctx.result_type.dtype != ctx.args[0].type.dtype:
        lhs, rhs = lhs.float(), rhs.float()
    out = torch.matmul(lhs, rhs)
    return TensorValue(data=out, type=ctx.result_type)
