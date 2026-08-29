from __future__ import annotations

from typing import Literal

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._helpers import broadcast_shapes, is_one, resolve_anchor_storage
from tilefoundry.ir.hir._shard_checks import check_multilinear_partials
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.shard_layout import shard_layout_of, split_target_axes
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    iterating,
    register_access_relation,
    relations_of,
    shape_from_relation,
)
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout


@register_op
class MatMul(Op):
    """Batched matrix multiplication with explicit physical matrix-axis order."""

    lhs = ParamDef(kind="input", pattern=Tensor)
    rhs = ParamDef(kind="input", pattern=Tensor)
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


def _broadcast_batch(lhs_batch, rhs_batch):
    """Right-aligned per-dim broadcast of two batch shapes (ranks may differ.

    Right-aligned per-dim broadcast of two batch shapes (ranks may differ —
    the shorter is padded on the left with 1s), or ``None`` when a dim pair is
    neither equal nor broadcastable.
    """
    return broadcast_shapes(tuple(lhs_batch), tuple(rhs_batch), raising=False)


def _operand_reads(
    shape: tuple,
    out_shape: tuple,
    inner: str,
    *,
    kept_axis: int,
    output_axis: int,
    contraction_axis: int,
) -> list[str]:
    """One coordinate per axis of an operand of the contraction.

    The two matrix axes are the contraction and the one the result keeps; which
    is which is the only difference between the two operands. Batch axes are
    right-aligned against the result's, and one the operand broadcasts reads its
    only coordinate rather than the result's.
    """
    rank = len(shape)
    kept_axis %= rank
    contraction_axis %= rank
    shift = len(out_shape) - rank
    reads: list[str] = []
    for axis in range(rank):
        if axis == contraction_axis:
            reads.append(inner)
        elif axis == kept_axis:
            reads.append(f"d{len(out_shape) + output_axis}")
        elif is_one(shape[axis]) and not is_one(out_shape[axis + shift]):
            reads.append("0")
        else:
            reads.append(f"d{axis + shift}")
    return reads


@register_access_relation(MatMul)
def _matmul_access_relation(call: "Call", ctx) -> AccessRelations:
    """Every coordinate of each operand a contraction reaches, read once.

    A product walks the axis it sums, so that axis is a coordinate this Op is
    asked by rather than something existential inside an image, and the result is
    accumulated over it. Reading an operand at the result's own coordinates would
    claim a shape it does not have the moment the summed and kept axes differ in
    extent. Which positions any of these coordinates are is the reader's
    question; the result's own extents follow from the operands.
    """
    lhs = ctx.type_of(call.args[0])
    rhs = ctx.type_of(call.args[1])
    a_m, a_k, b_n, b_k = matmul_axes(call.target)
    batch = _broadcast_batch(lhs.shape[:-2], rhs.shape[:-2])
    if batch is None:
        raise ValueError(
            f"MatMul batches {tuple(lhs.shape[:-2])} against "
            f"{tuple(rhs.shape[:-2])}, which do not broadcast"
        )
    out_shape = (*batch, lhs.shape[a_m], rhs.shape[b_n])
    summed = lhs.shape[a_k]
    rank = len(out_shape)
    dims = ", ".join((*(f"d{index}" for index in range(rank)), "k"))
    inner = "0" if is_one(summed) else "k"
    inputs = []
    for shape, kept_axis, output_axis, contraction_axis in (
        (tuple(lhs.shape), a_m, -2, a_k),
        (tuple(rhs.shape), b_n, -1, b_k),
    ):
        reads = _operand_reads(
            shape,
            out_shape,
            inner,
            kept_axis=kept_axis,
            output_axis=output_axis,
            contraction_axis=contraction_axis,
        )
        inputs.append(
            BoundaryRelation(AffineAccess(isl.map(f"{{ [{dims}] -> [{', '.join(reads)}] }}")))
        )
    accumulates = ", ".join(f"d{index}" for index in range(rank))
    return iterating(
        (*out_shape, summed),
        AccessRelations(
            inputs=tuple(inputs),
            outputs=(
                BoundaryRelation(AffineAccess(isl.map(f"{{ [{dims}] -> [{accumulates}] }}"))),
            ),
        ),
    )


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
    if _broadcast_batch(lhs.shape[:-2], rhs.shape[:-2]) is None:
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

    out_batch = _broadcast_batch(lhs.shape[:-2], rhs.shape[:-2])
    out_shape = shape_from_relation(
        relation, (*out_batch, lhs.shape[a_m], rhs.shape[b_n], lhs.shape[a_k])
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
    layout = shard if shard is not None else (shard_layout_of(lhs.layout) or lhs.layout)
    storage = resolve_anchor_storage(ctx, call, lhs.storage, rhs.storage)
    return TensorType(shape=out_shape, dtype=lhs.dtype, layout=layout, storage=storage)


@register_eval(MatMul)
def _eval_matmul(ctx):
    lhs = ctx.args[0].data
    rhs = ctx.args[1].data
    if ctx.op.a_layout == "KM":
        lhs = lhs.transpose(-1, -2)
    if ctx.op.b_layout == "NK":
        rhs = rhs.transpose(-1, -2)
    out = torch.matmul(lhs, rhs)
    return TensorValue(data=out, type=ctx.result_type)
