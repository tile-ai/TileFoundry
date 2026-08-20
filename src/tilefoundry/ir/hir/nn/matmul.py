from __future__ import annotations

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
    coordinates_of,
    iterating,
    logical_axes_of,
    register_access_relation,
    shape_from_relation,
)
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout


@register_op
class MatMul(Op):
    lhs = ParamDef(kind="input", pattern=Tensor)
    rhs = ParamDef(kind="input", pattern=Tensor)


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


def _held_axis(local, logical, axis: int) -> int:
    """How much of one logical axis this participant holds."""
    held = 1
    for position, owner in enumerate(logical_axes_of(local, logical)):
        if owner == axis:
            held *= local.shape[position]
    return held


def _operand_reads(
    shape: tuple, out_shape: tuple, inner: str, *, contracts_last: bool
) -> list[str]:
    """One coordinate per axis of an operand of the contraction.

    The two matrix axes are the contraction and the one the result keeps; which
    is which is the only difference between the two operands. Batch axes are
    right-aligned against the result's, and one the operand broadcasts reads its
    only coordinate rather than the result's.
    """
    rank = len(shape)
    shift = len(out_shape) - rank
    reads: list[str] = []
    for axis in range(rank):
        if axis == rank - 1:
            reads.append(inner if contracts_last else f"d{len(out_shape) - 1}")
        elif axis == rank - 2:
            reads.append(f"d{len(out_shape) - 2}" if contracts_last else inner)
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
    batch = _broadcast_batch(lhs.shape[:-2], rhs.shape[:-2])
    if batch is None:
        raise ValueError(
            f"MatMul batches {tuple(lhs.shape[:-2])} against "
            f"{tuple(rhs.shape[:-2])}, which do not broadcast"
        )
    out_shape = (*batch, lhs.shape[-2], rhs.shape[-1])
    summed = lhs.shape[-1]
    rank = len(out_shape)
    dims = ", ".join((*(f"d{index}" for index in range(rank)), "k"))
    inner = "0" if is_one(summed) else "k"
    inputs = []
    for shape, contracts_last, held in (
        (tuple(lhs.shape), True, lhs.shape[-1]),
        (tuple(rhs.shape), False, rhs.shape[-2]),
    ):
        reads = _operand_reads(shape, out_shape, inner, contracts_last=contracts_last)
        inputs.append(
            BoundaryRelation(AffineAccess(isl.map(f"{{ [{dims}] -> [{', '.join(reads)}] }}")))
        )
        del held
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
    if lhs.dtype != rhs.dtype:
        ctx.error(call, f"MatMul dtype mismatch: {lhs.dtype} vs {rhs.dtype}")
    if len(lhs.shape) < 2 or len(rhs.shape) < 2:
        ctx.error(call, "MatMul requires rank >= 2 on both operands")
    if _broadcast_batch(lhs.shape[:-2], rhs.shape[:-2]) is None:
        ctx.error(call, f"MatMul batch-dim mismatch {lhs.shape[:-2]} vs {rhs.shape[:-2]}")
    if lhs.shape[-1] != rhs.shape[-2]:
        ctx.error(
            call,
            f"MatMul contraction-dim mismatch: lhs K={lhs.shape[-1]} vs rhs K={rhs.shape[-2]}",
        )

    if _k_split_axes(lhs, len(lhs.shape) - 1) != _k_split_axes(rhs, len(rhs.shape) - 2):
        ctx.error(
            call,
            "MatMul contraction dim K must be split on the same mesh axes for both operands",
        )

    check_multilinear_partials(ctx, call, (("lhs", lhs), ("rhs", rhs)))

    relation = coordinates_of(call, ctx)

    out_batch = _broadcast_batch(lhs.shape[:-2], rhs.shape[:-2])
    out_shape = shape_from_relation(
        relation, (*out_batch, lhs.shape[-2], rhs.shape[-1], lhs.shape[-1])
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

    out = torch.matmul(ctx.args[0].data, ctx.args[1].data)
    return TensorValue(data=out, type=ctx.result_type)
