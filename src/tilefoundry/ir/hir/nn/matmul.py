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
    AccessRelationResult,
    AccessRelations,
    build_relation,
    elements_of,
    factored_image,
    logical_axes_of,
    logical_coordinates,
    moves,
    register_access_relation,
    register_type_relation,
    writes,
)
from tilefoundry.visitor_registry.isl_utility import to_domain
from tilefoundry.visitor_registry.relation_build import identity_access, shape_from_relation
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
    local, logical, carried: dict, out_logical, inner: str, *, contracts_last: bool
) -> list[str]:
    """One expression per logical axis of an operand of the contraction.

    The two matrix axes are the contraction and the one the result keeps; which
    is which is the only difference between the two operands. Batch axes are
    right-aligned against the result's, and one the operand broadcasts reads its
    only coordinate rather than the result's.
    """
    rank = len(logical.shape)
    shift = len(out_logical.shape) - rank
    reads: list[str] = []
    for axis in range(rank):
        if axis == rank - 1:
            reads.append(inner if contracts_last else carried.get(len(out_logical.shape) - 1, "0"))
        elif axis == rank - 2:
            reads.append(carried.get(len(out_logical.shape) - 2, "0") if contracts_last else inner)
        elif is_one(logical.shape[axis]) and not is_one(out_logical.shape[axis + shift]):
            reads.append("0")
        else:
            reads.append(carried.get(axis + shift, "0"))
    return reads


@register_access_relation(MatMul)
def _matmul_access_relation(call: "Call", ctx) -> AccessRelations:
    """Every coordinate of each operand a contraction reaches, read once.

    Reading each operand at the result's own coordinates claims a shape it does
    not have the moment the contracted and kept axes differ in extent: for
    `(M,K)` by `(K,N)` it gives the left operand N columns. The contracted
    extent is this participant's own, that axis being one a layout can split.
    """
    lhs_ty = ctx.local_type_of(call.args[0])
    rhs_ty = ctx.local_type_of(call.args[1])
    out_ty = ctx.local_type_of(call)
    logical_lhs = ctx.type_of(call.args[0])
    logical_rhs = ctx.type_of(call.args[1])
    logical_out = ctx.type_of(call)
    rank = len(out_ty.shape)
    carried = logical_coordinates(out_ty, logical_out)

    inputs = []
    for local, logical, contracts_last in (
        (lhs_ty, logical_lhs, True),
        (rhs_ty, logical_rhs, False),
    ):
        held = _held_axis(
            local, logical, len(logical.shape) - (1 if contracts_last else 2)
        )
        inner = "0" if held == 1 else "k"
        reads = _operand_reads(
            local, logical, carried, logical_out, inner, contracts_last=contracts_last
        )
        image = ", ".join(factored_image(reads, local, logical))
        dims = ", ".join(f"d{index}" for index in range(rank))
        guard = "" if held == 1 else f" : 0 <= k < {held}"
        inputs.append(
            moves(isl.map(f"{{ [{dims}] -> [{image}]{guard} }}"), elements_of(local))
        )
    return AccessRelations(
        inputs=tuple(inputs),
        outputs=(writes(identity_access(rank), elements_of(out_ty)),),
    )


@register_type_relation(MatMul)
def _matmul_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Forward access relation for ``(batch.., M, K) × (batch.., K, N)``.

    Iteration domain is ``[batch.., M, N, K]``; the output map drops K (the
    reduced contraction dim). A batch dim that this operand broadcasts (its
    extent is 1 while the output extent is larger) accesses a constant 0 rather
    than the iteration dim, so shard propagation treats it as a broadcast.
    """
    lhs, rhs = input_types
    lhs_batch = lhs.shape[:-2]
    rhs_batch = rhs.shape[:-2]
    out_batch = _broadcast_batch(lhs_batch, rhs_batch)
    b = len(out_batch)
    m, k, n = lhs.shape[-2], lhs.shape[-1], rhs.shape[-1]
    domain, param_map = to_domain((*out_batch, m, n, k))

    m_d, n_d, k_d = b, b + 1, b + 2
    in_dims = [f"d{i}" for i in range(b + 3)]

    def batch_access(in_batch):

        pad = b - len(in_batch)
        return [
            "0" if (is_one(in_batch[j]) and not is_one(out_batch[pad + j])) else f"d{pad + j}"
            for j in range(len(in_batch))
        ]

    lhs_out = batch_access(lhs_batch) + [f"d{m_d}", f"d{k_d}"]
    rhs_out = batch_access(rhs_batch) + [f"d{k_d}", f"d{n_d}"]
    out_out = [f"d{j}" for j in range(b)] + [f"d{m_d}", f"d{n_d}"]
    src = "[" + ", ".join(in_dims) + "]"
    maps = tuple(
        isl.map(f"{{ {src} -> [{', '.join(dst)}] }}") for dst in (lhs_out, rhs_out, out_out)
    )
    return AccessRelationResult(domain=domain, maps=maps, param_map=param_map)


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

    relation = build_relation(call, (lhs, rhs), ctx)

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
