"""Rotary Position Embedding (RoPE) HIR primitive.

SGLang baseline kernel K04. Applies position-dependent rotation to Q and K
along the last (head_dim) axis, using precomputed cos/sin caches indexed by
``pos_ids``.


Multi-output op: returns a tuple ``(q_rope, k_rope)``. Both share input shape /
dtype / layout / storage with their respective Q / K input.
"""

from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, TupleValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import check_multilinear_partials, reject_partials
from tilefoundry.ir.types import TupleType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    index_set,
    iterating,
    logical_coordinates,
    reached_at,
    register_access_relation,
)


@register_op
class RoPE(Op):
    """Rotary position embedding on Q and K. ``head_dim`` must be even."""

    q = ParamDef(kind="input", pattern=Tensor)
    k = ParamDef(kind="input", pattern=Tensor)
    cos_cache = ParamDef(kind="input", pattern=Tensor)
    sin_cache = ParamDef(kind="input", pattern=Tensor)
    pos_ids = ParamDef(kind="input", pattern=Tensor)


@register_typeinfer(RoPE)
def _(call: "Call", ctx: "TypeInferContext") -> TupleType:
    q_ty = ctx.type_of(call.args[0])
    k_ty = ctx.type_of(call.args[1])
    cos_ty = ctx.type_of(call.args[2])
    sin_ty = ctx.type_of(call.args[3])
    pos_ty = ctx.type_of(call.args[4])
    if not q_ty.shape or not k_ty.shape:
        ctx.error(call, "q and k must be at least rank-1")
    head_dim_q = q_ty.shape[-1]
    head_dim_k = k_ty.shape[-1]
    if isinstance(head_dim_q, int) and head_dim_q % 2 != 0:
        ctx.error(call, f"q head_dim {head_dim_q} must be even")
    if isinstance(head_dim_k, int) and head_dim_k % 2 != 0:
        ctx.error(call, f"k head_dim {head_dim_k} must be even")
    if isinstance(head_dim_q, int) and isinstance(head_dim_k, int) and head_dim_q != head_dim_k:
        ctx.error(call, f"q head_dim {head_dim_q} != k head_dim {head_dim_k}")

    check_multilinear_partials(
        ctx,
        call,
        (("q", q_ty), ("cos_cache", cos_ty), ("sin_cache", sin_ty)),
        anchor="q",
    )
    check_multilinear_partials(
        ctx,
        call,
        (("k", k_ty), ("cos_cache", cos_ty), ("sin_cache", sin_ty)),
        anchor="k",
    )

    reject_partials(ctx, call, "pos_ids", pos_ty.layout)
    return TupleType(fields=(q_ty, k_ty))


@register_access_relation(RoPE)
def _rope_access_relation(call: "Call", ctx: "TypeInferContext") -> AccessRelations:
    """GLOBAL level: a rotation per element, read out of a table by position.

    Rotating Q and rotating K are instances of the same work, so the space this
    Op walks says which: one coordinate names the value rotated, Q's boundaries
    answer where it is Q and K's where it is K, which keeps them apart even at
    equal width. Grouped-query K holds fewer heads and answers on that much. A
    table carries head_dim last and rows before it, so it is read at the rotated
    value's head_dim coordinate and at any row those axes could name -- the row
    is an element of `pos_ids`, which nothing here holds, read whole by both.
    """
    q_ty, k_ty = ctx.type_of(call.args[0]), ctx.type_of(call.args[1])
    logical_q = ctx.type_of(call.args[0])
    rank = len(q_ty.shape)
    head_dim = len(logical_q.shape) - 1
    carried = logical_coordinates(q_ty, logical_q)
    walked = ", ".join((*(f"d{index}" for index in range(rank)), f"d{rank}"))
    own = ", ".join(f"d{index}" for index in range(rank))
    value = isl.map(f"{{ [{walked}] -> [{own}] : d{rank} = 0 }}")
    grouped = isl.map(f"{{ [{walked}] -> [{own}] : d{rank} = 1 }}")
    narrower = index_set(tuple(k_ty.shape))
    if narrower is not None:
        grouped = grouped.intersect_range(narrower)
    value, grouped = AffineAccess(value), AffineAccess(grouped)
    positions = ctx.type_of(call.args[4])
    tables = []
    for operand in (2, 3):
        table = ctx.type_of(call.args[operand])
        logical_table = ctx.type_of(call.args[operand])
        rows = len(logical_table.shape) - 1
        tables.append(
            BoundaryRelation(reached_at(
                    rank + 1,
                    table,
                    logical_table,
                    {rows: carried.get(head_dim, "0")},
                    free=tuple(range(rows)),
                ))
        )
    return iterating(
        (*q_ty.shape, 2),
        AccessRelations(
            inputs=(
                BoundaryRelation(value),
                BoundaryRelation(grouped),
                *tables,
                BoundaryRelation(reached_at(
                        rank + 1,
                        positions,
                        ctx.type_of(call.args[4]),
                        {},
                        free=tuple(range(len(ctx.type_of(call.args[4]).shape))),
                    )),
            ),
            outputs=(
                BoundaryRelation(value),
                BoundaryRelation(grouped),
            ),
        ),
    )


@register_eval(RoPE)
def _eval_rope(ctx):

    q = ctx.args[0].data.float()
    k = ctx.args[1].data.float()
    pos = ctx.args[4].data.reshape(-1).long()
    cos = ctx.args[2].data[pos].float()[None, :, None, :]
    sin = ctx.args[3].data[pos].float()[None, :, None, :]

    def _rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return TupleValue(
        elements=(
            TensorValue(
                data=q_out.to(to_torch_dtype(ctx.result_type.fields[0].dtype)),
                type=ctx.result_type.fields[0],
            ),
            TensorValue(
                data=k_out.to(to_torch_dtype(ctx.result_type.fields[1].dtype)),
                type=ctx.result_type.fields[1],
            ),
        )
    )


__all__ = ["RoPE"]
