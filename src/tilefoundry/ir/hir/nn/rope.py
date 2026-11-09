"""Rotary Position Embedding (RoPE) HIR primitive.

SGLang baseline kernel K04. Applies position-dependent rotation to Q and K
along the last (head_dim) axis, using precomputed cos/sin caches indexed by
``pos_ids``.


Multi-output op: returns a tuple ``(q_rope, k_rope)``. Both share input shape /
dtype / layout / storage with their respective Q / K input.
"""

from __future__ import annotations

import isl

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import dim, expr, pattern, types
from tilefoundry.ir import utils as ir_utils


@expr.register_op(dialect='tf', category='nn')
class RoPE(expr.Op):
    """Rotary position embedding on Q and K. ``head_dim`` must be even."""

    q = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    k = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    cos_cache = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    sin_cache = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    pos_ids = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())


@typeinfer.register(RoPE)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TupleType:
    q_ty = ctx.type_of(call.args[0])
    k_ty = ctx.type_of(call.args[1])
    cos_ty = ctx.type_of(call.args[2])
    sin_ty = ctx.type_of(call.args[3])
    pos_ty = ctx.type_of(call.args[4])
    if not q_ty.shape or not k_ty.shape:
        ctx.error(call, "q and k must be at least rank-1")
    if len(q_ty.shape) != len(k_ty.shape):
        ctx.error(call, f"q rank {len(q_ty.shape)} != k rank {len(k_ty.shape)}")
    head_dim_q = q_ty.shape[-1]
    head_dim_k = k_ty.shape[-1]
    if isinstance(head_dim_q, int) and head_dim_q % 2 != 0:
        ctx.error(call, f"q head_dim {head_dim_q} must be even")
    if isinstance(head_dim_k, int) and head_dim_k % 2 != 0:
        ctx.error(call, f"k head_dim {head_dim_k} must be even")
    if isinstance(head_dim_q, int) and isinstance(head_dim_k, int) and head_dim_q != head_dim_k:
        ctx.error(call, f"q head_dim {head_dim_q} != k head_dim {head_dim_k}")

    ir_utils.check_multilinear_partials(
        (("q", q_ty), ("cos_cache", cos_ty), ("sin_cache", sin_ty)),
        ctx.target,
        kind=expr.BinaryKind.MUL,
        anchor="q",
    )
    ir_utils.check_multilinear_partials(
        (("k", k_ty), ("cos_cache", cos_ty), ("sin_cache", sin_ty)),
        ctx.target,
        kind=expr.BinaryKind.MUL,
        anchor="k",
    )

    ir_utils.reject_partials("pos_ids", pos_ty.layout)
    return types.TupleType(fields=(q_ty, k_ty))


@access.register_relation(RoPE)
def rope_access_relation(call: "expr.Call", ctx: "TypeInferContext") -> tuple[access.Relation, ...]:
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
    carried = access.logical_coordinates(q_ty, logical_q)
    walked = ", ".join((*(f"d{index}" for index in range(rank)), f"d{rank}"))
    own = ", ".join(f"d{index}" for index in range(rank))
    value = isl.map(f"{{ [{walked}] -> [{own}] : d{rank} = 0 }}")
    grouped = isl.map(f"{{ [{walked}] -> [{own}] : d{rank} = 1 }}")
    if expr.is_literal_shape(k_ty.shape):
        grouped = grouped.intersect_range(dim.shape_to_isl_set(tuple(k_ty.shape), {}))
    value, grouped = access.Relation(value), access.Relation(grouped)
    positions = ctx.type_of(call.args[4])
    tables = []
    for operand in (2, 3):
        table = ctx.type_of(call.args[operand])
        logical_table = ctx.type_of(call.args[operand])
        rows = len(logical_table.shape) - 1
        tables.append(
            access.reached_at(
                rank + 1,
                table,
                logical_table,
                {rows: carried.get(head_dim, "0")},
                free=tuple(range(rows)),
            )
        )
    return access.iterating(
        (*q_ty.shape, 2),
        (
            value,
            grouped,
            *tables,
            access.reached_at(
                rank + 1,
                positions,
                ctx.type_of(call.args[4]),
                {},
                free=tuple(range(len(ctx.type_of(call.args[4]).shape))),
            ),
            value,
            grouped,
        ),
    )


def rotate_half(x):
    half = x.shape[-1] // 2
    return utils.torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@evaluator.register(RoPE)
def eval_rope(ctx):
    fields = ctx.result_type.fields

    def rope(q, k, cos, sin, pos):
        pos = pos.reshape(-1).long()
        cos = cos[pos].float()[None, :, None, :]
        sin = sin[pos].float()[None, :, None, :]
        q, k = q.float(), k.float()
        return tuple(
            (x * cos + rotate_half(x) * sin).to(evaluator.to_torch_dtype(field.dtype))
            for x, field in zip((q, k), fields, strict=True)
        )

    xs = tuple(arg.tensor for arg in ctx.args)
    placements = evaluator.common_placements(*xs)
    if not placements:
        held = rope(*(x.tensor for x in xs))
    else:
        each = [(p, rope(*(x.to_local(p) for x in xs))) for p in placements]
        held = tuple(tuple((p, out[i]) for p, out in each) for i in range(len(fields)))
    return evaluator.TupleValue(
        tuple(evaluator.TensorValue(core.ShardTensor(field, data)) for field, data in zip(fields, held, strict=True))
    )


__all__ = ["RoPE"]


@cost.register(RoPE)
def rope_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    """Rotate q and k in place against the caches.

    Each rotated element costs one multiply against the cosine, one against
    the sine of its partner, and the add that combines them. The output is the
    pair, so its element count already covers both q and k.

    The position-indexed caches contribute one row per call.
    """
    query, key, cos_cache, sin_cache, pos_ids = cost.input_types(call, ctx)
    output = cost.output_type(call, ctx)
    if not isinstance(query, types.TensorType):
        raise ValueError("RoPE cost requires a tensor query input")
    return cost.Cost(
        {query.dtype: 3 * ir_utils.numel(output)},
        (
            cost.TrafficBytes(read=ir_utils.tensor_bytes(query)),
            cost.TrafficBytes(read=ir_utils.tensor_bytes(key)),
            cost.TrafficBytes(read=cost.row(cos_cache)),
            cost.TrafficBytes(read=cost.row(sin_cache)),
            cost.TrafficBytes(read=ir_utils.tensor_bytes(pos_ids)),
            cost.TrafficBytes(write=ir_utils.tensor_bytes(output)),
        ),
    )
