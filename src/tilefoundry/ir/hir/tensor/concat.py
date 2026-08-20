from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Expr, Op
from tilefoundry.ir.core.expr import Call
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._helpers import resolve_anchor_storage
from tilefoundry.ir.hir._shard_checks import (
    reject_dynamic_shards,
    require_compatible_meshes,
    require_uniform_partial_slices,
)
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import DimAdd, simplify_dim
from tilefoundry.ir.types.dim_isl import normalize_dim_entries
from tilefoundry.ir.types.shard import (
    Layout,
    Split,
    shard_layout_of,
    try_c_order_strides,
)
from tilefoundry.ir.types.shard.shard_layout import split_target_axes
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    coordinates_of,
    iterating,
    register_access_relation,
)
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout


@register_op
class Concat(Op):
    """Variadic input op.

    Variadic input op. `Call.args` is a
    plain `tuple[Expr, ...]` of rank-equal TensorType Exprs (NOT a TupleType
    Expr). The lone Param entry documents element type.
    """

    is_variadic: ClassVar[bool] = True

    inputs = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)


def _sum_dim(a: Expr, b: Expr) -> Expr:

    return simplify_dim(DimAdd, (a, b))


def _axis(call: "Call", ctx: "TypeInferContext", rank: int) -> int:
    raw_axis = call.target.axis
    axis = raw_axis + rank if raw_axis < 0 else raw_axis
    if not (0 <= axis < rank):
        ctx.error(call, f"axis {raw_axis} out of range for input rank {rank}")
    return axis




@register_access_relation(Concat)
def _concat_access(call: "Call", ctx) -> AccessRelations:
    """Each input reads its own segment of the result; the result is read whole.

    The segments are the same arithmetic the forward relation states, so the two
    cannot drift: one offset walk, one guard per input.
    """
    types = [ctx.type_of(arg) for arg in call.args]
    rank = len(types[0].shape)
    axis = _axis(call, ctx, rank)
    extents = tuple(type_.shape[axis] for type_ in types)
    if any(not isinstance(extent, int) or isinstance(extent, bool) for extent in extents):
        raise NotImplementedError(
            f"Concat access_relation: concat-axis extents must be static ints, got {extents}"
        )
    dims = [f"d{index}" for index in range(rank)]
    domain_text = ", ".join(dims)
    inputs, offset = [], 0
    for extent in extents:
        reads = list(dims)
        if offset:
            reads[axis] = f"d{axis} - {offset}"
        inputs.append(
            AffineAccess(
                isl.map(
                    f"{{ [{domain_text}] -> [{', '.join(reads)}] : "
                    f"{offset} <= d{axis} < {offset + extent} }}"
                )
            )
        )
        offset += extent
    out_shape = (
        *types[0].shape[:axis],
        sum(extents),
        *types[0].shape[axis + 1 :],
    )
    return iterating(
        out_shape,
    AccessRelations(
            inputs=tuple(
                BoundaryRelation(item)
                for item, type_ in zip(inputs, types)
            ),
            outputs=(
                BoundaryRelation(AffineAccess(isl.multi_aff(f"{{ [{domain_text}] -> [{domain_text}] }}"))),
            ),
        ),
    )


def _reject_concat_axis_splits(call, ctx, types, axis: int) -> None:
    for index, type_ in enumerate(types):
        layout = shard_layout_of(type_.layout)
        if layout is None:
            continue
        targets = split_target_axes(layout, type_.shape)
        if any(
            isinstance(attr, Split) and targets[mesh_axis] == axis
            for mesh_axis, attr in enumerate(layout.attrs)
        ):
            ctx.error(
                call,
                f"input {index} is Split along concat axis {axis}; use an "
                "explicit Reshard before Concat",
            )


@register_typeinfer(Concat)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    if not call.args:
        ctx.error(call, "Concat requires at least one input")
    types = [ctx.type_of(a) for a in call.args]
    base = types[0]
    axis = _axis(call, ctx, len(base.shape))
    for index, t in enumerate(types[1:], start=1):
        if len(t.shape) != len(base.shape):
            ctx.error(call, f"input {index} rank must match input 0")
        if t.dtype != base.dtype:
            ctx.error(call, f"input {index} dtype must match input 0")
        for dim, (actual, expected) in enumerate(zip(t.shape, base.shape, strict=True)):
            if dim != axis and actual != expected:
                ctx.error(
                    call,
                    f"input {index} shape must match input 0 outside axis {axis}",
                )
    new_shape = list(base.shape)
    for t in types[1:]:
        new_shape[axis] = _sum_dim(new_shape[axis], t.shape[axis])
    new_shape = normalize_dim_entries(tuple(new_shape))

    reject_dynamic_shards(ctx, call, types, "Concat")
    require_compatible_meshes(ctx, call, types, "Concat")
    _reject_concat_axis_splits(call, ctx, types, axis)
    try:
        relation = coordinates_of(call, ctx)
        layout = derive_output_shard_layout(tuple(types), relation, new_shape, fresh_strides=True)
    except ValueError as error:
        ctx.error(
            call,
            f"cannot derive input ownership: {error}; use an explicit Reshard before Concat",
        )
    if layout is not None:
        require_uniform_partial_slices(ctx, call, types, layout, "Concat")
        if getattr(layout.layout, "strides", None) is None:
            first = next(
                index
                for index, type_ in enumerate(types)
                if shard_layout_of(type_.layout) is not None
            )
            ctx.error(
                call,
                f"input {first} ownership produces an unrepresentable result "
                "layout; use an explicit Reshard before Concat",
            )
    else:
        layout = Layout(shape=new_shape, strides=try_c_order_strides(new_shape))
    storage = resolve_anchor_storage(ctx, call, *(t.storage for t in types))
    return TensorType(shape=new_shape, dtype=base.dtype, layout=layout, storage=storage)


@register_eval(Concat)
def _eval_concat(ctx):

    out = torch.cat([v.data for v in ctx.args], dim=ctx.op.axis)
    return TensorValue(data=out, type=ctx.result_type)
