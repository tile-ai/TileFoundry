from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.shard_layout import Broadcast, ShardLayout
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    identity_access,
    iterating,
    register_access_relation,
)


@register_op(name="repeat_interleave")
class RepeatInterleave(Op):
    """Repeat each element of ``x`` along ``axis`` ``repeats`` times, interleaved.

    Repeat each element of ``x`` along ``axis`` ``repeats`` times,
    interleaved (GQA head expansion). The named axis grows by ``repeats``;
    all other dims are unchanged.
    """

    x = ParamDef(kind="input", pattern=Tensor)
    repeats = ParamDef(kind="attribute", annotation=int)
    axis = ParamDef(kind="attribute", annotation=int)


def _normalize_axis(axis: int, rank: int) -> int:
    return axis if axis >= 0 else axis + rank


@register_typeinfer(RepeatInterleave)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    op = call.target
    shape = list(x_ty.shape)
    ax = _normalize_axis(op.axis, len(shape))
    if not (0 <= ax < len(shape)):
        ctx.error(call, f"RepeatInterleave: axis {op.axis} out of range for rank {len(shape)}")
    shape[ax] = shape[ax] * op.repeats

    new_layout = None
    if isinstance(x_ty.layout, ShardLayout) and any(
        not isinstance(a, Broadcast) for a in x_ty.layout.attrs
    ):
        ctx.error(
            call,
            "RepeatInterleave cannot express a sharded layout; reshard to a "
            "replicated layout first",
        )
    return TensorType(
        shape=tuple(shape),
        dtype=x_ty.dtype,
        layout=new_layout,
        storage=x_ty.storage,
    )


@register_eval(RepeatInterleave)
def _eval_repeat_interleave(ctx):

    out = torch.repeat_interleave(ctx.args[0].data, ctx.op.repeats, dim=ctx.op.axis)
    return TensorValue(data=out, type=ctx.result_type)


@register_access_relation(RepeatInterleave)
def _repeat_interleave_access(call: "Call", ctx) -> AccessRelations:
    """Several result coordinates read one source coordinate, which is read once.

    The pattern is many-to-one and the amount is its image. That three output
    positions depend on one element is the pattern's business; the element still
    crossed the boundary once.
    """
    source = ctx.type_of(call.args[0])
    rank = len(source.shape)
    axis = call.target.axis + rank if call.target.axis < 0 else call.target.axis
    repeats = call.target.repeats
    dims = [f"d{index}" for index in range(rank)]
    reads = list(dims)
    reads[axis] = f"floor(d{axis} / {repeats})" if repeats != 1 else dims[axis]
    domain = ", ".join(dims)
    out_shape = (
        *source.shape[:axis],
        source.shape[axis] * repeats,
        *source.shape[axis + 1 :],
    )
    produced = 1
    for extent in out_shape:
        produced *= extent if isinstance(extent, int) else 1
    return iterating(
        out_shape,
    AccessRelations(
            inputs=(
                BoundaryRelation(AffineAccess(isl.multi_aff(f"{{ [{domain}] -> [{', '.join(reads)}] }}"))),
            ),
            outputs=(BoundaryRelation(identity_access(rank)),),
        ),
    )
