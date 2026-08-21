"""Pure-value whole-slice indexed copy."""

from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.tensor.index_add import _infer_index_write
from tilefoundry.ir.hir.tensor.index_select import _norm_dim
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    BoundaryRelation,
    identity_access,
    iterating,
    logical_coordinates,
    reached_at,
    register_access_relation,
)


@register_op(name="index_copy")
class IndexCopy(Op):
    """Return ``dst`` with ``src`` slices copied to ``index`` along ``dim``."""

    dst = ParamDef(kind="input", pattern=Tensor)
    index = ParamDef(kind="input", pattern=Tensor)
    src = ParamDef(kind="input", pattern=Tensor)
    dim = ParamDef(kind="attribute", annotation=int, default=0)


@register_typeinfer(IndexCopy)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return _infer_index_write(
        call,
        ctx,
        op_name="IndexCopy",
        index_dtypes=(DType.i64,),
    )


@register_eval(IndexCopy)
def _eval_index_copy(ctx):
    dst, index, src = (arg.data for arg in ctx.args)
    dim = _norm_dim(ctx.op.dim, dst.dim())
    return TensorValue(
        data=dst.clone().index_copy_(dim, index, src),
        type=ctx.result_type,
    )


__all__ = ["IndexCopy"]


@register_access_relation(IndexCopy)
def _index_copy_access(call: "Call", ctx) -> AccessRelations:
    """The rows the index names are replaced; the container around them is kept.

    Two questions, and the index answers only one. Which rows are reached its
    values decide, so those boundaries cover every row the axis could legally
    name; no relation here holds the deciding element. Where the container lives
    it answers not at all: every coordinate of the destination is reached, being
    either a row this replaces or one it keeps. The payload's own row is `i`
    where the destination's is `index[i]`, so from here it too is a row nobody
    named, and reaching all of them is the payload.
    """
    dst = ctx.type_of(call.args[0])
    index = ctx.type_of(call.args[1])
    src = ctx.type_of(call.args[2])
    logical_dst = ctx.type_of(call.args[0])
    rank = len(dst.shape)
    dim = call.target.dim + rank if call.target.dim < 0 else call.target.dim
    identity = identity_access(rank)
    carried = logical_coordinates(dst, logical_dst)
    rows = reached_at(rank, dst, logical_dst, carried, free=(dim,))
    named = reached_at(rank, index, ctx.type_of(call.args[1]), {}, free=(0,))
    payload = reached_at(rank, src, ctx.type_of(call.args[2]), carried, free=(dim,))
    return iterating(
        dst.shape,
    AccessRelations(
            inputs=(
                BoundaryRelation(identity),
                BoundaryRelation(named),
                BoundaryRelation(payload),
            ),
            outputs=(
                BoundaryRelation(rows),
            ),
        ),
    )
