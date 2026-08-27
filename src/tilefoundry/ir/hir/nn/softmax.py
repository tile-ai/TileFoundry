from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    iterating,
    normalised_rows,
    register_access_relation,
)


@register_op
class SoftMax(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)


@register_access_relation(SoftMax)
def _softmax_access(call: "Call", ctx) -> AccessRelations:
    """One row normalised per iteration, read whole and written whole.

    Every element of a row needs the row's own maximum and sum before any of it
    can be written, so the axis being normalised is not a coordinate this Op is
    asked by: it is free in the images. An identity here would have a reader
    believe each output element depends on one input element, and tile an axis
    that cannot be tiled.
    """
    x_ty = ctx.type_of(call.args[0])
    logical_x = ctx.type_of(call.args[0])
    authored = call.target.axis
    axis = authored + len(logical_x.shape) if authored < 0 else authored
    rows, names, guards = normalised_rows(x_ty, logical_x, axis)
    domain = ", ".join(f"d{index}" for index in range(len(rows)))
    where = f" : {' and '.join(guards)}" if guards else ""
    row = AffineAccess(isl.map(f"{{ [{domain}] -> [{', '.join(names)}]{where} }}"))
    return iterating(
        rows,
        AccessRelations(
            inputs=(BoundaryRelation(row),),
            outputs=(BoundaryRelation(row),),
        ),
    )


@register_typeinfer(SoftMax)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    reject_partials(ctx, call, "x", x_ty.layout)
    return x_ty


@register_eval(SoftMax)
def _eval_softmax(ctx):

    out = torch.softmax(ctx.args[0].data.float(), dim=ctx.op.axis)
    return TensorValue(data=out.to(to_torch_dtype(ctx.result_type.dtype)), type=ctx.result_type)
