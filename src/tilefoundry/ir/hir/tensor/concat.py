from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Expr, Op
from tilefoundry.ir.core.expr import Call
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._helpers import resolve_anchor_storage
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import DimAdd, simplify_dim
from tilefoundry.visitor_registry import register_typeinfer


@register_op
class Concat(Op):
    """Variadic input op (§2.4). `Call.args` is a
    plain `tuple[Expr, ...]` of rank-equal TensorType Exprs (NOT a TupleType
    Expr). The lone Param entry documents element type."""
    is_variadic: ClassVar[bool] = True

    inputs = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)
def _sum_dim(a: Expr, b: Expr) -> Expr:
    # simplify_dim handles both the all-Constant fold and the
    # symbolic Call construction in one call.
    return simplify_dim(DimAdd, (a, b))

@register_typeinfer(Concat)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    if not call.args:
        ctx.error(call, "Concat requires at least one input")
    types = [ctx.type_of(a) for a in call.args]
    axis = call.target.axis
    base = types[0]
    for t in types[1:]:
        if len(t.shape) != len(base.shape):
            ctx.error(call, "Concat inputs must have matching rank")
        if t.dtype != base.dtype:
            ctx.error(call, f"Concat dtype mismatch: {t.dtype} vs {base.dtype}")
    new_shape = list(base.shape)
    for t in types[1:]:
        new_shape[axis] = _sum_dim(new_shape[axis], t.shape[axis])
    # Concatenation combines operands along an axis; a genuine sharding cannot
    # in general be re-expressed on the concatenated shape, so drop to an
    # unsharded output rather than carry one operand's layout. Re-expressing a
    # concat of sharded inputs is left to a follow-up.
    storage = resolve_anchor_storage(ctx, call, *(t.storage for t in types))
    return TensorType(
        shape=tuple(new_shape), dtype=base.dtype, layout=None, storage=storage
    )


@register_eval(Concat)
def _eval_concat(ctx):

    out = torch.cat([v.data for v in ctx.args], dim=ctx.op.axis)
    return TensorValue(data=out, type=ctx.result_type)
