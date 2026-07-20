from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TupleType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)


@register_op(name="tuple_get_item")
class TupleGetItem(Op):
    """Extracts a field of a tuple-typed Expr by static index.

    Replaces the former `core_ir.expr.TupleGetItem` Expr subclass. Using Call
    + Op here unifies the shape of multi-output op consumers (§8.6 hir = SSA
    DAG; only Call and leaves).
    """
    tuple_value = ParamDef(kind="input", pattern=Tensor)
    index = ParamDef(kind="attribute", annotation=int)


# GLOBAL-level: structural extractor, identity over the extracted field's
# own rank (`tuple_value` is a TupleType with no shape of its own).
register_access_relation(TupleGetItem)(identity_relations(1))
@register_typeinfer(TupleGetItem)
def _(call: "Call", ctx: "TypeInferContext"):
    tup_ty = ctx.type_of(call.args[0])
    if not isinstance(tup_ty, TupleType):
        ctx.error(call, "TupleGetItem on non-TupleType")
    idx = call.target.index
    if idx < 0 or idx >= len(tup_ty.fields):
        ctx.error(call, f"TupleGetItem index {idx} out of range")
    return tup_ty.fields[idx]


@register_eval(TupleGetItem)
def _eval_tuple_get_item(ctx):
    return ctx.args[0].elements[ctx.op.index]


__all__ = ["TupleGetItem"]
