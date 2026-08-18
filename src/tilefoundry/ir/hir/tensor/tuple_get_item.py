from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TupleType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    StorageEffectClaim,
    StorageEffectKind,
    StorageSpan,
    identity_relations,
    register_access_relation,
    static_bytes,
)


@register_op(name="tuple_get_item")
class TupleGetItem(Op):
    """Extract a field of a tuple-typed expression by static index.

    Representing extraction as a Call keeps multi-output consumers in the HIR
    SSA expression model. See [hir §1](docs/spec/hir.md#1-hir-expr-constructs).
    """

    tuple_value = ParamDef(kind="input", pattern=Tensor)
    index = ParamDef(kind="attribute", annotation=int)




def _tuple_get_item_storage(call: "Call", ctx) -> StorageEffectClaim | None:
    """One field is the run of the tuple the earlier fields do not cover."""
    tuple_type = ctx.type_of(call.args[0])
    if not isinstance(tuple_type, TupleType):
        return StorageEffectClaim(StorageEffectKind.FORWARD, (0,))
    sizes = [static_bytes(field) for field in tuple_type.fields]
    if any(size is None for size in sizes):
        return StorageEffectClaim(StorageEffectKind.FORWARD, (0,))
    index = call.target.index
    offset = sum(size for size in sizes[:index] if size is not None)
    return StorageEffectClaim(
        StorageEffectKind.FORWARD, (0,), (StorageSpan(0, offset, sizes[index] or 0),)
    )


register_access_relation(TupleGetItem)(identity_relations(1, _tuple_get_item_storage))


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
