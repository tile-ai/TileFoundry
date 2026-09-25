from __future__ import annotations

import isl

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError, TensorValue
from tilefoundry.ir.core import Constant, Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import Scalar, Tensor
from tilefoundry.ir.types import TupleType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    control_read,
    identity_access,
    iterating,
    leaf_span,
    leaves_of,
    register_access_relation,
)


@register_op(name="tuple_get_item")
class TupleGetItem(Op):
    """Extract a field of a tuple-typed expression by scalar index.

    Representing extraction as a Call keeps multi-output consumers in the HIR
    SSA expression model. See [hir §1](docs/spec/hir.md#1-hir-expr-constructs).
    """

    tuple_value = ParamDef(kind="input", pattern=Tensor)
    index = ParamDef(kind="input", pattern=Scalar)


@register_access_relation(TupleGetItem)
def _access_relations(call: "Call", ctx: "AccessContext") -> AccessRelations:
    held = ctx.type_of(call.args[0])
    if not isinstance(held, TupleType):
        raise ValueError("TupleGetItem access requires a TupleType operand")
    index = call.args[1]
    if isinstance(index, Constant):
        taken = index.value
        if (
            not isinstance(taken, int)
            or isinstance(taken, bool)
            or not 0 <= taken < len(held.fields)
        ):
            raise ValueError(f"TupleGetItem index {taken!r} out of range")
        result = held.fields[taken]
        begin, count = leaf_span(held, taken)
    else:
        if not held.fields:
            raise ValueError("TupleGetItem dynamic access requires a non-empty TupleType")
        result = held.fields[0]
        begin, count = 0, len(leaves_of(held))
    walks = getattr(result, "shape", ()) or ()
    rank = len(walks)
    coordinates = ", ".join(f"d{axis}" for axis in range(rank))
    reads = AffineAccess(isl.map(f"{{ [{coordinates}] -> [l] : {begin} <= l < {begin + count} }}"))
    return iterating(
        walks,
        AccessRelations(
            inputs=(
                BoundaryRelation(reads),
                BoundaryRelation(control_read(rank, ctx, index)),
            ),
            outputs=(BoundaryRelation(identity_access(rank)),),
        ),
    )


@register_typeinfer(TupleGetItem)
def _(call: "Call", ctx: "TypeInferContext"):
    tup_ty = ctx.type_of(call.args[0])
    if not isinstance(tup_ty, TupleType):
        ctx.error(call, "TupleGetItem on non-TupleType")
    index = call.args[1]
    if isinstance(index, Constant):
        taken = index.value
        if (
            not isinstance(taken, int)
            or isinstance(taken, bool)
            or taken < 0
            or taken >= len(tup_ty.fields)
        ):
            ctx.error(call, f"TupleGetItem index {taken!r} out of range")
        return tup_ty.fields[taken]
    if not tup_ty.fields:
        ctx.error(call, "TupleGetItem dynamic index requires a non-empty TupleType")
    first = tup_ty.fields[0]
    if any(field != first for field in tup_ty.fields[1:]):
        ctx.error(call, "TupleGetItem dynamic index requires homogeneous fields")
    return first


@register_eval(TupleGetItem)
def _eval_tuple_get_item(ctx):
    index = ctx.args[1]
    if not isinstance(index, TensorValue) or index.data.numel() != 1:
        raise EvalError("evaluator: TupleGetItem index is a single integer")
    taken = int(index.data.reshape(-1)[0].item())
    if not 0 <= taken < len(ctx.args[0].elements):
        raise EvalError(f"evaluator: TupleGetItem index {taken} out of range")
    return ctx.args[0].elements[taken]


__all__ = ["TupleGetItem"]
