"""HIR ``TupleGetItem`` Op: extract one field of a tuple value."""

from __future__ import annotations

import isl

from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types, utils


@expr.register_op(name="tuple_get_item", dialect='tf', category='tensor')
class TupleGetItem(expr.Op):
    """Extract a field of a tuple-typed expression by scalar index.

    Representing extraction as a Call keeps multi-output consumers in the HIR
    SSA expression model. See [hir §1](docs/spec/hir.md#1-hir-expr-constructs).
    """

    tuple_value = expr.ParamDef(kind="input")
    index = expr.ParamDef(kind="input", pattern=pattern.is_scalar_tensor())


@access.register_relation(TupleGetItem)
def access_relations(call: "expr.Call", ctx: "AccessContext") -> tuple[access.Relation, ...]:
    held = ctx.type_of(call.args[0])
    if not isinstance(held, types.TupleType):
        raise ValueError("TupleGetItem access requires a TupleType operand")
    index = call.args[1]
    if isinstance(index, expr.Constant):
        taken = index.value
        if (
            not isinstance(taken, int)
            or isinstance(taken, bool)
            or not 0 <= taken < len(held.fields)
        ):
            raise ValueError(f"TupleGetItem index {taken!r} out of range")
        result = held.fields[taken]
        begin, count = utils.leaf_span(held, taken)
    else:
        if not held.fields:
            raise ValueError("TupleGetItem dynamic access requires a non-empty TupleType")
        result = held.fields[0]
        begin, count = 0, len(utils.leaves_of(held))
    walks = getattr(result, "shape", ()) or ()
    rank = len(walks)
    coordinates = ", ".join(f"d{axis}" for axis in range(rank))
    reads = access.Relation(
        isl.map(f"{{ [{coordinates}] -> [l] : {begin} <= l < {begin + count} }}")
    )
    return access.iterating(
        walks,
        (reads, access.control_read(rank, ctx, index), access.identity(rank)),
    )


@typeinfer.register(TupleGetItem)
def _(call: "expr.Call", ctx: "TypeInferContext"):
    tup_ty = ctx.type_of(call.args[0])
    if not isinstance(tup_ty, types.TupleType):
        ctx.error(call, "TupleGetItem on non-TupleType")
    index = call.args[1]
    if isinstance(index, expr.Constant):
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


@evaluator.register(TupleGetItem)
def eval_tuple_get_item(ctx):
    index = ctx.args[1]
    if not isinstance(index, evaluator.TensorValue) or index.data.numel() != 1:
        raise evaluator.EvalError("evaluator: TupleGetItem index is a single integer")
    taken = int(index.data.reshape(-1)[0].item())
    if not 0 <= taken < len(ctx.args[0].elements):
        raise evaluator.EvalError(f"evaluator: TupleGetItem index {taken} out of range")
    return ctx.args[0].elements[taken]


__all__ = ["TupleGetItem"]


@cost.register(TupleGetItem)
def tuple_get_item_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    return cost.Cost({}, cost.idle(call))
