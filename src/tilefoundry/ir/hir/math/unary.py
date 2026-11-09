"""HIR generic value-form Unary Op (kind-tagged dispatch).

``Unary(UnaryKind.NEG, x)`` is the IR-level form behind the DSL
sugar names (``neg`` / ``abs`` / ``rsqrt`` / ``logical_not``).
"""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils


@expr.register_op(dialect="tf", category="math", aliases={
    "neg": {"kind": expr.UnaryKind.NEG},
    "abs": {"kind": expr.UnaryKind.ABS},
    "logical_not": {"kind": expr.UnaryKind.NOT},
    "rsqrt": {"kind": expr.UnaryKind.RSQRT},
    "exp": {"kind": expr.UnaryKind.EXP},
    "log": {"kind": expr.UnaryKind.LOG},
    "square": {"kind": expr.UnaryKind.SQUARE},
    "ceil": {"kind": expr.UnaryKind.CEIL},
    "round": {"kind": expr.UnaryKind.ROUND},
    "exp2": {"kind": expr.UnaryKind.EXP2},
    "log2": {"kind": expr.UnaryKind.LOG2},
})
class Unary(expr.Op):
    """Value-form pointwise unary operation."""

    x = expr.ParamDef(kind="input", pattern=pattern.Tensor())
    kind = expr.ParamDef(kind="attribute", annotation=expr.UnaryKind)


MONOTONE_INCREASING = frozenset(
    {
        expr.UnaryKind.EXP,
        expr.UnaryKind.LOG,
        expr.UnaryKind.RELU,
        expr.UnaryKind.CEIL,
        expr.UnaryKind.ROUND,
        expr.UnaryKind.EXP2,
        expr.UnaryKind.LOG2,
    }
)

LINEAR = frozenset({expr.UnaryKind.NEG})


@typeinfer.register(Unary)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    op = call.target
    if not isinstance(op.kind, expr.UnaryKind):
        ctx.error(call, f"Unary: kind must be UnaryKind, got {type(op.kind)}")
    x_ty = ctx.type_of(call.args[0])
    if op.kind is expr.UnaryKind.NOT and x_ty.dtype != expr.DType.bool:
        ctx.error(call, "Unary NOT: operand must be bool")
    if op.kind in MONOTONE_INCREASING:
        commutes_with = frozenset({"max", "min"})
    elif op.kind in LINEAR:
        commutes_with = frozenset({"sum"})
    else:
        commutes_with = frozenset()
    ir_utils.reject_partials("x", x_ty.layout, commutes_with=commutes_with)
    return types.TensorType(
        shape=x_ty.shape,
        dtype=x_ty.dtype,
        layout=x_ty.layout,
        storage=x_ty.storage,
    )


@evaluator.register(Unary)
def eval_unary(ctx):
    fn = {
        expr.UnaryKind.NEG: utils.torch.neg,
        expr.UnaryKind.ABS: utils.torch.abs,
        expr.UnaryKind.NOT: utils.torch.logical_not,
        expr.UnaryKind.RELU: utils.torch.relu,
        expr.UnaryKind.SQUARE: utils.torch.square,
        expr.UnaryKind.RSQRT: utils.torch.rsqrt,
        expr.UnaryKind.EXP: utils.torch.exp,
        expr.UnaryKind.LOG: utils.torch.log,
        expr.UnaryKind.CEIL: utils.torch.ceil,
        expr.UnaryKind.ROUND: utils.torch.round,
        expr.UnaryKind.EXP2: utils.torch.exp2,
        expr.UnaryKind.LOG2: utils.torch.log2,
    }[ctx.op.kind]
    x = ctx.args[0].tensor
    held = fn(x.tensor) if x.is_global else tuple((p, fn(t)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


__all__ = ["Unary"]


access.register_relation(Unary)(access.identity_relations)


SPECIAL = frozenset(
    {expr.UnaryKind.RSQRT, expr.UnaryKind.EXP, expr.UnaryKind.LOG, expr.UnaryKind.EXP2, expr.UnaryKind.LOG2}
)


@cost.register(Unary)
def unary_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    """An exponential is not a multiply, a negated truth is not arithmetic.

    ``_SPECIAL`` holds the kinds the machine answers on its
    special-function unit, at a rate of its own -- a quarter of the scalar one
    here. Counting one of those as a single FLOP would put it on the float pipe
    at four times the throughput the unit has, so it is a service rather than
    arithmetic. What is left for ``flops`` is the arithmetic that really is a
    multiply or an add.
    """
    kind = call.target.kind
    if kind in SPECIAL:
        return cost.serviced(call, ctx, "special")
    if kind is expr.UnaryKind.NOT:
        return cost.serviced(call, ctx, "predicate")
    return cost.elementwise(call, ctx)
