"""HIR generic value-form Unary Op (kind-tagged dispatch).

``Unary(UnaryKind.NEG, x)`` is the IR-level form behind the DSL
sugar names (``neg`` / ``abs`` / ``rsqrt`` / ``logical_not``).
"""

from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.kinds import UnaryKind
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.visitor_registry import register_typeinfer


@register_op
class Unary(Op):
    """Value-form pointwise unary operation."""
    x = ParamDef(kind="input", pattern=Tensor)
    kind = ParamDef(kind="attribute", annotation=UnaryKind)

# Monotone non-decreasing: commutes with max/min, not sum.
_MONOTONE_INCREASING = frozenset({
    UnaryKind.EXP, UnaryKind.LOG, UnaryKind.RELU,
    UnaryKind.CEIL, UnaryKind.ROUND, UnaryKind.EXP2, UnaryKind.LOG2,
})
# Linear negation: commutes with sum, not max/min (reverses order).
_LINEAR = frozenset({UnaryKind.NEG})

@register_typeinfer(Unary)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    op = call.target
    if not isinstance(op.kind, UnaryKind):
        ctx.error(call, f"Unary: kind must be UnaryKind, got {type(op.kind)}")
    x_ty = ctx.type_of(call.args[0])
    if op.kind is UnaryKind.NOT and x_ty.dtype != DType.bool:
        ctx.error(call, "Unary NOT: operand must be bool")
    if op.kind in _MONOTONE_INCREASING:
        commutes_with = frozenset({"max", "min"})
    elif op.kind in _LINEAR:
        commutes_with = frozenset({"sum"})
    else:
        commutes_with = frozenset()
    reject_partials(ctx, call, "x", x_ty.layout, commutes_with=commutes_with)
    return TensorType(
        shape=x_ty.shape,
        dtype=x_ty.dtype,
        layout=x_ty.layout,
        storage=x_ty.storage,
    )

@register_eval(Unary)
def _eval_unary(ctx):

    fns = {
        UnaryKind.NEG: torch.neg,
        UnaryKind.ABS: torch.abs,
        UnaryKind.NOT: torch.logical_not,
        UnaryKind.RELU: torch.relu,
        UnaryKind.SQUARE: torch.square,
        UnaryKind.RSQRT: torch.rsqrt,
        UnaryKind.EXP: torch.exp,
        UnaryKind.LOG: torch.log,
        UnaryKind.CEIL: torch.ceil,
        # Banker's rounding (round-half-to-even), matching torch.round's own
        # semantics -- not "round half away from zero".
        UnaryKind.ROUND: torch.round,
        UnaryKind.EXP2: torch.exp2,
        UnaryKind.LOG2: torch.log2,
    }
    return TensorValue(data=fns[ctx.op.kind](ctx.args[0].data), type=ctx.result_type)


__all__ = ["Unary"]
