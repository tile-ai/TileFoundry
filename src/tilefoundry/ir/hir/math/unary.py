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
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.shard_propagate import partial_reductions_by_axis


@register_op(dialect="tf", category="math")
class Unary(Op):
    """Value-form pointwise unary operation."""
    x = ParamDef(kind="input", pattern=Tensor)
    kind = ParamDef(kind="attribute", annotation=UnaryKind)

# Monotone non-decreasing: commutes with max/min, not sum.
_MONOTONE_INCREASING = frozenset({UnaryKind.EXP, UnaryKind.LOG, UnaryKind.RELU})
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
    for axis, reduction in enumerate(partial_reductions_by_axis(x_ty.layout)):
        if reduction is None:
            continue
        if op.kind in _MONOTONE_INCREASING and reduction == "sum":
            ctx.error(
                call,
                f"Unary {op.kind.name}: x carries Partial(sum) on mesh axis "
                f"{axis}, which does not commute; insert reshard(x, Broadcast) "
                "before this consumer",
            )
        elif op.kind in _LINEAR and reduction != "sum":
            ctx.error(
                call,
                f"Unary {op.kind.name}: x carries Partial({reduction}) on mesh "
                f"axis {axis}, which does not commute; insert reshard(x, "
                "Broadcast) before this consumer",
            )
        elif op.kind not in _MONOTONE_INCREASING and op.kind not in _LINEAR:
            ctx.error(
                call,
                f"Unary {op.kind.name}: x carries Partial({reduction}) on mesh "
                f"axis {axis}, which is not proven to commute; insert "
                "reshard(x, Broadcast) before this consumer",
            )
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
    }
    return TensorValue(data=fns[ctx.op.kind](ctx.args[0].data), type=ctx.result_type)


__all__ = ["Unary"]
