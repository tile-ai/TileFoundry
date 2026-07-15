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
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.shard_layout import partial_reductions


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
    reductions = partial_reductions(x_ty.layout)
    if reductions:
        if op.kind in _MONOTONE_INCREASING:
            if "sum" in reductions:
                ctx.error(
                    call,
                    f"Unary {op.kind.name}: Partial(sum) input on x is "
                    f"unsound ({op.kind.name.lower()} is nonlinear, does not "
                    "commute with sum) — insert reshard(x, Broadcast) before "
                    "this consumer",
                )
        elif op.kind in _LINEAR:
            if reductions - {"sum"}:
                ctx.error(
                    call,
                    f"Unary {op.kind.name}: Partial(max/min) input on x is "
                    "unsound (negation reverses order, does not commute "
                    "with max/min) — insert reshard(x, Broadcast) before "
                    "this consumer",
                )
        else:
            ctx.error(
                call,
                f"Unary {op.kind.name}: Partial input on x is unsound "
                f"({op.kind.name.lower()} is not proven to commute with any "
                "reduction) — insert reshard(x, Broadcast) before this "
                "consumer",
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
