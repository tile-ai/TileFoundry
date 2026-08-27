"""Fused SiLU/Swish HIR Op."""

from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)


@register_op
class Silu(Op):
    """Pointwise ``x * sigmoid(x)``, as one fused op."""

    x = ParamDef(kind="input", pattern=Tensor)




@register_typeinfer(Silu)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])

    reject_partials(ctx, call, "x", x_ty.layout)
    return x_ty


@register_eval(Silu)
def _eval_silu(ctx):
    return TensorValue(data=torch.nn.functional.silu(ctx.args[0].data), type=ctx.result_type)


__all__ = ["Silu"]


register_access_relation(Silu)(identity_relations(1))
