from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType


@register_op
class Rsqrt(Op):
    """Reciprocal square root (value form).

    Spec: hir.md §2.1
    """
    x = ParamDef(kind="input", pattern=Tensor)
@register_typeinfer(Rsqrt)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return ctx.type_of(call.args[0])


@register_eval(Rsqrt)
def _eval_rsqrt(ctx):

    return TensorValue(data=torch.rsqrt(ctx.args[0].data), type=ctx.result_type)
