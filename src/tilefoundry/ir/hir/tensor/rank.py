from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer


@register_op
class Rank(Op):
    x = ParamDef(kind="input", pattern=Tensor)


@register_typeinfer(Rank)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return TensorType.meta_scalar()


@register_eval(Rank)
def _eval_rank(ctx):
    data = torch.tensor(ctx.args[0].data.ndim, dtype=torch.int64)
    return TensorValue(data=data, type=ctx.result_type)
