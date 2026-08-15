from __future__ import annotations

import isl
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
    AccessRelationResult,
    register_type_relation,
)
from tilefoundry.visitor_registry.isl_utility import to_domain

_COMMUTES_WITH = frozenset({"max", "min"})


@register_op
class Softplus(Op):
    """Pointwise softplus ``log(1 + e**x)``."""

    x = ParamDef(kind="input", pattern=Tensor)


@register_type_relation(Softplus)
def _softplus_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Model Softplus as one elementwise read and write."""
    (x,) = input_types
    domain, param_map = to_domain(x.shape)
    dims = [f"d{i}" for i in range(len(x.shape))]
    src = "[" + ", ".join(dims) + "]"
    ident = isl.map(f"{{ {src} -> [{', '.join(dims)}] }}")
    return AccessRelationResult(domain=domain, maps=(ident, ident), param_map=param_map)


@register_typeinfer(Softplus)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    reject_partials(ctx, call, "x", x_ty.layout, commutes_with=_COMMUTES_WITH)
    return x_ty


@register_eval(Softplus)
def _eval_softplus(ctx):

    return TensorValue(data=torch.nn.functional.softplus(ctx.args[0].data), type=ctx.result_type)
