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
class Tanh(Op):
    x = ParamDef(kind="input", pattern=Tensor)


@register_type_relation(Tanh)
def _tanh_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Forward access relation for the elementwise Tanh: single input, no broadcast, no reduction.

    Forward access relation for the elementwise Tanh: single input, no
    broadcast, no reduction -- the iteration domain is the input shape and
    both the input map and the output map are the identity.
    """
    (x,) = input_types
    domain, param_map = to_domain(x.shape)
    dims = [f"d{i}" for i in range(len(x.shape))]
    src = "[" + ", ".join(dims) + "]"
    ident = isl.map(f"{{ {src} -> [{', '.join(dims)}] }}")
    return AccessRelationResult(domain=domain, maps=(ident, ident), param_map=param_map)


@register_typeinfer(Tanh)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    reject_partials(ctx, call, "x", x_ty.layout, commutes_with=_COMMUTES_WITH)
    return x_ty


@register_eval(Tanh)
def _eval_tanh(ctx):

    return TensorValue(data=torch.tanh(ctx.args[0].data), type=ctx.result_type)
