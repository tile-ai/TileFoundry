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

_COMMUTES_WITH = frozenset()


@register_op
class Gelu(Op):
    """Gaussian Error Linear Unit.

    Gaussian Error Linear Unit. ``approximate="tanh"`` is the tanh-based
    approximation (HF ``gelu_pytorch_tanh`` / Gemma-2 MLP activation).
    """

    x = ParamDef(kind="input", pattern=Tensor)
    approximate = ParamDef(kind="attribute", annotation=str, default="tanh")


@register_type_relation(Gelu)
def _gelu_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Model Gelu as one elementwise read and write."""
    (x,) = input_types
    domain, param_map = to_domain(x.shape)
    dims = [f"d{i}" for i in range(len(x.shape))]
    src = "[" + ", ".join(dims) + "]"
    ident = isl.map(f"{{ {src} -> [{', '.join(dims)}] }}")
    return AccessRelationResult(domain=domain, maps=(ident, ident), param_map=param_map)


@register_typeinfer(Gelu)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    reject_partials(ctx, call, "x", x_ty.layout, commutes_with=_COMMUTES_WITH)
    return x_ty


@register_eval(Gelu)
def _eval_gelu(ctx):

    return TensorValue(
        data=torch.nn.functional.gelu(ctx.args[0].data, approximate=ctx.op.approximate),
        type=ctx.result_type,
    )
