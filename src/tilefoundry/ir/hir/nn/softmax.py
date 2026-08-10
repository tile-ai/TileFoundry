from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    identity_relations,
    register_access_relation,
    register_type_relation,
)
from tilefoundry.visitor_registry.isl_utility import to_domain


@register_op
class SoftMax(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)


register_access_relation(SoftMax)(identity_relations(1))


@register_typeinfer(SoftMax)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    reject_partials(ctx, call, "x", x_ty.layout)
    return x_ty


@register_type_relation(SoftMax)
def _softmax_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Model fused row-wise softmax over the last axis.

    Batch axes form the domain and the reduced axis remains a range dimension,
    so one statement instance owns a whole row. Input and output reuse the same
    row map. Other reduction axes fail closed.
    """
    (x,) = input_types
    rank = len(x.shape)
    if rank < 1:
        raise NotImplementedError(
            f"SoftMax type_relation: x must be rank >= 1, got shape {x.shape}"
        )
    axis = call.target.axis
    norm_axis = axis % rank
    if norm_axis != rank - 1:
        raise NotImplementedError(
            "SoftMax type_relation: V1 only supports axis=-1 (the last "
            f"axis) -- got axis={axis} (normalized {norm_axis}) for a "
            f"rank-{rank} input; reducing any other axis has no "
            "access-relation modeling yet"
        )
    reduce_extent = x.shape[-1]
    if not isinstance(reduce_extent, int) or isinstance(reduce_extent, bool):
        raise NotImplementedError(
            "SoftMax type_relation: reduction axis must be a static int, "
            f"got {reduce_extent!r} -- a dynamic reduction axis has no isl "
            "representation here"
        )

    batch_shape = x.shape[:-1]
    domain, param_map = to_domain(batch_shape)
    dims = [f"d{i}" for i in range(len(batch_shape))]
    src = "[" + ", ".join(dims) + "]"
    row = ", ".join(dims + ["j"])
    row_map = isl.map(f"{{ {src} -> [{row}] : 0 <= j < {reduce_extent} }}")
    return AccessRelationResult(domain=domain, maps=(row_map, row_map), param_map=param_map)


@register_eval(SoftMax)
def _eval_softmax(ctx):

    out = torch.softmax(ctx.args[0].data.float(), dim=ctx.op.axis)
    return TensorValue(data=out.to(to_torch_dtype(ctx.result_type.dtype)), type=ctx.result_type)
