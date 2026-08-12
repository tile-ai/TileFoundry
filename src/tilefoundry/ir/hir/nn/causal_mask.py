"""HIR causal mask over a query-by-key score tile."""

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
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, split_target_axes
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)


@register_op(name="causal_mask")
class CausalMask(Op):
    """Mask keys whose global position is after the corresponding query."""

    scores = ParamDef(kind="input", pattern=Tensor)
    query_start = ParamDef(kind="input", pattern=Tensor)
    key_start = ParamDef(kind="input", pattern=Tensor)
    value = ParamDef(kind="attribute", annotation=float, default=float("-inf"))


register_access_relation(CausalMask)(identity_relations(3))


@register_typeinfer(CausalMask)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    scores = ctx.type_of(call.args[0])
    if len(scores.shape) < 2:
        ctx.error(call, f"scores must be rank >= 2, got shape {scores.shape}")
    for name, arg in zip(("query_start", "key_start"), call.args[1:]):
        type_ = ctx.type_of(arg)
        if type_.shape != () or type_.dtype.name not in ("i32", "i64"):
            ctx.error(call, f"{name} must be a rank-0 integer")

    reject_partials(ctx, call, "scores", scores.layout)
    if isinstance(scores.layout, ShardLayout):
        query_axis = len(scores.shape) - 2
        key_axis = len(scores.shape) - 1
        for mesh_axis, target in enumerate(
            split_target_axes(scores.layout, scores.shape)
        ):
            if target == query_axis:
                ctx.error(
                    call,
                    f"mesh axis {mesh_axis} splits the query axis; global causal "
                    "coordinates are not derivable from query_start alone",
                )
            if target == key_axis:
                ctx.error(
                    call,
                    f"mesh axis {mesh_axis} splits the key axis; global causal "
                    "coordinates are not derivable from key_start alone",
                )
    return scores


@register_eval(CausalMask)
def _eval_causal_mask(ctx):
    scores = ctx.args[0].data
    query_start = int(ctx.args[1].data.item())
    key_start = int(ctx.args[2].data.item())
    query = torch.arange(scores.shape[-2], device=scores.device) + query_start
    key = torch.arange(scores.shape[-1], device=scores.device) + key_start
    keep = key.unsqueeze(0) <= query.unsqueeze(1)
    value = torch.as_tensor(ctx.op.value, dtype=scores.dtype, device=scores.device)
    return TensorValue(data=torch.where(keep, scores, value), type=ctx.result_type)


__all__ = ["CausalMask"]
