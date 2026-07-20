from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.shard_layout import Broadcast, ShardLayout
from tilefoundry.visitor_registry import register_typeinfer


@register_op(name="repeat_interleave")
class RepeatInterleave(Op):
    """Repeat each element of ``x`` along ``axis`` ``repeats`` times,
    interleaved (GQA head expansion). The named axis grows by ``repeats``;
    all other dims are unchanged."""
    x = ParamDef(kind="input", pattern=Tensor)
    repeats = ParamDef(kind="attribute", annotation=int)
    axis = ParamDef(kind="attribute", annotation=int)
@register_typeinfer(RepeatInterleave)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    op = call.target
    shape = list(x_ty.shape)
    ax = op.axis if op.axis >= 0 else op.axis + len(shape)
    if not (0 <= ax < len(shape)):
        ctx.error(call, f"RepeatInterleave: axis {op.axis} out of range for rank {len(shape)}")
    shape[ax] = shape[ax] * op.repeats

    # The named axis grows, so the input layout no longer describes the
    # output; do not carry a stale sharded layout. An unsharded or fully
    # replicated input produces an unsharded output; a genuine sharding fails
    # closed (re-expressing a repeat across a Split would need an explicit
    # relation).
    new_layout = None
    if isinstance(x_ty.layout, ShardLayout) and any(
        not isinstance(a, Broadcast) for a in x_ty.layout.attrs
    ):
        ctx.error(
            call,
            "RepeatInterleave cannot express a sharded layout; reshard to a "
            "replicated layout first",
        )
    return TensorType(
        shape=tuple(shape),
        dtype=x_ty.dtype,
        layout=new_layout,
        storage=x_ty.storage,
    )


@register_eval(RepeatInterleave)
def _eval_repeat_interleave(ctx):

    out = torch.repeat_interleave(ctx.args[0].data, ctx.op.repeats, dim=ctx.op.axis)
    return TensorValue(data=out, type=ctx.result_type)
