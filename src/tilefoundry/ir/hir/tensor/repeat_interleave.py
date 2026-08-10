from __future__ import annotations

import isl
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
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    register_type_relation,
)
from tilefoundry.visitor_registry.relation_build import build_domain


@register_op(name="repeat_interleave")
class RepeatInterleave(Op):
    """Repeat each element of ``x`` along ``axis`` ``repeats`` times, interleaved.

    Repeat each element of ``x`` along ``axis`` ``repeats`` times,
    interleaved (GQA head expansion). The named axis grows by ``repeats``;
    all other dims are unchanged.
    """

    x = ParamDef(kind="input", pattern=Tensor)
    repeats = ParamDef(kind="attribute", annotation=int)
    axis = ParamDef(kind="attribute", annotation=int)


def _normalize_axis(axis: int, rank: int) -> int:
    return axis if axis >= 0 else axis + rank


@register_type_relation(RepeatInterleave)
def _repeat_interleave_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Forward relation for RepeatInterleave.

    Forward relation for RepeatInterleave: the iteration domain is the
    *output* shape (the named axis already expanded to ``in_extent *
    repeats``); the output map is identity -- every domain point writes
    exactly one output element. The input map reads the source element at
    ``out_idx // repeats`` along the named axis (``repeats`` consecutive
    output positions alias the same input element); every other axis is
    identity.
    """
    (x,) = input_types
    op = call.target
    rank = len(x.shape)
    ax = _normalize_axis(op.axis, rank)
    repeats = op.repeats

    out_shape = list(x.shape)
    out_shape[ax] = out_shape[ax] * repeats

    dims = [f"d{i}" for i in range(rank)]
    src = "[" + ", ".join(dims) + "]"
    in_dims = [f"floor({dims[i]}/{repeats})" if i == ax else dims[i] for i in range(rank)]
    in_map = isl.map(f"{{ {src} -> [{', '.join(in_dims)}] }}")
    out_map = isl.map(f"{{ {src} -> [{', '.join(dims)}] }}")
    return AccessRelationResult(domain=build_domain(tuple(out_shape)), maps=(in_map, out_map))


@register_typeinfer(RepeatInterleave)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    op = call.target
    shape = list(x_ty.shape)
    ax = _normalize_axis(op.axis, len(shape))
    if not (0 <= ax < len(shape)):
        ctx.error(call, f"RepeatInterleave: axis {op.axis} out of range for rank {len(shape)}")
    shape[ax] = shape[ax] * op.repeats

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
