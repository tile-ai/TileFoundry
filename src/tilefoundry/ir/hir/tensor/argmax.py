"""ArgMax HIR primitive.

SGLang baseline kernel H3 (greedy sampling). Returns int64 indices along the
reduction axis; ``keepdim=False``.

"""

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
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import (
    Layout,
    canonical_shard_layout,
    try_c_order_strides,
)
from tilefoundry.ir.types.shard.shard_layout import Split, shard_layout_of, split_target_axes
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    coordinates_of,
    identity_access,
    iterating,
    register_access_relation,
)
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout


@register_op
class ArgMax(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int, default=-1)


@register_typeinfer(ArgMax)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    if not x_ty.shape:
        ctx.error(call, "x must be at least rank-1")
    rank = len(x_ty.shape)
    axis = call.target.axis
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        ctx.error(call, f"axis {call.target.axis} out of range for rank {rank}")

    source_shard = shard_layout_of(x_ty.layout)
    if source_shard is not None:
        targets = split_target_axes(source_shard, x_ty.shape)
        if any(
            isinstance(attr, Split) and targets[mesh_axis] == axis
            for mesh_axis, attr in enumerate(source_shard.attrs)
        ):
            ctx.error(
                call,
                f"reduction axis {axis} must not be Split-sharded; use an "
                "explicit Reshard before ArgMax",
            )
    reject_partials(ctx, call, "x", x_ty.layout)
    out_shape = tuple(d for i, d in enumerate(x_ty.shape) if i != axis)
    new_layout = (
        None
        if x_ty.layout is None
        else Layout(shape=out_shape, strides=try_c_order_strides(out_shape))
    )
    if source_shard is not None:
        relation = coordinates_of(call, ctx)
        derived = derive_output_shard_layout(
            (x_ty,),
            relation,
            out_shape,
            complete_reduction_dims=frozenset({axis}),
            fresh_strides=True,
        )
        new_layout = (
            derived
            if derived is not None
            else canonical_shard_layout(out_shape, source_shard.mesh, source_shard.attrs)
        )
    return TensorType(
        shape=out_shape,
        dtype=DType.i64,
        layout=new_layout,
        storage=x_ty.storage,
    )


@register_eval(ArgMax)
def _eval_argmax(ctx):
    out = torch.argmax(ctx.args[0].data, dim=ctx.op.axis)
    return TensorValue(data=out, type=ctx.result_type)


@register_access_relation(ArgMax)
def _argmax_access_relation(call: "Call", ctx: "TypeInferContext") -> AccessRelations:
    """A scan walks what it reads, and collapses the axis it scanned on the way out.

    The coordinates are the source's own, because reading every element of the
    reduced axis is the whole of what this does. The result names one fewer of
    them, which is the collapse.
    """
    x_ty = ctx.type_of(call.args[0])
    rank = len(x_ty.shape)
    axis = call.target.axis
    if axis < 0:
        axis += rank
    dims = [f"d{index}" for index in range(rank)]
    walked = ", ".join(dims)
    kept = ", ".join(dim for index, dim in enumerate(dims) if index != axis)
    return iterating(
        x_ty.shape,
        AccessRelations(
            inputs=(BoundaryRelation(identity_access(rank)),),
            outputs=(BoundaryRelation(AffineAccess(isl.map(f"{{ [{walked}] -> [{kept}] }}"))),),
        ),
    )


__all__ = ["ArgMax"]
