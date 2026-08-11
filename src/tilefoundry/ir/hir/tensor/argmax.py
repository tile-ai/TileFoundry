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
    ShardLayout,
    canonical_shard_layout,
    try_c_order_strides,
)
from tilefoundry.ir.types.shard.shard_layout import Split, split_target_axes
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    AccessRelations,
    build_relation,
    register_access_relation,
    register_type_relation,
)
from tilefoundry.visitor_registry.relation_build import build_domain
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

    if isinstance(x_ty.layout, ShardLayout):
        targets = split_target_axes(x_ty.layout, x_ty.shape)
        if any(
            isinstance(attr, Split) and targets[mesh_axis] == axis
            for mesh_axis, attr in enumerate(x_ty.layout.attrs)
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
    if isinstance(x_ty.layout, ShardLayout):
        relation = build_relation(call, (x_ty,), ctx)
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
            else canonical_shard_layout(out_shape, x_ty.layout.mesh, x_ty.layout.attrs)
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


@register_type_relation(ArgMax)
def _argmax_type_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    (x,) = input_types
    rank = len(x.shape)
    axis = call.target.axis % rank
    dims = [f"d{i}" for i in range(rank)]
    source = f"[{', '.join(dims)}]"
    output = [dim for i, dim in enumerate(dims) if i != axis]
    return AccessRelationResult(
        domain=build_domain(x.shape),
        maps=(
            isl.map(f"{{ {source} -> [{', '.join(dims)}] }}"),
            isl.map(f"{{ {source} -> [{', '.join(output)}] }}"),
        ),
    )


@register_access_relation(ArgMax)
def _argmax_access_relation(call: "Call", ctx: "TypeInferContext") -> AccessRelations:
    """GLOBAL: input scanned over the reduction axis (isl.map).

    GLOBAL: input scanned over the reduction axis (isl.map). Output is
    identity over the leading dims (axis collapsed away).
    """
    x_ty = ctx.type_of(call.args[0])
    rank = len(x_ty.shape)
    axis = call.target.axis
    if axis < 0:
        axis += rank
    in_dims = ", ".join(f"i{i}" for i in range(rank))
    leading = [f"i{i}" for i in range(rank) if i != axis]
    out_dims = ", ".join(leading) if leading else ""
    if out_dims:
        in_rel = isl.map(f"{{ [{out_dims}] -> [{in_dims}] }}")
        out_id = isl.multi_aff(f"{{ [{out_dims}] -> [{out_dims}] }}")
    else:
        in_rel = isl.map(f"{{ [] -> [{in_dims}] }}")
        out_id = isl.multi_aff("{ [] -> [] }")
    return AccessRelations(inputs=(in_rel,), outputs=(out_id,))


__all__ = ["ArgMax"]
