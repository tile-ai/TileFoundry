from __future__ import annotations

import torch.nn.functional as F

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, split_target_axes
from tilefoundry.visitor_registry import register_typeinfer


@register_op(name="layer_norm")
class LayerNorm(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    weight = ParamDef(kind="input", pattern=Tensor)
    bias = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)
    eps = ParamDef(kind="attribute", annotation=float)


def _normalized_axis(call: "Call", ctx: "TypeInferContext", rank: int) -> int:
    axis = call.target.axis
    normalized = axis + rank if axis < 0 else axis
    if not (0 <= normalized < rank):
        ctx.error(call, f"axis {axis} out of range for rank {rank}")
    return normalized


def _reject_normalized_splits(call, ctx, name, type_, first_axis: int) -> None:
    if not isinstance(type_.layout, ShardLayout):
        return
    for mesh_axis, logical_axis in enumerate(
        split_target_axes(type_.layout, type_.shape)
    ):
        if logical_axis is not None and logical_axis >= first_axis:
            ctx.error(
                call,
                f"{name} normalized axis {logical_axis} is Split-sharded on "
                f"mesh axis {mesh_axis}; use an explicit Reshard before LayerNorm",
            )


@register_typeinfer(LayerNorm)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    weight_ty = ctx.type_of(call.args[1])
    bias_ty = ctx.type_of(call.args[2])
    axis = _normalized_axis(call, ctx, len(x_ty.shape))
    normalized_shape = x_ty.shape[axis:]
    for name, affine_ty in (("weight", weight_ty), ("bias", bias_ty)):
        if affine_ty.shape != normalized_shape:
            ctx.error(
                call,
                f"{name} shape {affine_ty.shape} must equal x.shape[{axis}:] "
                f"{normalized_shape} for normalized axis {axis}",
            )

    if x_ty.dtype not in (DType.f32, DType.f16, DType.bf16):
        ctx.error(call, f"x dtype must be f32, f16, or bf16, got {x_ty.dtype}")
    if weight_ty.dtype != bias_ty.dtype:
        ctx.error(
            call,
            f"weight dtype {weight_ty.dtype} must match bias dtype {bias_ty.dtype}",
        )
    affine_dtype = weight_ty.dtype
    if affine_dtype != x_ty.dtype and not (
        x_ty.dtype in (DType.f16, DType.bf16) and affine_dtype == DType.f32
    ):
        ctx.error(
            call,
            f"affine dtype {affine_dtype} must match x dtype {x_ty.dtype}, or be "
            "f32 when x is f16/bf16",
        )

    for arg, ty in (("x", x_ty), ("weight", weight_ty), ("bias", bias_ty)):
        reject_partials(ctx, call, arg, ty.layout)
    _reject_normalized_splits(call, ctx, "x", x_ty, axis)
    _reject_normalized_splits(call, ctx, "weight", weight_ty, 0)
    _reject_normalized_splits(call, ctx, "bias", bias_ty, 0)
    return x_ty


@register_eval(LayerNorm)
def _eval_layer_norm(ctx):
    x, weight, bias = (arg.data for arg in ctx.args)
    rank = x.ndim
    axis = ctx.op.axis + rank if ctx.op.axis < 0 else ctx.op.axis
    out = F.layer_norm(x, tuple(x.shape[axis:]), weight, bias, ctx.op.eps)
    return TensorValue(data=out, type=ctx.result_type)
