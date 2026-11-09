"""HIR ``LayerNorm`` Op."""

from __future__ import annotations

import isl

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils


@expr.register_op(name="layer_norm", dialect='tf', category='nn')
class LayerNorm(expr.Op):
    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    weight = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    bias = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    axis = expr.ParamDef(kind="attribute", annotation=int)
    eps = expr.ParamDef(kind="attribute", annotation=float)


def normalized_axis(call: "expr.Call", ctx: "TypeInferContext", rank: int) -> int:
    axis = call.target.axis
    normalized = axis + rank if axis < 0 else axis
    if not (0 <= normalized < rank):
        ctx.error(call, f"axis {axis} out of range for rank {rank}")
    return normalized


def reject_normalized_splits(call, ctx, name, type_, first_axis: int) -> None:
    if not isinstance(type_.layout, types.ShardLayout):
        return
    for mesh_axis, logical_axis in enumerate(types.split_target_axes(type_.layout, type_.shape)):
        if logical_axis is not None and logical_axis >= first_axis:
            ctx.error(
                call,
                f"{name} normalized axis {logical_axis} is Split-sharded on "
                f"mesh axis {mesh_axis}; use an explicit Reshard before LayerNorm",
            )


@typeinfer.register(LayerNorm)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])
    weight_ty = ctx.type_of(call.args[1])
    bias_ty = ctx.type_of(call.args[2])
    axis = normalized_axis(call, ctx, len(x_ty.shape))
    normalized_shape = x_ty.shape[axis:]
    for name, affine_ty in (("weight", weight_ty), ("bias", bias_ty)):
        if affine_ty.shape != normalized_shape:
            ctx.error(
                call,
                f"{name} shape {affine_ty.shape} must equal x.shape[{axis}:] "
                f"{normalized_shape} for normalized axis {axis}",
            )

    if x_ty.dtype not in (expr.DType.f32, expr.DType.f16, expr.DType.bf16):
        ctx.error(call, f"x dtype must be f32, f16, or bf16, got {x_ty.dtype}")
    if weight_ty.dtype != bias_ty.dtype:
        ctx.error(
            call,
            f"weight dtype {weight_ty.dtype} must match bias dtype {bias_ty.dtype}",
        )
    affine_dtype = weight_ty.dtype
    if affine_dtype != x_ty.dtype and not (
        x_ty.dtype in (expr.DType.f16, expr.DType.bf16) and affine_dtype == expr.DType.f32
    ):
        ctx.error(
            call,
            f"affine dtype {affine_dtype} must match x dtype {x_ty.dtype}, or be "
            "f32 when x is f16/bf16",
        )

    for arg, ty in (("x", x_ty), ("weight", weight_ty), ("bias", bias_ty)):
        ir_utils.reject_partials(arg, ty.layout)
    reject_normalized_splits(call, ctx, "x", x_ty, axis)
    reject_normalized_splits(call, ctx, "weight", weight_ty, 0)
    reject_normalized_splits(call, ctx, "bias", bias_ty, 0)
    return x_ty


@evaluator.register(LayerNorm)
def eval_layer_norm(ctx):
    def layer_norm(x, weight, bias):
        axis = ctx.op.axis + x.ndim if ctx.op.axis < 0 else ctx.op.axis
        return utils.F.layer_norm(x, tuple(x.shape[axis:]), weight, bias, ctx.op.eps)

    xs = tuple(arg.tensor for arg in ctx.args)
    placements = evaluator.common_placements(*xs)
    held = (
        layer_norm(*(x.tensor for x in xs))
        if not placements
        else tuple((p, layer_norm(*(x.to_local(p) for x in xs))) for p in placements)
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


@access.register_relation(LayerNorm)
def layer_norm_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """One row normalised per iteration; the parameters read across the suffix.

    Normalising needs the whole suffix before any of it can be written, so those
    axes are not coordinates this Op is asked by. The parameters match the whole
    suffix rather than one axis of it, which is what this Op's own type contract
    already requires of them; the verifier refuses a split at or beyond that
    axis, so
    their footprint is the suffix's product in every view.
    """
    x = ctx.type_of(call.args[0])
    authored = call.target.axis
    axis = authored + len(x.shape) if authored < 0 else authored
    rows, names, guards = access.normalised_rows(x, x, axis)
    domain = ", ".join(f"d{index}" for index in range(len(rows)))
    where = f" : {' and '.join(guards)}" if guards else ""
    row = access.Relation(isl.map(f"{{ [{domain}] -> [{', '.join(names)}]{where} }}"))
    belongs = access.logical_axes_of(x, x)
    suffix = (
        ", ".join(names[position] for position, owner in enumerate(belongs) if owner >= axis) or "0"
    )
    across = access.Relation(isl.map(f"{{ [{domain}] -> [{suffix}]{where} }}"))
    return access.iterating(
        rows,
        (row, across, across, row),
    )


@cost.register(LayerNorm)
def layer_norm_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    source = cost.input_types(call, ctx)[0]
    output = cost.output_type(call, ctx)
    if not isinstance(source, types.TensorType):
        raise ValueError("LayerNorm cost requires a tensor input")
    return cost.Cost({expr.DType.f32: 8 * ir_utils.numel(source)}, cost.traffic(cost.input_types(call, ctx), output))


__all__ = ["LayerNorm"]
