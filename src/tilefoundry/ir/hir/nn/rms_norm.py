"""Define fused last-axis RMS normalization.

The rank-one weight matches the final input extent; preceding dimensions pass
through, including symbolic expressions. Computation accumulates in f32 and
casts back to the input dtype, while the weight may use another dtype.
"""

from __future__ import annotations

import isl

from tilefoundry import core, utils
from tilefoundry.interfaces import access, autodiff, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils


def identity(rank: int) -> "isl.multi_aff":
    if rank == 0:
        return access.Relation(isl.map("{ [] -> [] }"))
    dims = ", ".join(f"i{i}" for i in range(rank))
    return access.Relation(isl.map(f"{{ [{dims}] -> [{dims}] }}"))


@expr.register_op(name="rms_norm", dialect='tf', category='nn')
class RMSNorm(expr.Op):
    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    weight = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    eps = expr.ParamDef(kind="attribute", annotation=float, default=1e-6)


@typeinfer.register(RMSNorm)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])
    w_ty = ctx.type_of(call.args[1])

    if len(x_ty.shape) < 1:
        ctx.error(call, f"x must be rank ≥ 1, got shape {x_ty.shape}")
    if len(w_ty.shape) != 1:
        ctx.error(call, f"weight must be rank-1, got shape {w_ty.shape}")
    if x_ty.shape[-1] != w_ty.shape[0]:
        ctx.error(call, f"x last dim {x_ty.shape[-1]} != weight dim {w_ty.shape[0]}")

    for arg, ty in (("x", x_ty), ("weight", w_ty)):
        ir_utils.reject_partials(arg, ty.layout)

    return typeinfer.combine_operand_attrs(x_ty, typeinfer.shard_read_types(ctx, call.args)[1:], ctx)


@access.register_relation(RMSNorm)
def rms_norm_relation(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """GLOBAL level: one row normalised per iteration, read whole.

    A normalisation is asked once per row, because every element of a row needs
    the whole row's sum before any of it can be written. So the axis it
    normalises is not a coordinate this Op is asked by: it is free in the images,
    and the row is what each boundary reaches. The weight matches that axis and
    nothing else, so every row reaches all of it -- one weight element each.
    """
    x_ty = ctx.type_of(call.args[0])
    w_ty = ctx.type_of(call.args[1])
    logical_x = ctx.type_of(call.args[0])
    normalised = len(logical_x.shape) - 1
    rows, names, guards = access.normalised_rows(x_ty, logical_x, normalised)
    domain = ", ".join(f"d{index}" for index in range(len(rows)))
    where = f" : {' and '.join(guards)}" if guards else ""
    element = f"{{ [{domain}] -> [{', '.join(names)}]{where} }}"
    across = ", ".join(
        access.factored_image(
            [access.logical_term(names, x_ty, logical_x, normalised)],
            w_ty,
            ctx.type_of(call.args[1]),
        )
    )
    return access.iterating(
        rows,
        (
            access.Relation(isl.map(element)),
            access.Relation(isl.map(f"{{ [{domain}] -> [{across}]{where} }}")),
            access.Relation(isl.map(element)),
        ),
    )


@evaluator.register(RMSNorm)
def eval_rms_norm(ctx):
    def rms_norm(x, weight):
        xf = x.float()
        ms = xf.pow(2).mean(dim=-1, keepdim=True)
        out = xf * utils.torch.rsqrt(ms + ctx.op.eps) * weight.float()
        return out.to(evaluator.to_torch_dtype(ctx.result_type.dtype))

    xs = tuple(arg.tensor for arg in ctx.args)
    placements = evaluator.common_placements(*xs)
    held = (
        rms_norm(*(x.tensor for x in xs))
        if not placements
        else tuple((p, rms_norm(*(x.to_local(p) for x in xs))) for p in placements)
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


@cost.register(RMSNorm)
def rms_norm_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    inputs = cost.input_types(call, ctx)
    output = cost.output_type(call, ctx)
    source = inputs[0]
    if not isinstance(source, types.TensorType):
        raise ValueError("RMSNorm cost requires a tensor input")
    return cost.Cost({expr.DType.f32: 8 * ir_utils.numel(source)}, cost.traffic(inputs, output))


@autodiff.register(RMSNorm)
def _(call: expr.Call, cotangent: expr.Expr, ctx: autodiff.Context) -> tuple:
    """Written with the input x, so a zero in the weight loses nothing.

    ``inv = rsqrt(mean(x * x) + eps)``, ``n = x * inv``, ``gn = g * weight``;
    ``dx = inv * (gn - n * mean(gn * n))`` and the weight takes ``sum(g * n)``
    over every other axis.
    """
    x, weight = call.args
    dtype = ctx.type_of(x).dtype

    def mean(value):
        return ctx.call("reduce", value, axes=(-1,), keepdim=True, kind=expr.ReduceKind.MEAN)

    inverse = ctx.call(
        "rsqrt", ctx.call("add", mean(ctx.call("mul", x, x)), ctx.constant(call.target.eps, dtype))
    )
    normed = ctx.call("mul", x, inverse)
    scaled = ctx.call("mul", cotangent, weight)
    centered = ctx.call("sub", scaled, ctx.call("mul", normed, mean(ctx.call("mul", scaled, normed))))
    return (
        ctx.call("mul", inverse, centered),
        ctx.sum_to(ctx.call("mul", cotangent, normed), ctx.type_of(weight)),
    )


__all__ = ["RMSNorm"]
