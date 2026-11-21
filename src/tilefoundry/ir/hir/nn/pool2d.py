"""HIR two-dimensional pooling ``Pool2D`` Op (torch max_pool2d / avg_pool2d)."""

from __future__ import annotations

import isl

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils


@expr.register_op(dialect="tf", category="nn", aliases={
    "max_pool2d": {"kind": expr.PoolKind.MAX},
    "avg_pool2d": {"kind": expr.PoolKind.AVG},
})
class Pool2D(expr.Op):
    input = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    kernel_size = expr.ParamDef(kind="attribute", annotation=tuple)
    stride = expr.ParamDef(kind="attribute", annotation=tuple)
    padding = expr.ParamDef(kind="attribute", annotation=tuple, default=(0, 0))
    ceil_mode = expr.ParamDef(kind="attribute", annotation=bool, default=False)
    count_include_pad = expr.ParamDef(kind="attribute", annotation=bool, default=True)
    kind = expr.ParamDef(kind="attribute", annotation=expr.PoolKind)


def int_pair(call, ctx, name: str, values: tuple, *, positive: bool) -> tuple:
    if len(values) != 2:
        ctx.error(call, f"{name} must be length-2, got {values}")
    bound = 1 if positive else 0
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= bound for v in values):
        qualifier = "positive" if positive else "non-negative"
        ctx.error(call, f"{name} values must be {qualifier}, got {values}")
    return values


def validate_pool2d(call, ctx, x) -> tuple:
    op = call.target
    if not isinstance(op.kind, expr.PoolKind):
        ctx.error(call, f"kind must be PoolKind, got {type(op.kind)}")
    if len(x.shape) != 4:
        ctx.error(call, f"expects rank-4 NCHW input, got shape {x.shape}")
    kernel_size = int_pair(call, ctx, "kernel_size", op.kernel_size, positive=True)
    stride = int_pair(call, ctx, "stride", op.stride, positive=True)
    padding = int_pair(call, ctx, "padding", op.padding, positive=False)
    if any(pad > kernel // 2 for pad, kernel in zip(padding, kernel_size)):
        ctx.error(call, f"padding {padding} must be at most half of kernel_size {kernel_size}")
    if any(expr.static_dim_value(extent) is None for extent in x.shape[2:]):
        ctx.error(call, f"input height and width must be static, got {x.shape[2:]}")
    return kernel_size, stride, padding


def spatial_extent(in_dim: int, k: int, s: int, p: int, ceil_mode: bool) -> int:
    """Window count along one axis.

    With ``ceil_mode`` the division rounds up, and a last window that starts at
    or beyond ``in_dim + p`` is dropped.
    """
    out = (in_dim + 2 * p - k + (s - 1 if ceil_mode else 0)) // s + 1
    return out - 1 if ceil_mode and (out - 1) * s >= in_dim + p else out


def output_shape(x, kernel_size, stride, padding, ceil_mode) -> tuple:
    height, width = (expr.static_dim_value(extent) for extent in x.shape[2:])
    return (
        x.shape[0],
        x.shape[1],
        spatial_extent(height, kernel_size[0], stride[0], padding[0], ceil_mode),
        spatial_extent(width, kernel_size[1], stride[1], padding[1], ceil_mode),
    )


@typeinfer.register(Pool2D)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x = ctx.type_of(call.args[0])
    kernel_size, stride, padding = validate_pool2d(call, ctx, x)
    out_shape = output_shape(x, kernel_size, stride, padding, call.target.ceil_mode)
    for name, extent in zip(("height", "width"), out_shape[2:]):
        if extent <= 0:
            ctx.error(call, f"output {name} extent must be positive, got {extent}")

    ir_utils.reject_partials("input", x.layout)
    read = tuple(map(ctx.type_of, call.args))
    relation = access.relations_of(call, ctx)
    try:
        shard = typeinfer.derive_output_shard_layout(
            read,
            relation,
            out_shape,
            fresh_strides=True,
            ctx=ctx,
            storage=x.storage,
        )
    except ValueError as error:
        ctx.error(
            call,
            f"cannot derive input ownership: {error}; use an explicit Reshard before Pool2D",
        )
    layout = shard or types.Layout(out_shape)
    return types.TensorType(
        shape=out_shape,
        dtype=x.dtype,
        layout=layout,
        storage=x.storage,
    )


@evaluator.register(Pool2D)
def eval_pool2d(ctx):
    def pool2d(input_):
        op = ctx.op
        if op.kind is expr.PoolKind.MAX:
            return utils.F.max_pool2d(
                input_, op.kernel_size, op.stride, op.padding, ceil_mode=op.ceil_mode
            )
        return utils.F.avg_pool2d(
            input_,
            op.kernel_size,
            op.stride,
            op.padding,
            ceil_mode=op.ceil_mode,
            count_include_pad=op.count_include_pad,
        )

    x = ctx.args[0].tensor
    placements = evaluator.common_placements(x)
    held = (
        pool2d(x.tensor)
        if not placements
        else tuple((p, pool2d(x.to_local(p))) for p in placements)
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


def window_term(out_dim: str, tap: str, stride: int, pad: int) -> str:
    scaled = out_dim if stride == 1 else f"{stride}{out_dim}"
    walked = f"{scaled} + {tap}"
    return walked if pad == 0 else f"{walked} - {pad}"


@access.register_relation(Pool2D)
def pool2d_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """The pooling window, as one affine relation over the space it walks.

    The window sits where the output coordinate puts it, so the input access is
    a relation guarded to the input extent. Padding positions read nothing.
    The space walked is the result plus the two window taps.
    """
    x = ctx.type_of(call.args[0])
    op = call.target
    k_h, k_w = op.kernel_size
    result = output_shape(x, op.kernel_size, op.stride, op.padding, op.ceil_mode)
    domain = ", ".join(("n", "c", "oh", "ow", "kh", "kw"))

    height = window_term("oh", "kh", op.stride[0], op.padding[0])
    width = window_term("ow", "kw", op.stride[1], op.padding[1])
    in_h, in_w = (expr.static_dim_value(extent) for extent in x.shape[2:])
    guard = " and ".join((f"0 <= {height} < {in_h}", f"0 <= {width} < {in_w}"))
    reached = access.Relation(
        isl.map(f"{{ [{domain}] -> [n, c, {height}, {width}] : {guard} }}")
    )
    return access.iterating(
        (*result, k_h, k_w),
        (
            reached,
            access.Relation(isl.multi_aff(f"{{ [{domain}] -> [n, c, oh, ow] }}")),
        ),
    )


@cost.register(Pool2D)
def pool2d_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    (input_,) = cost.input_types(call, ctx)
    output = cost.output_type(call, ctx)
    if not all(isinstance(type_, types.TensorType) for type_ in (input_, output)):
        raise ValueError("Pool2D cost requires tensor input and output")
    k_h, k_w = call.target.kernel_size
    flops = ir_utils.numel(output) * k_h * k_w
    return cost.Cost({input_.dtype: flops}, cost.traffic((input_,), output))


__all__ = ["Pool2D"]
