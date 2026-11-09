"""HIR two-dimensional convolution ``Conv2D`` Op."""

from __future__ import annotations

import isl

from tilefoundry import core, services, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import dim, expr, pattern, types
from tilefoundry.ir import utils as ir_utils
from tilefoundry.ir.types import int_tuple


@expr.register_op(dialect='tf', category='nn')
class Conv2D(expr.Op):
    input = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    weight = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    bias = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    stride = expr.ParamDef(kind="attribute", annotation=tuple)
    padding = expr.ParamDef(kind="attribute", annotation=tuple)
    dilation = expr.ParamDef(kind="attribute", annotation=tuple)
    groups = expr.ParamDef(kind="attribute", annotation=int)


def i64(value: int) -> expr.Constant:
    return dim.i64_const(value)


def as_expr(v):
    if isinstance(v, expr.Expr):
        return v
    return i64(int(v))


def out_spatial(in_dim: expr.Expr, k: int, s: int, p: int, d: int) -> expr.Expr:
    """Compute (in + 2*p - d*(k-1) - 1) // s + 1, keeping symbolic dims alive.

    If `in_dim` is a Constant the result is also a Constant; otherwise we
    build a `dim.*` Expr tree so downstream passes can simplify.
    """
    eff_k = d * (k - 1) + 1

    add_pad = services.simplify_dim("add", (in_dim, i64(2 * p)))
    sub_k = services.simplify_dim("sub", (add_pad, i64(eff_k)))
    div_s = services.simplify_dim("floordiv", (sub_k, i64(s)))
    plus_1 = services.simplify_dim("add", (div_s, i64(1)))
    return services.normalize_dim(plus_1)


def pair(call, ctx, name: str, values: tuple, *, positive: bool) -> tuple[int, int]:
    if len(values) != 2:
        ctx.error(call, f"{name} must be length-2, got {values}")
    for value in values:
        valid = isinstance(value, int) and not isinstance(value, bool)
        valid = valid and (value > 0 if positive else value >= 0)
        if not valid:
            qualifier = "positive" if positive else "non-negative"
            ctx.error(call, f"{name} values must be {qualifier}, got {values}")
    return values


def static_extent(call, ctx, name: str, dim_value) -> int:
    value = expr.static_dim_value(dim_value)
    if value is None:
        ctx.error(call, f"{name} must be static, got {dim_value}")
    return value


def validate_conv2d(call, ctx, x, weight, bias):
    op = call.target
    if len(x.shape) != 4 or len(weight.shape) != 4:
        ctx.error(call, "expects rank-4 input and weight (NCHW / OIHW)")
    if len(bias.shape) != 1:
        ctx.error(call, f"bias must be rank-1, got shape {bias.shape}")
    if x.dtype != weight.dtype:
        ctx.error(call, f"weight dtype {weight.dtype} must match input dtype {x.dtype}")
    if bias.dtype != x.dtype:
        ctx.error(call, f"bias dtype {bias.dtype} must match input dtype {x.dtype}")

    stride = pair(call, ctx, "stride", op.stride, positive=True)
    padding = pair(call, ctx, "padding", op.padding, positive=False)
    dilation = pair(call, ctx, "dilation", op.dilation, positive=True)
    if not isinstance(op.groups, int) or isinstance(op.groups, bool) or op.groups <= 0:
        ctx.error(call, f"groups must be positive, got {op.groups}")

    in_channels = expr.static_dim_value(x.shape[1])
    out_channels = expr.static_dim_value(weight.shape[0])
    k_h = static_extent(call, ctx, "kernel height", weight.shape[2])
    k_w = static_extent(call, ctx, "kernel width", weight.shape[3])
    if k_h <= 0 or k_w <= 0:
        ctx.error(call, f"kernel extents must be positive, got {(k_h, k_w)}")
    if op.groups == 1:
        expected_weight_channels = x.shape[1]
    else:
        if in_channels is None:
            ctx.error(call, f"input channels must be static for groups={op.groups}")
        if out_channels is None:
            ctx.error(call, f"output channels must be static for groups={op.groups}")
        if in_channels % op.groups:
            ctx.error(
                call,
                f"input channels {in_channels} must be divisible by groups {op.groups}",
            )
        if out_channels % op.groups:
            ctx.error(
                call,
                f"output channels {out_channels} must be divisible by groups {op.groups}",
            )
        expected_weight_channels = in_channels // op.groups
    if weight.shape[1] != expected_weight_channels:
        ctx.error(
            call,
            f"weight input-channel extent {weight.shape[1]} must equal input "
            f"channels/groups {expected_weight_channels}",
        )
    if bias.shape[0] != weight.shape[0]:
        ctx.error(
            call,
            f"bias extent {bias.shape[0]} must equal output channels {weight.shape[0]}",
        )
    return stride, padding, dilation, op.groups, k_h, k_w


def output_shape(x, weight, stride, padding, dilation, k_h, k_w) -> tuple:
    height = out_spatial(x.shape[2], k_h, stride[0], padding[0], dilation[0])
    width = out_spatial(x.shape[3], k_w, stride[1], padding[1], dilation[1])
    static_height = expr.static_dim_value(height)
    static_width = expr.static_dim_value(width)
    return (
        x.shape[0],
        weight.shape[0],
        static_height if static_height is not None else height,
        static_width if static_width is not None else width,
    )


def require_group_aligned_output_split(call, ctx, weight, bias, groups: int) -> None:
    """A split of the output channels must not cut across a group.

    A shard straddling a boundary reads two groups where its neighbour reads
    one, and nothing here can tell those participants apart: a projection knows
    a shard's extent and not its offset. Twelve channels in three groups over
    four shards would report one footprint for all four where the real ones are
    1, 2, 2 and 1 groups, so it is refused where the author can align it.
    """
    if groups == 1:
        return
    channels = expr.static_dim_value(weight.shape[0])
    if channels is None:
        return
    per_group = channels // groups
    for name, type_ in (("weight", weight), ("bias", bias)):
        layout = types.shard_layout_of(type_.layout)
        if layout is None:
            continue
        targets = types.split_target_axes(layout, type_.shape)
        for mesh_axis, attr in enumerate(layout.attrs):
            if not isinstance(attr, types.Split) or targets[mesh_axis] != 0:
                continue
            shards = layout.layout.shape[attr.axis]
            if not isinstance(shards, int) or shards <= 0:
                continue
            held = channels // shards if name == "bias" else channels // shards
            if held == 0 or (per_group % held and held % per_group):
                ctx.error(
                    call,
                    f"{name} splits {channels} output channels {shards} ways "
                    f"across {groups} groups of {per_group}, so a participant "
                    "can straddle a group boundary and no projection here can "
                    "say which; align the split to the group size",
                )


def require_exact_partial_state(call, ctx, x, weight, bias) -> None:
    named = (("input", x), ("weight", weight), ("bias", bias))
    placed = [
        (name, layout)
        for name, type_ in named
        if (layout := types.shard_layout_of(type_.layout)) is not None
    ]
    mesh = placed[0][1].mesh if placed else None

    ir_utils.check_multilinear_partials(
        (("input", x), ("weight", weight)), ctx.target, kind=expr.BinaryKind.MUL
    )
    contraction_splits: dict[int, tuple[str, int]] = {}
    channel_split_axes: dict[str, set[int]] = {"input": set(), "weight": set()}
    for name, type_, axes in (
        ("input", x, {1: 4}),
        ("weight", weight, {1: 4, 2: 5, 3: 6}),
    ):
        layout = types.shard_layout_of(type_.layout)
        if layout is None:
            continue
        targets = types.split_target_axes(layout, type_.shape)
        for mesh_axis, logical_axis in enumerate(targets):
            if logical_axis not in axes:
                continue
            domain_dim = axes[logical_axis]
            if logical_axis == 1:
                channel_split_axes[name].add(mesh_axis)
            previous = contraction_splits.get(mesh_axis)
            if previous is not None and previous[1] != domain_dim:
                ctx.error(
                    call,
                    f"{name} contraction Split on mesh axis {mesh_axis} conflicts "
                    f"with {previous[0]}; use an explicit Reshard before Conv2D",
                )
            contraction_splits[mesh_axis] = (name, domain_dim)

    for mesh_axis in sorted(channel_split_axes["input"] ^ channel_split_axes["weight"]):
        missing = "weight" if mesh_axis in channel_split_axes["input"] else "input"
        ctx.error(
            call,
            f"{missing} must carry a matching input-channel Split on mesh axis "
            f"{mesh_axis}; use an explicit Reshard before Conv2D",
        )

    required_partial_axes = set(contraction_splits)
    for type_ in (x, weight):
        required_partial_axes.update(
            axis
            for axis, reduction in enumerate(types.partial_reductions_by_axis(type_.layout))
            if reduction is not None
        )

    bias_reductions = types.partial_reductions_by_axis(bias.layout)
    for mesh_axis in sorted(required_partial_axes):
        reduction = bias_reductions[mesh_axis] if mesh_axis < len(bias_reductions) else None
        if reduction != "sum" or not (
            (bias_layout := types.shard_layout_of(bias.layout)) is not None and bias_layout.mesh == mesh
        ):
            ctx.error(
                call,
                f"bias must carry Partial(sum) on mesh axis {mesh_axis} for "
                "the partial convolution result; use an explicit Reshard before Conv2D",
            )
    for mesh_axis, reduction in enumerate(bias_reductions):
        if reduction is not None and mesh_axis not in required_partial_axes:
            ctx.error(
                call,
                f"bias carries Partial({reduction}) on mesh axis {mesh_axis} "
                "without a partial convolution result; use an explicit Reshard "
                "before Conv2D",
            )


@typeinfer.register(Conv2D)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x = ctx.type_of(call.args[0])
    w = ctx.type_of(call.args[1])
    bias = ctx.type_of(call.args[2])
    stride, padding, dilation, group_count, k_h, k_w = validate_conv2d(call, ctx, x, w, bias)
    out_shape = output_shape(x, w, stride, padding, dilation, k_h, k_w)
    for name, extent in zip(("height", "width"), out_shape[2:]):
        static = expr.static_dim_value(extent)
        if static is not None and static <= 0:
            ctx.error(call, f"output {name} extent must be positive, got {static}")

    require_exact_partial_state(call, ctx, x, w, bias)
    require_group_aligned_output_split(call, ctx, w, bias, group_count)
    relation = access.relations_of(call, ctx)
    try:
        shard = typeinfer.derive_output_shard_layout(
            (x, w, bias),
            relation,
            out_shape,
            partial_reduction_dims=frozenset({4, 5, 6}),
            fresh_strides=True,
            ctx=ctx,
            storage=x.storage,
        )
    except ValueError as error:
        ctx.error(
            call,
            f"cannot derive input ownership: {error}; use an explicit Reshard before Conv2D",
        )
    layout = shard or types.Layout(out_shape)
    return types.TensorType(
        shape=out_shape,
        dtype=x.dtype,
        layout=layout,
        storage=x.storage,
    )


@evaluator.register(Conv2D)
def eval_conv2d(ctx):
    def conv2d(input_, weight, bias):
        return utils.F.conv2d(
            input_,
            weight,
            bias,
            stride=ctx.op.stride,
            padding=ctx.op.padding,
            dilation=ctx.op.dilation,
            groups=ctx.op.groups,
        )

    xs = tuple(arg.tensor for arg in ctx.args)
    placements = evaluator.common_placements(*xs)
    held = (
        conv2d(*(x.tensor for x in xs))
        if not placements
        else tuple((p, conv2d(*(x.to_local(p) for x in xs))) for p in placements)
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


def spatial_term(out_dim, kernel_dim, stride, pad, dilation, extent):
    terms = [out_dim if stride == 1 else f"{stride}{out_dim}"]
    if extent != 1:
        terms.append(kernel_dim if dilation == 1 else f"{dilation}{kernel_dim}")
    walked = " + ".join(terms)
    return walked if pad == 0 else f"{walked} - {pad}"


@access.register_relation(Conv2D)
def conv2d_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """The receptive field, as one affine relation over the space it walks.

    Where a window sits moves affinely with the output coordinate, so this is a
    relation and not a window: the offset is not a value, it is the coordinate.
    The space walked is what the result has, plus the contraction and the two
    kernel taps, all of them derived from the operands and the Op's own strides
    because the result's Type is what this answer produces.
    """
    x = ctx.type_of(call.args[0])
    weight = ctx.type_of(call.args[1])
    op = call.target
    k_h = expr.static_dim_value(weight.shape[2])
    k_w = expr.static_dim_value(weight.shape[3])
    contraction = weight.shape[1]
    result = output_shape(x, weight, op.stride, op.padding, op.dilation, k_h, k_w)
    dims = ["n", "co", "oh", "ow", "ci", "kh", "kw"]
    domain = ", ".join(dims)

    height = spatial_term("oh", "kh", op.stride[0], op.padding[0], op.dilation[0], k_h)
    width = spatial_term("ow", "kw", op.stride[1], op.padding[1], op.dilation[1], k_w)
    guard = " and ".join((f"0 <= {height} < {x.shape[2]}", f"0 <= {width} < {x.shape[3]}"))
    per_group_out = max(weight.shape[0] // op.groups, 1) if op.groups != 1 else 1
    channel = "ci" if op.groups == 1 else f"floor(co/{per_group_out})*{contraction}+ci"
    reached = access.Relation(
        isl.map(f"{{ [{domain}] -> [n, {channel}, {height}, {width}] : {guard} }}")
    )
    return access.iterating(
        (*result, contraction, k_h, k_w),
        (
            reached,
            access.Relation(isl.multi_aff(f"{{ [{domain}] -> [co, ci, kh, kw] }}")),
            access.Relation(isl.multi_aff(f"{{ [{domain}] -> [co] }}")),
            access.Relation(isl.multi_aff(f"{{ [{domain}] -> [n, co, oh, ow] }}")),
        ),
    )


@cost.register(Conv2D)
def conv2d_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    input_, weight, bias = cost.input_types(call, ctx)
    output = cost.output_type(call, ctx)
    if not all(
        isinstance(type_, types.TensorType)
        for type_ in (input_, weight, bias, output)
    ):
        raise ValueError("Conv2D cost requires tensor inputs and output")
    global_weight = ctx.type_of(call.args[1])
    logical_weight_shape = weight.shape
    if isinstance(global_weight.layout, types.ShardLayout):
        logical_weight_shape = [1] * len(global_weight.shape)
        axes = int_tuple.mode_axes(int_tuple.compatible_profile(global_weight.layout.layout.shape, global_weight.shape))
        for extent, logical_axis in zip(weight.shape, axes):
            logical_weight_shape[logical_axis] *= extent
    flops = (
        2
        * ir_utils.numel(output)
        * logical_weight_shape[1]
        * logical_weight_shape[2]
        * logical_weight_shape[3]
    )
    return cost.Cost(
        {input_.dtype: flops}, cost.traffic((input_, weight, bias), output)
    )


__all__ = ["Conv2D"]
