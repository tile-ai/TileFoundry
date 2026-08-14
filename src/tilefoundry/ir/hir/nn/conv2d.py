from __future__ import annotations

import isl
import torch.nn.functional as F

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Expr, Op
from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import check_multilinear_partials
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import DimAdd, DimFloorDiv, DimSub, simplify_dim
from tilefoundry.ir.types.dim_isl import normalize_dim
from tilefoundry.ir.types.shape_helpers import i64_const, static_dim_value
from tilefoundry.ir.types.shard import Layout, try_c_order_strides
from tilefoundry.ir.types.shard.shard_layout import shard_layout_of, split_target_axes
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    build_relation,
    register_type_relation,
)
from tilefoundry.visitor_registry.isl_utility import to_domain
from tilefoundry.visitor_registry.shard_propagate import (
    derive_output_shard_layout,
    partial_reductions_by_axis,
)


@register_op
class Conv2D(Op):
    input = ParamDef(kind="input", pattern=Tensor)
    weight = ParamDef(kind="input", pattern=Tensor)
    bias = ParamDef(kind="input", pattern=Tensor)
    stride = ParamDef(kind="attribute", annotation=tuple)
    padding = ParamDef(kind="attribute", annotation=tuple)
    dilation = ParamDef(kind="attribute", annotation=tuple)
    groups = ParamDef(kind="attribute", annotation=int)


def _i64(value: int) -> Constant:
    return i64_const(value)


def _as_expr(v):
    if isinstance(v, Expr):
        return v
    return _i64(int(v))


def _out_spatial(in_dim: Expr, k: int, s: int, p: int, d: int) -> Expr:
    """Compute (in + 2*p - d*(k-1) - 1) // s + 1, keeping symbolic dims alive.

    If `in_dim` is a Constant the result is also a Constant; otherwise we
    build a `dim.*` Expr tree so downstream passes can simplify.
    """
    eff_k = d * (k - 1) + 1

    add_pad = simplify_dim(DimAdd, (in_dim, _i64(2 * p)))
    sub_k = simplify_dim(DimSub, (add_pad, _i64(eff_k)))
    div_s = simplify_dim(DimFloorDiv, (sub_k, _i64(s)))
    plus_1 = simplify_dim(DimAdd, (div_s, _i64(1)))
    return normalize_dim(plus_1)


def _pair(call, ctx, name: str, values: tuple, *, positive: bool) -> tuple[int, int]:
    if len(values) != 2:
        ctx.error(call, f"{name} must be length-2, got {values}")
    for value in values:
        valid = isinstance(value, int) and not isinstance(value, bool)
        valid = valid and (value > 0 if positive else value >= 0)
        if not valid:
            qualifier = "positive" if positive else "non-negative"
            ctx.error(call, f"{name} values must be {qualifier}, got {values}")
    return values


def _static_extent(call, ctx, name: str, dim) -> int:
    value = static_dim_value(dim)
    if value is None:
        ctx.error(call, f"{name} must be static, got {dim}")
    return value


def _validate_conv2d(call, ctx, x, weight, bias):
    op = call.target
    if len(x.shape) != 4 or len(weight.shape) != 4:
        ctx.error(call, "expects rank-4 input and weight (NCHW / OIHW)")
    if len(bias.shape) != 1:
        ctx.error(call, f"bias must be rank-1, got shape {bias.shape}")
    if x.dtype != weight.dtype:
        ctx.error(call, f"weight dtype {weight.dtype} must match input dtype {x.dtype}")
    if bias.dtype != x.dtype:
        ctx.error(call, f"bias dtype {bias.dtype} must match input dtype {x.dtype}")

    stride = _pair(call, ctx, "stride", op.stride, positive=True)
    padding = _pair(call, ctx, "padding", op.padding, positive=False)
    dilation = _pair(call, ctx, "dilation", op.dilation, positive=True)
    if not isinstance(op.groups, int) or isinstance(op.groups, bool) or op.groups <= 0:
        ctx.error(call, f"groups must be positive, got {op.groups}")

    in_channels = static_dim_value(x.shape[1])
    out_channels = static_dim_value(weight.shape[0])
    k_h = _static_extent(call, ctx, "kernel height", weight.shape[2])
    k_w = _static_extent(call, ctx, "kernel width", weight.shape[3])
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


def _output_shape(x, weight, stride, padding, dilation, k_h, k_w) -> tuple:
    height = _out_spatial(
        x.shape[2], k_h, stride[0], padding[0], dilation[0]
    )
    width = _out_spatial(
        x.shape[3], k_w, stride[1], padding[1], dilation[1]
    )
    static_height = static_dim_value(height)
    static_width = static_dim_value(width)
    return (
        x.shape[0],
        weight.shape[0],
        static_height if static_height is not None else height,
        static_width if static_width is not None else width,
    )


def _require_exact_partial_state(call, ctx, x, weight, bias) -> None:
    named = (("input", x), ("weight", weight), ("bias", bias))
    placed = [
        (index, name, layout)
        for index, (name, type_) in enumerate(named)
        if (layout := shard_layout_of(type_.layout)) is not None
    ]
    if placed:
        mesh = placed[0][2].mesh
        for index, name, layout in placed[1:]:
            if layout.mesh != mesh:
                ctx.error(
                    call,
                    f"{name} (input {index}) references a different mesh; use "
                    "an explicit Reshard before Conv2D",
                )
    else:
        mesh = None

    check_multilinear_partials(ctx, call, (("input", x), ("weight", weight)))
    contraction_splits: dict[int, tuple[str, int]] = {}
    channel_split_axes: dict[str, set[int]] = {"input": set(), "weight": set()}
    for name, type_, axes in (
        ("input", x, {1: 4}),
        ("weight", weight, {1: 4, 2: 5, 3: 6}),
    ):
        layout = shard_layout_of(type_.layout)
        if layout is None:
            continue
        targets = split_target_axes(layout, type_.shape)
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

    for mesh_axis in sorted(
        channel_split_axes["input"] ^ channel_split_axes["weight"]
    ):
        missing = (
            "weight"
            if mesh_axis in channel_split_axes["input"]
            else "input"
        )
        ctx.error(
            call,
            f"{missing} must carry a matching input-channel Split on mesh axis "
            f"{mesh_axis}; use an explicit Reshard before Conv2D",
        )

    required_partial_axes = set(contraction_splits)
    for type_ in (x, weight):
        required_partial_axes.update(
            axis
            for axis, reduction in enumerate(partial_reductions_by_axis(type_.layout))
            if reduction is not None
        )

    bias_reductions = partial_reductions_by_axis(bias.layout)
    for mesh_axis in sorted(required_partial_axes):
        reduction = (
            bias_reductions[mesh_axis]
            if mesh_axis < len(bias_reductions)
            else None
        )
        if reduction != "sum" or not (
            (bias_layout := shard_layout_of(bias.layout)) is not None
            and bias_layout.mesh == mesh
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


@register_type_relation(Conv2D)
def _conv2d_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    x, weight, _bias = input_types
    op = call.target
    k_h = static_dim_value(weight.shape[2])
    k_w = static_dim_value(weight.shape[3])
    in_per_group = (
        x.shape[1]
        if op.groups == 1
        else static_dim_value(x.shape[1]) // op.groups
    )
    out_per_group = (
        None
        if op.groups == 1
        else static_dim_value(weight.shape[0]) // op.groups
    )
    out_shape = _output_shape(
        x, weight, op.stride, op.padding, op.dilation, k_h, k_w
    )
    domain, param_map = to_domain((*out_shape, in_per_group, k_h, k_w))
    dims = [f"d{i}" for i in range(7)]
    source = f"[{', '.join(dims)}]"
    input_channel = (
        "d4"
        if op.groups == 1
        else f"floor(d1/{out_per_group})*{in_per_group}+d4"
    )

    def spatial(out_dim: str, kernel_dim: str, stride: int, pad: int, dilation: int, k: int):
        terms = [out_dim if stride == 1 else f"{stride}*{out_dim}"]
        if k != 1:
            terms.append(kernel_dim if dilation == 1 else f"{dilation}*{kernel_dim}")
        expression = "+".join(terms)
        return expression if pad == 0 else f"{expression}-{pad}"

    input_map = isl.map(
        f"{{ {source} -> [d0, {input_channel}, "
        f"{spatial('d2', 'd5', op.stride[0], op.padding[0], op.dilation[0], k_h)}, "
        f"{spatial('d3', 'd6', op.stride[1], op.padding[1], op.dilation[1], k_w)}] }}"
    )
    weight_map = isl.map(f"{{ {source} -> [d1, d4, d5, d6] }}")
    bias_map = isl.map(f"{{ {source} -> [d1] }}")
    output_map = isl.map(f"{{ {source} -> [d0, d1, d2, d3] }}")
    return AccessRelationResult(
        domain=domain,
        maps=(input_map, weight_map, bias_map, output_map),
        param_map=param_map,
    )


@register_typeinfer(Conv2D)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x = ctx.type_of(call.args[0])
    w = ctx.type_of(call.args[1])
    bias = ctx.type_of(call.args[2])
    stride, padding, dilation, _groups, k_h, k_w = _validate_conv2d(
        call, ctx, x, w, bias
    )
    out_shape = _output_shape(x, w, stride, padding, dilation, k_h, k_w)
    for name, extent in zip(("height", "width"), out_shape[2:]):
        static = static_dim_value(extent)
        if static is not None and static <= 0:
            ctx.error(call, f"output {name} extent must be positive, got {static}")

    _require_exact_partial_state(call, ctx, x, w, bias)
    relation = build_relation(call, (x, w, bias), ctx)
    try:
        shard = derive_output_shard_layout(
            (x, w, bias),
            relation,
            out_shape,
            partial_reduction_dims=frozenset({4, 5, 6}),
            fresh_strides=True,
        )
    except ValueError as error:
        ctx.error(
            call,
            f"cannot derive input ownership: {error}; use an explicit Reshard "
            "before Conv2D",
        )
    layout = shard or Layout(
        shape=out_shape, strides=try_c_order_strides(out_shape)
    )
    return TensorType(
        shape=out_shape,
        dtype=x.dtype,
        layout=layout,
        storage=x.storage,
    )


@register_eval(Conv2D)
def _eval_conv2d(ctx):
    input_, weight, bias = (arg.data for arg in ctx.args)
    out = F.conv2d(
        input_,
        weight,
        bias,
        stride=ctx.op.stride,
        padding=ctx.op.padding,
        dilation=ctx.op.dilation,
        groups=ctx.op.groups,
    )
    return TensorValue(data=out, type=ctx.result_type)
