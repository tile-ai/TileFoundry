"""Quantize last-axis groups to FP8 vectors with one f32 scale per group.

The result is ``(x_q, x_scale)``. ``x_q`` preserves the input shape;
``x_scale`` replaces the final extent with ``extent // group``.
"""

from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError, TensorValue, TupleValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.ir.types.dim import DimFloorDiv, simplify_dim
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.shard import (
    Layout,
    ShardLayout,
    canonical_shard_layout,
    try_c_order_strides,
)
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Split,
    layout_axis_to_tensor_axis,
    shard_layout_local_shape,
    split_target_axes,
)
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    register_access_relation,
)


@register_op
class Quant(Op):
    """Per-token-group FP8 quantize. Multi-output (x_q, x_scale)."""

    x = ParamDef(kind="input", pattern=Tensor)
    scheme = ParamDef(kind="attribute", annotation=str, default="per_token_group")
    group = ParamDef(kind="attribute", annotation=int, default=128)
    target_dtype = ParamDef(kind="attribute", annotation=DType, default=DType.fp8e4m3)


def _dense_extent(factors) -> int | None:
    extent = 1
    for size, stride in sorted(factors, key=lambda item: item[1]):
        if size == 1:
            continue
        if size <= 0 or stride != extent:
            return None
        extent *= size
    return extent


def _logical_shard_attrs(call, ctx, x_ty, group: int):
    layout = x_ty.layout
    targets = split_target_axes(layout, x_ty.shape)
    last_axis = len(x_ty.shape) - 1
    if last_axis in targets:
        try:
            local_shape = shard_layout_local_shape(layout)
            layout_axes = layout_axis_to_tensor_axis(layout.layout.shape, x_ty.shape)
            strides = layout.layout.strides
            global_factors = []
            local_factors = []
            for physical_axis, logical_axis in enumerate(layout_axes):
                if logical_axis != last_axis or strides is None:
                    continue
                size = static_dim_value(layout.layout.shape[physical_axis])
                local_size = static_dim_value(local_shape[physical_axis])
                stride = static_dim_value(strides[physical_axis])
                if size is None or local_size is None or stride is None:
                    raise ValueError("dynamic last-axis factor")
                global_factors.append((size, stride))
                local_factors.append((local_size, stride))
            global_last = _dense_extent(global_factors)
            local_last = _dense_extent(local_factors)
            aligned = (
                global_last == static_dim_value(x_ty.shape[last_axis])
                and local_last is not None
                and local_last % group == 0
            )
            for mesh_axis, (attr, target) in enumerate(zip(layout.attrs, targets)):
                if not isinstance(attr, Split) or target != last_axis:
                    continue
                split_extent = static_dim_value(layout.layout.shape[attr.axis])
                mesh_extent = static_dim_value(layout.mesh.layout.shape[mesh_axis])
                aligned = (
                    aligned
                    and split_extent is not None
                    and mesh_extent is not None
                    and mesh_extent > 0
                    and split_extent % mesh_extent == 0
                )
        except (IndexError, TypeError, ValueError):
            aligned = False
        if not aligned:
            ctx.error(
                call,
                f"last axis {last_axis} Split cuts through group={group}; use "
                "an explicit Reshard before Quant",
            )
    return tuple(
        Split(target) if isinstance(attr, Split) else attr
        for attr, target in zip(layout.attrs, targets)
    )


def _result_layouts(call, ctx, x_ty, scale_shape, group: int):
    if isinstance(x_ty.layout, ShardLayout):
        if all(isinstance(attr, Broadcast) for attr in x_ty.layout.attrs):
            return None, None
        attrs = _logical_shard_attrs(call, ctx, x_ty, group)
        try:
            return (
                canonical_shard_layout(x_ty.shape, x_ty.layout.mesh, attrs),
                canonical_shard_layout(scale_shape, x_ty.layout.mesh, attrs),
            )
        except ValueError as error:
            ctx.error(
                call,
                f"cannot derive result sharding: {error}; use an explicit "
                "Reshard before Quant",
            )
    if x_ty.layout is None:
        return None, None
    return (
        Layout(shape=x_ty.shape, strides=try_c_order_strides(x_ty.shape)),
        Layout(shape=scale_shape, strides=try_c_order_strides(scale_shape)),
    )


@register_typeinfer(Quant)
def _(call: "Call", ctx: "TypeInferContext") -> TupleType:
    x_ty = ctx.type_of(call.args[0])
    if not x_ty.shape:
        ctx.error(call, "x must be at least rank-1")

    op = call.target
    if op.scheme != "per_token_group":
        ctx.error(
            call,
            f"scheme must be 'per_token_group', got {op.scheme!r}",
        )
    group = op.group
    if isinstance(group, bool) or not isinstance(group, int) or group <= 0:
        ctx.error(
            call,
            f"group must be a positive non-boolean integer, got {group!r}",
        )
    if op.target_dtype != DType.fp8e4m3:
        ctx.error(
            call,
            f"target_dtype must be fp8e4m3, got {op.target_dtype}",
        )

    reject_partials(ctx, call, "x", x_ty.layout)
    last = x_ty.shape[-1]
    if isinstance(last, int):
        if last % group != 0:
            ctx.error(call, f"last dim {last} not divisible by group={group}")
        scale_last = last // group
    else:
        scale_last = simplify_dim(DimFloorDiv, (last, group))
    scale_shape = x_ty.shape[:-1] + (scale_last,)
    q_layout, scale_layout = _result_layouts(call, ctx, x_ty, scale_shape, group)
    x_q_ty = TensorType(
        shape=x_ty.shape,
        dtype=op.target_dtype,
        layout=q_layout,
        storage=x_ty.storage,
    )
    scale_ty = TensorType(
        shape=scale_shape,
        dtype=DType.f32,
        layout=scale_layout,
        storage=x_ty.storage,
    )
    return TupleType(fields=(x_q_ty, scale_ty))


@register_eval(Quant)
def _eval_quant(ctx):
    x = ctx.args[0].data.float()
    group = ctx.op.group
    last = x.shape[-1]
    if last % group:
        raise EvalError(
            f"Quant: runtime last dim {last} not divisible by group={group}"
        )
    grouped = x.reshape(*x.shape[:-1], last // group, group)
    absmax = grouped.abs().amax(dim=-1)
    scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / 448.0)
    quantized = (
        (grouped / scale.unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .reshape(x.shape)
        .to(to_torch_dtype(DType.fp8e4m3))
    )
    return TupleValue(
        elements=(
            TensorValue(data=quantized, type=ctx.result_type.fields[0]),
            TensorValue(data=scale, type=ctx.result_type.fields[1]),
        )
    )


@register_access_relation(Quant)
def _quant_access_relation(call: "Call", ctx: "TypeInferContext") -> AccessRelations:
    """GLOBAL black-box quant.

    - input ``x`` is read element-wise → identity multi_aff over the rank-N
      domain.
    - output ``x_q`` is element-wise identity (same shape).
    - output ``x_scale`` reduces over the in-group offset (last dim divided by
      ``group``); expressed as an isl map ``[..., j] -> [..., j // group]``.
    """
    x_ty = ctx.type_of(call.args[0])
    rank = len(x_ty.shape)
    group = call.target.group

    dims = ", ".join(f"i{k}" for k in range(rank))
    ident = isl.multi_aff(f"{{ [{dims}] -> [{dims}] }}")

    if rank == 0:
        scale_rel = ident  # pragma: no cover
    else:
        outer = ", ".join(f"i{k}" for k in range(rank - 1))
        last = f"i{rank - 1}"
        out_dims = (outer + ", ") if outer else ""
        scale_rel = isl.map(f"{{ [{dims}] -> [{out_dims}floor({last}/{group})] }}")

    return AccessRelations(
        inputs=(ident,),
        outputs=(ident, scale_rel),
    )


__all__ = ["Quant"]
