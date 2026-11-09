"""Quantize last-axis groups to FP8 vectors with one f32 scale per group.

The result is ``(x_q, x_scale)``. ``x_q`` preserves the input shape;
``x_scale`` replaces the final extent with ``extent // group``.
"""

from __future__ import annotations

import isl

from tilefoundry import core, services, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils
from tilefoundry.ir.types import int_tuple


@expr.register_op(dialect='tf', category='tensor')
class Quant(expr.Op):
    """Per-token-group FP8 quantize. Multi-output (x_q, x_scale)."""

    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    scheme = expr.ParamDef(kind="attribute", annotation=str, default="per_token_group")
    group = expr.ParamDef(kind="attribute", annotation=int, default=128)
    target_dtype = expr.ParamDef(kind="attribute", annotation=expr.DType, default=expr.DType.fp8e4m3)


def dense_extent(factors) -> int | None:
    extent = 1
    for size, stride in sorted(factors, key=lambda item: item[1]):
        if size == 1:
            continue
        if size <= 0 or stride != extent:
            return None
        extent *= size
    return extent


def logical_shard_attrs(call, ctx, x_ty, group: int):
    layout = x_ty.layout
    targets = types.split_target_axes(layout, x_ty.shape)
    last_axis = len(x_ty.shape) - 1
    if last_axis in targets:
        try:
            local_shape = types.shard_layout_local_shape(layout)
            layout_axes = int_tuple.mode_axes(int_tuple.compatible_profile(layout.layout.shape, x_ty.shape))
            strides = layout.layout.strides
            global_factors = []
            local_factors = []
            for physical_axis, logical_axis in enumerate(layout_axes):
                if logical_axis != last_axis:
                    continue
                size = expr.static_dim_value(layout.layout.shape[physical_axis])
                local_size = expr.static_dim_value(local_shape[physical_axis])
                stride = expr.static_dim_value(strides[physical_axis])
                if size is None or local_size is None or stride is None:
                    raise ValueError("dynamic last-axis factor")
                global_factors.append((size, stride))
                local_factors.append((local_size, stride))
            global_last = dense_extent(global_factors)
            local_last = dense_extent(local_factors)
            aligned = (
                global_last == expr.static_dim_value(x_ty.shape[last_axis])
                and local_last is not None
                and local_last % group == 0
            )
            for mesh_axis, (attr, target) in enumerate(zip(layout.attrs, targets)):
                if not isinstance(attr, types.Split) or target != last_axis:
                    continue
                split_extent = expr.static_dim_value(layout.layout.shape[attr.axis])
                mesh_extent = expr.static_dim_value(types.flatten(layout.mesh.layout).shape[mesh_axis])
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
        types.Split(target) if isinstance(attr, types.Split) else attr
        for attr, target in zip(layout.attrs, targets)
    )


def result_layouts(call, ctx, x_ty, scale_shape, group: int):
    if isinstance(x_ty.layout, types.ShardLayout):
        attrs = logical_shard_attrs(call, ctx, x_ty, group)
        try:
            return (
                types.ShardLayout.from_logical(
                    x_ty.shape,
                    x_ty.layout.mesh,
                    attrs,
                    storage=x_ty.storage,
                    owners=ctx.target.storage_owners
                    if ctx is not None and ctx.target is not None
                    else expr.StorageOwners((), {}),
                ),
                types.ShardLayout.from_logical(
                    scale_shape,
                    x_ty.layout.mesh,
                    attrs,
                    storage=x_ty.storage,
                    owners=ctx.target.storage_owners
                    if ctx is not None and ctx.target is not None
                    else expr.StorageOwners((), {}),
                ),
            )
        except ValueError as error:
            ctx.error(
                call,
                f"cannot derive result sharding: {error}; use an explicit Reshard before Quant",
            )
    return (
        types.Layout(x_ty.shape),
        types.Layout(scale_shape),
    )


@typeinfer.register(Quant)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TupleType:
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
    if op.target_dtype != expr.DType.fp8e4m3:
        ctx.error(
            call,
            f"target_dtype must be fp8e4m3, got {op.target_dtype}",
        )

    ir_utils.reject_partials("x", x_ty.layout)
    last = x_ty.shape[-1]
    if isinstance(last, int):
        if last % group != 0:
            ctx.error(call, f"last dim {last} not divisible by group={group}")
        scale_last = last // group
    else:
        scale_last = services.simplify_dim("floordiv", (last, group))
    scale_shape = x_ty.shape[:-1] + (scale_last,)
    q_layout, scale_layout = result_layouts(call, ctx, x_ty, scale_shape, group)
    x_q_ty = types.TensorType(
        shape=x_ty.shape,
        dtype=op.target_dtype,
        layout=q_layout,
        storage=x_ty.storage,
    )
    scale_ty = types.TensorType(
        shape=scale_shape,
        dtype=expr.DType.f32,
        layout=scale_layout,
        storage=x_ty.storage,
    )
    return types.TupleType(fields=(x_q_ty, scale_ty))


@evaluator.register(Quant)
def eval_quant(ctx):
    fields = ctx.result_type.fields

    def quant(x):
        x = x.float()
        group = ctx.op.group
        last = x.shape[-1]
        if last % group:
            raise evaluator.EvalError(f"Quant: runtime last dim {last} not divisible by group={group}")
        grouped = x.reshape(*x.shape[:-1], last // group, group)
        absmax = grouped.abs().amax(dim=-1)
        scale = utils.torch.where(absmax == 0, utils.torch.ones_like(absmax), absmax / 448.0)
        quantized = (
            (grouped / scale.unsqueeze(-1))
            .clamp(-448.0, 448.0)
            .reshape(x.shape)
            .to(evaluator.to_torch_dtype(expr.DType.fp8e4m3))
        )
        return quantized, scale

    x = ctx.args[0].tensor
    if x.is_global:
        held = quant(x.tensor)
    else:
        each = [(p, quant(t)) for p, t in x.tensor]
        held = tuple(tuple((p, out[i]) for p, out in each) for i in range(len(fields)))
    return evaluator.TupleValue(
        tuple(evaluator.TensorValue(core.ShardTensor(field, data)) for field, data in zip(fields, held, strict=True))
    )


@access.register_relation(Quant)
def quant_access_relation(call: "expr.Call", ctx: "TypeInferContext") -> tuple[access.Relation, ...]:
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
    ident = access.Relation(isl.map(f"{{ [{dims}] -> [{dims}] }}"))

    if rank == 0:
        scale_rel = ident
    else:
        outer = ", ".join(f"i{k}" for k in range(rank - 1))
        last = f"i{rank - 1}"
        out_dims = (outer + ", ") if outer else ""
        scale_rel = access.Relation(isl.map(f"{{ [{dims}] -> [{out_dims}floor({last}/{group})] }}"))

    return access.iterating(
        x_ty.shape,
        (ident, ident, scale_rel),
    )


__all__ = ["Quant"]


@cost.register(Quant)
def quant_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    inputs = cost.input_types(call, ctx)
    output = cost.output_type(call, ctx)
    source = inputs[0]
    if not isinstance(source, types.TensorType):
        raise ValueError("Quant cost requires a tensor input")
    return cost.Cost({expr.DType.f32: 4 * ir_utils.numel(source)}, cost.traffic(inputs, output))
