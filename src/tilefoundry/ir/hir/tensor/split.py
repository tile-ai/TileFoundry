from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, TupleValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType, TupleType
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.shard import (
    Broadcast,
    Layout,
    Partial,
    ShardLayout,
    canonical_shard_layout,
    shard_layout_of,
    try_c_order_strides,
)
from tilefoundry.ir.types.shard.shard_layout import Split as ShardSplit
from tilefoundry.ir.types.shard.shard_layout import Split as SplitAttr
from tilefoundry.ir.types.shard.shard_layout import layout_axis_to_tensor_axis, split_target_axes
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    identity_access,
    iterating,
    register_access_relation,
)


@register_op
class Split(Op):
    """Multi-output op. `Call.type` is `TupleType` ([types §5](docs/spec/types.md#5-tupletype))."""

    x = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)
    num_splits = ParamDef(kind="attribute", annotation=int)


def _reject_redistribution(ctx, call: "Call", x_ty, axis: int, parts: int) -> None:
    """Refuse dividing an axis the layout already hands out in slices.

    Each participant would hold one whole part, and saying which one means
    knowing where that participant starts. This wave carries extents and not
    offsets, so the redistribution is refused at the contract the author wrote
    against rather than surfacing later as an analysis that cannot answer.
    """
    layout = shard_layout_of(x_ty.layout)
    if layout is None:
        return
    targets = split_target_axes(layout, x_ty.shape)
    mesh = layout.mesh.layout.shape if layout.mesh is not None else ()
    for mesh_axis, attr in enumerate(layout.attrs):
        divides = mesh_axis < len(mesh) and mesh[mesh_axis] > 1
        if isinstance(attr, ShardSplit) and divides and targets[mesh_axis] == axis:
            ctx.error(
                call,
                f"Split: axis {axis} is divided into {parts} parts and is "
                "already Split across participants, so which part a participant "
                "holds depends on an offset this analysis does not carry; "
                "reshard to a replicated layout before the split",
            )


@register_typeinfer(Split)
def _(call: "Call", ctx: "TypeInferContext") -> TupleType:
    x_ty = ctx.type_of(call.args[0])
    raw_axis = call.target.axis
    rank = len(x_ty.shape)
    axis = raw_axis + rank if raw_axis < 0 else raw_axis
    if not (0 <= axis < rank):
        ctx.error(call, f"axis {raw_axis} out of range for rank {rank}")
    n = call.target.num_splits
    if n <= 0:
        ctx.error(call, f"num_splits must be positive, got {n}")
    _reject_redistribution(ctx, call, x_ty, axis, n)
    orig = x_ty.shape[axis]
    v = static_dim_value(orig)
    if v is not None:
        if v % n != 0:
            ctx.error(call, f"axis {axis} extent {v} not divisible by {n}")
        part_len = v // n
    else:
        part_len = orig
    part_shape = list(x_ty.shape)
    part_shape[axis] = part_len
    part_shape = tuple(part_shape)
    if isinstance(x_ty.layout, ShardLayout):
        layout_to_tensor = layout_axis_to_tensor_axis(x_ty.layout.layout.shape, x_ty.shape)
        attrs = tuple(
            SplitAttr(layout_to_tensor[attr.axis])
            if isinstance(attr, SplitAttr)
            else Partial(attr.reduction)
            if isinstance(attr, Partial)
            else Broadcast()
            for attr in x_ty.layout.attrs
        )
        part_layout = canonical_shard_layout(part_shape, x_ty.layout.mesh, attrs)
    elif x_ty.layout is None:
        part_layout = None
    else:
        part_layout = Layout(shape=part_shape, strides=try_c_order_strides(part_shape))
    part_ty = TensorType(
        shape=part_shape, dtype=x_ty.dtype, layout=part_layout, storage=x_ty.storage
    )
    return TupleType(fields=tuple(part_ty for _ in range(n)))


@register_eval(Split)
def _eval_split(ctx):
    source = ctx.args[0].data
    axis = ctx.op.axis + source.ndim if ctx.op.axis < 0 else ctx.op.axis
    n = ctx.op.num_splits
    extent = source.shape[axis]
    if extent % n:
        raise ValueError(f"Split: runtime axis {axis} extent {extent} is not divisible by {n}")
    parts = torch.chunk(source, n, dim=axis)
    return TupleValue(
        tuple(
            TensorValue(data=part, type=field_type)
            for part, field_type in zip(parts, ctx.result_type.fields)
        )
    )


@register_access_relation(Split)
def _split_access(call: "Call", ctx) -> AccessRelations:
    """One space, the source's, with each part written on its own run of it.

    Every element of the source becomes an element of exactly one part, so the
    space to walk is the source's and each output is partial in it: the run of
    the split axis that part covers, shifted to that part's own coordinates.
    Walking a part's own coordinates instead would say every part reads the
    first one.
    """
    source = ctx.type_of(call.args[0])
    rank = len(source.shape)
    axis = call.target.axis + rank if call.target.axis < 0 else call.target.axis
    parts = call.target.num_splits
    extent = source.shape[axis]
    if not isinstance(extent, int) or isinstance(extent, bool) or extent % parts:
        raise NotImplementedError(
            f"Split access_relation: axis {axis} extent {extent!r} does not "
            f"divide into {parts} static parts"
        )
    chunk = extent // parts
    domain = ", ".join(f"d{index}" for index in range(rank))
    written = []
    for part in range(parts):
        writes = [f"d{index}" for index in range(rank)]
        begin = part * chunk
        if begin:
            writes[axis] = f"d{axis} - {begin}"
        written.append(
            AffineAccess(
                isl.map(
                    f"{{ [{domain}] -> [{', '.join(writes)}] : "
                    f"{begin} <= d{axis} < {begin + chunk} }}"
                )
            )
        )

    return iterating(
        source.shape,
        AccessRelations(
            inputs=(BoundaryRelation(identity_access(rank)),),
            outputs=tuple(BoundaryRelation(item) for item in written),
        ),
    )
