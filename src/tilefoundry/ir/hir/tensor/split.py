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
    try_c_order_strides,
)
from tilefoundry.ir.types.shard.shard_layout import Split as SplitAttr
from tilefoundry.ir.types.shard.shard_layout import layout_axis_to_tensor_axis
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    elements_of,
    moves,
    register_access_relation,
)


@register_op
class Split(Op):
    """Multi-output op. `Call.type` is `TupleType` ([types §5](docs/spec/types.md#5-tupletype))."""

    x = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)
    num_splits = ParamDef(kind="attribute", annotation=int)


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
    """Each part reads its own run of the split axis, offset by the parts before it.

    The offsets are the same arithmetic the type inference uses to shape the
    parts, so a part cannot claim a run that its Type does not have. The source
    is read once across all of them, not once per part.
    """
    source = ctx.local_type_of(call.args[0])
    rank = len(source.shape)
    axis = call.target.axis + rank if call.target.axis < 0 else call.target.axis
    extent = source.shape[axis]
    parts = call.target.num_splits
    if not isinstance(extent, int) or isinstance(extent, bool) or extent % parts:
        raise NotImplementedError(
            f"Split access_relation: axis {axis} extent {extent!r} does not "
            f"divide into {parts} static parts"
        )
    chunk = extent // parts
    dims = [f"d{index}" for index in range(rank)]
    domain = ", ".join(dims)
    outputs = []
    for part in range(parts):
        reads = list(dims)
        if part:
            reads[axis] = f"d{axis} + {part * chunk}"
        outputs.append(
            isl.map(
                f"{{ [{domain}] -> [{', '.join(reads)}] : 0 <= d{axis} < {chunk} }}"
            )
        )
    per_part = elements_of(source) // parts
    return AccessRelations(
        inputs=(moves(outputs[0], elements_of(source)),),
        outputs=tuple(moves(item, per_part) for item in outputs),
    )
