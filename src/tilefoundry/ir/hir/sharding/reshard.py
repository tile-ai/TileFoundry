from __future__ import annotations

from dataclasses import replace
from itertools import zip_longest
from math import prod

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Call, Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.ir.types import ComposedLayout, TensorType
from tilefoundry.ir.types.int_tuple import compatible_profile, mode_axes
from tilefoundry.ir.types.layout import Layout, LayoutBase, flatten
from tilefoundry.ir.types.layout_algebra import coalesce, domain_offset
from tilefoundry.ir.types.shard_layout import (
    ShardLayout,
    Split,
    shard_layout_of,
)
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.types.utils import is_literal_shape, static_dim_value
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    identity_access,
    iterating,
    register_access_relation,
)
from tilefoundry.visitor_registry.buffer_alias import aliased_operand, register_buffer_alias
from tilefoundry.visitor_registry.shard_propagate import partial_reductions_by_axis


@register_op(dialect='tf', category='sharding')
class Reshard(Op):
    """Convert *x* to a target layout / storage in place, preserving the logical shape."""

    scope_inputs = ()

    x = ParamDef(kind="input", pattern=is_ranked_tensor())
    layout = ParamDef(kind="attribute", annotation=LayoutBase, default=None)
    storage = ParamDef(kind="attribute", default=None)


@register_buffer_alias(Reshard)
def _buffer_alias(call: Call) -> int | None:
    source, result = call.args[0].type, call.type
    if result.storage is not source.storage or not is_literal_shape(result.shape):
        return None
    if any(
        before != after
        for before, after in zip_longest(
            partial_reductions_by_axis(source.layout),
            partial_reductions_by_axis(result.layout),
        )
    ):
        return None
    from tilefoundry.ir.hir.tensor.slice import Slice  # noqa: PLC0415

    offset, view = 0, call.args[0]
    while isinstance(view, Call):
        physical = view.type.layout
        physical = physical.layout if isinstance(physical, ShardLayout) else physical
        if isinstance(physical, ComposedLayout):
            break
        position = aliased_operand(view)
        if position is None:
            break
        parent = view.args[position]
        if isinstance(view.target, Slice):
            starts = view.args[1].elements
            if all(static_dim_value(start) is not None for start in starts):
                _, moved = domain_offset(starts, parent.type.layout, parent.type.shape)
                offset += moved
        view = parent
    if offset != 0:
        source = replace(source, layout=ComposedLayout(None, offset, source.layout))
    boxes = []
    for held in (source, result):
        layout = held.layout
        shard = shard_layout_of(layout)
        physical = shard.layout if shard is not None else layout
        if shard is not None and isinstance(layout, ComposedLayout):
            physical = replace(layout, outer=physical)
        flat = flatten(physical)
        profile = compatible_profile(flat.shape, held.shape)
        axes = mode_axes(profile)
        independent, splits = set(), []
        if shard is not None:
            for mesh_axis, attr in enumerate(shard.attrs):
                if (
                    isinstance(attr, Split)
                    and static_dim_value(tuple(flatten(shard.mesh.layout.shape))[mesh_axis]) != 1
                    and flat.strides[attr.axis] == 0
                ):
                    position = attr.axis
                    suffix = prod(flat.shape[index] for index in range(position + 1, len(axes))
                                  if axes[index] == axes[position])
                    splits.append((mesh_axis, axes[position], suffix, flat.shape[position]))
                    independent.add(position)
        kept = tuple(
            (extent, step)
            for index, (extent, step) in enumerate(zip(flat.shape, flat.strides, strict=True))
            if index not in independent
        )
        shape, strides = zip(*kept) if kept else ((1,), (0,))
        address = coalesce(Layout(tuple(shape), tuple(strides)), major="row")
        if isinstance(physical, ComposedLayout) and (
            physical.inner is not None or physical.offset != 0
        ):
            address = replace(physical, outer=address)
        boxes.append((tuple(splits), address))
    return 0 if boxes[0] == boxes[1] else None


@register_access_relation(Reshard)
def _reshard_access(call: "Call", ctx) -> tuple[AccessRelation, ...]:
    """Every logical index reads itself. Where those bytes go is a separate fact.

    A reshard moves a value between storages or redistributes it across
    positions; neither changes which logical element the result's index came
    from, so the boundary is exactly identity and never opaque. Which positions
    those indices are is the reader's question, not this one's.
    """
    logical = ctx.type_of(call.args[0])
    rank = len(logical.shape)
    reads = identity_access(rank)
    return iterating(
        logical.shape,
        (reads, identity_access(rank)),
    )


@register_typeinfer(Reshard)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    if not call.args:
        ctx.error(call, "missing required input 'x'")
    x_ty = ctx.type_of(call.args[0])
    op = call.target
    if op.storage is StorageKind.UMAT:
        ctx.error(
            call,
            "Reshard: destination storage cannot be unmaterialized (umat); "
            "reshard targets a concrete residency",
        )
    storage_changed = op.storage is not None and op.storage != x_ty.storage
    if op.layout is None:
        if storage_changed:
            ctx.error(
                call,
                "Reshard: storage change requires an explicit `layout=` argument.",
            )
        return TensorType(
            shape=x_ty.shape,
            dtype=x_ty.dtype,
            layout=x_ty.layout,
            storage=x_ty.storage,
        )
    new_storage = op.storage if op.storage is not None else x_ty.storage
    new_layout = op.layout
    return TensorType(shape=x_ty.shape, dtype=x_ty.dtype, layout=new_layout, storage=new_storage)


@register_eval(Reshard)
def _eval_reshard(ctx):

    return TensorValue(data=ctx.args[0].data, type=ctx.result_type)
