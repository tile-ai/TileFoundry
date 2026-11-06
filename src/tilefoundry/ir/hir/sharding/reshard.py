from __future__ import annotations

from dataclasses import replace
from math import prod

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Call, Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.mesh_scope import device_layout
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.ir.types import ComposedLayout, TensorType
from tilefoundry.ir.types.int_tuple import compatible_profile, mode_axes
from tilefoundry.ir.types.layout import Layout, LayoutBase, flatten
from tilefoundry.ir.types.layout_algebra import coalesce, domain_offset
from tilefoundry.ir.types.mesh import axis_keys
from tilefoundry.ir.types.shard_layout import (
    Broadcast,
    Conversion,
    Dynamic,
    Partial,
    ShardAttr,
    ShardLayout,
    Split,
    Varying,
    attrs_by_axis,
    register_strides,
    shard_layout_of,
)
from tilefoundry.ir.types.storage import StorageKind, StorageOwners
from tilefoundry.ir.types.utils import is_literal_shape, static_dim_value
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    identity_access,
    iterating,
    register_access_relation,
)
from tilefoundry.visitor_registry.buffer_alias import aliased_operand, register_buffer_alias


@register_op(dialect='tf', category='sharding')
class Reshard(Op):
    """Convert *x* to a target layout / storage in place, preserving the logical shape."""

    scope_inputs = ()

    x = ParamDef(kind="input", pattern=is_ranked_tensor())
    layout = ParamDef(kind="attribute", annotation=LayoutBase, default=None)
    storage = ParamDef(kind="attribute", default=None)


@register_buffer_alias(Reshard)
def _buffer_alias(call: Call) -> int | None:
    """0 when both sides' structural boxes agree, so the result is a view; None for a copy.

    Each thread holds its registers privately, so the box of an ``rmem`` ShardLayout also
    holds the threads its mesh numbers, and steps by register, where every Split steps 0
    and is recorded with its mesh axis. A plain register end reads as Varying
    on the other end's mesh while comparing those boxes.
    """
    source, result = call.args[0].type, call.type
    if result.storage is not source.storage or not is_literal_shape(result.shape):
        return None
    layouts = tuple(shard_layout_of(held.layout) for held in (source, result))
    states = tuple(attrs_by_axis(sl) for sl in layouts)
    for key in set().union(*(state or {} for state in states)):
        before, after = (state.get(key, Broadcast()) if state is not None else None for state in states)
        if isinstance(before, Partial) or isinstance(after, Partial):
            try:
                if ShardAttr.convert(before, after, True) not in (Conversion.NONE, Conversion.VIEW):
                    return None
            except ValueError:
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
        if shard is None and held.storage is StorageKind.RMEM:
            peer = next((sl for sl in layouts if sl is not None), None)
            if peer is not None:
                shard = ShardLayout(layout, tuple(Varying() for _ in axis_keys(peer.mesh)), peer.mesh)
                layout = shard
        physical = shard.layout if shard is not None else layout
        if shard is not None and isinstance(layout, ComposedLayout):
            physical = replace(layout, outer=physical)
        flat, threads = flatten(physical), None
        if held.storage is StorageKind.RMEM and shard is not None:
            mesh = shard.mesh
            start = mesh.layout.offset if isinstance(mesh.layout, ComposedLayout) else 0
            threads = (mesh.topologies, device_layout(mesh), start)
            sizes = (getattr(topology, "size", 1) for topology in mesh.topologies)
            if any(isinstance(attr, Dynamic) for attr in shard.attrs) or not all(
                isinstance(value, int)
                for value in (start, *sizes, *threads[1].shape, *threads[1].strides)
            ):
                return None
        steps = flat.strides if threads is None else register_strides(shard)
        profile = compatible_profile(flat.shape, held.shape)
        axes = mode_axes(profile)
        independent, splits = set(), []
        if shard is not None:
            for mesh_axis, attr in enumerate(shard.attrs):
                if (
                    isinstance(attr, Split)
                    and static_dim_value(tuple(flatten(shard.mesh.layout.shape))[mesh_axis]) != 1
                    and steps[attr.axis] == 0
                ):
                    position = attr.axis
                    suffix = prod(flat.shape[index] for index in range(position + 1, len(axes))
                                  if axes[index] == axes[position])
                    splits.append((mesh_axis, axes[position], suffix, flat.shape[position]))
                    independent.add(position)
        kept = tuple(
            (extent, step)
            for index, (extent, step) in enumerate(zip(flat.shape, steps, strict=True))
            if index not in independent
        )
        shape, strides = zip(*kept) if kept else ((1,), (0,))
        address = coalesce(Layout(tuple(shape), tuple(strides)), major="row")
        if isinstance(physical, ComposedLayout) and (
            physical.inner is not None or physical.offset != 0
        ):
            address = replace(physical, outer=address)
        boxes.append((tuple(splits), address, threads))
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
    target = getattr(ctx, "target", None)
    owners = target.storage_owners if target else StorageOwners((), {})
    layouts = tuple(shard_layout_of(layout) for layout in (x_ty.layout, new_layout))
    states = tuple(attrs_by_axis(sl) for sl in layouts)
    for key in set().union(*(state or {} for state in states)):
        level = key[0]
        src, dst = (
            ShardAttr.of(state.get(key, Broadcast()) if state is not None else None,
                         owners.private(storage, level))
            for state, storage in zip(states, (x_ty.storage, new_storage), strict=True)
        )
        try:
            ShardAttr.convert(src, dst, owners.private(x_ty.storage, level))
        except ValueError as error:
            ctx.error(call, f"Reshard mesh axis {key}: {error}")
    return TensorType(shape=x_ty.shape, dtype=x_ty.dtype, layout=new_layout, storage=new_storage)


@register_eval(Reshard)
def _eval_reshard(ctx):
    """Retype a packed value; gather one that differs per unit by its own type."""
    return TensorValue(ctx.args[0].tensor.reshard(ctx.result_type))
