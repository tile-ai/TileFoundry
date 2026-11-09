"""HIR ``Reshard`` Op: convert a tensor to a target layout or storage."""

from __future__ import annotations

from dataclasses import replace
from itertools import zip_longest
from math import prod

from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types, utils
from tilefoundry.ir.hir import tensor
from tilefoundry.ir.types import int_tuple, layout_algebra


@expr.register_op(dialect='tf', category='sharding')
class Reshard(expr.Op):
    """Convert *x* to a target layout / storage in place, preserving the logical shape."""

    scope_inputs = ()

    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    layout = expr.ParamDef(kind="attribute", annotation=types.LayoutBase, default=None)
    storage = expr.ParamDef(kind="attribute", default=None)


@access.register_alias(Reshard)
def buffer_alias(call: expr.Call) -> int | None:
    """0 when both sides' structural boxes agree, so the result is a view; None for a copy.

    Each thread holds its registers privately, so the box of an ``rmem`` ShardLayout also
    holds the threads its mesh numbers, and steps by register, where every Split steps 0
    and is recorded with its mesh axis. A plain register end reads as Varying
    on the other end's mesh while comparing those boxes.
    """
    source, result = call.args[0].type, call.type
    if result.storage is not source.storage or not expr.is_literal_shape(result.shape):
        return None
    layouts = tuple(types.shard_layout_of(held.layout) for held in (source, result))
    frame = next((layout.mesh for layout in layouts if layout is not None), None)
    states = tuple(
        utils.attrs_by_axis(held, None if layout is not None else frame)
        for held, layout in zip((source, result), layouts)
    )
    for key in states[0].keys() | states[1].keys():
        before, after = (state.get(key, types.Broadcast()) for state in states)
        if isinstance(before, types.Partial) or isinstance(after, types.Partial):
            if not (
                before == after
                or (isinstance(before, types.Varying) and isinstance(after, types.Partial))
            ):
                return None
    offset, view = 0, call.args[0]
    while isinstance(view, expr.Call):
        physical = view.type.layout
        physical = physical.layout if isinstance(physical, types.ShardLayout) else physical
        if isinstance(physical, types.ComposedLayout):
            break
        position = access.aliased_operand(view)
        if position is None:
            break
        parent = view.args[position]
        if isinstance(view.target, tensor.Slice):
            starts = view.args[1].elements
            if all(expr.static_dim_value(start) is not None for start in starts):
                _, moved = types.domain_offset(starts, parent.type.layout, parent.type.shape)
                offset += moved
        view = parent
    if offset != 0:
        source = replace(source, layout=types.ComposedLayout(None, offset, source.layout))
    boxes = []
    for held in (source, result):
        layout = held.layout
        shard = types.shard_layout_of(layout)
        if shard is None and held.storage is expr.StorageKind.RMEM:
            peer = next((sl for sl in layouts if sl is not None), None)
            if peer is not None:
                shard = types.ShardLayout(
                    layout, tuple(types.Varying() for _ in peer.attrs), peer.mesh
                )
                layout = shard
        physical = shard.layout if shard is not None else layout
        if shard is not None and isinstance(layout, types.ComposedLayout):
            physical = replace(layout, outer=physical)
        flat, threads = types.flatten(physical), None
        if held.storage is expr.StorageKind.RMEM and shard is not None:
            mesh = shard.mesh
            start = mesh.layout.offset if isinstance(mesh.layout, types.ComposedLayout) else 0
            threads = (mesh.topologies, utils.device_layout(mesh), start)
            sizes = (getattr(topology, "size", 1) for topology in mesh.topologies)
            if not all(
                isinstance(value, int)
                for value in (start, *sizes, *threads[1].shape, *threads[1].strides)
            ):
                return None
        steps = flat.strides if threads is None else utils.register_strides(shard)
        profile = int_tuple.compatible_profile(flat.shape, held.shape)
        axes = int_tuple.mode_axes(profile)
        independent, splits = set(), []
        if shard is not None:
            for mesh_axis, attr in enumerate(shard.attrs):
                if (
                    isinstance(attr, types.Split)
                    and expr.static_dim_value(tuple(types.flatten(shard.mesh.layout.shape))[mesh_axis]) != 1
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
        address = layout_algebra.coalesce(types.Layout(tuple(shape), tuple(strides)), major="row")
        if isinstance(physical, types.ComposedLayout) and (
            physical.inner is not None or physical.offset != 0
        ):
            address = replace(physical, outer=address)
        boxes.append((tuple(splits), address, threads))
    return 0 if boxes[0] == boxes[1] else None


@access.register_relation(Reshard)
def reshard_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """Every logical index reads itself. Where those bytes go is a separate fact.

    A reshard moves a value between storages or redistributes it across
    positions; neither changes which logical element the result's index came
    from, so the boundary is exactly identity and never opaque. Which positions
    those indices are is the reader's question, not this one's.
    """
    logical = ctx.type_of(call.args[0])
    rank = len(logical.shape)
    reads = access.identity(rank)
    return access.iterating(
        logical.shape,
        (reads, access.identity(rank)),
    )


@typeinfer.register(Reshard)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    if not call.args:
        ctx.error(call, "missing required input 'x'")
    x_ty = ctx.type_of(call.args[0])
    op = call.target
    if op.storage is expr.StorageKind.UMAT:
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
        return types.TensorType(
            shape=x_ty.shape,
            dtype=x_ty.dtype,
            layout=x_ty.layout,
            storage=x_ty.storage,
        )
    new_storage = op.storage if op.storage is not None else x_ty.storage
    new_layout = op.layout
    target = getattr(ctx, "target", None)
    owners = target.storage_owners if target else expr.StorageOwners((), {})
    layouts = tuple(types.shard_layout_of(layout) for layout in (x_ty.layout, new_layout))
    result = types.TensorType(
        shape=x_ty.shape, dtype=x_ty.dtype, layout=new_layout, storage=new_storage
    )
    frame = next((layout.mesh for layout in layouts if layout is not None), None)
    states = tuple(
        utils.attrs_by_axis(held, None if layout is not None else frame, owners)
        for held, layout in zip((x_ty, result), layouts)
    )
    for key in states[0].keys() | states[1].keys():
        src, dst = (state.get(key, types.Broadcast()) for state in states)
        if isinstance(dst, types.Partial) and src != dst and not isinstance(src, types.Varying):
            ctx.error(
                call,
                f"Reshard mesh axis {key}: {src!r} cannot become {dst!r}; "
                "only Varying or the same Partial becomes Partial",
            )
    return result


@evaluator.register(Reshard)
def eval_reshard(ctx):
    """Retype a packed value; gather one that differs per unit by its own type."""
    return evaluator.TensorValue(ctx.args[0].tensor.reshard(ctx.result_type))


def split_axes(type_) -> "dict[int, int] | None":
    """Which tensor axis each mesh axis splits, or ``None`` when undistributed."""
    layout = types.shard_layout_of(type_.layout)
    if layout is None:
        return None
    return {
        mesh_axis: tensor_axis
        for mesh_axis, tensor_axis in enumerate(types.split_target_axes(layout, type_.shape))
        if tensor_axis is not None
    }


def sent_traffic(source, destination, *, logical_source=None, logical_destination=None) -> tuple[tuple[str, cost.TrafficBytes], ...]:
    """Count boundary movement for split changes and completed partial reductions.

    Whole logical layouts select communicating mesh axes; projected types
    supply each unit's source bytes. At each boundary, retain one uniform
    share across the selected axes and charge the remaining bytes.
    """
    logical_source = source if logical_source is None else logical_source
    logical_destination = destination if logical_destination is None else logical_destination
    before = split_axes(logical_source)
    after = split_axes(logical_destination)
    if before is None or after is None:
        return ()
    completed = {
        axis
        for axis, (pending, remaining) in enumerate(zip_longest(
            types.partial_reductions_by_axis(logical_source.layout),
            types.partial_reductions_by_axis(logical_destination.layout),
        ))
        if pending is not None and remaining is None
    }
    mesh = types.shard_layout_of(source.layout).mesh
    extents = types.flatten(types.flatten(mesh.layout).shape)
    stated = mesh.layout.outer if isinstance(mesh.layout, types.ComposedLayout) else mesh.layout
    levels = types.flatten(
        tuple(
            int_tuple.repeat_like(mode, topology.name)
            for mode, topology in zip(stated.shape, mesh.topologies, strict=True)
        )
    )
    held = utils.tensor_bytes(source)
    moved: dict[str, int] = {}
    for topology in mesh.topologies:
        units = 1
        for mesh_axis, level in enumerate(levels):
            if level != topology.name:
                continue
            if before.get(mesh_axis) != after.get(mesh_axis) or mesh_axis in completed:
                units *= extents[mesh_axis]
        if units > 1:
            moved[topology.name] = held - held // units
    return tuple(
        (name, cost.TrafficBytes(read=value, write=value)) for name, value in sorted(moved.items())
    )


@cost.register(Reshard)
def reshard_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    if access.aliased_operand(call) is not None:
        return cost.Cost({}, cost.idle(call))
    source = cost.input_types(call, ctx)[0]
    destination = cost.output_type(call, ctx)
    sent = sent_traffic(
        source, destination,
        logical_source=ctx.type_of(call.args[0]), logical_destination=ctx.type_of(call),
    )
    return cost.Cost({}, (
        cost.TrafficBytes(read=utils.tensor_bytes(source)),
        cost.TrafficBytes(write=utils.tensor_bytes(destination)),
    ), sent=sent)


__all__ = ["Reshard"]
