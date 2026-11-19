"""HIR ``Reshard`` Op: convert a tensor to a target layout or storage."""

from __future__ import annotations

from dataclasses import replace
from itertools import zip_longest
from math import prod

from tilefoundry.interfaces import access, autodiff, cost, evaluator, typeinfer
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


def moved_units(source, destination, owners: expr.StorageOwners) -> "utils.UnitChange | None":
    """How the units holding *destination* differ from those holding *source*.

    Only topologies that hold their own copy of the destination storage count
    ([shard §6.1](docs/spec/shard.md#61-shard-attr-semantics)): elsewhere every
    unit reads the same bytes. None when either end is plain or both are held
    by the same units.
    """
    layouts = tuple(types.shard_layout_of(held.layout) for held in (source, destination))
    if any(layout is None for layout in layouts):
        return None
    owning = {
        name
        for name in utils.topology_names(layouts[1].mesh)
        if owners.private(destination.storage, name)
    }
    return utils.unit_change(layouts[0].mesh, layouts[1].mesh, owning)


def added_units(call: expr.Call, owners: expr.StorageOwners) -> int:
    """Units a Reshard's result holds that its source does not: each receives one copy."""
    change = moved_units(call.args[0].type, call.type, owners)
    return 0 if change is None else change.added


@access.register_alias(Reshard)
def buffer_alias(call: expr.Call, owners: expr.StorageOwners) -> int | None:
    """0 when both sides' structural boxes agree, so the result is a view; None for a copy.

    Each thread holds its registers privately, so the box of an ``rmem`` ShardLayout also
    holds the threads its mesh numbers, and steps by register, where every Split steps 0
    and is recorded with its mesh axis. A plain register end reads as Varying
    on the other end's mesh while comparing those boxes.
    """
    source, result = call.args[0].type, call.type
    if result.storage is not source.storage or not expr.is_literal_shape(result.shape):
        return None
    if added_units(call, owners):
        return None
    layouts = tuple(types.shard_layout_of(held.layout) for held in (source, result))
    shard_mesh = next((layout.mesh for layout in layouts if layout is not None), None)
    states = tuple(
        utils.attrs_by_axis(held, None if layout is not None else shard_mesh)
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
        position = access.aliased_operand(view, owners)
        if position is None:
            break
        parent = view.args[position]
        if isinstance(view.target, tensor.Slice):
            starts = view.args[1].elements
            if all(expr.static_dim_value(start) is not None for start in starts):
                held = parent.type.layout
                held = held.layout if isinstance(held, types.ShardLayout) else held
                grouped = types.unflatten(types.flatten(held), int_tuple.compatible_profile(held.shape, parent.type.shape))
                _, moved = types.domain_offset(tuple(starts), grouped)
                offset += moved
        view = parent
    if offset:
        source = replace(source, layout=types.make_composed_layout(None, offset, source.layout))
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
        flat, threads = types.flatten(types.plain_layout(physical)), None
        if held.storage is expr.StorageKind.RMEM and shard is not None:
            mesh = shard.mesh
            first_index = mesh.layout(0)
            threads = (mesh.topologies, types.plain_layout(utils.flat_layout(mesh)), first_index)
            sizes = (getattr(topology, "size", 1) for topology in mesh.topologies)
            if not all(
                isinstance(value, int)
                for value in (*sizes, first_index, *threads[1].shape, *threads[1].strides)
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
                    and expr.static_dim_value(shard.mesh.shape[mesh_axis]) != 1
                    and steps[attr.axis] == 0
                ):
                    mode = attr.axis
                    suffix = prod(flat.shape[index] for index in range(mode + 1, len(axes))
                                  if axes[index] == axes[mode])
                    splits.append((mesh_axis, axes[mode], suffix, flat.shape[mode]))
                    independent.add(mode)
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
    units; neither changes which logical element the result's index came
    from, so the boundary is exactly identity and never opaque. Which modes
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
    shard_mesh = next((layout.mesh for layout in layouts if layout is not None), None)
    states = tuple(
        utils.attrs_by_axis(held, None if layout is not None else shard_mesh, owners)
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
    change = moved_units(x_ty, result, owners)
    if change is not None and change.overlap_in_part and any(
        isinstance(attr, (types.Split, types.Partial))
        for layout in layouts
        for attr in layout.attrs
    ):
        ctx.error(
            call,
            f"mesh slices that share some {change.topology} units, but not all, cannot "
            "exchange a Split or Partial value; reshard it to Broadcast on its own "
            "units first, or move it to units it does not hold",
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


def sent_traffic(
    source,
    destination,
    *,
    logical_source=None,
    logical_destination=None,
    whole=False,
    owners: expr.StorageOwners,
) -> tuple[tuple[str, cost.TrafficBytes], ...]:
    """Count boundary movement for split changes and completed partial reductions.

    Whole logical layouts select communicating mesh axes; projected types
    supply each unit's source bytes. At each boundary, retain one uniform
    share across the selected mesh axes and charge the remaining bytes. A
    destination holding units the source does not instead charges one added
    unit's received bytes at the coarsest topology whose units differ; the
    *whole* view counts every added unit's copy.
    """
    logical_source = source if logical_source is None else logical_source
    logical_destination = destination if logical_destination is None else logical_destination
    change = moved_units(logical_source, logical_destination, owners)
    if change is not None and change.added:
        received = utils.tensor_bytes(destination)
        if whole:
            mesh = types.shard_layout_of(destination.layout).mesh
            held = utils.local_type_of(
                destination, topology=change.topology, topologies=mesh.topologies
            )
            received = utils.tensor_bytes(held) * change.added
        return ((change.topology, cost.TrafficBytes(read=received, write=received)),)
    before = split_axes(logical_source)
    after = split_axes(logical_destination)
    if before is None or after is None:
        return ()
    completed = {
        mesh_axis
        for mesh_axis, (pending, remaining) in enumerate(zip_longest(
            types.partial_reductions_by_axis(logical_source.layout),
            types.partial_reductions_by_axis(logical_destination.layout),
        ))
        if pending is not None and remaining is None
    }
    mesh = types.shard_layout_of(source.layout).mesh
    extents = mesh.shape
    axis_topologies = tuple(topology for topology, _ in utils.axis_keys(mesh))
    held = utils.tensor_bytes(source)
    moved: dict[str, int] = {}
    for topology in mesh.topologies:
        units = 1
        for mesh_axis, axis_topology in enumerate(axis_topologies):
            if axis_topology != topology.name:
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
    owners = access.target_owners(ctx.target)
    if access.aliased_operand(call, owners) is not None:
        return cost.Cost({}, cost.idle(call))
    source = cost.input_types(call, ctx)[0]
    destination = cost.output_type(call, ctx)
    sent = sent_traffic(
        source, destination,
        logical_source=ctx.type_of(call.args[0]), logical_destination=ctx.type_of(call),
        whole=ctx.topology is None,
        owners=owners,
    )
    return cost.Cost({}, (
        cost.TrafficBytes(read=utils.tensor_bytes(source)),
        cost.TrafficBytes(write=utils.tensor_bytes(destination)),
    ), sent=sent)


@autodiff.register(Reshard)
def _(call: expr.Call, cotangent: expr.Expr, ctx: autodiff.Context) -> tuple:
    """The gradient resharded back to the input's layout, with ``Partial`` read as ``Broadcast``.

    This is DTensor's ``_redistribute_backward``: one Reshard, whatever mesh
    axes it changes.
    """
    source = ctx.type_of(call.args[0])
    wanted = autodiff.gradient_layout(source.layout)
    held = ctx.type_of(cotangent)
    if (held.layout, held.storage) == (wanted, source.storage):
        return (cotangent,)
    return (ctx.call("reshard", cotangent, layout=types.normalize_layout(wanted), storage=source.storage),)


__all__ = ["Reshard", "added_units"]
