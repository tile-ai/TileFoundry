"""HIR ``Bitcast`` Op: reinterpret a tensor's bytes under a new shape and layout."""

from __future__ import annotations

from functools import cache
from math import prod

import isl

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils
from tilefoundry.ir.types import int_tuple


@expr.register_op(name="bitcast", dialect='tf', category='tensor')
class Bitcast(expr.Op):
    """Reinterpret a tensor's bytes under a new shape and layout."""

    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    layout = expr.ParamDef(kind="attribute", annotation=types.LayoutBase)


def address_points(layout, count: int, logical_shape: tuple):
    """Yield logical index, physical unit ids, full address, and local address.

    Both byte reinterpretation and storage aliasing read the same factored
    shard domain; their reducers decide whether repeated locations are legal.
    Varying axes use a representative copy at coordinate zero; every copy uses
    the same mapping within its own registers.
    """
    shard = types.shard_layout_of(layout)
    base = shard.layout if shard is not None else layout
    whole = {}
    flat = types.flatten(base)
    grouped = types.unflatten(types.Layout(flat.shape, types.compact_col_major(flat.shape)),
                        int_tuple.compatible_profile(base.shape, logical_shape))
    logical_steps = types.compact_col_major(logical_shape)
    for coordinate in range(count):
        logical = types.idx2crd(coordinate, logical_shape, logical_steps)
        address = types.apply(base, types.crd2idx(logical, grouped.shape, grouped.strides))
        if shard is None:
            yield coordinate, (), address, address
        else:
            whole[address] = coordinate
    if shard is None:
        return
    shift = layout.offset if isinstance(layout, types.ComposedLayout) else 0
    layout = shard
    if len(whole) != count:
        raise ValueError(f"rmem layout address function is not injective: {layout!r}")
    for placement in types.Placement.from_mesh(layout.mesh):
        ids = placement.program_ids(layout.mesh.topologies)
        coordinates = types.mesh_coords(layout.mesh, ids)
        if any(
            isinstance(attr, types.Varying) and coord != 0
            for attr, coord in zip(layout.attrs, coordinates, strict=True)
        ):
            continue
        if any(isinstance(attr, types.Varying) for attr in layout.attrs):
            unit = tuple(
                (axis, coord)
                for axis, (attr, coord) in enumerate(zip(layout.attrs, coordinates, strict=True))
                if not isinstance(attr, types.Varying)
            )
        else:
            unit = tuple(
                sorted(
                    (topology.name, start + index)
                    for topology, start, index in zip(
                        layout.mesh.topologies, types.starts(layout.mesh), ids, strict=True
                    )
                )
            )
        local, offset = types.local_layout_and_offset(
            layout, tuple(layout.layout.shape), coordinates
        )
        for coord in range(prod(local.shape)):
            held = types.apply(local, coord)
            logical = whole.get(offset + held)
            if logical is None:
                raise ValueError(
                    f"local shard addresses extend outside the rmem layout: {layout!r}"
                )
            yield logical, unit, shift + offset + held, shift + held


def address_table(layout, count: int, logical_shape: tuple):
    """Require one unique address per element within each Varying copy."""
    if not isinstance(layout, types.ShardLayout):
        return tuple(point[2] for point in address_points(layout, count, logical_shape))
    layout = ir_utils.participant_layout(layout)
    addresses = [None] * count
    for logical, unit, full, held in address_points(layout, count, logical_shape):
        if addresses[logical] is not None:
            raise ValueError(f"local shards overlap in rmem layout: {layout!r}")
        addresses[logical] = (unit, held)
    if any(address is None for address in addresses):
        raise ValueError(f"local shards do not cover the rmem layout: {layout!r}")
    return tuple(addresses)


@cache
def address_mapping(
    source, result, count: int, source_shape: tuple, result_shape: tuple
) -> tuple[int, ...]:
    """Result colex coordinates mapped bijectively within the same units' copies."""
    shards = tuple(types.shard_layout_of(layout) for layout in (source, result))
    varying = tuple(
        tuple(axis for axis, attr in enumerate(shard.attrs) if isinstance(attr, types.Varying))
        if shard is not None
        else ()
        for shard in shards
    )
    if varying[0] != varying[1]:
        raise ValueError("source and result Varying axes must match")
    if varying[0] and shards[0].mesh != shards[1].mesh:
        raise ValueError("source and result Varying copies must use the same mesh")
    source_at = {
        address: coord for coord, address in enumerate(address_table(source, count, source_shape))
    }
    if len(source_at) != count:
        raise ValueError("source layout address function is not injective")
    result_at = address_table(result, count, result_shape)
    if len(set(result_at)) != count:
        raise ValueError("result layout address function is not injective")
    if not set(result_at).issubset(source_at):
        raise ValueError("result layout addresses extend outside the source address set")
    return tuple(source_at[address] for address in result_at)


@typeinfer.register(Bitcast)
def _(call: expr.Call, ctx) -> types.TensorType:
    source = ctx.type_of(call.args[0])
    stated_layout = call.target.layout
    if not expr.is_literal_shape(source.shape):
        ctx.error(call, "source tensor shape must be literal")
    sharded = isinstance(source.layout, types.ShardLayout) or isinstance(stated_layout, types.ShardLayout)
    shape = source.shape if sharded else tuple(prod(types.flatten(mode)) for mode in stated_layout.shape)
    result = types.TensorType(shape, source.dtype, stated_layout, source.storage)
    layout = result.layout
    if sharded:
        if source.storage is not expr.StorageKind.RMEM:
            ctx.error(call, "sharded bitcast requires rmem storage")
        if not isinstance(source.layout, types.ShardLayout) or not isinstance(layout, types.ShardLayout):
            ctx.error(call, "sharded bitcast requires ShardLayout at both ends")
        source_layout = ir_utils.participant_layout(source.layout)
        result_layout = ir_utils.participant_layout(layout)
        if source_layout.mesh != result_layout.mesh:
            ctx.error(call, "source and result ShardLayout must use the same mesh")
    else:
        source_layout, result_layout = source.layout, layout
    for end, held in (("source", source_layout), ("result", result_layout)):
        if sharded:
            inner = held.layout
            if (
                not isinstance(inner, types.Layout)
                or any(isinstance(stride, tuple) for stride in inner.strides)
            ):
                ctx.error(
                    call,
                    f"{end} rmem layout requires an inner Layout with flat stated strides: {held!r}",
                )
            if not all(isinstance(attr, (types.Split, types.Varying)) for attr in held.attrs):
                ctx.error(call, f"{end} rmem layout must contain only Split or Varying attributes")
            try:
                axes = int_tuple.mode_axes(int_tuple.compatible_profile(held.layout.shape, source.shape))
                types.split_target_axes(held, source.shape)
                extents = tuple(prod(types.flatten(mode)) for mode in held.layout.shape)
                if any(
                    prod(
                        extent
                        for extent, assigned in zip(extents, axes, strict=True)
                        if assigned == axis
                    )
                    != size
                    for axis, size in enumerate(source.shape)
                ):
                    raise ValueError("layout modes do not factor the logical tensor shape")
            except (IndexError, ValueError) as error:
                ctx.error(call, f"{end} ShardLayout is invalid for shape {source.shape}: {error}")
            held = held.layout
        if not isinstance(held, (types.Layout, types.ComposedLayout)) or types.shard_layout_of(held) is not None:
            ctx.error(call, f"{end} layout must be plain (without ShardLayout): {held!r}")
        if not expr.is_literal_shape(types.flatten(held.shape)):
            ctx.error(call, f"{end} layout shape must be literal")
    count = prod(source.shape)
    layout_count = prod(types.flatten(layout.shape))
    if layout_count != count:
        ctx.error(call, f"element count differs: source {count}, result layout {layout_count}")
    if prod(shape) != count:
        ctx.error(call, f"element count differs: source {count}, result {prod(shape)}")
    try:
        address_mapping(source.layout, layout, count, source.shape, shape)
    except ValueError as error:
        ctx.error(call, str(error))
    return result


@access.register_alias(Bitcast)
def buffer_alias(call: expr.Call) -> int:
    return 0


def quasi_affine_equations(coordinates: tuple, result_shape: tuple) -> list[str] | None:
    """Compress a binary-digit sum only after proving every mapped coordinate."""
    if not coordinates or any(extent <= 0 or extent & (extent - 1) for extent in result_shape):
        return None
    origin = coordinates[0]
    terms = []
    stride = 1
    for axis, extent in enumerate(result_shape):
        for bit in range(extent.bit_length() - 1):
            power = 1 << bit
            step = tuple(
                value - base
                for value, base in zip(coordinates[stride * power], origin, strict=True)
            )
            terms.append((axis, power, step))
        stride *= extent
    if not all(
        source
        == tuple(
            base + sum(step[axis] * ((digits[index] // power) % 2) for index, power, step in terms)
            for axis, base in enumerate(origin)
        )
        for coord, source in enumerate(coordinates)
        for digits in (types.idx2crd(coord, result_shape, types.compact_col_major(result_shape)),)
    ):
        return None
    return [
        f"s{axis} = {base}"
        + "".join(
            f" + {step[axis]} * (floor(d{index}/{power}) % 2)"
            for index, power, step in terms
            if step[axis]
        )
        for axis, base in enumerate(origin)
    ]


@cache
def address_relation(source_layout, result_layout, source_shape: tuple, result_shape: tuple):
    pairs = address_mapping(
        source_layout, result_layout, prod(result_shape), source_shape, result_shape
    )
    domain = ", ".join(f"d{axis}" for axis in range(len(result_shape)))
    image = ", ".join(f"s{axis}" for axis in range(len(source_shape)))
    coordinates = tuple(types.idx2crd(coord, source_shape, types.compact_col_major(source_shape)) for coord in pairs)
    if coordinates:
        origin = coordinates[0]
        steps = []
        stride = 1
        for extent in result_shape:
            steps.append(
                tuple(value - base for value, base in zip(coordinates[stride], origin, strict=True))
                if extent > 1
                else (0,) * len(source_shape)
            )
            stride *= extent
        if all(
            source
            == tuple(
                base + sum(step[axis] * digit for step, digit in zip(steps, digits, strict=True))
                for axis, base in enumerate(origin)
            )
            for coord, source in enumerate(coordinates)
            for digits in (types.idx2crd(coord, result_shape, types.compact_col_major(result_shape)),)
        ):
            equations = [
                f"s{axis} = {base}"
                + "".join(
                    f" + {step[axis]} * d{index}" for index, step in enumerate(steps) if step[axis]
                )
                for axis, base in enumerate(origin)
            ]
            return isl.map(
                f"{{ [{domain}] -> [{image}] : " + (" and ".join(equations) or "true") + " }"
            )
    equations = quasi_affine_equations(coordinates, result_shape)
    if equations is not None:
        return isl.map(
            f"{{ [{domain}] -> [{image}] : " + (" and ".join(equations) or "true") + " }"
        )
    points = []
    for coord, source_coord in enumerate(pairs):
        constraints = [
            *(f"d{axis} = {value}" for axis, value in enumerate(types.idx2crd(coord, result_shape, types.compact_col_major(result_shape)))),
            *(
                f"s{axis} = {value}"
                for axis, value in enumerate(types.idx2crd(source_coord, source_shape, types.compact_col_major(source_shape)))
            ),
        ]
        points.append(f"[{domain}] -> [{image}] : " + (" and ".join(constraints) or "true"))
    relation = isl.map("{ " + "; ".join(points or [f"[{domain}] -> [{image}] : false"]) + " }")
    return relation.coalesce()


@access.register_relation(Bitcast)
def bitcast_access(call: expr.Call, ctx) -> tuple[access.Relation, ...]:
    source = ctx.type_of(call.args[0])
    result = ctx.type_of(call)
    relation = address_relation(source.layout, result.layout, source.shape, result.shape)
    return access.iterating(result.shape, (access.Relation(relation), access.identity(len(result.shape))))


@evaluator.register(Bitcast)
def eval_bitcast(ctx):
    """Apply the proven local mapping independently to each placement's data."""
    x = ctx.args[0].tensor
    if x.is_global:
        held = bitcast(x.tensor, x.type, ctx.result_type)
    else:
        source, result = core.local_type(x.type), core.local_type(ctx.result_type)
        held = tuple((p, bitcast(t, source, result)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


def bitcast(data, source, result):
    """*data* of type *source* read through the layout of *result*."""
    pairs = address_mapping(source.layout, result.layout, prod(result.shape), source.shape, result.shape)
    indices = [0] * len(pairs)
    for coord, source_coord in enumerate(pairs):
        result_index = source_index = 0
        for digit, extent in zip(types.idx2crd(coord, result.shape, types.compact_col_major(result.shape)), result.shape, strict=True):
            result_index = result_index * extent + digit
        for digit, extent in zip(
            types.idx2crd(source_coord, source.shape, types.compact_col_major(source.shape)), source.shape, strict=True
        ):
            source_index = source_index * extent + digit
        indices[result_index] = source_index
    index = utils.torch.tensor(indices, dtype=utils.torch.int64, device=data.device)
    return data.reshape(-1).index_select(0, index).reshape(result.shape)


@cost.register(Bitcast)
def bitcast_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    return cost.Cost({}, cost.idle(call))


__all__ = ["Bitcast", "address_mapping", "address_relation", "eval_bitcast"]
