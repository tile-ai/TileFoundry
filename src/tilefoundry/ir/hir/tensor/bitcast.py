from __future__ import annotations

from functools import cache
from itertools import product
from math import prod

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Call, Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.sharding.reshard import _plain
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.ir.types import (
    ComposedLayout,
    Layout,
    LayoutBase,
    ShardLayout,
    Split,
    StorageKind,
    TensorType,
)
from tilefoundry.ir.types.int_tuple import flatten
from tilefoundry.ir.types.layout import apply, get, rank
from tilefoundry.ir.types.shard_layout import (
    _positions,
    layout_axis_to_tensor_axis,
    local_layout_and_offset,
    shard_layout_of,
    split_target_axes,
)
from tilefoundry.ir.types.utils import is_literal_shape
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    identity_access,
    iterating,
    register_access_relation,
)
from tilefoundry.visitor_registry.buffer_alias import Alias, register_buffer_alias


@register_op(name="bitcast")
class Bitcast(Op):
    """Reinterpret a tensor's bytes under a new shape and layout."""

    x = ParamDef(kind="input", pattern=is_ranked_tensor())
    layout = ParamDef(kind="attribute", annotation=LayoutBase)


def _addresses(layout, count: int):
    """Enumerate each unit's local addresses using the shared shard definition.

    Shard ownership uses the layout's factored domain, independently of its
    grouping into logical tensor axes. Whole addresses identify colex elements.
    """
    if not isinstance(layout, ShardLayout):
        return tuple(apply(layout, coord) for coord in range(count))
    whole = {apply(layout.layout, coord): coord for coord in range(count)}
    if len(whole) != count:
        raise ValueError(f"rmem layout address function is not injective: {layout!r}")
    stated = layout.mesh.layout
    if isinstance(stated, ComposedLayout):
        stated = stated.outer
    levels = tuple(get(stated, index) for index in range(rank(stated)))
    addresses = [None] * count
    for coordinates in product(*(range(prod(flatten(level.shape))) for level in levels)):
        ids = tuple(apply(level, coord) for level, coord in zip(levels, coordinates, strict=True))
        unit = tuple(_positions(layout, ids).values())
        local, offset = local_layout_and_offset(layout, tuple(layout.layout.shape), ids)
        for coord in range(prod(local.shape)):
            held = apply(local, coord)
            logical = whole.get(offset + held)
            if logical is None:
                raise ValueError(f"local shard addresses extend outside the rmem layout: {layout!r}")
            if addresses[logical] is not None:
                raise ValueError(f"local shards overlap in rmem layout: {layout!r}")
            addresses[logical] = (unit, held)
    if any(address is None for address in addresses):
        raise ValueError(f"local shards do not cover the rmem layout: {layout!r}")
    return tuple(addresses)


@cache
def _address_mapping(source, result, count: int) -> tuple[int, ...]:
    """Result colex coordinates mapped bijectively to source colex coordinates."""
    source_at = {address: coord for coord, address in enumerate(_addresses(source, count))}
    if len(source_at) != count:
        raise ValueError("source layout address function is not injective")
    result_at = _addresses(result, count)
    if len(set(result_at)) != count:
        raise ValueError("result layout address function is not injective")
    if not set(result_at).issubset(source_at):
        raise ValueError("result layout addresses extend outside the source address set")
    return tuple(source_at[address] for address in result_at)


def _coordinates(coord: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    digits = []
    for extent in shape:
        digits.append(coord % extent)
        coord //= extent
    return tuple(digits)


@register_typeinfer(Bitcast)
def _(call: Call, ctx) -> TensorType:
    source = ctx.type_of(call.args[0])
    layout = call.target.layout
    if not is_literal_shape(source.shape):
        ctx.error(call, "source tensor shape must be literal")
    sharded = isinstance(source.layout, ShardLayout) or isinstance(layout, ShardLayout)
    if sharded:
        if source.storage is not StorageKind.RMEM:
            ctx.error(call, "sharded bitcast requires rmem storage")
        if not isinstance(source.layout, ShardLayout) or not isinstance(layout, ShardLayout):
            ctx.error(call, "sharded bitcast requires ShardLayout at both ends")
        if source.layout.mesh != layout.mesh:
            ctx.error(call, "source and result ShardLayout must use the same mesh")
    for end, held in (("source", source.layout), ("result", layout)):
        if sharded:
            inner = held.layout
            if (
                not isinstance(inner, Layout)
                or inner.strides is None
                or any(isinstance(stride, tuple) for stride in inner.strides)
            ):
                ctx.error(
                    call,
                    f"{end} rmem layout requires an inner Layout with flat stated strides: {held!r}",
                )
            if not all(isinstance(attr, Split) for attr in held.attrs):
                ctx.error(call, f"{end} rmem layout must contain only Split attributes")
            try:
                axes = layout_axis_to_tensor_axis(held.layout.shape, source.shape)
                split_target_axes(held, source.shape)
                extents = tuple(prod(flatten(mode)) for mode in held.layout.shape)
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
        if not isinstance(held, (Layout, ComposedLayout)) or shard_layout_of(held) is not None:
            ctx.error(call, f"{end} layout must be plain (without ShardLayout): {held!r}")
        if not _plain(held):
            ctx.error(call, f"{end} layout must have strides stated in every Layout component")
        if not is_literal_shape(flatten(held.shape)):
            ctx.error(call, f"{end} layout shape must be literal")
    shape = source.shape if sharded else tuple(prod(flatten(mode)) for mode in layout.shape)
    count = prod(source.shape)
    layout_count = prod(flatten(layout.shape))
    if layout_count != count:
        ctx.error(call, f"element count differs: source {count}, result layout {layout_count}")
    if prod(shape) != count:
        ctx.error(call, f"element count differs: source {count}, result {prod(shape)}")
    try:
        _address_mapping(source.layout, layout, count)
    except ValueError as error:
        ctx.error(call, str(error))
    return TensorType(shape, source.dtype, layout, source.storage)


@register_buffer_alias(Bitcast)
def _buffer_alias(call: Call) -> Alias:
    return Alias(0)


def _quasi_affine_equations(coordinates: tuple, result_shape: tuple) -> list[str] | None:
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
        for digits in (_coordinates(coord, result_shape),)
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
def _address_relation(source_layout, result_layout, source_shape: tuple, result_shape: tuple):
    pairs = _address_mapping(source_layout, result_layout, prod(result_shape))
    domain = ", ".join(f"d{axis}" for axis in range(len(result_shape)))
    image = ", ".join(f"s{axis}" for axis in range(len(source_shape)))
    coordinates = tuple(_coordinates(coord, source_shape) for coord in pairs)
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
            for digits in (_coordinates(coord, result_shape),)
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
    equations = _quasi_affine_equations(coordinates, result_shape)
    if equations is not None:
        return isl.map(
            f"{{ [{domain}] -> [{image}] : " + (" and ".join(equations) or "true") + " }"
        )
    points = []
    for coord, source_coord in enumerate(pairs):
        constraints = [
            *(f"d{axis} = {value}" for axis, value in enumerate(_coordinates(coord, result_shape))),
            *(
                f"s{axis} = {value}"
                for axis, value in enumerate(_coordinates(source_coord, source_shape))
            ),
        ]
        points.append(f"[{domain}] -> [{image}] : " + (" and ".join(constraints) or "true"))
    relation = isl.map("{ " + "; ".join(points or [f"[{domain}] -> [{image}] : false"]) + " }")
    return relation.coalesce()


@register_access_relation(Bitcast)
def _bitcast_access(call: Call, ctx) -> tuple[AccessRelation, ...]:
    source = ctx.type_of(call.args[0])
    result = ctx.type_of(call)
    relation = _address_relation(source.layout, result.layout, source.shape, result.shape)
    return iterating(result.shape, (AccessRelation(relation), identity_access(len(result.shape))))


@register_eval(Bitcast)
def _eval_bitcast(ctx):
    source = ctx.args[0]
    result = ctx.result_type
    pairs = _address_mapping(source.type.layout, result.layout, prod(result.shape))
    indices = [0] * len(pairs)
    for coord, source_coord in enumerate(pairs):
        result_index = source_index = 0
        for digit, extent in zip(_coordinates(coord, result.shape), result.shape, strict=True):
            result_index = result_index * extent + digit
        for digit, extent in zip(
            _coordinates(source_coord, source.type.shape), source.type.shape, strict=True
        ):
            source_index = source_index * extent + digit
        indices[result_index] = source_index
    index = torch.tensor(indices, dtype=torch.int64, device=source.data.device)
    data = source.data.reshape(-1).index_select(0, index).reshape(result.shape)
    return TensorValue(data=data, type=result)
