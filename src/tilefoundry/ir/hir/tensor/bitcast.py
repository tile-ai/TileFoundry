from __future__ import annotations

from functools import cache
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
from tilefoundry.ir.types.layout import apply
from tilefoundry.ir.types.shard_layout import (
    layout_axis_to_tensor_axis,
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
    """Pair a sharded element's owning mesh coordinate with its local offset.

    Local views retain the whole layout's strides and composed address function.
    """
    if not isinstance(layout, ShardLayout):
        return tuple(apply(layout, coord) for coord in range(count))
    shape = tuple(prod(flatten(mode)) for mode in layout.layout.shape)
    mesh_shape = tuple(flatten(layout.mesh.layout.shape))
    if len(layout.attrs) != len(mesh_shape):
        raise ValueError("shard attributes must match the mesh axes")
    divisors = [1] * len(shape)
    for attr, extent in zip(layout.attrs, mesh_shape, strict=True):
        if not isinstance(attr, Split):
            raise ValueError("rmem layout must contain only Split attributes")
        if not 0 <= attr.axis < len(shape):
            raise ValueError("Split axis is outside the layout")
        divisors[attr.axis] *= extent
    if any(extent % divisor for extent, divisor in zip(shape, divisors, strict=True)):
        raise ValueError("Split mesh extents must divide the layout modes")
    addresses = []
    for coord in range(count):
        digits = _coordinates(coord, shape)
        local = [
            digit % (extent // divisor)
            for digit, extent, divisor in zip(digits, shape, divisors, strict=True)
        ]
        unit = []
        for mesh_axis, (attr, extent) in enumerate(zip(layout.attrs, mesh_shape, strict=True)):
            inner = prod(
                mesh_shape[other]
                for other in range(mesh_axis + 1, len(mesh_shape))
                if layout.attrs[other].axis == attr.axis
            )
            unit.append(
                digits[attr.axis] // (shape[attr.axis] // divisors[attr.axis] * inner) % extent
            )
        origin = tuple(digit - held for digit, held in zip(digits, local, strict=True))
        origin_coord = 0
        stride = 1
        for digit, extent in zip(origin, shape, strict=True):
            origin_coord += digit * stride
            stride *= extent
        offset = apply(layout.layout, coord) - apply(layout.layout, origin_coord)
        addresses.append((tuple(unit), offset))
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
