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
from tilefoundry.ir.types import ComposedLayout, Layout, LayoutBase, TensorType
from tilefoundry.ir.types.int_tuple import flatten
from tilefoundry.ir.types.layout import apply
from tilefoundry.ir.types.shard_layout import shard_layout_of
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


@cache
def _address_mapping(source, result, count: int) -> tuple[int, ...]:
    """Result colex coordinates mapped bijectively to source colex coordinates."""
    source_at = {apply(source, coord): coord for coord in range(count)}
    if len(source_at) != count:
        raise ValueError("source layout address function is not injective")
    result_at = tuple(apply(result, coord) for coord in range(count))
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
    for end, held in (("source", source.layout), ("result", layout)):
        if not isinstance(held, (Layout, ComposedLayout)) or shard_layout_of(held) is not None:
            ctx.error(call, f"{end} layout must be plain (without ShardLayout): {held!r}")
        if not _plain(held):
            ctx.error(call, f"{end} layout must have strides stated in every Layout component")
        if not is_literal_shape(flatten(held.shape)):
            ctx.error(call, f"{end} layout shape must be literal")
    if not is_literal_shape(source.shape):
        ctx.error(call, "source tensor shape must be literal")
    shape = tuple(prod(flatten(mode)) for mode in layout.shape)
    count = prod(source.shape)
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
                if extent > 1 else (0,) * len(source_shape)
            )
            stride *= extent
        if all(
            source == tuple(
                base + sum(step[axis] * digit for step, digit in zip(steps, digits, strict=True))
                for axis, base in enumerate(origin)
            )
            for coord, source in enumerate(coordinates)
            for digits in (_coordinates(coord, result_shape),)
        ):
            equations = [
                f"s{axis} = {base}" + "".join(
                    f" + {step[axis]} * d{index}" for index, step in enumerate(steps)
                    if step[axis]
                )
                for axis, base in enumerate(origin)
            ]
            return isl.map(f"{{ [{domain}] -> [{image}] : " + (" and ".join(equations) or "true") + " }")
    points = []
    for coord, source_coord in enumerate(pairs):
        constraints = [
            *(f"d{axis} = {value}" for axis, value in enumerate(_coordinates(coord, result_shape))),
            *(f"s{axis} = {value}" for axis, value in enumerate(_coordinates(source_coord, source_shape))),
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
        for digit, extent in zip(_coordinates(source_coord, source.type.shape), source.type.shape, strict=True):
            source_index = source_index * extent + digit
        indices[result_index] = source_index
    index = torch.tensor(indices, dtype=torch.int64, device=source.data.device)
    data = source.data.reshape(-1).index_select(0, index).reshape(result.shape)
    return TensorValue(data=data, type=result)
