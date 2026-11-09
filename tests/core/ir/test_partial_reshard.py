"""A Partial -> Broadcast reshard is not a view; Broadcast -> plain still aliases."""

from dataclasses import replace

import pytest

from tilefoundry.interfaces import access
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import sharding


def _types():
    cta = types.Mesh((types.Topology("cta", 2),), types.Layout((2,), (1,)))
    src = types.TensorType(
        (64, 64),
        expr.DType.f32,
        types.ShardLayout(types.Layout((64, 64), (64, 1)), (types.Partial("sum"),), cta),
        expr.StorageKind.SMEM,
    )
    dst = types.TensorType((64, 64), expr.DType.f32, types.Layout((64, 64), (64, 1)), expr.StorageKind.GMEM)
    return src, dst


@pytest.mark.parametrize("move_axis", (False, True), ids=("broadcast", "move_axis"))
def test_partial_to_broadcast_not_aliased(move_axis):
    src, _ = _types()
    if move_axis:
        mesh = types.Mesh((types.Topology("cta", 4),), types.Layout((2, 2), (2, 1)))
        src = replace(
            src, layout=replace(src.layout, attrs=(types.Partial("sum"), types.Broadcast()), mesh=mesh)
        )
        attrs = (types.Broadcast(), types.Partial("sum"))
    else:
        attrs = (types.Broadcast(),)
    dst = replace(src, layout=replace(src.layout, attrs=attrs))
    call = expr.Call(sharding.Reshard(layout=dst.layout, storage=dst.storage), (expr.Var("src", type=src),), type=dst)
    assert access.aliased_operand(call) is None


def test_broadcast_to_plain_aliases():
    src, _ = _types()
    src = replace(src, layout=replace(src.layout, attrs=(types.Broadcast(),)))
    dst = replace(src, layout=src.layout.layout)
    call = expr.Call(
        sharding.Reshard(layout=dst.layout, storage=dst.storage), (expr.Var("src", type=src),), type=dst
    )
    assert access.aliased_operand(call) == 0


@pytest.mark.parametrize(
    "before,after,expected",
    (
        (types.Partial("sum"), types.Partial("sum"), 0),
        (types.Varying(), types.Partial("sum"), 0),
        (None, types.Partial("sum"), 0),
        (types.Partial("sum"), types.Partial("max"), None),
        (types.Broadcast(), types.Partial("sum"), None),
    ),
)
def test_partial_alias_states(before, after, expected):
    src, _ = _types()
    layout = src.layout
    src = replace(src, layout=layout.layout if before is None else replace(layout, attrs=(before,)))
    dst = replace(src, layout=replace(layout, attrs=(after,)))
    call = expr.Call(
        sharding.Reshard(layout=dst.layout, storage=dst.storage), (expr.Var("src", type=src),), type=dst
    )
    assert access.aliased_operand(call) == expected
