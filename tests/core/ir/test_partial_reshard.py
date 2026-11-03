"""A Partial -> Broadcast reshard is not a view; Broadcast -> plain still aliases."""

from dataclasses import replace

import pytest

from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import (
    Broadcast,
    DType,
    Layout,
    Mesh,
    Partial,
    ShardLayout,
    StorageKind,
    TensorType,
    Topology,
)
from tilefoundry.visitor_registry.buffer_alias import aliased_operand


def _types():
    cta = Mesh((Topology("cta", 2),), Layout((2,), (1,)))
    src = TensorType(
        (64, 64),
        DType.f32,
        ShardLayout(Layout((64, 64), (64, 1)), (Partial("sum"),), cta),
        StorageKind.SMEM,
    )
    dst = TensorType((64, 64), DType.f32, Layout((64, 64), (64, 1)), StorageKind.GMEM)
    return src, dst


@pytest.mark.parametrize("move_axis", (False, True), ids=("broadcast", "move_axis"))
def test_partial_to_broadcast_not_aliased(move_axis):
    src, _ = _types()
    if move_axis:
        mesh = Mesh((Topology("cta", 4),), Layout((2, 2), (2, 1)))
        src = replace(
            src, layout=replace(src.layout, attrs=(Partial("sum"), Broadcast()), mesh=mesh)
        )
        attrs = (Broadcast(), Partial("sum"))
    else:
        attrs = (Broadcast(),)
    dst = replace(src, layout=replace(src.layout, attrs=attrs))
    call = Call(Reshard(layout=dst.layout, storage=dst.storage), (Var("src", type=src),), type=dst)
    assert aliased_operand(call) is None


def test_broadcast_to_plain_aliases():
    src, _ = _types()
    src = replace(src, layout=replace(src.layout, attrs=(Broadcast(),)))
    dst = replace(src, layout=src.layout.layout)
    call = Call(Reshard(layout=dst.layout, storage=dst.storage), (Var("src", type=src),), type=dst)
    assert aliased_operand(call) == 0
