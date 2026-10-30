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
    Split,
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


@pytest.mark.parametrize("stride", (0, 4))
def test_device_gather_materializes_even_with_the_same_global_address_map(stride):
    mesh = Mesh((Topology("gpu", 2),), Layout((2,), (1,)), ("dp",))
    source = TensorType((4, 4), DType.f32, ShardLayout(Layout((4, 4), (stride, 1)), (Split(0),), mesh), StorageKind.GMEM)
    result = replace(source, layout=ShardLayout(Layout((4, 4), (4, 1)), (Broadcast(),), mesh))
    argument = Var("weight", type=source)
    gather = Call(Reshard(layout=result.layout), (argument,), type=result)
    unchanged = Call(Reshard(layout=source.layout), (argument,), type=source)
    assert aliased_operand(gather) is None
    assert aliased_operand(unchanged) == 0


def test_device_rank_permutation_changes_the_storage_instance():
    mesh = Mesh((Topology("gpu", 4),), Layout((2, 2), (2, 1)), ("row", "col"))
    source = TensorType((4, 4), DType.f32, ShardLayout(Layout((4, 4)), (Split(0), Split(1)), mesh), StorageKind.GMEM)
    permuted = replace(mesh, layout=Layout((2, 2), (1, 2)))
    result = replace(source, layout=replace(source.layout, mesh=permuted))
    moved = Call(Reshard(layout=result.layout), (Var("x", type=source),), type=result)
    assert aliased_operand(moved) is None
