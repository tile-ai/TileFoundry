"""The thread that lowering writes shared memory from when no operand splits it."""

from __future__ import annotations

import pytest

from tilefoundry.dsl import DimVar
from tilefoundry.ir import expr, types
from tilefoundry.passes import transforms

CTA, THREAD = types.Topology("cta", 2), types.Topology("thread", 32)
TILE = types.TensorType((8,), expr.DType.f32, types.Layout((8,), (1,)), expr.StorageKind.SMEM)
BLOCKS = types.Mesh((CTA,), types.Layout((2,), (1,)), ("b",))


def writer(*around):
    """The innermost region lowering opens around a fill of TILE in *around*, innermost first."""
    meshes = transforms.transfer_meshes((TILE,), TILE, around, (CTA, THREAD))
    return meshes[-1] if meshes else None


def threads(start, count):
    return types.Mesh((THREAD,), types.ComposedLayout(None, start, types.Layout((count,), (1,))), ("t",))


def test_the_writer_is_the_first_thread_of_the_innermost_thread_mesh() -> None:
    mesh = writer(BLOCKS, threads(16, 16), threads(0, 32))
    assert mesh.topologies == (THREAD,)
    assert (mesh.layout.offset, types.size(types.levels(mesh)[0])) == (16, 1)


def test_without_a_thread_mesh_the_writer_is_the_first_thread_of_the_cta() -> None:
    mesh = writer(BLOCKS)
    assert (mesh.layout.offset, types.size(types.levels(mesh)[0])) == (0, 1)


@pytest.mark.parametrize(
    "enclosing",
    [
        threads(16, 1),
        threads(DimVar("start", 0, 31), 1),
        types.Mesh(
            (types.Topology("cta", DimVar("blocks", 1, 64)), THREAD),
            types.ComposedLayout(None, 16, types.Layout(((1,), (16,)), ((1,), (1,)))),
            ("b", "t"),
        ),
    ],
    ids=["one thread already", "a start that is not a number", "levels whose sizes are not all numbers"],
)
def test_no_writer_without_a_constant_first_thread(enclosing) -> None:
    assert transforms.transfer_meshes((TILE,), TILE, (enclosing,), (CTA, THREAD)) == ()
