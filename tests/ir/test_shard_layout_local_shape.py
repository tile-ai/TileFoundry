"""Unit tests for ``shard_layout_local_shape``.

"""

from __future__ import annotations

from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.mesh import Mesh, Topology
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    ShardLayout,
    Split,
    shard_layout_local_shape,
)


def test_seq2_canonical() -> None:
    """seq_2 reshard ``(2 @ x, 12 @ y, 128 @ t)`` produces global
    ``(2, 12, 128)`` → per-thread ``(1, 3, 4)``."""
    mesh = Mesh(
        Topology("thread", 8 * 32),
        Layout(shape=(2, 4, 32), strides=(128, 32, 1)),
        names=("x", "y", "t"),
    )
    sl = ShardLayout(
        layout=Layout(shape=(2, 12, 128), strides=(1536, 128, 1)),
        attrs=(Split(0), Split(1), Split(2)),
        mesh=mesh,
    )
    assert shard_layout_local_shape(sl) == (1, 3, 4)


def test_rmsnorm_single_axis_two_splits() -> None:
    """rmsnorm: global ``(1, 1536)`` with two Splits on layout dim 1
    (y=4, t=32 both splitting axis 1) — wait, that's the old
    convention. Under new spec, layout.shape should be the
    unsharded shape per attrs. A simple single-layout-dim case:
    global ``(128,)`` with two mesh axes (y=4, t=32) both splitting
    axis 0 → per-thread ``(1,)``."""
    mesh = Mesh(
        Topology("thread", 4 * 32),
        Layout(shape=(4, 32), strides=(32, 1)),
        names=("y", "t"),
    )
    sl = ShardLayout(
        layout=Layout(shape=(128,), strides=(1,)),
        attrs=(Split(0), Split(0)),
        mesh=mesh,
    )
    assert shard_layout_local_shape(sl) == (1,)


def test_broadcast_does_not_divide() -> None:
    mesh = Mesh(
        Topology("thread", 2 * 4),
        Layout(shape=(2, 4), strides=(4, 1)),
        names=("x", "t"),
    )
    sl = ShardLayout(
        layout=Layout(shape=(4,), strides=(1,)),
        attrs=(Broadcast(), Split(0)),
        mesh=mesh,
    )
    # Only t (mesh axis 1, extent 4) splits layout dim 0.
    assert shard_layout_local_shape(sl) == (1,)


def test_partial_does_not_divide_layout_dim() -> None:
    """``Partial`` is a mesh-axis value state with no layout axis: it does
    NOT divide any layout dim (each shard keeps the full local shape)."""
    mesh = Mesh(
        topology=Topology("thread", 4),
        layout=Layout(shape=(4,), strides=(1,)),
        names=("t",),
    )
    sl = ShardLayout(
        layout=Layout(shape=(8,), strides=(1,)),
        attrs=(Partial(),),
        mesh=mesh,
    )
    assert shard_layout_local_shape(sl) == (8,)


def test_residue_when_mesh_does_not_fully_consume() -> None:
    """global ``(8,)`` with one mesh axis t=4 → per-thread ``(2,)``
    (local residue 2)."""
    mesh = Mesh(
        topology=Topology("thread", 4),
        layout=Layout(shape=(4,), strides=(1,)),
        names=("t",),
    )
    sl = ShardLayout(
        layout=Layout(shape=(8,), strides=(1,)),
        attrs=(Split(0),),
        mesh=mesh,
    )
    assert shard_layout_local_shape(sl) == (2,)
