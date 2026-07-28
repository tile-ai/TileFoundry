"""``shard_layout_local_shape`` — which mesh axes divide a layout dim.

The plain one-Split-per-axis case is asserted on every sharded op type
(``tests/ops`` compares local extents). What is kept here is the two ways the
answer is *not* "layout shape // mesh shape": several mesh axes splitting one
layout dim, and mesh axes that own no layout dim at all. Getting either wrong
sizes a register allocation.
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


def test_two_mesh_axes_splitting_one_layout_dim_divide_together() -> None:
    """Global ``(128,)`` with two mesh axes (y=4, t=32) both splitting axis 0 →
    per-thread ``(1,)``: the divisors compose rather than the last one winning."""
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


def test_broadcast_and_partial_do_not_divide_a_layout_dim() -> None:
    """``Broadcast`` replicates and ``Partial`` is a mesh-axis value state; neither
    owns a layout axis, so neither divides one — each shard keeps the full local
    extent for those axes."""
    mesh = Mesh(
        Topology("thread", 2 * 4),
        Layout(shape=(2, 4), strides=(4, 1)),
        names=("x", "t"),
    )
    broadcast = ShardLayout(
        layout=Layout(shape=(4,), strides=(1,)),
        attrs=(Broadcast(), Split(0)),
        mesh=mesh,
    )
    # Only t (mesh axis 1, extent 4) splits layout dim 0.
    assert shard_layout_local_shape(broadcast) == (1,)

    partial = ShardLayout(
        layout=Layout(shape=(8,), strides=(1,)),
        attrs=(Partial(),),
        mesh=Mesh(
            topology=Topology("thread", 4),
            layout=Layout(shape=(4,), strides=(1,)),
            names=("t",),
        ),
    )
    assert shard_layout_local_shape(partial) == (8,)
