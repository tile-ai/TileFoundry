"""``shard_layout_local_shape`` — which mesh axes divide a layout dim.

The plain one-Split-per-axis case is asserted on every sharded op type
(``tests/ops`` compares local extents). What is kept here is the two ways the
answer is *not* "layout shape // mesh shape": several mesh axes splitting one
layout dim, and mesh axes that own no layout dim at all. Getting either wrong
sizes a register allocation.
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.mesh import Mesh, Topology
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    ShardLayout,
    Split,
    shard_layout_local_shape,
)

#: Global ``(128,)`` under two mesh axes (y=4, t=32) that both split axis 0 becomes
#: per-thread ``(1,)``: the divisors compose rather than the last one winning. The
#: other two rows are the axes that divide nothing -- ``Broadcast`` replicates and
#: ``Partial`` is a mesh-axis value state, so neither owns a layout axis, and each
#: shard keeps the full local extent for them. In the Broadcast row only t (mesh
#: axis 1, extent 4) divides layout dim 0.
DIVIDED = [
    pytest.param(
        ShardLayout(
            layout=Layout(shape=(128,), strides=(1,)),
            attrs=(Split(0), Split(0)),
            mesh=Mesh(
                Topology("thread", 4 * 32),
                Layout(shape=(4, 32), strides=(32, 1)),
                names=("y", "t"),
            ),
        ),
        (1,),
        id="two_mesh_axes_on_one_dim",
    ),
    pytest.param(
        ShardLayout(
            layout=Layout(shape=(4,), strides=(1,)),
            attrs=(Broadcast(), Split(0)),
            mesh=Mesh(
                Topology("thread", 2 * 4),
                Layout(shape=(2, 4), strides=(4, 1)),
                names=("x", "t"),
            ),
        ),
        (1,),
        id="broadcast_divides_nothing",
    ),
    pytest.param(
        ShardLayout(
            layout=Layout(shape=(8,), strides=(1,)),
            attrs=(Partial(),),
            mesh=Mesh(
                Topology("thread", 4),
                Layout(shape=(4,), strides=(1,)),
                names=("t",),
            ),
        ),
        (8,),
        id="partial_divides_nothing",
    ),
]


@pytest.mark.parametrize(("sharded", "expected"), DIVIDED)
def test_only_a_mesh_axis_that_owns_a_dim_divides_it(sharded, expected) -> None:
    assert shard_layout_local_shape(sharded) == expected
