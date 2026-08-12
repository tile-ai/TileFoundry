"""``shard_layout_local_shape`` — which mesh axes divide a layout dim.

The plain one-Split-per-axis case is asserted on every sharded op type
(``tests/ops`` compares local extents). What is kept here is the two ways the
answer is *not* "layout shape // mesh shape": several mesh axes splitting one
layout dim, and mesh axes that own no layout dim at all. Getting either wrong
sizes a register allocation.
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.mesh import Mesh, Topology
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    ShardLayout,
    Split,
    shard_layout_local_shape,
)

DIVIDED = [
    pytest.param(
        ShardLayout(
            layout=Layout(shape=(128,), strides=(1,)),
            attrs=(Split(0), Split(0)),
            mesh=Mesh(
                (Topology("thread", 4 * 32),),
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
                (Topology("thread", 2 * 4),),
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
                (Topology("thread", 4),),
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


_N = DimVar("local_n", 1, 65)
_M = DimVar("mesh_n", 1, 65)


def _symbolic_layout(axis_extent, mesh_extent, *, split: bool) -> ShardLayout:
    return ShardLayout(
        layout=Layout(shape=(axis_extent, 8), strides=None),
        attrs=(Split(0) if split else Broadcast(),),
        mesh=Mesh(
            (Topology("cta", mesh_extent),),
            Layout(shape=(mesh_extent,), strides=(1,)),
        ),
    )


def test_unconsumed_symbolic_axis_is_available_to_type_inference() -> None:
    layout = _symbolic_layout(_N, 8, split=False)

    assert shard_layout_local_shape(layout, require_static=False) == (_N, 8)
    with pytest.raises(ValueError, match="not static after sharding"):
        shard_layout_local_shape(layout)


@pytest.mark.parametrize("require_static", [False, True], ids=["typeinfer", "lowering"])
def test_matching_symbolic_split_has_one_local_element(require_static: bool) -> None:
    layout = _symbolic_layout(_N, _N, split=True)

    assert shard_layout_local_shape(layout, require_static=require_static) == (1, 8)


def test_unresolved_symbolic_split_is_rejected() -> None:
    layout = _symbolic_layout(_N, _M, split=True)

    with pytest.raises(ValueError, match="divisibility.*bind symbolic dimensions"):
        shard_layout_local_shape(layout, require_static=False)
