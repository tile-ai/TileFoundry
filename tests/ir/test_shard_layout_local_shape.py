"""``shard_layout_local_shape`` — which mesh axes divide a layout dim.

The plain one-Split-per-axis case is asserted on every sharded op type
(``tests/ops`` compares local extents). What is kept here is the two ways the
answer is *not* "layout shape // mesh shape": several mesh axes splitting one
layout dim, and mesh axes that own no layout dim at all. Getting either wrong
sizes a register allocation.
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.core.expr import Call
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shape_helpers import i64_const, static_dim_value
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


def test_an_extent_written_as_arithmetic_over_constants_is_static() -> None:
    """``A + B`` is an extent, not an open dimension.

    A body-local name for a constant expression is an IR ``Binary`` and stays
    one, so 4096 + 4096 was not 8192 to this projection and the program was
    refused for a dimension nothing had left open.
    """
    half = i64_const(4096)
    whole = Call(type=half.type, target=Binary(kind=BinaryKind.ADD), args=(half, half))
    assert static_dim_value(whole) == 8192

    layout = ShardLayout(
        layout=Layout(shape=(whole,), strides=(1,)),
        attrs=(Split(0),),
        mesh=Mesh(
            topologies=(Topology("cta", 128),),
            layout=Layout(shape=(128,), strides=(1,)),
            names=("unit",),
        ),
    )
    assert shard_layout_local_shape(layout) == (64,)


def test_an_undecidable_extent_is_named_the_way_it_was_written() -> None:
    """The refusal has to be findable in the reader's own program.

    ``repr`` of an IR expression is a node dump, so an author who left one
    dimension open got a paragraph of internals and no name.
    """
    layout = _symbolic_layout(_N, _M, split=True)

    with pytest.raises(ValueError) as raised:
        shard_layout_local_shape(layout, require_static=False)
    message = str(raised.value)
    assert "TensorType" not in message
    assert _N.name in message or _M.name in message
