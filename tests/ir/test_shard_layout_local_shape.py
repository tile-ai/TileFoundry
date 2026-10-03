"""``shard_layout_local_shape`` — which mesh axes divide a layout dim.

The plain one-Split-per-axis case is asserted on every sharded op type
(``tests/ops`` compares local extents). What is kept here is the two ways the
answer is *not* "layout shape // mesh shape": several mesh axes splitting one
layout dim, and mesh axes that own no layout dim at all. Getting either wrong
sizes a register allocation.
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.types.dim import DimMul, DimVar, simplify_dim
from tilefoundry.ir.types.layout import Layout
from tilefoundry.ir.types.mesh import Mesh, Topology
from tilefoundry.ir.types.shard_layout import (
    Broadcast,
    Partial,
    ShardLayout,
    Split,
    canonical_shard_layout,
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


_N = DimVar("local_n", 1, 64)
_M = DimVar("mesh_n", 1, 64)


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
@pytest.mark.parametrize(
    ("extent", "expected", "canonical"),
    [
        pytest.param(_N, (1, 8), False, id="equal"),
        pytest.param(simplify_dim(DimMul, (_N, 64)), (64, 8), False, id="symbolic_factor_left"),
        pytest.param(simplify_dim(DimMul, (64, _N)), (64, 8), False, id="symbolic_factor_right"),
        pytest.param(_N * 64 + 1, None, False, id="not_divisible"),
        pytest.param(_N * 64, (1, 1, 8), True, id="canonical_exact"),
        pytest.param(_N * 128, (1, 1, 2, 8), True, id="canonical_residual"),
        pytest.param(_N * 64 + 1, None, True, id="canonical_undecidable"),
    ],
)
def test_matching_symbolic_split_has_decidable_local_extent(
    require_static, extent, expected, canonical
) -> None:
    if canonical:
        mesh = Mesh(
            (Topology("thread", _N * 64),),
            Layout(shape=(_N, 64), strides=(64, 1)),
        )
        if expected is None:
            with pytest.raises(ValueError, match="dynamic; cannot factorize across multiple mesh axes"):
                canonical_shard_layout((extent, 8), mesh, (Split(0), Split(0)))
            return
        layout = canonical_shard_layout((extent, 8), mesh, (Split(0), Split(0)))
    else:
        layout = _symbolic_layout(extent, _N, split=True)

    if expected is None:
        with pytest.raises(ValueError, match="do not have a decidable divisibility relation"):
            shard_layout_local_shape(layout, require_static=require_static)
    else:
        assert shard_layout_local_shape(layout, require_static=require_static) == expected


def test_unresolved_symbolic_split_is_rejected() -> None:
    layout = _symbolic_layout(_N, _M, split=True)

    with pytest.raises(ValueError, match="divisibility.*bind symbolic dimensions"):
        shard_layout_local_shape(layout, require_static=False)
