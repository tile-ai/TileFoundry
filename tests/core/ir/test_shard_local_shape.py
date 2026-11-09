"""``shard_layout_local_shape`` — which mesh axes divide a layout dim.

The plain one-Split-per-axis case is asserted on every sharded op type
(``tests/ops`` compares local extents). What is kept here is the two ways the
answer is *not* "layout shape // mesh shape": several mesh axes splitting one
layout dim, and mesh axes that own no layout dim at all. Getting either wrong
sizes a register allocation.
"""

from __future__ import annotations

import pytest

from tilefoundry import services
from tilefoundry.ir import expr, types

DIVIDED = [
    pytest.param(
        types.ShardLayout(
            layout=types.Layout(shape=(128,), strides=(1,)),
            attrs=(types.Split(0), types.Split(0)),
            mesh=types.Mesh(
                (types.Topology("thread", 4 * 32),),
                types.Layout(shape=(4, 32), strides=(32, 1)),
                names=("y", "t"),
            ),
        ),
        (1,),
        id="two_mesh_axes_on_one_dim",
    ),
    pytest.param(
        types.ShardLayout(
            layout=types.Layout(shape=(4,), strides=(1,)),
            attrs=(types.Broadcast(), types.Split(0)),
            mesh=types.Mesh(
                (types.Topology("thread", 2 * 4),),
                types.Layout(shape=(2, 4), strides=(4, 1)),
                names=("x", "t"),
            ),
        ),
        (1,),
        id="broadcast_divides_nothing",
    ),
    pytest.param(
        types.ShardLayout(
            layout=types.Layout(shape=(8,), strides=(1,)),
            attrs=(types.Partial(),),
            mesh=types.Mesh(
                (types.Topology("thread", 4),),
                types.Layout(shape=(4,), strides=(1,)),
                names=("t",),
            ),
        ),
        (8,),
        id="partial_divides_nothing",
    ),
]


@pytest.mark.parametrize(("sharded", "expected"), DIVIDED)
def test_owning_mesh_axis_divides_dim(sharded, expected) -> None:
    assert types.shard_layout_local_shape(sharded) == expected


_N = expr.DimVar("local_n", 1, 64)
_M = expr.DimVar("mesh_n", 1, 64)


def _symbolic_layout(axis_extent, mesh_extent, *, split: bool) -> types.ShardLayout:
    return types.ShardLayout(
        layout=types.Layout(shape=(axis_extent, 8), strides=None),
        attrs=(types.Split(0) if split else types.Broadcast(),),
        mesh=types.Mesh(
            (types.Topology("cta", mesh_extent),),
            types.Layout(shape=(mesh_extent,), strides=(1,)),
        ),
    )


def test_unconsumed_symbolic_axis_reaches_typeinfer() -> None:
    layout = _symbolic_layout(_N, 8, split=False)

    assert types.shard_layout_local_shape(layout, require_static=False) == (_N, 8)
    with pytest.raises(ValueError, match="not static after sharding"):
        types.shard_layout_local_shape(layout)


@pytest.mark.parametrize("require_static", [False, True], ids=["typeinfer", "lowering"])
@pytest.mark.parametrize(
    ("extent", "expected", "canonical"),
    [
        pytest.param(_N, (1, 8), False, id="equal"),
        pytest.param(services.simplify_dim("mul", (_N, 64)), (64, 8), False, id="symbolic_factor_left"),
        pytest.param(services.simplify_dim("mul", (64, _N)), (64, 8), False, id="symbolic_factor_right"),
        pytest.param(_N * 64 + 1, None, False, id="not_divisible"),
        pytest.param(_N * 64, (1, 1, 8), True, id="canonical_exact"),
        pytest.param(_N * 128, (1, 1, 2, 8), True, id="canonical_residual"),
        pytest.param(_N * 64 + 1, None, True, id="canonical_undecidable"),
    ],
)
def test_symbolic_split_local_extent_decidable(
    require_static, extent, expected, canonical
) -> None:
    if canonical:
        mesh = types.Mesh(
            (types.Topology("thread", _N * 64),),
            types.Layout(shape=(_N, 64), strides=(64, 1)),
        )
        if expected is None:
            with pytest.raises(
                ValueError, match="dynamic; cannot factorize across multiple mesh axes"
            ):
                types.ShardLayout.from_logical(
                    (extent, 8),
                    mesh,
                    (types.Split(0), types.Split(0)),
                    storage=expr.StorageKind.GMEM,
                    owners=expr.StorageOwners((), {}),
                )
            return
        layout = types.ShardLayout.from_logical(
            (extent, 8),
            mesh,
            (types.Split(0), types.Split(0)),
            storage=expr.StorageKind.GMEM,
            owners=expr.StorageOwners((), {}),
        )
    else:
        layout = _symbolic_layout(extent, _N, split=True)

    if expected is None:
        with pytest.raises(ValueError, match="do not have a decidable divisibility relation"):
            types.shard_layout_local_shape(layout, require_static=require_static)
    else:
        assert types.shard_layout_local_shape(layout, require_static=require_static) == expected


def test_unresolved_symbolic_split_is_rejected() -> None:
    layout = _symbolic_layout(_N, _M, split=True)

    with pytest.raises(ValueError, match="divisibility.*bind symbolic dimensions"):
        types.shard_layout_local_shape(layout, require_static=False)
