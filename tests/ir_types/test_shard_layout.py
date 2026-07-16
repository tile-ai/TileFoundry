"""Spec 003 shard primitives smoke coverage."""

from __future__ import annotations

import pytest

from tilefoundry.ir.types.shard import (
    B,
    Broadcast,
    ComposedLayout,
    Layout,
    Mesh,
    P,
    Partial,
    S,
    ShardLayout,
    Split,
    Topology,
)
from tilefoundry.ir.types.shard import layout_algebra as la


def _mesh() -> Mesh:
    return Mesh(topology=Topology("cta", 128), layout=Layout(shape=(128,), strides=(1,)))


def test_s_returns_split_with_axis():
    a = S(2)
    assert isinstance(a, Split)
    assert a.axis == 2


def test_p_returns_partial_with_reduction():
    a = P("sum")
    assert isinstance(a, Partial)
    assert a.reduction == "sum"


def test_b_returns_broadcast():
    assert isinstance(B(), Broadcast)


def test_shard_layout_keeps_layout_and_mesh():
    layout = Layout(shape=(4, 8), strides=(8, 1))
    sl = ShardLayout(layout=layout, attrs=(S(0), S(1)), mesh=_mesh())
    assert sl.layout is layout
    assert sl.mesh == _mesh()


def test_composed_layout_nests_layouts():
    inner = Layout(shape=(8,), strides=(1,))
    outer = Layout(shape=(4,), strides=(1,))
    c = ComposedLayout(inner=inner, offset=0, outer=outer)
    assert c.inner is inner
    assert c.outer is outer


# --- layout algebra: CuTe left/right inverse (ported, ``layout_algebra``) ----

_INJECTIVE_LAYOUTS = [
    Layout(shape=(4, 32), strides=(32, 1)),
    Layout(shape=(4, 32), strides=(1, 4)),
    Layout(shape=(128,), strides=(2,)),
    Layout(shape=(8,), strides=(3,)),
    Layout(shape=(3, 5), strides=(5, 1)),
    Layout(shape=(16,), strides=(1,)),
]


@pytest.mark.parametrize("layout", _INJECTIVE_LAYOUTS, ids=lambda l: f"{l.shape}:{l.strides}")
def test_left_inverse_round_trip(layout):
    """``left_inverse(L)(L(c)) == c`` for every domain coord (CuTe contract)."""
    inv = la.left_inverse(layout)
    for c in range(la.size(layout)):
        assert la.apply(inv, la.apply(layout, c)) == c


@pytest.mark.parametrize("layout", _INJECTIVE_LAYOUTS, ids=lambda l: f"{l.shape}:{l.strides}")
def test_right_inverse_round_trip(layout):
    """``L(right_inverse(L)(i)) == i`` over the right-inverse domain."""
    rinv = la.right_inverse(layout)
    for i in range(la.size(rinv)):
        assert la.apply(layout, la.apply(rinv, i)) == i


def test_left_inverse_explicit_values():
    # Hand-verified against the pure-Python CuTe reference (pycute).
    assert la.left_inverse(Layout(shape=(128,), strides=(2,))) == Layout(
        shape=(2, 128), strides=(128, 1)
    )
    assert la.left_inverse(Layout(shape=(4, 32), strides=(32, 1))) == Layout(
        shape=(32, 4), strides=(4, 1)
    )


# --- mesh execution scope: contains / project (all via one helper) ----------


def _check_scope(scope, n_threads, member, coord):
    """Brute-force ``la.contains`` / ``la.project`` over the whole capacity
    against an independent membership + coordinate oracle. Every case routes
    through the same ``la.contains`` / ``la.project`` (no per-case predicate)."""
    for t in range(n_threads):
        assert la.contains(scope, t) is member(t), f"contains t={t}"
        if member(t):
            assert la.project(scope, t) == coord(t), f"project t={t}"


def test_contains_contiguous_offset_view():
    # offset=128 over a (4,32) warp/lane layout: threads 128..255 active.
    scope = ComposedLayout(inner=None, offset=128, outer=Layout(shape=(4, 32), strides=(32, 1)))
    _check_scope(
        scope, 256,
        member=lambda t: 128 <= t < 256,
        coord=lambda t: ((t - 128) // 32, (t - 128) % 32),
    )


def test_contains_stride2_odd():
    # cta[1::2]: offset=1, stride=2 over 256 threads -> odd ids only.
    scope = ComposedLayout(inner=None, offset=1, outer=Layout(shape=(128,), strides=(2,)))
    _check_scope(scope, 256, member=lambda t: t % 2 == 1, coord=lambda t: ((t - 1) // 2,))


def test_contains_column_view():
    # column y==5 of a (4,32) row-major grid: image = 5 + 32*x, x in [0,4).
    scope = ComposedLayout(inner=None, offset=5, outer=Layout(shape=(4,), strides=(32,)))
    _check_scope(
        scope, 128,
        member=lambda t: t % 32 == 5 and t // 32 < 4,
        coord=lambda t: ((t - 5) // 32,),
    )


def test_contains_rectangle_view():
    # rectangle [2:5, 3:7] of an 8x8 row-major grid.
    Y, x0, x1, y0, y1 = 8, 2, 5, 3, 7
    scope = ComposedLayout(
        inner=None, offset=x0 * Y + y0, outer=Layout(shape=(x1 - x0, y1 - y0), strides=(Y, 1))
    )
    _check_scope(
        scope, 8 * Y,
        member=lambda t: x0 <= t // Y < x1 and y0 <= t % Y < y1,
        coord=lambda t: (t // Y - x0, t % Y - y0),
    )


def test_warp_groups_disjoint():
    # cta[0::2] (WG0) and cta[1::2] (WG1): images partition the capacity.
    wg0 = ComposedLayout(inner=None, offset=0, outer=Layout(shape=(128,), strides=(2,)))
    wg1 = ComposedLayout(inner=None, offset=1, outer=Layout(shape=(128,), strides=(2,)))
    for t in range(256):
        assert la.contains(wg0, t) != la.contains(wg1, t)
        assert la.contains(wg0, t) or la.contains(wg1, t)


def test_non_projectable_inner_rejected():
    # A non-identity inner is not an admissible execution scope in v1.
    scope = ComposedLayout(
        inner=Layout(shape=(4,), strides=(7,)), offset=0, outer=Layout(shape=(4,), strides=(1,))
    )
    with pytest.raises(la.NotProjectable):
        la.project(scope, 0)


@pytest.mark.parametrize(
    "outer",
    [
        Layout(shape=(2, 2), strides=(1, 1)),  # c0 and c1 collide in the image
        Layout(shape=(4,), strides=(0,)),      # broadcast: all coords -> 0
        Layout(shape=(2, 2), strides=(2, 2)),  # overlapping strides collide
    ],
    ids=lambda l: f"{l.shape}:{l.strides}",
)
def test_non_injective_outer_rejected(outer):
    # A non-projectable outer must fail closed, not pick a colliding representative.
    assert not la.is_inverse_projectable(outer)
    scope = ComposedLayout(inner=None, offset=0, outer=outer)
    with pytest.raises(la.NotProjectable):
        la.contains(scope, 0)
    with pytest.raises(la.NotProjectable):
        la.project(scope, 0)


@pytest.mark.parametrize(
    "bad",
    [
        Layout(shape=(2, 2), strides=(1, 1)),
        Layout(shape=(4,), strides=(0,)),
        Layout(shape=(2, 2), strides=(2, 2)),
        Layout(shape=(5, 3), strides=(3, 8)),  # injective but not CuTe-projectable
    ],
    ids=lambda l: f"{l.shape}:{l.strides}",
)
def test_public_layout_inverse_fails_closed(bad):
    # Public left/right inverse must not hand out a bogus inverse for a layout
    # that is_inverse_projectable rejects.
    assert not la.is_inverse_projectable(bad)
    with pytest.raises(la.NotProjectable):
        la.left_inverse(bad)
    with pytest.raises(la.NotProjectable):
        la.right_inverse(bad)


def test_composed_left_inverse_round_trip():
    # left_inverse(ComposedLayout) recovers the domain coord across the image.
    scope = ComposedLayout(inner=None, offset=128, outer=Layout(shape=(4, 32), strides=(32, 1)))
    inv = la.left_inverse(scope)
    assert isinstance(inv, ComposedLayout)
    for c in range(la.size(scope.outer)):
        t = la.image(scope, c)
        assert la._apply_any(inv, t) == c
